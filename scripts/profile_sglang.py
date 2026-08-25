#!/usr/bin/env python3
"""Capture and analyze a bounded SGLang baseline with Nsight Systems."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from autotune import (
    command_manifest,
    enable_child_subreaper,
    execution_errors,
    increase_benchmark_request_count,
    latest_log_message,
    reap_exited_children,
    sanitized_environment,
    set_cli_option,
    startup_failure_detail,
    summarize_jsonl,
    wait_port_available,
    wait_ready,
    write_json,
)
from inferopt import dump_json, load_json
from sglang_runtime import summarize_sglang_log


NSYS_ROUTING_REPORTS = (
    "cuda_gpu_kern_sum",
    "cuda_api_sum",
    "cuda_gpu_mem_time_sum",
)


def emit_progress(
    progress: Callable[[dict[str, Any]], None] | None,
    phase: str, message: str, *, completed: int | None = None,
    total: int | None = None,
) -> None:
    if progress is not None:
        progress({
            "phase": phase, "message": message,
            "completed": completed, "total": total,
        })

NSYS_DETAILED_REPORTS = NSYS_ROUTING_REPORTS + (
    "cuda_gpu_trace",
    "cuda_kern_exec_sum",
)


def run_command(
    command: list[str], *, cwd: str | None = None, timeout: float = 120,
    progress: Callable[[dict[str, Any]], None] | None = None,
    progress_label: str | None = None,
) -> dict[str, Any]:
    try:
        with tempfile.TemporaryDirectory(prefix="inferopt-command-") as temporary:
            stdout_path = Path(temporary) / "stdout"
            stderr_path = Path(temporary) / "stderr"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                process = subprocess.Popen(
                    command, cwd=cwd, stdout=stdout, stderr=stderr, text=True
                )
                started = time.monotonic()
                last_heartbeat = started
                while process.poll() is None:
                    now = time.monotonic()
                    if now - started >= timeout:
                        process.kill()
                        process.wait(timeout=10)
                        return {
                            "command": command, "returncode": 124,
                            "stderr": f"timeout after {timeout} seconds", "stdout": "",
                        }
                    if progress is not None and now - last_heartbeat >= 30:
                        emit_progress(
                            progress, "command",
                            f"{progress_label or Path(command[0]).name} still running; "
                            f"elapsed={now - started:.0f}s",
                        )
                        last_heartbeat = now
                    time.sleep(0.2)
            return {
                "command": command,
                "returncode": process.returncode,
                "stdout": stdout_path.read_text(encoding="utf-8", errors="replace"),
                "stderr": stderr_path.read_text(encoding="utf-8", errors="replace"),
            }
    except FileNotFoundError:
        return {"command": command, "returncode": 127, "stderr": "command not found", "stdout": ""}


def nsys_inventory() -> dict[str, Any]:
    version = run_command(["nsys", "--version"], timeout=15)
    devices = run_command(["nsys", "profile", "--gpu-metrics-devices=help"], timeout=30)
    ncu_path = next((candidate for candidate in (
        shutil.which("ncu"),
        "/opt/nvidia/nsight-compute/2025.2.1/ncu",
        "/usr/local/cuda/bin/ncu",
        "/usr/local/cuda-12.9/bin/ncu",
    ) if candidate and Path(candidate).is_file()), None)
    ncu_version = run_command([ncu_path, "--version"], timeout=15) if ncu_path else {
        "returncode": 127, "stdout": "", "stderr": "ncu not found"
    }
    ncu_probe = run_command([ncu_path, "--query-metrics", "--devices", "0"], timeout=30) if ncu_path else ncu_version
    ncu_text = ncu_probe["stdout"] + ncu_probe["stderr"]
    return {
        "available": version["returncode"] == 0,
        "version": (version["stdout"] or version["stderr"]).strip(),
        "gpu_metrics_available": devices["returncode"] == 0 and "cuda-visible" in (
            devices["stdout"] + devices["stderr"]
        ),
        "gpu_metrics_help": (devices["stdout"] + devices["stderr"])[-4000:],
        "ncu": {
            "available": ncu_version["returncode"] == 0,
            "path": ncu_path,
            "version": (ncu_version["stdout"] or ncu_version["stderr"]).strip(),
            "performance_counter_access": ncu_probe["returncode"] == 0 and "ERR_NVGPUCTRPERM" not in ncu_text,
            "permission_error": "ERR_NVGPUCTRPERM" in ncu_text,
            "probe_detail": ncu_text[-1000:],
        },
    }


def roofline_diagnosis(
    ncu: dict[str, Any], top_kernel: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Classify a hotspot only from NCU counters, never from Nsys share alone."""
    if not ncu.get("available"):
        return {
            "status": "roofline_unavailable_ncu_missing",
            "reason": "Nsight Compute is not installed",
        }
    if not ncu.get("performance_counter_access"):
        return {
            "status": "roofline_unavailable_permission",
            "reason": "GPU performance-counter access is not enabled",
            "top_kernel": top_kernel,
            "next_step": "enable NVIDIA profiler permissions, then collect shape-matched NCU metrics",
        }
    if not metrics:
        return {
            "status": "roofline_pending_capture",
            "reason": "NCU is available but no shape-matched metrics were captured",
            "top_kernel": top_kernel,
        }
    achieved_flops = metrics.get("achieved_flops")
    achieved_bandwidth = metrics.get("achieved_bandwidth_bytes_s")
    peak_flops = metrics.get("peak_flops")
    peak_bandwidth = metrics.get("peak_bandwidth_bytes_s")
    intensity = (
        achieved_flops / achieved_bandwidth
        if isinstance(achieved_flops, (int, float))
        and isinstance(achieved_bandwidth, (int, float))
        and achieved_bandwidth > 0 else None
    )
    compute_util = (
        achieved_flops / peak_flops
        if isinstance(achieved_flops, (int, float))
        and isinstance(peak_flops, (int, float)) and peak_flops > 0 else None
    )
    bandwidth_util = (
        achieved_bandwidth / peak_bandwidth
        if isinstance(achieved_bandwidth, (int, float))
        and isinstance(peak_bandwidth, (int, float)) and peak_bandwidth > 0 else None
    )
    classification = (
        "memory_bound" if bandwidth_util is not None and bandwidth_util >= 0.7
        and (compute_util is None or compute_util < 0.6)
        else "compute_bound" if compute_util is not None and compute_util >= 0.7
        else "mixed_or_latency_bound"
    )
    routing = {
        "memory_bound": "consider KV/cache precision only with quality validation, page layout, larger batch pressure, and memory-traffic fusion",
        "compute_bound": "consider low-precision GEMM/MoE kernels, tile selection, Tensor Core utilization, and compile/fusion",
        "mixed_or_latency_bound": "inspect occupancy and warp stalls before changing serving parameters",
    }
    return {
        "status": "available",
        "top_kernel": top_kernel,
        "metrics": metrics,
        "arithmetic_intensity_flops_per_byte": intensity,
        "compute_utilization": compute_util,
        "memory_bandwidth_utilization": bandwidth_util,
        "classification": classification,
        "routing": routing[classification],
    }


def parse_ncu_roofline_csv(path: str | Path) -> dict[str, float]:
    """Parse a caller-provided shape-matched NCU CSV export.

    NCU metric names vary by architecture/version, so the caller supplies a
    small normalized CSV with ``metric,value`` rows. This avoids treating an
    arbitrary NCU export as a roofline result.
    """
    values: dict[str, float] = {}
    with Path(path).expanduser().open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            metric = row.get("metric")
            value = row.get("value")
            if not metric or value is None:
                continue
            try:
                values[metric] = float(str(value).replace(",", ""))
            except ValueError:
                continue
    required = {
        "achieved_flops", "achieved_bandwidth_bytes_s",
        "peak_flops", "peak_bandwidth_bytes_s",
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError("roofline metrics CSV is missing: " + ", ".join(missing))
    return values


def profile_commands(spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    trial = {"name": "profile-baseline", "config": spec["search"]["baseline"]}
    manifest = command_manifest(spec, trial, output_dir)
    nsys = nsys_inventory()
    report_base = output_dir / "baseline"
    server = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=process-tree",
        "--cpuctxsw=process-tree",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--kill=none",
        "--wait=all",
        "--force-overwrite=true",
        "--cuda-memory-usage=true",
    ]
    if nsys["gpu_metrics_available"]:
        server.extend(["--gpu-metrics-devices=cuda-visible", "--gpu-metrics-frequency=1000"])
    server.extend(["--output", str(report_base), *manifest["server"]])
    benchmark = [
        *manifest["benchmark"],
        "--profile",
        "--profile-activities",
        "CUDA_PROFILER",
        "--profile-output-dir",
        str(output_dir / "framework-profile"),
    ]
    return {"nsys": nsys, "server": server, "benchmark": benchmark, "report_base": str(report_base)}


def parse_csv(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header_index = next((index for index, line in enumerate(lines) if "," in line), None)
    if header_index is None:
        return []
    return [dict(row) for row in csv.DictReader(lines[header_index:])]


def numeric(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        cleaned = value.strip().replace(",", "").replace("%", "")
        try:
            return float(cleaned)
        except ValueError:
            continue
    return None


def row_name(row: dict[str, str]) -> str:
    for key in ("Name", "Kernel Name", "Operation", "API Name"):
        if row.get(key):
            return row[key]
    return ""


def row_time_pct(row: dict[str, str]) -> float:
    return numeric(row, "Time (%)", "Time %", "Percentage") or 0.0


def group_share(rows: list[dict[str, str]], patterns: tuple[str, ...]) -> float:
    return sum(
        row_time_pct(row)
        for row in rows
        if any(pattern in row_name(row).lower() for pattern in patterns)
    )


def kernel_category(name: str) -> str | None:
    """Assign one exclusive category to a kernel using semantic priority."""
    normalized = name.lower()
    categories = (
        ("communication", ("nccl", "allreduce", "all_reduce", "alltoall", "all_to_all", "all_gather")),
        ("moe", ("moe", "expert", "grouped", "deepgemm", "deepep")),
        # FlashAttention 3 kernels are CUTLASS device kernels, so attention
        # must be recognized before the generic CUTLASS/GEMM fallback.
        ("attention", ("attention", "flashattn", "flash_attn", "flash::", "flash_fwd", "fmha", "paged", "mla")),
        ("gemm", ("gemm", "matmul", "cutlass", "cublas")),
    )
    for category, patterns in categories:
        if any(pattern in normalized for pattern in patterns):
            return category
    return None


def kernel_family(name: str) -> str:
    """Collapse implementation variants into optimization-sized operator families."""
    normalized = name.lower()
    if "chunk_gated_delta_rule" in normalized or "fused_gdn" in normalized:
        return "gdn_delta_rule"
    if "recompute_w_u" in normalized or "chunk_fwd_kernel_o" in normalized:
        return "gdn_delta_rule"
    if any(token in normalized for token in ("flashattn", "flash_attn", "flash::", "fmha")):
        return "flash_attention"
    if any(token in normalized for token in ("nccl", "allreduce", "all_reduce", "alltoall", "all_to_all")):
        return "collective_communication"
    if any(token in normalized for token in ("moe", "expert", "grouped", "deepgemm", "deepep")):
        return "moe"
    return name.split("(", 1)[0].strip()[:160] or "unknown"


def kernel_family_summaries(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row_name(row)
        family = kernel_family(name)
        item = families.setdefault(family, {"name": family, "time_pct": 0.0, "instances": 0.0, "members": []})
        item["time_pct"] += row_time_pct(row)
        item["instances"] += numeric(row, "Instances", "Count") or 0.0
        if name and len(item["members"]) < 6:
            item["members"].append(name)
    return sorted(families.values(), key=lambda item: item["time_pct"], reverse=True)


def kernel_group_share(rows: list[dict[str, str]], category: str) -> float:
    return sum(
        row_time_pct(row)
        for row in rows
        if kernel_category(row_name(row)) == category
    )


def analyze_reports(reports: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    kernels = reports.get("cuda_gpu_kern_sum", [])
    apis = reports.get("cuda_api_sum", [])
    memops = reports.get("cuda_gpu_mem_time_sum", [])
    communication_pct = kernel_group_share(kernels, "communication")
    attention_pct = kernel_group_share(kernels, "attention")
    gemm_pct = kernel_group_share(kernels, "gemm")
    moe_pct = kernel_group_share(kernels, "moe")
    sync_pct = group_share(apis, ("synchronize", "eventquery", "streamwait", "eventsynchronize"))
    allocation_pct = group_share(apis, ("malloc", "free", "alloc"))
    kernel_total_ns = sum(numeric(row, "Total Time (ns)") or 0 for row in kernels)
    memop_total_ns = sum(numeric(row, "Total Time (ns)") or 0 for row in memops)
    memop_share_of_gpu_activity_pct = (
        memop_total_ns / (kernel_total_ns + memop_total_ns) * 100
        if kernel_total_ns + memop_total_ns > 0
        else None
    )
    intervals = []
    for row in reports.get("cuda_gpu_trace", []):
        start = numeric(row, "Start (ns)", "Start")
        duration = numeric(row, "Duration (ns)", "Duration")
        if start is not None and duration is not None and duration >= 0:
            intervals.append((start, start + duration))
    active_pct = None
    gap_pct = None
    if intervals:
        intervals.sort()
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        span = merged[-1][1] - merged[0][0]
        active = sum(end - start for start, end in merged)
        if span > 0:
            active_pct = active / span * 100
            gap_pct = 100 - active_pct

    launch_rows = reports.get("cuda_kern_exec_sum", [])
    avg_launch_latency_ns = None
    if launch_rows:
        weighted = []
        for row in launch_rows:
            latency = numeric(row, "AAvg (ns)", "Avg API Dur (ns)", "Avg API Duration (ns)")
            instances = numeric(row, "Count", "Instances") or 1
            if latency is not None:
                weighted.append((latency, instances))
        if weighted:
            avg_launch_latency_ns = sum(value * count for value, count in weighted) / sum(
                count for _, count in weighted
            )
    queue_latency_ns = []
    for row in launch_rows:
        latency = numeric(row, "QAvg (ns)", "Avg Queue Dur (ns)")
        instances = numeric(row, "Count", "Instances") or 1
        if latency is not None:
            queue_latency_ns.append((latency, instances))
    avg_queue_latency_ns = (
        sum(value * count for value, count in queue_latency_ns) / sum(count for _, count in queue_latency_ns)
        if queue_latency_ns else None
    )

    top_kernels = [
        {"name": row_name(row), "time_pct": row_time_pct(row), "instances": numeric(row, "Instances", "Count")}
        for row in kernels[:20]
    ]
    top_kernel_families = kernel_family_summaries(kernels)[:12]
    top_apis = [
        {"name": row_name(row), "time_pct": row_time_pct(row), "instances": numeric(row, "Num Calls", "Calls", "Count")}
        for row in apis[:15]
    ]

    secondary = []
    if sync_pct >= 20:
        secondary.append("cuda_synchronization")
    if moe_pct >= 25:
        secondary.append("moe_compute")
    if attention_pct >= 25:
        secondary.append("attention")
    if communication_pct >= 10:
        secondary.append("communication")
    if memop_share_of_gpu_activity_pct is not None and memop_share_of_gpu_activity_pct >= 15:
        secondary.append("memory_transfer")
    if active_pct is not None and active_pct < 65 and sync_pct >= 20:
        primary = "host_or_scheduler_stall"
    elif communication_pct >= 15:
        primary = "communication"
    elif memop_share_of_gpu_activity_pct is not None and memop_share_of_gpu_activity_pct >= 15:
        primary = "memory_transfer"
    elif sync_pct >= 20:
        primary = "cpu_gpu_synchronization"
    elif moe_pct >= 25:
        primary = "moe_compute"
    elif attention_pct >= 30:
        primary = "attention"
    elif gemm_pct >= 30:
        primary = "gemm_compute"
    else:
        primary = "mixed_gpu_compute"

    return {
        "primary_bottleneck": primary,
        "shares_pct": {
            "communication_kernels": communication_pct,
            "attention_kernels": attention_pct,
            "gemm_kernels": gemm_pct,
            "moe_kernels": moe_pct,
            "cuda_sync_apis": sync_pct,
            "cuda_allocation_apis": allocation_pct,
            "gpu_memops_within_activity": memop_share_of_gpu_activity_pct,
        },
        "avg_launch_latency_ns": avg_launch_latency_ns,
        "avg_kernel_queue_latency_ns": avg_queue_latency_ns,
        "gpu_timeline_active_pct": active_pct,
        "gpu_timeline_gap_pct": gap_pct,
        "top_kernels": top_kernels,
        "top_kernel_families": top_kernel_families,
        "top_cuda_apis": top_apis,
        "secondary_bottlenecks": secondary,
        "profiling_run_performance_comparable": False,
        "evidence_quality": (
            "nsys_cuda_timeline" if reports.get("cuda_gpu_trace")
            else "nsys_routing_summaries"
        ),
    }


def collect_stats(
    report: Path, output_dir: Path, *, include_detailed_timeline: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    """Collect Nsys routing summaries without repeatedly exporting a large trace.

    `cuda_gpu_trace` and `cuda_kern_exec_sum` can be hundreds of megabytes for
    long-context serving. They are useful for a manually requested deep dive,
    but are not required for the default parameter-routing decision. Once the
    first summary has materialized SQLite, subsequent reports query that file.
    """
    parsed: dict[str, list[dict[str, str]]] = {}
    statuses: dict[str, Any] = {}
    report_names = NSYS_DETAILED_REPORTS if include_detailed_timeline else NSYS_ROUTING_REPORTS
    sqlite_report = report.with_suffix(".sqlite")
    source = sqlite_report if sqlite_report.is_file() else report
    for index, report_name in enumerate(report_names, 1):
        emit_progress(
            progress, "stats",
            f"running nsys stats report {report_name} ({index}/{len(report_names)})",
            completed=index - 1, total=len(report_names),
        )
        queried_source = source
        result = run_command(
            [
                "nsys", "stats", "--report", report_name,
                "--format", "csv", "--output", "-", str(queried_source),
            ],
            timeout=300, progress=progress,
            progress_label=f"nsys stats {report_name}",
        )
        # The first invocation exports the .nsys-rep if necessary. Point later
        # summary queries at the stable SQLite artifact instead of re-exporting.
        if source == report and sqlite_report.is_file():
            source = sqlite_report
        (output_dir / f"{report_name}.stdout.csv").write_text(result["stdout"], encoding="utf-8")
        (output_dir / f"{report_name}.stderr.log").write_text(result["stderr"], encoding="utf-8")
        parsed[report_name] = parse_csv(result["stdout"]) if result["returncode"] == 0 else []
        statuses[report_name] = {
            "returncode": result["returncode"], "rows": len(parsed[report_name]),
            "source": str(queried_source),
        }
        emit_progress(
            progress, "stats",
            f"completed {report_name}: {len(parsed[report_name])} rows",
            completed=index, total=len(report_names),
        )
    return parsed, statuses


def collect_prometheus(url: str, output_dir: Path) -> dict[str, Any]:
    result = collect_prometheus_sample(url)
    if not result["available"]:
        return result
    raw = result.pop("raw")
    (output_dir / "prometheus.txt").write_text(raw, encoding="utf-8")
    return result


def collect_prometheus_sample(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except (OSError, TimeoutError) as exc:
        return {"available": False, "error": type(exc).__name__}
    selected = []
    keywords = ("queue", "running", "token", "cache", "retract", "util", "prefill", "decode", "expert")
    for line in raw.splitlines():
        if line.startswith("#") or not any(keyword in line.lower() for keyword in keywords):
            continue
        selected.append(line)
    return {"available": True, "selected_samples": selected[:500], "raw": raw}


def steady_state_preflight(
    command: list[str], spec: dict[str, Any], output_dir: Path, env: dict[str, str],
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Warm the serving stack and expand request count before tracing it."""
    benchmark = list(command)
    raw_path = output_dir / "preflight-result.jsonl"
    result_index = benchmark.index("--output-file") + 1
    benchmark[result_index] = str(raw_path)
    minimum_duration = float(spec["benchmark"].get("min_measurement_seconds", 0))
    # The profiled benchmark runs immediately after this preflight and can be
    # materially faster once shared prefixes have populated the radix cache.
    # Size the capture from a hot-cache observation and leave enough margin for
    # normal run-to-run variance; otherwise a cold preflight can pass while the
    # actual Nsight capture is shorter than the same validity gate.
    capture_duration_target = minimum_duration * 1.25
    require_hot_cache_observation = (
        spec["benchmark"].get("dataset_name") == "generated-shared-prefix"
    )
    hot_cache_primed = not require_hot_cache_observation
    attempts: list[dict[str, Any]] = []
    max_preflight_attempts = 2 if require_hot_cache_observation else 1
    for attempt_index in range(1, max_preflight_attempts + 1):
        prompts = int(benchmark[benchmark.index("--num-prompts") + 1])
        emit_progress(
            progress, "preflight",
            f"preflight attempt {attempt_index}/{max_preflight_attempts}; num_prompts={prompts}",
            completed=attempt_index - 1, total=max_preflight_attempts,
        )
        log_path = output_dir / f"preflight-{attempt_index:02d}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                benchmark, cwd=spec["repository"], env=env,
                stdout=log, stderr=subprocess.STDOUT, text=True,
            )
            preflight_started = time.monotonic()
            last_heartbeat = preflight_started
            timeout = float(spec["execution"].get("benchmark_timeout_sec", 1800))
            while process.poll() is None:
                now = time.monotonic()
                if now - preflight_started >= timeout:
                    process.kill()
                    process.wait(timeout=10)
                    raise RuntimeError("profile preflight benchmark timed out")
                if now - last_heartbeat >= 30:
                    emit_progress(
                        progress, "preflight",
                        f"preflight attempt {attempt_index} running; num_prompts={prompts}, "
                        f"elapsed={now - preflight_started:.0f}s",
                        completed=attempt_index - 1, total=max_preflight_attempts,
                    )
                    last_heartbeat = now
                time.sleep(1)
        if process.returncode != 0:
            raise RuntimeError(f"profile preflight benchmark exited with code {process.returncode}")
        summary = summarize_jsonl(raw_path, spec)
        duration = summary["measurement_validity"].get("duration_sec") or 0
        duration_target_passed = duration >= capture_duration_target
        cache_state = "hot_observation" if hot_cache_primed else "cache_priming"
        attempts.append({
            "attempt": attempt_index,
            "num_prompts": int(benchmark[benchmark.index("--num-prompts") + 1]),
            "measurement_validity": summary["measurement_validity"],
            "cache_state": cache_state,
            "capture_duration_target_sec": capture_duration_target,
            "capture_duration_target_passed": duration_target_passed,
        })
        if hot_cache_primed and duration_target_passed:
            return benchmark, attempts
        short_path = output_dir / f"preflight-short-attempt-{attempt_index}.jsonl"
        raw_path.replace(short_path)
        attempts[-1]["result_file"] = short_path.name
        current_prompts = int(benchmark[benchmark.index("--num-prompts") + 1])
        if not hot_cache_primed:
            # Do not size a shared-prefix capture from its first cold-cache
            # pass. Repeat the same request window once the cache is primed.
            hot_cache_primed = True
            next_prompts = current_prompts
        else:
            if duration <= 0:
                raise RuntimeError(
                    "profile preflight did not report a positive measurement duration"
                )
            multiplier = (
                max(1.25, (capture_duration_target / duration) * 1.1)
            )
            next_prompts = max(
                current_prompts + 1,
                int(math.ceil(current_prompts * multiplier)),
            )
        attempts[-1]["next_effective_num_prompts"] = increase_benchmark_request_count(
            benchmark, next_prompts
        )
        if hot_cache_primed and cache_state == "hot_observation":
            # The next benchmark is the actual Nsight capture. Avoid a third
            # unprofiled pass: the target includes 25% duration headroom and
            # the multiplier adds another 10% variance allowance.
            attempts[-1]["capture_window_estimated"] = True
            return benchmark, attempts
    raise RuntimeError("profile preflight could not produce a capture window")


def capture_benchmark_with_metrics(
    command: list[str], spec: dict[str, Any], output_dir: Path, env: dict[str, str],
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Capture workload-time metrics instead of relying on a terminal queue sample."""
    host = spec["execution"].get("host", "127.0.0.1")
    port = int(spec["execution"].get("port", 30000))
    samples: list[dict[str, Any]] = []
    with (output_dir / "benchmark.log").open("w", encoding="utf-8") as bench_log:
        process = subprocess.Popen(command, cwd=spec["repository"], env=env, stdout=bench_log,
                                   stderr=subprocess.STDOUT, text=True)
        started = time.monotonic()
        last_heartbeat = started
        deadline = started + float(spec["execution"].get("benchmark_timeout_sec", 1800))
        while process.poll() is None:
            if time.monotonic() >= deadline:
                process.kill()
                process.wait(timeout=10)
                raise RuntimeError("profile benchmark timed out")
            sample = collect_prometheus_sample(f"http://{host}:{port}/metrics")
            sample.pop("raw", None)
            samples.append({"elapsed_sec": round(time.monotonic() - started, 3), **sample})
            now = time.monotonic()
            if now - last_heartbeat >= 30:
                prompts = int(command[command.index("--num-prompts") + 1])
                emit_progress(
                    progress, "capture",
                    f"Nsys capture benchmark running; num_prompts={prompts}, "
                    f"elapsed={now - started:.0f}s",
                )
                last_heartbeat = now
            time.sleep(2.0)
        returncode = process.wait()
    write_json(output_dir / "prometheus-samples.json", samples)
    return returncode, samples


def summarize_prometheus_text(raw: str) -> dict[str, Any]:
    keywords = ("queue", "running", "token", "cache", "retract", "util", "prefill", "decode", "expert")
    selected = [
        line for line in raw.splitlines()
        if not line.startswith("#") and any(keyword in line.lower() for keyword in keywords)
    ]
    return {"available": True, "selected_samples": selected[:500]}


def collect_json(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return {"available": True, "value": json.loads(response.read().decode("utf-8"))}
    except (OSError, TimeoutError, json.JSONDecodeError) as exc:
        return {"available": False, "error": type(exc).__name__}


def runtime_admission_capacity(server_info: dict[str, Any]) -> int | None:
    """Find SGLang's resolved admission capacity in version-varying API JSON."""
    def visit(value: Any, key: str) -> int | None:
        if isinstance(value, dict):
            candidate = value.get(key)
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                return candidate
            for child in value.values():
                found = visit(child, key)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = visit(child, key)
                if found is not None:
                    return found
        return None

    if not server_info.get("available"):
        return None
    payload = server_info.get("value")
    # Newer SGLang revisions leave the configured max_running_requests null
    # under auto admission, but publish the scheduler's resolved per-DP value.
    return visit(payload, "max_running_requests") or visit(
        payload, "effective_max_running_requests_per_dp"
    )


def startup_server_capacity(host: str, port: int) -> dict[str, Any]:
    """Query both known SGLang info endpoints immediately after readiness."""
    def positive_integer(value: Any, keys: tuple[str, ...]) -> int | None:
        if isinstance(value, dict):
            for key in keys:
                candidate = value.get(key)
                if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                    return candidate
            for child in value.values():
                found = positive_integer(child, keys)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = positive_integer(child, keys)
                if found is not None:
                    return found
        return None

    attempts: list[dict[str, Any]] = []
    for endpoint in ("/get_server_info", "/server_info"):
        response = collect_json(f"http://{host}:{port}{endpoint}")
        capacity = runtime_admission_capacity(response)
        token_capacity = positive_integer(
            response.get("value"), ("max_total_num_tokens", "max_total_tokens")
        ) if response.get("available") else None
        attempts.append({
            "endpoint": endpoint,
            "available": response.get("available", False),
            "capacity": capacity,
            "max_total_tokens": token_capacity,
            "error": response.get("error"),
        })
        if capacity is not None:
            return {
                "available": True, "source": endpoint,
                "max_running_requests": capacity,
                "max_total_tokens": token_capacity,
                "attempts": attempts,
            }
    return {
        "available": False, "source": None, "max_running_requests": None,
        "max_total_tokens": None, "attempts": attempts,
    }


def bounded_profile_request_target(
    current_prompts: int, group_floor: int, admission_capacity: int | None,
    token_capacity: int | None, tokens_per_request: int,
) -> dict[str, Any]:
    """Size an unbounded diagnostic trace without inheriting a huge backlog."""
    practical_capacity = (
        max(1, int(token_capacity) // max(1, tokens_per_request))
        if isinstance(token_capacity, int) and token_capacity > 0 else None
    )
    target = max(current_prompts, group_floor)
    if isinstance(practical_capacity, int):
        pressure = practical_capacity
        if isinstance(admission_capacity, int) and admission_capacity > 0:
            pressure = min(pressure, admission_capacity)
        target = max(target, min(256, pressure * 3))
        policy = "three_practical_kv_waves_capped_256"
    elif isinstance(admission_capacity, int) and admission_capacity > 0:
        target = max(target, min(256, admission_capacity))
        policy = "bounded_admission_fallback_capped_256"
    else:
        policy = "configured_profile_window"
    return {
        "target_prompts": min(256, target),
        "practical_request_capacity": practical_capacity,
        "policy": policy,
    }


def bounded_profile_step_window(
    capture_prompts: int, output_tokens: int, practical_capacity: int | None,
) -> dict[str, int]:
    """Choose an auto-stopping CUDA-profiler window for SGLang 0.5.x."""
    output_tokens = max(1, int(output_tokens))
    steps = max(1, min(64, output_tokens))
    start = (
        min(int(practical_capacity), max(0, int(capture_prompts) // 2))
        if output_tokens >= 32
        and isinstance(practical_capacity, int)
        and practical_capacity > 0
        else 0
    )
    return {"start_step": start, "steps": steps}


def effective_server_config(server_info: dict[str, Any]) -> dict[str, Any]:
    """Locate the resolved ServerArgs object in version-dependent server-info JSON."""
    def visit(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            if "model_path" in value and "tp_size" in value and "chunked_prefill_size" in value:
                return value
            for child in value.values():
                found = visit(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = visit(child)
                if found is not None:
                    return found
        return None

    config = visit(server_info.get("value")) if server_info.get("available") else {}
    if not isinstance(config, dict):
        return {}
    graph_config = config.get("cuda_graph_config", {})
    if isinstance(graph_config, dict):
        for phase in ("decode", "prefill"):
            phase_config = graph_config.get(phase, {})
            key = f"cuda_graph_max_bs_{phase}"
            if config.get(key) is None and isinstance(phase_config, dict):
                resolved = phase_config.get("max_bs")
                if isinstance(resolved, int):
                    config[key] = resolved
    return config


def stop_profile_process_group(
    process: subprocess.Popen[Any], timeout: float,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"method": "process_group_already_exited", "returncode": process.poll()}
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    last_heartbeat = started
    while time.monotonic() < deadline:
        reap_exited_children(timeout=0.05)
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            if process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            return {"method": "sigterm_process_group", "returncode": process.poll()}
        now = time.monotonic()
        if now - last_heartbeat >= 30:
            emit_progress(
                progress, "cleanup",
                f"waiting for profiled process tree to exit; elapsed={now - started:.0f}s",
            )
            last_heartbeat = now
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    return {"method": "sigkill_process_group", "returncode": process.poll()}


def install_profile_interrupt_handlers() -> dict[int, Any]:
    """Turn terminal/session shutdown into a normal finally-path cleanup."""
    previous: dict[int, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise RuntimeError(f"profile interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        inherited = signal.getsignal(signum)
        # Respect nohup/systemd supervision. Replacing an inherited SIG_IGN
        # for SIGHUP makes a long remote experiment fail when its SSH control
        # connection is recycled even though the user explicitly detached it.
        if signum == signal.SIGHUP and inherited == signal.SIG_IGN:
            continue
        previous[signum] = inherited
        signal.signal(signum, interrupted)
    return previous


def restore_profile_interrupt_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def run_profile(
    spec: dict[str, Any], output_dir: Path, *, include_detailed_timeline: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    errors = execution_errors(spec)
    if errors:
        raise ValueError("; ".join(errors))
    if not spec.get("scope", {}).get("allow_profiling"):
        raise ValueError("scope.allow_profiling must be true")
    if output_dir.exists():
        raise ValueError("profile output directory already exists")
    output_dir.mkdir(parents=True, mode=0o700)
    commands = profile_commands(spec, output_dir)
    if not commands["nsys"]["available"]:
        raise RuntimeError("Nsight Systems is not available")
    write_json(output_dir / "spec.json", spec)
    write_json(output_dir / "commands.json", commands)
    enable_child_subreaper()
    env = sanitized_environment(spec)
    env["SGLANG_TORCH_PROFILER_DIR"] = str(output_dir / "framework-profile")
    host = spec["execution"].get("host", "127.0.0.1")
    port = int(spec["execution"].get("port", 30000))
    process: subprocess.Popen[Any] | None = None
    started = time.monotonic()
    accelerator_elapsed_sec: float | None = None
    status: dict[str, Any] = {"state": "starting"}
    startup_capacity: dict[str, Any] = {
        "available": False, "source": None, "max_running_requests": None, "attempts": [],
    }
    previous_handlers = install_profile_interrupt_handlers()
    try:
        emit_progress(progress, "server_launch", f"waiting for profile port {host}:{port}", completed=0, total=6)
        wait_port_available(host, port, float(spec["execution"].get("shutdown_timeout_sec", 60)))
        with (output_dir / "server-nsys.log").open("w", encoding="utf-8") as log:
            emit_progress(progress, "server_launch", "starting SGLang under Nsys", completed=0, total=6)
            process = subprocess.Popen(
                commands["server"], cwd=spec["repository"], env=env,
                stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
            ready, detail = wait_ready(
                f"http://{host}:{port}/v1/models", process,
                float(spec["execution"].get("startup_timeout_sec", 1200)),
                heartbeat=lambda elapsed: emit_progress(
                    progress, "server_startup",
                    f"{latest_log_message(output_dir / 'server-nsys.log', 'model load/KV allocation/CUDA Graph capture')}; "
                    f"elapsed={elapsed:.0f}s",
                    completed=0, total=6,
                ),
            )
            if not ready:
                raise RuntimeError(startup_failure_detail(
                    detail, output_dir / "server-nsys.log"
                ))
            startup_capacity = startup_server_capacity(host, port)
            emit_progress(
                progress, "server_ready",
                "profiled service is ready; resolving practical request capacity",
                completed=1, total=6,
            )
            write_json(output_dir / "startup-capacity.json", startup_capacity)
            if spec["benchmark"].get("unbounded_concurrency", False):
                current_prompts = int(commands["benchmark"][commands["benchmark"].index("--num-prompts") + 1])
                group_floor = int(spec["benchmark"].get("gsp_num_groups", 1)) * 4
                capacity = startup_capacity.get("max_running_requests")
                target_prompts = max(current_prompts, group_floor)
                token_capacity = startup_capacity.get("max_total_tokens")
                tokens_per_request = max(
                    1,
                    int(spec["benchmark"].get("random_input_len", 1))
                    + int(spec["benchmark"].get("random_output_len", 1)),
                )
                # max_running_requests is an admission ceiling, often 2048,
                # not the number of long-context requests that fit in KV. A
                # profile must apply pressure without inheriting a 5x 2048
                # confirmation backlog. Use three practical KV waves and cap
                # the diagnostic capture at 256 requests.
                sizing = bounded_profile_request_target(
                    current_prompts, group_floor, capacity, token_capacity,
                    tokens_per_request,
                )
                target_prompts = int(sizing["target_prompts"])
                effective_prompts = increase_benchmark_request_count(commands["benchmark"], target_prompts)
                startup_capacity.update({
                    "initial_request_policy": sizing["policy"],
                    "practical_request_capacity": sizing["practical_request_capacity"],
                    "tokens_per_request": tokens_per_request,
                    "requested_num_prompts": target_prompts,
                    "effective_num_prompts": effective_prompts,
                })
                write_json(output_dir / "startup-capacity.json", startup_capacity)
            profile_benchmark, preflight_attempts = steady_state_preflight(
                commands["benchmark"][:commands["benchmark"].index("--profile")],
                spec, output_dir, env, progress=progress,
            )
            emit_progress(
                progress, "preflight", "steady-state preflight completed",
                completed=2, total=6,
            )
            capture_output_index = commands["benchmark"].index("--output-file") + 1
            profile_prompt_index = profile_benchmark.index("--num-prompts") + 1
            commands["benchmark"][capture_output_index] = str(output_dir / "result.jsonl")
            capture_prompts = int(profile_benchmark[profile_prompt_index])
            # generated-shared-prefix derives the real sample count from
            # groups * prompts-per-group. Copying only --num-prompts leaves
            # that dataset at its original size even though logs claim the
            # capture was expanded.
            increase_benchmark_request_count(commands["benchmark"], capture_prompts)
            profile_window = bounded_profile_step_window(
                capture_prompts,
                int(spec["benchmark"].get("random_output_len", 1)),
                startup_capacity.get("practical_request_capacity"),
            )
            profile_steps = profile_window["steps"]
            profile_start_step = profile_window["start_step"]
            # SGLang 0.5.x can block the benchmark client indefinitely while
            # /stop_profile waits for cudaProfilerStop under Nsys. A bounded
            # automatic window avoids that endpoint and captures the
            # prefill-to-decode transition after the warm preflight.
            set_cli_option(commands["benchmark"], "--profile-start-step", profile_start_step)
            set_cli_option(commands["benchmark"], "--profile-steps", profile_steps)
            startup_capacity.update({
                "profile_start_step": profile_start_step,
                "profile_steps": profile_steps,
                "profile_stop_policy": "automatic_scheduler_steps",
            })
            write_json(output_dir / "startup-capacity.json", startup_capacity)
            capture_prompt_index = commands["benchmark"].index("--num-prompts") + 1
            status["state"] = "profiling"
            returncode, prometheus_samples = capture_benchmark_with_metrics(
                commands["benchmark"], spec, output_dir, env, progress=progress
            )
            if returncode != 0:
                raise RuntimeError(f"profile benchmark exited with code {returncode}")
            summary = summarize_jsonl(output_dir / "result.jsonl", spec)
            if not summary["measurement_validity"]["duration_gate_passed"]:
                validity = summary["measurement_validity"]
                raise RuntimeError(
                    "Nsight Systems capture did not meet the steady-state duration gate "
                    f"(observed {validity.get('duration_sec')} sec, required "
                    f"{validity.get('minimum_duration_sec')} sec, "
                    f"num_prompts={commands['benchmark'][capture_prompt_index]})"
                )
            write_json(output_dir / "benchmark-summary.json", summary)
            emit_progress(
                progress, "capture", "bounded CUDA capture completed; collecting runtime metadata",
                completed=3, total=6,
            )
            prometheus = collect_prometheus(f"http://{host}:{port}/metrics", output_dir)
            prometheus["capture_samples"] = prometheus_samples
            prometheus["preflight_attempts"] = preflight_attempts
            server_info = collect_json(f"http://{host}:{port}/get_server_info")
            if not server_info["available"]:
                server_info = collect_json(f"http://{host}:{port}/server_info")
            write_json(output_dir / "server-info.json", server_info)
            effective_config = effective_server_config(server_info)
            write_json(output_dir / "effective-server-config.json", effective_config)
            status["shutdown"] = stop_profile_process_group(
                process, float(spec["execution"].get("shutdown_timeout_sec", 60)),
                progress=progress,
            )
            emit_progress(
                progress, "export", "profiled server stopped; waiting for Nsys report export",
                completed=4, total=6,
            )
            # Nsight's report export can take minutes for a large CUDA trace
            # after the server process and all GPU work have stopped. Keep
            # wall elapsed for observability, but expose an accelerator-bound
            # duration so the controller does not spend its GPU-hour budget
            # on CPU-only CSV parsing.
            accelerator_elapsed_sec = time.monotonic() - started
            runtime_observations = summarize_sglang_log(
                (output_dir / "server-nsys.log").read_text(encoding="utf-8", errors="replace")
            )
            write_json(output_dir / "runtime-observations.json", runtime_observations)
            report = output_dir / "baseline.nsys-rep"
            if not report.is_file():
                candidates = sorted(output_dir.glob("baseline*.nsys-rep"))
                if not candidates:
                    raise RuntimeError("nsys did not produce an .nsys-rep artifact")
                report = candidates[0]
            parsed, report_status = collect_stats(
                report, output_dir,
                include_detailed_timeline=include_detailed_timeline,
                progress=progress,
            )
            diagnosis = analyze_reports(parsed)
            roofline = roofline_diagnosis(
                commands["nsys"].get("ncu", {}),
                diagnosis.get("top_kernels", [None])[0],
            )
            status.update({"state": "completed", "report": str(report), "stats": report_status})
            emit_progress(
                progress, "diagnosis", "Nsys reports parsed and bottleneck evidence generated",
                completed=6, total=6,
            )
            final = {
                "schema_version": 1,
                "run_dir": str(output_dir),
                "elapsed_sec": time.monotonic() - started,
                "accelerator_elapsed_sec": accelerator_elapsed_sec,
                "tool": commands["nsys"],
                "status": status,
                "benchmark": summary,
                "prometheus": prometheus,
                "server_info": server_info,
                "startup_capacity": startup_capacity,
                "effective_server_config": effective_config,
                "runtime_observations": runtime_observations,
                "diagnosis": diagnosis,
                "roofline": roofline,
            }
            write_json(output_dir / "nsys-diagnosis.json", final)
            return final
    finally:
        if process is not None and "shutdown" not in status:
            status["shutdown"] = stop_profile_process_group(
                process, float(spec["execution"].get("shutdown_timeout_sec", 60)),
                progress=progress,
            )
            if accelerator_elapsed_sec is None:
                accelerator_elapsed_sec = time.monotonic() - started
        status["accelerator_elapsed_sec"] = accelerator_elapsed_sec
        status["reaped_descendants"] = (
            reap_exited_children()
            if spec.get("execution", {}).get("process_wide_child_reaping", True)
            else []
        )
        status["elapsed_sec"] = time.monotonic() - started
        write_json(output_dir / "status.json", status)
        restore_profile_interrupt_handlers(previous_handlers)


def resolve_profile_spec(value: dict[str, Any]) -> dict[str, Any]:
    """Accept either an execution spec or an autopilot plan for direct use."""
    if isinstance(value.get("profiling"), dict) and isinstance(value["profiling"].get("spec"), dict):
        return value["profiling"]["spec"]
    return value


def diagnose_existing(
    profile_dir: Path, *, include_detailed_timeline: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    profile_dir = Path(profile_dir).expanduser()
    if not profile_dir.is_absolute() or not profile_dir.is_dir():
        raise ValueError("profile_dir must be an existing absolute directory")
    report = next(iter(sorted(profile_dir.glob("*.nsys-rep"))), None)
    if report is None:
        raise ValueError("profile_dir contains no .nsys-rep artifact")
    parsed, statuses = collect_stats(
        report, profile_dir, include_detailed_timeline=include_detailed_timeline,
        progress=progress,
    )
    server_info = load_json(profile_dir / "server-info.json") if (profile_dir / "server-info.json").is_file() else {}
    effective_config = effective_server_config(server_info)
    write_json(profile_dir / "effective-server-config.json", effective_config)
    prometheus_path = profile_dir / "prometheus.txt"
    prometheus = summarize_prometheus_text(prometheus_path.read_text(encoding="utf-8")) if prometheus_path.is_file() else {
        "available": False, "error": "artifact_missing"
    }
    server_log = profile_dir / "server-nsys.log"
    runtime_observations = summarize_sglang_log(
        server_log.read_text(encoding="utf-8", errors="replace")
    ) if server_log.is_file() else {"available": False, "error": "artifact_missing"}
    write_json(profile_dir / "runtime-observations.json", runtime_observations)
    diagnosis = analyze_reports(parsed)
    tool = nsys_inventory()
    final = {
        "schema_version": 1,
        "run_dir": str(profile_dir),
        "report": str(report),
        "status": {"state": "completed", "report": str(report), "stats": statuses, "reused": True},
        "tool": tool,
        "stats": statuses,
        "diagnosis": diagnosis,
        "roofline": roofline_diagnosis(
            tool.get("ncu", {}), diagnosis.get("top_kernels", [None])[0]
        ),
        "benchmark": load_json(profile_dir / "benchmark-summary.json") if (profile_dir / "benchmark-summary.json").is_file() else None,
        "prometheus": prometheus,
        "effective_server_config": effective_config,
        "runtime_observations": runtime_observations,
    }
    write_json(profile_dir / "nsys-diagnosis.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "diagnose"], nargs="?", default="run")
    parser.add_argument("--spec")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--detailed", action="store_true",
        help="also export costly GPU timeline and API-kernel execution reports",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.command == "run" and not args.yes:
        print("error: profiling requires --yes", file=sys.stderr)
        return 2
    try:
        output_dir = Path(args.output_dir).expanduser()
        if args.command == "diagnose":
            final = diagnose_existing(output_dir, include_detailed_timeline=args.detailed)
        else:
            if not args.spec:
                raise ValueError("run requires --spec")
            final = run_profile(
                resolve_profile_spec(load_json(args.spec)), output_dir,
                include_detailed_timeline=args.detailed,
            )
        dump_json(final, args.output)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
