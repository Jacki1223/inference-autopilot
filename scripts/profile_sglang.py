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
import time
import urllib.request
from pathlib import Path
from typing import Any

from autotune import (
    command_manifest,
    enable_child_subreaper,
    execution_errors,
    increase_benchmark_request_count,
    reap_exited_children,
    sanitized_environment,
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

NSYS_DETAILED_REPORTS = NSYS_ROUTING_REPORTS + (
    "cuda_gpu_trace",
    "cuda_kern_exec_sum",
)


def run_command(command: list[str], *, cwd: str | None = None, timeout: float = 120) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except FileNotFoundError:
        return {"command": command, "returncode": 127, "stderr": "command not found", "stdout": ""}
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stderr": f"timeout after {exc.timeout} seconds",
            "stdout": exc.stdout or "",
        }


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
        "top_cuda_apis": top_apis,
        "secondary_bottlenecks": secondary,
        "profiling_run_performance_comparable": False,
        "evidence_quality": (
            "nsys_cuda_timeline" if reports.get("cuda_gpu_trace")
            else "nsys_routing_summaries"
        ),
    }


def collect_stats(
    report: Path, output_dir: Path, *, include_detailed_timeline: bool = False
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
    for report_name in report_names:
        queried_source = source
        result = run_command(
            [
                "nsys", "stats", "--report", report_name,
                "--format", "csv", "--output", "-", str(queried_source),
            ],
            timeout=300,
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
    command: list[str], spec: dict[str, Any], output_dir: Path, env: dict[str, str]
) -> tuple[list[str], list[dict[str, Any]]]:
    """Warm the serving stack and expand request count before tracing it."""
    benchmark = list(command)
    raw_path = output_dir / "preflight-result.jsonl"
    result_index = benchmark.index("--output-file") + 1
    benchmark[result_index] = str(raw_path)
    minimum_duration = float(spec["benchmark"].get("min_measurement_seconds", 0))
    attempts: list[dict[str, Any]] = []
    for attempt_index in range(1, 6):
        result = subprocess.run(
            benchmark, cwd=spec["repository"], env=env, capture_output=True,
            text=True, timeout=float(spec["execution"].get("benchmark_timeout_sec", 1800)), check=False,
        )
        (output_dir / f"preflight-{attempt_index:02d}.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        if result.returncode != 0:
            raise RuntimeError(f"profile preflight benchmark exited with code {result.returncode}")
        summary = summarize_jsonl(raw_path, spec)
        attempts.append({
            "attempt": attempt_index,
            "num_prompts": int(benchmark[benchmark.index("--num-prompts") + 1]),
            "measurement_validity": summary["measurement_validity"],
        })
        if summary["measurement_validity"]["duration_gate_passed"]:
            return benchmark, attempts
        if attempt_index == 5:
            break
        short_path = output_dir / f"preflight-short-attempt-{attempt_index}.jsonl"
        raw_path.replace(short_path)
        attempts[-1]["result_file"] = short_path.name
        duration = summary["measurement_validity"].get("duration_sec") or 0
        current_prompts = int(benchmark[benchmark.index("--num-prompts") + 1])
        multiplier = max(2.0, (minimum_duration / duration) * 1.2) if duration > 0 else 2.0
        increase_benchmark_request_count(benchmark, max(current_prompts + 1, int(math.ceil(current_prompts * multiplier))))
    raise RuntimeError("profile preflight did not reach the minimum steady-state measurement duration")


def capture_benchmark_with_metrics(
    command: list[str], spec: dict[str, Any], output_dir: Path, env: dict[str, str]
) -> tuple[int, list[dict[str, Any]]]:
    """Capture workload-time metrics instead of relying on a terminal queue sample."""
    host = spec["execution"].get("host", "127.0.0.1")
    port = int(spec["execution"].get("port", 30000))
    samples: list[dict[str, Any]] = []
    with (output_dir / "benchmark.log").open("w", encoding="utf-8") as bench_log:
        process = subprocess.Popen(command, cwd=spec["repository"], env=env, stdout=bench_log,
                                   stderr=subprocess.STDOUT, text=True)
        started = time.monotonic()
        deadline = started + float(spec["execution"].get("benchmark_timeout_sec", 1800))
        while process.poll() is None:
            if time.monotonic() >= deadline:
                process.kill()
                process.wait(timeout=10)
                raise RuntimeError("profile benchmark timed out")
            sample = collect_prometheus_sample(f"http://{host}:{port}/metrics")
            sample.pop("raw", None)
            samples.append({"elapsed_sec": round(time.monotonic() - started, 3), **sample})
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
    attempts: list[dict[str, Any]] = []
    for endpoint in ("/get_server_info", "/server_info"):
        response = collect_json(f"http://{host}:{port}{endpoint}")
        capacity = runtime_admission_capacity(response)
        attempts.append({
            "endpoint": endpoint,
            "available": response.get("available", False),
            "capacity": capacity,
            "error": response.get("error"),
        })
        if capacity is not None:
            return {"available": True, "source": endpoint, "max_running_requests": capacity, "attempts": attempts}
    return {"available": False, "source": None, "max_running_requests": None, "attempts": attempts}


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


def stop_profile_process_group(process: subprocess.Popen[Any], timeout: float) -> dict[str, Any]:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"method": "process_group_already_exited", "returncode": process.poll()}
    deadline = time.monotonic() + timeout
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
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupted)
    return previous


def restore_profile_interrupt_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def run_profile(
    spec: dict[str, Any], output_dir: Path, *, include_detailed_timeline: bool = False
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
        wait_port_available(host, port, float(spec["execution"].get("shutdown_timeout_sec", 60)))
        with (output_dir / "server-nsys.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                commands["server"], cwd=spec["repository"], env=env,
                stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
            ready, detail = wait_ready(
                f"http://{host}:{port}/v1/models", process,
                float(spec["execution"].get("startup_timeout_sec", 1200)),
            )
            if not ready:
                raise RuntimeError(detail or "profile server failed health check")
            startup_capacity = startup_server_capacity(host, port)
            write_json(output_dir / "startup-capacity.json", startup_capacity)
            if spec["benchmark"].get("unbounded_concurrency", False):
                current_prompts = int(commands["benchmark"][commands["benchmark"].index("--num-prompts") + 1])
                group_floor = int(spec["benchmark"].get("gsp_num_groups", 1)) * 4
                capacity = startup_capacity.get("max_running_requests")
                target_prompts = max(current_prompts, group_floor)
                if isinstance(capacity, int) and capacity > 0:
                    target_prompts = max(target_prompts, capacity * 5)
                effective_prompts = increase_benchmark_request_count(commands["benchmark"], target_prompts)
                startup_capacity.update({
                    "initial_request_policy": "five_admission_waves" if capacity else "minimum_prefix_reuse_fallback",
                    "requested_num_prompts": target_prompts,
                    "effective_num_prompts": effective_prompts,
                })
                write_json(output_dir / "startup-capacity.json", startup_capacity)
            profile_benchmark, preflight_attempts = steady_state_preflight(
                commands["benchmark"][:commands["benchmark"].index("--profile")], spec, output_dir, env
            )
            capture_output_index = commands["benchmark"].index("--output-file") + 1
            capture_prompt_index = commands["benchmark"].index("--num-prompts") + 1
            profile_prompt_index = profile_benchmark.index("--num-prompts") + 1
            commands["benchmark"][capture_output_index] = str(output_dir / "result.jsonl")
            commands["benchmark"][capture_prompt_index] = profile_benchmark[profile_prompt_index]
            status["state"] = "profiling"
            returncode, prometheus_samples = capture_benchmark_with_metrics(
                commands["benchmark"], spec, output_dir, env
            )
            if returncode != 0:
                raise RuntimeError(f"profile benchmark exited with code {returncode}")
            summary = summarize_jsonl(output_dir / "result.jsonl", spec)
            if not summary["measurement_validity"]["duration_gate_passed"]:
                raise RuntimeError("Nsight Systems capture did not meet the steady-state duration gate")
            write_json(output_dir / "benchmark-summary.json", summary)
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
                process, float(spec["execution"].get("shutdown_timeout_sec", 60))
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
                report, output_dir, include_detailed_timeline=include_detailed_timeline
            )
            diagnosis = analyze_reports(parsed)
            status.update({"state": "completed", "report": str(report), "stats": report_status})
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
            }
            write_json(output_dir / "nsys-diagnosis.json", final)
            return final
    finally:
        if process is not None and "shutdown" not in status:
            status["shutdown"] = stop_profile_process_group(
                process, float(spec["execution"].get("shutdown_timeout_sec", 60))
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


def diagnose_existing(profile_dir: Path, *, include_detailed_timeline: bool = False) -> dict[str, Any]:
    if not profile_dir.is_absolute() or not profile_dir.is_dir():
        raise ValueError("profile_dir must be an existing absolute directory")
    report = next(iter(sorted(profile_dir.glob("*.nsys-rep"))), None)
    if report is None:
        raise ValueError("profile_dir contains no .nsys-rep artifact")
    parsed, statuses = collect_stats(
        report, profile_dir, include_detailed_timeline=include_detailed_timeline
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
    final = {
        "schema_version": 1,
        "run_dir": str(profile_dir),
        "report": str(report),
        "status": {"state": "completed", "report": str(report), "stats": statuses, "reused": True},
        "tool": nsys_inventory(),
        "stats": statuses,
        "diagnosis": analyze_reports(parsed),
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
