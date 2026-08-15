#!/usr/bin/env python3
"""Private inference optimization utilities with no command execution surface."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


CANONICAL_ALIASES = {
    "request_throughput_rps": ["request_throughput", "request_throughput_rps"],
    "output_throughput_tps": ["output_throughput", "output_throughput_tps"],
    "total_throughput_tps": ["total_throughput", "total_token_throughput", "total_throughput_tps"],
    "request_goodput_rps": ["request_goodput", "request_goodput_rps"],
    "mean_ttft_ms": ["mean_ttft_ms", "mean_ttft"],
    "median_ttft_ms": ["median_ttft_ms", "median_ttft"],
    "p99_ttft_ms": ["p99_ttft_ms", "p99_ttft"],
    "mean_tpot_ms": ["mean_tpot_ms", "mean_tpot"],
    "median_tpot_ms": ["median_tpot_ms", "median_tpot"],
    "p99_tpot_ms": ["p99_tpot_ms", "p99_tpot"],
    "mean_itl_ms": ["mean_itl_ms", "mean_itl"],
    "median_itl_ms": ["median_itl_ms", "median_itl"],
    "p99_itl_ms": ["p99_itl_ms", "p99_itl"],
    "mean_e2e_latency_ms": ["mean_e2e_latency_ms", "mean_e2e_latency"],
    "p99_e2e_latency_ms": ["p99_e2e_latency_ms", "p99_e2e_latency"],
}

SLO_MAPPING = {
    "p99_ttft_ms": ("p99_ttft_ms", "max"),
    "p99_tpot_ms": ("p99_tpot_ms", "max"),
    "p99_itl_ms": ("p99_itl_ms", "max"),
    "p99_e2e_latency_ms": ("p99_e2e_latency_ms", "max"),
    "max_error_rate": ("error_rate", "max"),
    "min_request_throughput_rps": ("request_throughput_rps", "min"),
    "min_output_throughput_tps": ("output_throughput_tps", "min"),
    "min_request_goodput_rps": ("request_goodput_rps", "min"),
}

METRIC_DIRECTIONS = {
    "request_throughput_rps": "maximize",
    "output_throughput_tps": "maximize",
    "total_throughput_tps": "maximize",
    "request_goodput_rps": "maximize",
    "mean_ttft_ms": "minimize",
    "median_ttft_ms": "minimize",
    "p99_ttft_ms": "minimize",
    "mean_tpot_ms": "minimize",
    "median_tpot_ms": "minimize",
    "p99_tpot_ms": "minimize",
    "mean_itl_ms": "minimize",
    "median_itl_ms": "minimize",
    "p99_itl_ms": "minimize",
    "mean_e2e_latency_ms": "minimize",
    "p99_e2e_latency_ms": "minimize",
    "error_rate": "minimize",
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def dump_json(value: Any, output: str | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def safe_command(command: list[str], timeout: int = 10) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if not executable:
        return {"available": False}
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "error": type(exc).__name__}


def inventory() -> dict[str, Any]:
    commands = {
        "nvidia_smi_query": [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id,power.limit",
            "--format=csv,noheader,nounits",
        ],
        "nvidia_topology": ["nvidia-smi", "topo", "-m"],
        "rocm_info": ["rocminfo"],
        "cpu": ["sysctl", "-n", "machdep.cpu.brand_string"]
        if platform.system() == "Darwin"
        else ["lscpu", "-J"],
        "numa": ["numactl", "--hardware"],
        "infiniband": ["ibstat"],
        "cuda_compiler": ["nvcc", "--version"],
        "nsys": ["nsys", "--version"],
        "ncu": ["ncu", "--version"],
    }
    return {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "tools": {name: safe_command(cmd) for name, cmd in commands.items()},
    }


def validate_spec(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("name", "mode", "framework", "model", "workload", "slo", "objective", "budget", "scope"):
        if key not in spec:
            errors.append(f"missing required field: {key}")
    if spec.get("mode") not in {"dry_run", "shadow", "execute"}:
        errors.append("mode must be dry_run, shadow, or execute")
    for section in ("model", "workload", "slo", "objective", "budget", "scope"):
        if section in spec and not isinstance(spec[section], dict):
            errors.append(f"{section} must be an object")
    if isinstance(spec.get("model"), dict) and not spec["model"].get("path"):
        errors.append("model.path is required")
    if isinstance(spec.get("objective"), dict):
        objective_metric = spec["objective"].get("metric")
        if not objective_metric:
            errors.append("objective.metric is required")
        elif objective_metric not in METRIC_DIRECTIONS:
            errors.append(f"unsupported objective.metric: {objective_metric}")
        if spec["objective"].get("direction") not in {"maximize", "minimize"}:
            errors.append("objective.direction must be maximize or minimize")
        elif objective_metric in METRIC_DIRECTIONS and spec["objective"]["direction"] != METRIC_DIRECTIONS[objective_metric]:
            errors.append(f"objective.direction for {objective_metric} must be {METRIC_DIRECTIONS[objective_metric]}")
        for key in ("min_improvement_pct", "max_regression_pct"):
            value = spec["objective"].get(key, 0)
            if not isinstance(value, (int, float)) or value < 0:
                errors.append(f"objective.{key} must be non-negative")
        if objective_metric == "request_goodput_rps":
            goodput_slo = spec["objective"].get("goodput_slo")
            if not isinstance(goodput_slo, dict) or not goodput_slo:
                errors.append("objective.goodput_slo is required for request_goodput_rps")
            elif not any(key in goodput_slo for key in ("max_ttft_ms", "max_tpot_ms")):
                errors.append("objective.goodput_slo requires max_ttft_ms or max_tpot_ms")
    slo = spec.get("slo")
    if isinstance(slo, dict):
        for key, value in slo.items():
            if key not in SLO_MAPPING:
                errors.append(f"unsupported slo: {key}")
            elif not isinstance(value, (int, float)) or value < 0:
                errors.append(f"slo.{key} must be non-negative")
    budget = spec.get("budget", {})
    for key in ("max_trials", "max_gpu_hours", "max_wall_time_minutes"):
        if not isinstance(budget.get(key), (int, float)) or budget.get(key, 0) <= 0:
            errors.append(f"budget.{key} must be positive")
    scope = spec.get("scope", {})
    if not scope.get("output_dir"):
        errors.append("scope.output_dir is required")
    if scope.get("production") and spec.get("mode") != "dry_run":
        errors.append("production scope is limited to dry_run analysis by this MVP")
    if spec.get("mode") in {"shadow", "execute"} and not scope.get("allow_launch"):
        errors.append("shadow/execute mode requires scope.allow_launch=true")
    return errors


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def first_value(record: dict[str, Any], aliases: list[str]) -> float | None:
    for key in aliases:
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(flatten_numbers(item))
        return flattened
    return []


def request_goodput(record: dict[str, Any], spec: dict[str, Any] | None) -> float | None:
    if not spec:
        return None
    thresholds = spec.get("objective", {}).get("goodput_slo")
    duration = record.get("duration")
    ttfts = record.get("ttfts")
    itls = record.get("itls")
    if not isinstance(thresholds, dict) or not isinstance(duration, (int, float)) or duration <= 0:
        return None
    if not isinstance(ttfts, list) or not isinstance(itls, list) or len(ttfts) != len(itls):
        return None
    errors = record.get("errors")
    if not isinstance(errors, list) or len(errors) != len(ttfts):
        errors = [""] * len(ttfts)
    good = 0
    for ttft, request_itls, error in zip(ttfts, itls, errors):
        if error or not isinstance(ttft, (int, float)):
            continue
        tpot_values = flatten_numbers(request_itls)
        tpot_ms = mean(tpot_values) * 1000 if tpot_values else None
        ttft_limit = thresholds.get("max_ttft_ms")
        tpot_limit = thresholds.get("max_tpot_ms")
        ttft_pass = not isinstance(ttft_limit, (int, float)) or ttft * 1000 <= ttft_limit
        tpot_pass = not isinstance(tpot_limit, (int, float)) or (tpot_ms is not None and tpot_ms <= tpot_limit)
        good += int(ttft_pass and tpot_pass)
    return good / float(duration)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if isinstance(value, dict):
                records.append(value)
    if not records:
        raise ValueError(f"No JSON objects found in {path}")
    return records


def summarize(records: list[dict[str, Any]], spec: dict[str, Any] | None = None) -> dict[str, Any]:
    numeric: dict[str, list[float]] = {key: [] for key in CANONICAL_ALIASES}
    completed = 0.0
    failed = 0.0
    for record in records:
        for canonical, aliases in CANONICAL_ALIASES.items():
            value = first_value(record, aliases)
            if value is not None:
                numeric[canonical].append(value)
        completed += float(record.get("completed", record.get("num_prompts", 0)) or 0)
        errors = record.get("errors")
        detailed_failures = sum(bool(error) for error in errors) if isinstance(errors, list) else 0
        failed += float(record.get("failed", detailed_failures) or 0)
        goodput = request_goodput(record, spec)
        if goodput is not None:
            numeric["request_goodput_rps"].append(goodput)

    metrics = {key: mean(values) for key, values in numeric.items() if values}
    if completed + failed > 0:
        metrics["error_rate"] = failed / (completed + failed)

    latest = records[-1]
    for source, prefix in (("ttfts", "ttft"), ("tpots", "tpot"), ("itls", "itl")):
        values = latest.get(source)
        if isinstance(values, list):
            flat = flatten_numbers(values)
            if flat:
                # Detailed SGLang arrays are seconds; canonical latency is milliseconds.
                metrics[f"mean_{prefix}_ms"] = mean(flat) * 1000
                metrics[f"median_{prefix}_ms"] = median(flat) * 1000
                p99 = percentile(flat, 0.99)
                if p99 is not None:
                    metrics[f"p99_{prefix}_ms"] = p99 * 1000
    return {
        "schema_version": 1,
        "source_records": len(records),
        "metrics": metrics,
        "raw_completed": completed,
        "raw_failed": failed,
    }


def slo_results(summary: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    checks = []
    for slo_name, limit in spec.get("slo", {}).items():
        if slo_name not in SLO_MAPPING or not isinstance(limit, (int, float)):
            continue
        metric, kind = SLO_MAPPING[slo_name]
        observed = metrics.get(metric)
        passed = observed is not None and (observed <= limit if kind == "max" else observed >= limit)
        checks.append({"slo": slo_name, "metric": metric, "observed": observed, "limit": limit, "passed": passed})
    # An empty SLO object means the user intentionally requested objective-only
    # tuning. It is not a failed acceptance gate.
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def optimization_plan(spec: dict[str, Any]) -> dict[str, Any]:
    workload = spec.get("workload", {})
    model = spec.get("model", {})
    hardware = spec.get("hardware", {})
    stages = [
        {"stage": "baseline", "purpose": "Pin environment, verify correctness, and measure low/mid/target load."},
        {"stage": "topology", "purpose": "Screen feasible parallelism and aggregated versus disaggregated layouts."},
        {"stage": "memory", "purpose": "Tune KV capacity, dtype, page size, and maximum running requests."},
        {"stage": "scheduler", "purpose": "Tune chunked prefill, batching, CUDA Graph coverage, and overlap."},
        {"stage": "backends", "purpose": "Compare attention, GEMM, MoE, and collective backends."},
    ]
    if workload.get("prefix_reuse_ratio", 0) > 0.1:
        stages.append({"stage": "cache", "purpose": "Evaluate radix/cache-aware routing and hierarchical cache."})
    if model.get("architecture") == "moe":
        stages.append({"stage": "moe", "purpose": "Tune EP, All-to-All, overlap, EPLB, and redundant experts."})
    if workload.get("output_tokens", {}).get("p50", 0) >= 128:
        stages.append({"stage": "speculative", "purpose": "Screen AR versus available speculative algorithms and depths."})
    stages.append({"stage": "profile", "purpose": "Profile only the first unresolved SLO bottleneck."})
    stages.append({"stage": "final_validation", "purpose": "Repeat winner, validate SLO/correctness, and test workload variation."})
    return {
        "schema_version": 1,
        "mode": spec.get("mode"),
        "framework": spec.get("framework"),
        "hardware_summary": hardware,
        "budget": spec.get("budget"),
        "objective": spec.get("objective"),
        "stages": stages,
        "execution_enabled": False,
        "note": "Review and execute approved commands manually; this MVP does not launch processes.",
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    objective = spec["objective"]
    metric = objective["metric"]
    direction = objective["direction"]
    base_value = baseline.get("metrics", {}).get(metric)
    cand_value = candidate.get("metrics", {}).get(metric)
    improvement = None
    if isinstance(base_value, (int, float)) and isinstance(cand_value, (int, float)) and base_value != 0:
        raw = (cand_value - base_value) / abs(base_value) * 100
        improvement = raw if direction == "maximize" else -raw
    candidate_slo = slo_results(candidate, spec)
    minimum = float(objective.get("min_improvement_pct", 0))
    maximum_regression = float(objective.get("max_regression_pct", 0))
    regression_checks = []
    baseline_metrics = baseline.get("metrics", {})
    candidate_metrics = candidate.get("metrics", {})
    for secondary_metric, metric_direction in METRIC_DIRECTIONS.items():
        if secondary_metric == metric:
            continue
        base_secondary = baseline_metrics.get(secondary_metric)
        cand_secondary = candidate_metrics.get(secondary_metric)
        if not isinstance(base_secondary, (int, float)) or not isinstance(cand_secondary, (int, float)):
            continue
        if base_secondary == 0:
            regression = 0.0 if cand_secondary == 0 else (float("inf") if metric_direction == "minimize" else 0.0)
        elif metric_direction == "maximize":
            regression = (base_secondary - cand_secondary) / abs(base_secondary) * 100
        else:
            regression = (cand_secondary - base_secondary) / abs(base_secondary) * 100
        regression_checks.append({
            "metric": secondary_metric,
            "baseline": base_secondary,
            "candidate": cand_secondary,
            "regression_pct": regression,
            "limit_pct": maximum_regression,
            "passed": regression <= maximum_regression,
        })
    regressions_passed = all(check["passed"] for check in regression_checks)
    accepted = (
        candidate_slo["passed"]
        and regressions_passed
        and improvement is not None
        and improvement >= minimum
    )
    return {
        "objective_metric": metric,
        "direction": direction,
        "baseline": base_value,
        "candidate": cand_value,
        "improvement_pct": improvement,
        "minimum_improvement_pct": minimum,
        "maximum_regression_pct": maximum_regression,
        "candidate_slo": candidate_slo,
        "secondary_regressions_passed": regressions_passed,
        "secondary_regression_checks": regression_checks,
        "accepted": accepted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--spec", required=True)

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--output")

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--spec", required=True)
    plan_parser.add_argument("--output")

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--input", required=True)
    analyze_parser.add_argument("--spec")
    analyze_parser.add_argument("--output")

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--spec", required=True)
    compare_parser.add_argument("--output")

    args = parser.parse_args()
    try:
        if args.command == "validate":
            errors = validate_spec(load_json(args.spec))
            dump_json({"valid": not errors, "errors": errors}, None)
            return 0 if not errors else 2
        if args.command == "inventory":
            dump_json(inventory(), args.output)
            return 0
        if args.command == "plan":
            spec = load_json(args.spec)
            errors = validate_spec(spec)
            if errors:
                raise ValueError("; ".join(errors))
            dump_json(optimization_plan(spec), args.output)
            return 0
        if args.command == "analyze":
            spec = load_json(args.spec) if args.spec else None
            result = summarize(read_jsonl(args.input), spec)
            if spec:
                result["slo"] = slo_results(result, spec)
            dump_json(result, args.output)
            return 0
        if args.command == "compare":
            dump_json(compare(load_json(args.baseline), load_json(args.candidate), load_json(args.spec)), args.output)
            return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
