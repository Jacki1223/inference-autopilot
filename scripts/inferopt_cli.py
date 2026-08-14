#!/usr/bin/env python3
"""Standalone CLI for private, evidence-driven SGLang inference optimization."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import autopilot
import inferopt


def write_json(value: Any, output: str | Path) -> None:
    target = Path(output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def ask(label: str, default: str | None = None) -> str:
    prompt = f"{label}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    value = input(prompt).strip()
    return value or (default or "")


def parse_concurrency_points(value: str) -> list[int]:
    if not value.strip():
        return []
    try:
        points = [int(item) for item in re.split(r"[\s,]+", value.strip()) if item]
    except ValueError as exc:
        raise ValueError("concurrency points must be positive integers separated by commas or spaces") from exc
    if not points or any(point <= 0 for point in points):
        raise ValueError("concurrency points must be positive integers separated by commas or spaces")
    points = sorted(set(points))
    if len(points) > 16:
        raise ValueError("at most 16 explicit concurrency points are supported")
    return points


def init_task(args: argparse.Namespace) -> dict[str, Any]:
    interactive = not args.non_interactive
    if interactive and not sys.stdin.isatty():
        raise ValueError("init needs a TTY, or pass --non-interactive with all required options")

    def value(name: str, label: str, default: str | None = None) -> str:
        current = getattr(args, name, None)
        return current if current else (ask(label, default) if interactive else (default or ""))

    repository = value("repository", "SGLang repository (checkout used to discover current server flags)", os.getcwd())
    python = value("python", "Python executable (interpreter that runs SGLang)", sys.executable)
    model_path = value("model_path", "Local model directory (already available checkpoint)")
    output_dir = value("output_dir", "Private artifact directory (logs, traces, and results)", str(Path.cwd() / "inference-autopilot-runs"))
    name = value("name", "Experiment name (used in result-directory names)", "single-gpu-serving")
    mode = value("deployment_mode", "Deployment mode: online_latency (tail SLOs) or offline_throughput (batch throughput)", "online_latency")
    input_tokens = int(value("input_tokens", "Input tokens per request", "256"))
    output_tokens = int(value("output_tokens", "Output tokens per request", "64"))
    default_concurrency = "8" if mode == "online_latency" else "64"
    max_concurrency = int(value("max_concurrency", "Target concurrency (highest point for final tuning)", default_concurrency))
    concurrency_points = parse_concurrency_points(value(
        "concurrency_points", "Concurrency points to measure, comma or space separated (blank = automatic 1,2,4,... sweep)", ""
    ))
    if concurrency_points and concurrency_points[-1] != max_concurrency:
        raise ValueError("explicit concurrency points must include the target concurrency as their largest value")
    shared_prefix_tokens = int(value(
        "shared_prefix_tokens", "Shared prefix tokens (common input prefix; 0 disables prefix-cache testing)", "0"
    ))
    experiment_mode = value("experiment_mode", "Experiment intensity: fast (coarse), balanced (default), or rigorous (final decision)", "balanced")
    experiment_profiles = {
        "fast": {
            "search_depth": "evidence_guided", "max_trials": 14, "max_gpu_hours": 1,
            "max_wall_time_minutes": 90, "confirmation_repetitions": 2,
            "warmup_multiplier": 2, "warmup_floor": 16,
            "request_multiplier": 16, "request_floor": 128, "duration": 20,
        },
        "balanced": {
            "search_depth": "thorough", "max_trials": 24, "max_gpu_hours": 4,
            "max_wall_time_minutes": 360, "confirmation_repetitions": 3,
            "warmup_multiplier": 4, "warmup_floor": 32,
            "request_multiplier": 32, "request_floor": 512, "duration": 45,
        },
        "rigorous": {
            "search_depth": "thorough", "max_trials": 36, "max_gpu_hours": 8,
            "max_wall_time_minutes": 720, "confirmation_repetitions": 5,
            "warmup_multiplier": 8, "warmup_floor": 64,
            "request_multiplier": 64, "request_floor": 1000, "duration": 120,
        },
    }
    if experiment_mode not in experiment_profiles:
        raise ValueError("experiment intensity must be fast, balanced, or rigorous")
    profile = experiment_profiles[experiment_mode]

    calibration_steps = 1
    calibration_value = 1
    while calibration_value < max_concurrency:
        calibration_value *= 2
        calibration_steps += 1
    task: dict[str, Any] = {
        "name": name,
        "repository": str(Path(repository).expanduser().resolve()),
        "python": str(Path(python).expanduser().resolve()),
        "model_path": str(Path(model_path).expanduser().resolve()),
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "deployment_mode": mode,
        "experiment_mode": experiment_mode,
        "search_depth": profile["search_depth"],
        "workload": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "max_concurrency": max_concurrency,
            "request_rate": "inf",
            "num_prompts": max(512, max_concurrency * 128),
        },
        "slo": ({"p99_ttft_ms": 1000, "p99_tpot_ms": 100, "p99_e2e_latency_ms": 2000}
                if mode == "online_latency" else {"max_error_rate": 0.0}),
        "objective": {"metric": "request_throughput_rps", "direction": "maximize", "min_improvement_pct": 3, "max_regression_pct": 5},
        "budget": {
            "max_trials": profile["max_trials"],
            "max_gpu_hours": profile["max_gpu_hours"],
            "max_wall_time_minutes": profile["max_wall_time_minutes"],
        },
        "profiling": {"enabled": True},
        "confirmation_repetitions": profile["confirmation_repetitions"],
        "measurement": {
            "warmup_requests": max(profile["warmup_floor"], max_concurrency * profile["warmup_multiplier"]),
            "min_measurement_requests": max(profile["request_floor"], max_concurrency * profile["request_multiplier"]),
            "min_measurement_seconds": profile["duration"],
        },
        "calibration": {
            "enabled": True, "min_concurrency": 1, "max_concurrency": max_concurrency,
            "max_steps": calibration_steps, "stop_on_slo_failure": True,
            **({"concurrencies": concurrency_points, "max_steps": len(concurrency_points)} if concurrency_points else {}),
        },
        "offline": True,
        "allow_download": False,
        "deployment": {"allow_model_variant_recommendations": True, "allow_auto_model_switch": False},
        "quality": {},
        "env": {"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices or "0"},
    }
    if shared_prefix_tokens:
        prefix = shared_prefix_tokens
        if not 0 < prefix < input_tokens:
            raise ValueError("--shared-prefix-tokens must be between 1 and input tokens - 1")
        task["workload"]["prefix_reuse_ratio"] = prefix / input_tokens
        task["workload"]["shared_prefix"] = {
            "groups": 8,
            "prompts_per_group": max(64, task["workload"]["num_prompts"] // 8),
            "system_prompt_tokens": prefix,
            "question_tokens": input_tokens - prefix,
            "ordered": False,
        }
    errors = autopilot.validate_task(task)
    if errors:
        raise ValueError("; ".join(errors))
    return task


def doctor(task: dict[str, Any]) -> dict[str, Any]:
    errors = autopilot.validate_task(task)
    if errors:
        return {"status": "invalid_task", "errors": errors}
    hardware = autopilot.parse_nvidia_inventory() or autopilot.parse_amd_inventory()
    if hardware is None:
        return {"status": "no_supported_accelerator", "errors": ["no NVIDIA or AMD accelerator inventory available"]}
    model = autopilot.model_inventory(task["model_path"])
    framework = autopilot.framework_evidence(task)
    feasibility = autopilot.single_gpu_feasibility(task, hardware, model)
    variants = []
    for candidate in task.get("model_variants", []):
        candidate_model = autopilot.model_inventory(candidate["model_path"])
        candidate_feasibility = autopilot.single_gpu_feasibility(task, hardware, candidate_model)
        variants.append({
            "name": candidate["name"],
            "model_path": candidate["model_path"],
            "declared_quantization": candidate.get("quantization"),
            "detected_quantization": candidate_model.get("quantization"),
            "model_weight_gib": candidate_model.get("weight_gib"),
            "feasibility": candidate_feasibility,
            "quality_gate": {
                "state": "pending" if task.get("quality", {}).get("evaluation_dataset") else "missing_evaluation_dataset",
                "policy": "a deployable variant is not selected until it clears the explicit quality gate",
            },
        })
    profiler = inferopt.inventory().get("tools", {})
    return {
        "schema_version": 1,
        "status": "ready" if framework.get("launch_server_help_available") and feasibility.get("status") == "deployable_as_is" else "attention_required",
        "hardware": hardware,
        "model": model,
        "framework": framework,
        "single_gpu_feasibility": feasibility,
        "local_model_variants": variants,
        "profiler_tools": {key: profiler.get(key) for key in ("nsys", "ncu")},
        "next_command": "inferopt plan --task TASK.json" if feasibility.get("status") == "deployable_as_is" else "review single_gpu_feasibility before launching a benchmark",
    }


def markdown_report(final: dict[str, Any]) -> str:
    recommendation = final.get("recommended_configuration") or {}
    profile = final.get("profiling", {}) if isinstance(final.get("profiling"), dict) else {}
    diagnosis = profile.get("diagnosis", {}) if isinstance(profile.get("diagnosis"), dict) else {}
    lines = [
        "# Inference Autopilot Report",
        "",
        f"- Run directory: `{final.get('run_dir', 'unknown')}`",
        f"- Decision: `{final.get('recommendation_status', 'unknown')}`",
        f"- Deployable: `{final.get('deployable', False)}`",
        f"- Primary diagnosis: `{diagnosis.get('primary_bottleneck', 'unavailable')}`",
        "",
        "## Recommended Configuration",
        "",
        "```json",
        json.dumps(recommendation.get("config", recommendation), indent=2, sort_keys=True),
        "```",
    ]
    command = final.get("deployment_command")
    if isinstance(command, list):
        lines.extend(["", "## Deployment Command", "", "```bash", " ".join(str(item) for item in command), "```"])
    bottleneck = final.get("bottleneck", {}) if isinstance(final.get("bottleneck"), dict) else {}
    mechanism = bottleneck.get("screening_mechanism", {}) if isinstance(bottleneck.get("screening_mechanism"), dict) else {}
    if mechanism:
        lines.extend(["", "## Evidence", "", f"- Screening classification: `{mechanism.get('classification', 'unavailable')}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create a single-GPU autopilot task")
    init.add_argument("--output", required=True)
    init.add_argument("--non-interactive", action="store_true")
    init.add_argument("--repository")
    init.add_argument("--python")
    init.add_argument("--model-path")
    init.add_argument("--output-dir")
    init.add_argument("--name")
    init.add_argument("--deployment-mode")
    init.add_argument("--input-tokens")
    init.add_argument("--output-tokens")
    init.add_argument("--max-concurrency")
    init.add_argument("--concurrency-points", help="comma-separated capacity/SLO measurement points; must end at max concurrency")
    init.add_argument("--shared-prefix-tokens")
    init.add_argument("--experiment-mode", choices=["fast", "balanced", "rigorous"])
    init.add_argument("--cuda-visible-devices")
    for name in ("doctor", "feasibility", "plan", "run", "validate"):
        item = commands.add_parser(name)
        item.add_argument("--task", required=True)
        item.add_argument("--output")
        if name == "run":
            item.add_argument("--yes", action="store_true")
    report = commands.add_parser("report", help="render a human-readable completed-run report")
    report.add_argument("--result", required=True)
    report.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            write_json(init_task(args), args.output)
            return 0
        if args.command == "report":
            target = Path(args.output).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(markdown_report(inferopt.load_json(args.result)), encoding="utf-8")
            return 0
        task = inferopt.load_json(args.task)
        if args.command == "validate":
            errors = autopilot.validate_task(task)
            inferopt.dump_json({"valid": not errors, "errors": errors}, args.output)
            return 0 if not errors else 2
        if args.command in {"doctor", "feasibility"}:
            result = doctor(task)
            if args.command == "feasibility":
                result = result.get("single_gpu_feasibility", result)
            inferopt.dump_json(result, args.output)
            return 0 if result.get("status") not in {"invalid_task", "no_supported_accelerator"} else 2
        if args.command == "plan":
            inferopt.dump_json(autopilot.build_plan(task), args.output)
            return 0
        if args.command == "run":
            if not args.yes:
                raise ValueError("run requires --yes after reviewing doctor and plan")
            inferopt.dump_json(autopilot.run_autopilot(task), args.output)
            return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
