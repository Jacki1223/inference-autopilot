#!/usr/bin/env python3
"""Standalone CLI for private, evidence-driven SGLang inference optimization."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

import autopilot
import autotune
import generate_moe_config
import inferopt
import sglang_runtime


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


def parse_nonnegative_number(name: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative number") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return parsed


def visibility_environment(value: str) -> dict[str, str]:
    """Translate an explicit GPU selection while allowing the runtime default."""
    selection = value.strip()
    if not selection or selection.lower() == "all":
        for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
            if os.environ.get(key):
                return {key: os.environ[key]}
        return {}
    identifiers = [item.strip() for item in selection.split(",")]
    if any(not item for item in identifiers):
        raise ValueError("visible GPUs must be 'all' or comma-separated indexes/UUIDs")
    return {"CUDA_VISIBLE_DEVICES": ",".join(identifiers)}


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
    name = value("name", "Experiment name (used in result-directory names)", "single-host-serving")
    mode = value("deployment_mode", "Deployment mode: online_latency (tail SLOs) or offline_throughput (batch throughput)", "online_latency")
    input_tokens = int(value("input_tokens", "Input tokens per request", "256"))
    output_tokens = int(value("output_tokens", "Output tokens per request", "64"))
    if mode not in {"online_latency", "offline_throughput"}:
        raise ValueError("deployment mode must be online_latency or offline_throughput")
    # These are optional hard acceptance gates, not benchmark durations. The
    # same questions are asked for both deployment modes so a task can retain
    # a latency budget while changing only its optimization objective.
    optional_latency_slos = {
        "p99_e2e_latency_ms": value(
            "p99_e2e_latency_ms",
            "Optional p99 E2E latency limit in ms (request start to final token; blank or 0 = no limit)",
            "",
        ),
        "p99_ttft_ms": value(
            "p99_ttft_ms",
            "Optional p99 TTFT limit in ms (request start to first token; blank or 0 = no limit)",
            "",
        ),
        "p99_tpot_ms": value(
            "p99_tpot_ms",
            "Optional p99 TPOT limit in ms (average time per generated token; blank or 0 = no limit)",
            "",
        ),
    }
    slo = {
        name: limit
        for name, raw in optional_latency_slos.items()
        if raw and (limit := parse_nonnegative_number(name, raw)) > 0
    }
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
    visible_gpus = value(
        "cuda_visible_devices",
        "GPUs to use (all = every GPU visible to this process; otherwise comma-separated indexes or UUIDs)",
        "all",
    )
    experiment_profiles = {
        "fast": {
            "search_depth": "evidence_guided", "max_trials": 14, "max_gpu_hours": 1,
            "max_wall_time_minutes": 90, "confirmation_repetitions": 2,
            "warmup_multiplier": 2, "warmup_floor": 16,
            "request_multiplier": 16, "request_floor": 128, "duration": 20,
        },
        "balanced": {
            "search_depth": "evidence_guided", "max_trials": 18, "max_gpu_hours": 3,
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
        "slo": slo,
        "objective": {"metric": "request_throughput_rps", "direction": "maximize", "min_improvement_pct": 1, "max_regression_pct": 5},
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
            "strategy": "adaptive", "max_steps": calibration_steps, "stop_on_slo_failure": True,
            **({"concurrencies": concurrency_points, "max_steps": len(concurrency_points)} if concurrency_points else {}),
        },
        "offline": True,
        "allow_download": False,
        "deployment": {"allow_model_variant_recommendations": True, "allow_auto_model_switch": False},
        "quality": {},
        "env": visibility_environment(visible_gpus),
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
            # A shared-prefix workload must retain group locality. Randomizing
            # groups defeats the cache/scheduling behavior the task requests.
            "ordered": True,
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
    single_gpu = autopilot.single_gpu_feasibility(task, hardware, model)
    feasibility = autopilot.deployment_feasibility(task, hardware, model)
    variants = []
    for candidate in task.get("model_variants", []):
        candidate_model = autopilot.model_inventory(candidate["model_path"])
        candidate_feasibility = autopilot.deployment_feasibility(task, hardware, candidate_model)
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
    nsys = profiler.get("nsys", {})
    nsys_ready = bool(
        isinstance(nsys, dict)
        and nsys.get("available")
        and nsys.get("returncode") == 0
        and not nsys.get("error")
    )
    blocking_errors = []
    if hardware.get("vendor") != "nvidia":
        blocking_errors.append(
            "automatic run currently requires NVIDIA; AMD discovery is supported but its profiling executor is not implemented"
        )
    if not framework.get("launch_server_help_available"):
        blocking_errors.append("the selected Python/SGLang checkout did not expose launch_server --help")
    if feasibility.get("status") != "deployable_as_is":
        blocking_errors.append(feasibility.get("reason", "the selected GPUs cannot deploy this checkpoint"))
    if not nsys_ready:
        blocking_errors.append("nsys is required for inferopt run but was not found or did not execute successfully")
    return {
        "schema_version": 1,
        "status": "ready" if not blocking_errors else "attention_required",
        "blocking_errors": blocking_errors,
        "hardware": hardware,
        "model": model,
        "framework": framework,
        "deployment_feasibility": feasibility,
        "single_gpu_feasibility": single_gpu,
        "local_model_variants": variants,
        "profiler_tools": {key: profiler.get(key) for key in ("nsys", "ncu")},
        "next_command": (
            "inferopt plan --task TASK.json"
            if not blocking_errors
            else "resolve blocking_errors before launching a benchmark"
        ),
    }


def tune_moe(task: dict[str, Any], profile_path: str, result_path: str,
             output_dir: str, timeout_minutes: float, max_batch_sizes: int,
             validate_end_to_end: bool = True,
             topk_ids_dir: str | None = None) -> dict[str, Any]:
    errors = autopilot.validate_task(task)
    if errors:
        raise ValueError("; ".join(errors))
    profile = inferopt.load_json(profile_path)
    completed_run = inferopt.load_json(result_path)
    profile_dir = Path(str(profile.get("run_dir", ""))).expanduser()
    server_log = profile_dir / "server-nsys.log"
    if server_log.is_file():
        profile["runtime_observations"] = sglang_runtime.summarize_sglang_log(
            server_log.read_text(encoding="utf-8", errors="replace")
        )
    execution_task = json.loads(json.dumps(task))
    execution_task["kernel_tuning"] = {
        "mode": "execute",
        "timeout_minutes": timeout_minutes,
        "max_batch_sizes": max_batch_sizes,
    }
    if topk_ids_dir:
        execution_task["kernel_tuning"]["topk_ids_dir"] = str(Path(topk_ids_dir).expanduser().resolve())
    discovery = autopilot.discover(execution_task)
    plan = autopilot.moe_kernel_optimization_plan(execution_task, discovery, profile)
    progress = autopilot.ProgressReporter()
    progress.emit(
        "optional-moe",
        "starting explicitly approved high-cost kernel search; this is outside inferopt run",
    )
    execution = autopilot.execute_moe_kernel_tuning(
        execution_task, plan, Path(output_dir).expanduser().resolve()
    )
    validation = None
    deployment_command = None
    deployment_environment: dict[str, Any] = {}
    generated_config_deployable = False
    if execution.get("status") == "completed" and validate_end_to_end:
        recommendation = completed_run.get("recommended_configuration") or {}
        baseline_config = recommendation.get("config", recommendation)
        if not isinstance(baseline_config, dict) or not baseline_config:
            baseline_config = completed_run.get("profiled_initial_configuration") or {
                "tp_size": discovery["derived"]["minimum_tp_size"]
            }
        baseline_environment = completed_run.get("deployment_environment", {})
        if not isinstance(baseline_environment, dict):
            baseline_environment = {}
        if "SGLANG_MOE_CONFIG_DIR" in baseline_environment:
            raise ValueError(
                "the source baseline already uses SGLANG_MOE_CONFIG_DIR; a clean baseline result is required"
            )
        validation_task = json.loads(json.dumps(task))
        validation_task["env"] = {**validation_task.get("env", {}), **baseline_environment}
        repetitions = max(3, int(task.get("confirmation_repetitions", 3)))
        validation_spec = autopilot.explicit_configuration_spec(
            validation_task,
            discovery,
            stage_name="optional-fused-moe-validation",
            baseline=baseline_config,
            configurations=[{
                "name": "fused-moe-autotuned-config",
                "config": baseline_config,
                "env": {"SGLANG_MOE_CONFIG_DIR": execution["config_root"]},
            }],
            max_trials=repetitions * 2,
            repetitions=repetitions,
            remaining_gpu_hours=float(task["budget"]["max_gpu_hours"]),
            remaining_wall_minutes=float(task["budget"]["max_wall_time_minutes"]),
        )
        progress.emit(
            "optional-moe",
            f"kernel config generated; running {repetitions} interleaved baseline/candidate repetitions under the original workload and SLOs",
        )
        validation = autopilot.execute_with_progress(
            validation_spec, progress, "optional fused MoE A/B"
        )
        winner = validation.get("winner")
        generated_config_deployable = bool(
            isinstance(winner, dict)
            and winner.get("configuration_name") == "fused-moe-autotuned-config"
            and winner.get("confirmed")
            and execution.get("paired_config_complete", True)
        )
        if generated_config_deployable:
            deployment_command = autopilot.final_server_command(validation_spec, winner)
            deployment_environment = winner.get("env", {})
    return {
        "schema_version": 1,
        "operation": "optional_fused_moe_kernel_tuning",
        "plan": plan,
        "execution": execution,
        "end_to_end_validation": validation,
        "generated_config_deployable": generated_config_deployable,
        "deployment_command": deployment_command,
        "deployment_environment": deployment_environment,
        "next_step": (
            "use the emitted environment and deployment command; the generated config passed the original workload, SLO, improvement, stability, and repetition gates"
            if generated_config_deployable
            else "do not deploy the generated config; it did not complete or did not beat the baseline through every end-to-end gate"
            if validate_end_to_end
            else "run end-to-end baseline/candidate validation before deploying the generated config"
            if execution.get("status") == "completed"
            else "inspect the execution reason; the normal deployment recommendation is unchanged"
        ),
    }


def markdown_report(final: dict[str, Any]) -> str:
    recommendation = final.get("recommended_configuration") or {}
    profile = final.get("profiling", {}) if isinstance(final.get("profiling"), dict) else {}
    diagnosis = profile.get("diagnosis", {}) if isinstance(profile.get("diagnosis"), dict) else {}
    search_plan = final.get("search_plan", {}) if isinstance(final.get("search_plan"), dict) else {}
    routing = search_plan.get("routing_evidence", {}) if isinstance(search_plan.get("routing_evidence"), dict) else {}
    routing_primary = routing.get("primary_bottleneck", diagnosis.get("primary_bottleneck", "unavailable"))
    lines = [
        "# Inference Autopilot Report",
        "",
        f"- Run directory: `{final.get('run_dir', 'unknown')}`",
        f"- Decision: `{final.get('recommendation_status', 'unknown')}`",
        f"- Deployable: `{final.get('deployable', False)}`",
        f"- Parameter-routing diagnosis: `{routing_primary}`",
        f"- Raw Nsight diagnosis: `{diagnosis.get('primary_bottleneck', 'unavailable')}`",
        f"- Nsight timing comparable to unprofiled baseline: `{diagnosis.get('profiling_run_performance_comparable', 'unknown')}`",
        "",
    ]
    if recommendation:
        lines.extend([
            "## Recommended Configuration",
            "",
            "```json",
            json.dumps(recommendation.get("config", recommendation), indent=2, sort_keys=True),
            "```",
        ])
    else:
        lines.extend([
            "## Recommendation",
            "",
            "No deployment command is recommended.",
            f"Status: `{final.get('recommendation_status', 'unknown')}`.",
            f"Reason: {final.get('recommendation_reason', 'insufficient deployable evidence')}",
        ])
    command = final.get("deployment_command")
    if isinstance(command, list):
        deployment_env = final.get("deployment_environment", {})
        rendered_env = " ".join(
            f"{key}={shlex.quote(str(value))}" for key, value in sorted(deployment_env.items())
        )
        rendered_command = shlex.join(str(item) for item in command)
        if rendered_env:
            rendered_command = f"{rendered_env} {rendered_command}"
        lines.extend(["", "## Deployment Command", "", "```bash", rendered_command, "```"])
    model = final.get("discovery", {}).get("model", {})
    if isinstance(model, dict) and (model.get("weight_quantization") or model.get("checkpoint_dtype")):
        lines.extend([
            "", "## Model Precision", "",
            f"- Weight format: `{model.get('weight_quantization') or model.get('quantization') or 'unquantized'}`",
            f"- Checkpoint/activation dtype: `{model.get('checkpoint_dtype') or model.get('dtype') or 'auto'}`",
            "- Launch policy: SGLang reads checkpoint metadata automatically; no dtype or quantization flag is injected unless the task explicitly requests one.",
        ])
    cookbook = final.get("cookbook_initial_screen", {})
    if isinstance(cookbook, dict):
        aggregates = cookbook.get("aggregates", [])
        baseline = next((item for item in aggregates if item.get("kind") == "baseline"), None)
        cookbook_candidates = [item for item in aggregates if item.get("kind") == "candidate" and item.get("metrics")]
        if baseline or cookbook_candidates:
            lines.extend(["", "## Cookbook Comparison", ""])
            if baseline:
                metrics = baseline.get("metrics", {})
                lines.append(
                    "- SGLang-default baseline: "
                    f"`{metrics.get('request_throughput_rps', 'unknown')} RPS`, "
                    f"p99 E2E `{metrics.get('p99_e2e_latency_ms', 'unknown')} ms`"
                )
            for candidate in cookbook_candidates:
                metrics = candidate.get("metrics", {})
                comparison = candidate.get("comparison", {})
                lines.append(
                    f"- `{candidate.get('configuration_name')}`: "
                    f"`{metrics.get('request_throughput_rps', 'unknown')} RPS`, "
                    f"p99 E2E `{metrics.get('p99_e2e_latency_ms', 'unknown')} ms`, "
                    f"screening change `{comparison.get('improvement_pct', 'unknown')}%`"
                )
    screening = final.get("screening", {})
    if isinstance(screening, dict):
        candidates = [
            item for item in screening.get("aggregates", [])
            if item.get("kind") == "candidate" and item.get("metrics")
            and item.get("comparison", {}).get("improvement_pct") is not None
        ]
        if candidates:
            best_observed = max(candidates, key=lambda item: item["comparison"]["improvement_pct"])
            lines.extend([
                "", "## Best Observed Delta", "",
                "This is not a deployment recommendation unless it passed confirmation.",
                "```json",
                json.dumps(best_observed.get("config", {}), indent=2, sort_keys=True),
                "```",
                f"- Screening change: `{best_observed['comparison']['improvement_pct']:.3f}%`",
                f"- Rejection reasons: `{', '.join(best_observed.get('rejection_reasons', [])) or 'none'}`",
            ])
    bottleneck = final.get("bottleneck", {}) if isinstance(final.get("bottleneck"), dict) else {}
    mechanism = bottleneck.get("screening_mechanism", {}) if isinstance(bottleneck.get("screening_mechanism"), dict) else {}
    if mechanism:
        lines.extend(["", "## Evidence", "", f"- Screening classification: `{mechanism.get('classification', 'unavailable')}`"])
    parameter_search = final.get("parameter_search", {})
    if isinstance(parameter_search, dict):
        lines.extend([
            "", "## Parameter Search", "",
            f"- Attempted parameter candidates: `{parameter_search.get('attempted_parameter_candidates', 'unknown')}`",
            f"- Executed parameter candidates: `{parameter_search.get('executed_parameter_candidates', 'unknown')}`",
            f"- Failed parameter candidates: `{parameter_search.get('failed_parameter_candidates', 'unknown')}`",
            f"- Evidence sufficient for a deployment recommendation: `{parameter_search.get('sufficient_evidence', False)}`",
        ])
    resolved = search_plan.get("resolved_baseline", {})
    if recommendation and isinstance(resolved, dict):
        effective_recommendation = {
            **resolved,
            **recommendation.get("config", recommendation),
        }
        lines.extend([
            "", "## Effective Runtime Settings", "",
            "The launch command emits only measured deltas; omitted values remain the resolved defaults of the tested SGLang version.",
            "```json",
            json.dumps(effective_recommendation, indent=2, sort_keys=True),
            "```",
        ])
    ranked = search_plan.get("ranked_parameter_groups", [])
    screening_priority = search_plan.get("screening_priority_order", [])
    if isinstance(screening_priority, list) and screening_priority:
        priority_index = {name: index for index, name in enumerate(screening_priority)}
        ranked = sorted(
            ranked,
            key=lambda item: priority_index.get(item.get("parameter"), len(priority_index)),
        )
    excluded_chunks = search_plan.get("excluded_chunked_prefill_candidates", [])
    chunk_strategy = search_plan.get("chunked_prefill_strategy", {})
    audit = search_plan.get("parameter_audit", {})
    screening_aggregates = screening.get("aggregates", []) if isinstance(screening, dict) else []
    attempted = [item for item in screening_aggregates if item.get("kind") == "candidate"]
    if resolved or ranked or attempted or excluded_chunks:
        lines.extend(["", "## Parameter Selection Evidence", ""])
        shares = diagnosis.get("shares_pct", {})
        if isinstance(shares, dict):
            lines.append(
                "- GPU kernel-time shares: "
                f"attention `{shares.get('attention_kernels', 'unknown')}%`, "
                f"MoE `{shares.get('moe_kernels', 'unknown')}%`, "
                f"GEMM `{shares.get('gemm_kernels', 'unknown')}%`, "
                f"communication `{shares.get('communication_kernels', 'unknown')}%`."
            )
        if diagnosis.get("profiling_run_performance_comparable") is False:
            lines.append(
                "- Nsight changed request throughput by "
                f"`{diagnosis.get('profile_throughput_regression_pct', 'unknown')}%`; host-gap and CUDA API timing were excluded from parameter routing, while GPU kernel shares remained usable."
            )
        if resolved:
            lines.append(f"- Resolved SGLang baseline: `{json.dumps(resolved, sort_keys=True)}`")
        if isinstance(chunk_strategy, dict) and chunk_strategy.get("strategy"):
            lines.append(
                f"- Chunked-prefill strategy: `{chunk_strategy['strategy']}`; "
                f"candidate order `{chunk_strategy.get('ordered_candidates', [])}`. "
                f"{chunk_strategy.get('reason', '')}"
            )
        if screening_priority:
            lines.append(
                f"- Workload/trace screening priority: `{screening_priority}`. "
                "Trial budget may stop before lower-priority families."
            )
        for item in ranked[:8]:
            lines.append(
                f"- Planned `{item.get('parameter')}` values `{item.get('values', [])}`: {item.get('reason', 'selected by evidence routing')}"
            )
        for item in attempted:
            comparison = item.get("comparison", {})
            lines.append(
                f"- Measured `{item.get('configuration_name')}`: objective change "
                f"`{comparison.get('improvement_pct', 'unavailable')}%`; "
                f"confirmed `{item.get('confirmed', False)}`; "
                f"rejections `{', '.join(item.get('rejection_reasons', [])) or 'none'}`."
            )
        for item in excluded_chunks:
            lines.append(
                f"- Excluded `chunked_prefill_size={item.get('chunked_prefill_size')}` before launch: "
                f"predicted mem-fraction `{item.get('predicted_mem_fraction_static')}` is below the estimated model floor "
                f"`{item.get('minimum_estimated_static_fraction')}`."
            )
        if isinstance(audit, dict) and audit.get("summary"):
            lines.append(
                f"- Current-version ServerArgs audit: `{json.dumps(audit['summary'], sort_keys=True)}`. "
                "Parameters not tested were classified as inapplicable, control-plane/diagnostic, incompatible, or lower priority rather than silently ignored."
            )
    fused_moe = final.get("kernel_optimization", {}).get("fused_moe", {})
    if isinstance(fused_moe, dict) and fused_moe.get("status") == "candidate_required":
        fused_moe_execution = final.get("kernel_optimization", {}).get("fused_moe_execution", {})
        run_dir = Path(str(final.get("run_dir", ".")))
        tune_command = " ".join([
            "inferopt tune-moe",
            "--task", shlex.quote(str(run_dir / "task.json")),
            "--profile", shlex.quote(str(run_dir / "profile" / "nsys-diagnosis.json")),
            "--result", shlex.quote(str(run_dir / "final.json")),
            "--output-dir", shlex.quote(str(run_dir / "optional-fused-moe-tuning")),
            "--yes",
            "--output", shlex.quote(str(run_dir / "optional-fused-moe-tuning.json")),
        ])
        lines.extend([
            "", "## Fused MoE Kernel", "",
            "SGLang used a generic fused MoE fallback because no config matched this GPU, model shape, quantization, parallel layout, and Triton version. This warning is an optimization opportunity, not proof of an end-to-end regression.",
            "",
            f"- Priority: `{fused_moe.get('priority')}`",
            f"- Reason: {fused_moe.get('reason')}",
            f"- Missing tuned configs: `{len(fused_moe.get('missing_config_files', []))}`",
            f"- Shape-matched batch sizes: `{fused_moe.get('shape_matched_batch_sizes', [])}`",
            f"- Policy: {fused_moe.get('application_policy')}",
            f"- Autotuning execution: `{fused_moe_execution.get('status', 'unknown')}`",
            f"- Autotuning result: {fused_moe_execution.get('reason') or fused_moe_execution.get('validation_policy') or 'see kernel-tuning.json'}",
            "- Cost policy: not part of `inferopt run`; execute only after explicit user approval.",
            "- Standalone behavior: compile shape-matched Triton candidates, then automatically run at least three interleaved baseline/candidate repetitions with the original workload and SLOs.",
            "- Adoption rule: the generated config remains rejected unless it clears the configured objective-improvement threshold, all SLOs, secondary-regression limits, variation limit, and confirmation count.",
            "- Result interpretation: `generated_config_deployable=true` means use the emitted environment and command; `false` means retain the original deployment.",
            "",
            "```bash",
            tune_command,
            "```",
        ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create a single-host autopilot task")
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
    init.add_argument("--p99-e2e-latency-ms")
    init.add_argument("--p99-ttft-ms")
    init.add_argument("--p99-tpot-ms")
    init.add_argument("--max-concurrency")
    init.add_argument("--concurrency-points", help="comma-separated capacity/SLO measurement points; must end at max concurrency")
    init.add_argument("--shared-prefix-tokens")
    init.add_argument("--experiment-mode", choices=["fast", "balanced", "rigorous"])
    init.add_argument(
        "--cuda-visible-devices",
        help="comma-separated GPU indexes/UUIDs, or 'all' (default) to keep every currently visible GPU",
    )
    for name in ("doctor", "feasibility", "plan", "run", "validate"):
        item = commands.add_parser(name)
        item.add_argument("--task", required=True)
        item.add_argument("--output")
        if name == "run":
            item.add_argument("--yes", action="store_true")
    report = commands.add_parser("report", help="render a human-readable completed-run report")
    report.add_argument("--result", required=True)
    report.add_argument("--output", required=True)
    tune_moe_parser = commands.add_parser(
        "tune-moe",
        help="optionally run the high-cost fused MoE kernel tuner outside the normal workflow",
        description=(
            "Compile shape-matched fused MoE configs, then compare the generated config "
            "with the completed run's baseline under the original workload and SLOs."
        ),
        epilog=(
            "This command is never called by inferopt run. It can take a long time. "
            "Only generated_config_deployable=true authorizes the emitted environment and command; "
            "otherwise keep the original deployment."
        ),
    )
    tune_moe_parser.add_argument("--task", required=True, help="task.json from the completed inferopt run")
    tune_moe_parser.add_argument(
        "--profile", required=True,
        help="completed RUN_DIR/profile/nsys-diagnosis.json used to recover MoE shapes and warnings",
    )
    tune_moe_parser.add_argument(
        "--result", required=True,
        help="completed RUN_DIR/final.json whose recommendation becomes the clean A/B baseline",
    )
    tune_moe_parser.add_argument(
        "--output-dir", required=True,
        help="private directory for generated configs, tuner logs, and A/B artifacts",
    )
    tune_moe_parser.add_argument(
        "--timeout-minutes", type=float, default=120,
        help="maximum kernel-search time in minutes (default: 120)",
    )
    tune_moe_parser.add_argument(
        "--max-batch-sizes", type=int, default=4,
        help="maximum representative decode/prefill shapes to tune (default: 4)",
    )
    tune_moe_parser.add_argument(
        "--topk-ids-dir",
        help="directory produced by SGLang's official top-k capture workflow; required when logs request an _down config",
    )
    tune_moe_parser.add_argument(
        "--yes", action="store_true",
        help="acknowledge that this optional operation is GPU-intensive and may run for hours",
    )
    tune_moe_parser.add_argument(
        "--no-validate", action="store_true",
        help="generate configs only; never deploy them until a separate end-to-end A/B test passes",
    )
    tune_moe_parser.add_argument("--output", help="JSON decision file; stdout is used when omitted")
    generate_moe_parser = commands.add_parser(
        "generate-moe-config",
        help="generate SGLang fused-MoE Triton config files",
        description=(
            "Generate a paired normal and _down config by default. Paired mode uses "
            "SGLang's official separate tuner and requires top-k capture files."
        ),
    )
    generate_moe_parser.add_argument("--repository", required=True, help="SGLang source checkout")
    generate_moe_parser.add_argument("--python", default=sys.executable, help="Python interpreter used by SGLang")
    generate_moe_parser.add_argument("--model-path", required=True, help="local model checkpoint")
    generate_moe_parser.add_argument("--output-dir", required=True, help="private output directory")
    generate_moe_parser.add_argument("--output", help="summary JSON path")
    generate_moe_parser.add_argument("--mode", choices=["paired", "standard"], default="paired")
    generate_moe_parser.add_argument("--topk-ids-dir", help="required in paired mode")
    generate_moe_parser.add_argument("--tp-size", type=int, default=1)
    generate_moe_parser.add_argument("--ep-size", type=int, default=1)
    generate_moe_parser.add_argument(
        "--dtype", choices=["auto", "fp8_w8a8", "int8_w8a8", "int8_w8a16", "int4_w4a16"], default="auto"
    )
    generate_moe_parser.add_argument("--batch-sizes", type=int, nargs="+", default=generate_moe_config.DEFAULT_BATCH_SIZES)
    generate_moe_parser.add_argument("--search-space-file")
    generate_moe_parser.add_argument("--per-channel-quant", action="store_true")
    generate_moe_parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    generate_moe_parser.add_argument("--timeout-minutes", type=float, default=120)
    generate_moe_parser.add_argument("--yes", action="store_true")
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
        if args.command == "tune-moe":
            if not args.yes:
                raise ValueError("tune-moe is high cost and requires --yes")
            if args.timeout_minutes <= 0 or args.max_batch_sizes <= 0:
                raise ValueError("tune-moe timeout and max batch sizes must be positive")
            if args.topk_ids_dir and (
                not Path(args.topk_ids_dir).expanduser().is_absolute()
                or not Path(args.topk_ids_dir).expanduser().is_dir()
            ):
                raise ValueError("--topk-ids-dir must be an existing absolute directory")
            result = tune_moe(
                task, args.profile, args.result, args.output_dir,
                args.timeout_minutes, args.max_batch_sizes,
                validate_end_to_end=not args.no_validate,
                topk_ids_dir=args.topk_ids_dir,
            )
            inferopt.dump_json(result, args.output)
            return 0 if (
                result["generated_config_deployable"]
                or args.no_validate and result["execution"].get("status") == "completed"
            ) else 2
        if args.command == "generate-moe-config":
            generate_moe_config._run(args)
            return 0
        if args.command == "validate":
            errors = autopilot.validate_task(task)
            inferopt.dump_json({"valid": not errors, "errors": errors}, args.output)
            return 0 if not errors else 2
        if args.command in {"doctor", "feasibility"}:
            diagnosis = doctor(task)
            if args.command == "feasibility":
                result = diagnosis.get("deployment_feasibility", diagnosis)
                success = result.get("status") == "deployable_as_is"
            else:
                result = diagnosis
                success = result.get("status") == "ready"
            inferopt.dump_json(result, args.output)
            return 0 if success else 2
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
