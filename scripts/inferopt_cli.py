#!/usr/bin/env python3
"""Standalone CLI for private, evidence-driven SGLang inference optimization."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
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
import profile_sglang
import parameter_evolution
import sglang_catalog
import sglang_runtime


def source_version() -> str:
    version_path = Path(__file__).resolve().parents[1] / "VERSION"
    value = version_path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise ValueError(f"invalid VERSION value: {value!r}")
    return value


def package_version() -> str:
    source_path = Path(__file__).resolve().parents[1] / "VERSION"
    if source_path.is_file():
        return source_version()
    try:
        return importlib.metadata.version("inference-autopilot")
    except importlib.metadata.PackageNotFoundError:
        return source_version()


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


def parse_yes_no(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"y", "yes", "true", "1"}:
        return True
    if normalized in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"{name} must be yes or no")


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
    dataset_name = value(
        "dataset_name",
        "Workload data: synthetic (fixed token shape), custom (real JSONL), or sharegpt (real JSON)",
        "synthetic",
    ).strip().lower()
    if dataset_name not in {"synthetic", "custom", "sharegpt"}:
        raise ValueError("dataset name must be synthetic, custom, or sharegpt")
    dataset_path = ""
    apply_chat_template = False
    if dataset_name in {"custom", "sharegpt"}:
        dataset_path = value(
            "dataset_path",
            "Absolute dataset path on this server (custom = JSONL; sharegpt = JSON array)",
        )
        raw_apply_template = getattr(args, "apply_chat_template", None)
        if raw_apply_template is None:
            raw_apply_template = (
                ask("Apply the model chat template to each real prompt (yes/no)", "yes")
                if interactive else "yes"
            )
            apply_chat_template = parse_yes_no("apply-chat-template", raw_apply_template)
        else:
            apply_chat_template = bool(raw_apply_template)
    if mode not in {"online_latency", "offline_throughput"}:
        raise ValueError("deployment mode must be online_latency or offline_throughput")
    # Latency limits are optional hard acceptance gates, not benchmark
    # durations. A task must use one statistic family so that it cannot
    # accidentally mix a tail E2E gate with an average TTFT gate.
    p99_argument_names = (
        "p99_e2e_latency_ms", "p99_ttft_ms", "p99_tpot_ms",
    )
    avg_argument_names = (
        "avg_e2e_latency_ms", "avg_ttft_ms", "avg_tpot_ms",
    )
    supplied_p99 = any(getattr(args, field, None) not in {None, "", "0", 0} for field in p99_argument_names)
    supplied_avg = any(getattr(args, field, None) not in {None, "", "0", 0} for field in avg_argument_names)
    if supplied_p99 and supplied_avg:
        raise ValueError("latency limits must use either p99 or avg, not both")
    requested_statistic = getattr(args, "latency_slo_statistic", None)
    if requested_statistic is None and interactive and not (supplied_p99 or supplied_avg):
        requested_statistic = ask(
            "Latency SLO statistic: p99 (tail) or avg (arithmetic mean); choose one family for all limits; blank = no latency SLO",
            "",
        )
    statistic = str(requested_statistic or ("p99" if supplied_p99 else "avg" if supplied_avg else "")).strip().lower()
    if statistic in {"", "none"}:
        statistic = ""
    if statistic not in {"", "p99", "avg"}:
        raise ValueError("latency SLO statistic must be p99, avg, or blank")
    if statistic == "p99" and supplied_avg or statistic == "avg" and supplied_p99:
        raise ValueError("latency SLO statistic does not match the supplied latency limit options")
    latency_fields = {
        "p99": (
            ("p99_e2e_latency_ms", "p99 E2E"),
            ("p99_ttft_ms", "p99 TTFT"),
            ("p99_tpot_ms", "p99 TPOT"),
        ),
        "avg": (
            ("avg_e2e_latency_ms", "average E2E"),
            ("avg_ttft_ms", "average TTFT"),
            ("avg_tpot_ms", "average TPOT"),
        ),
    }
    optional_latency_slos: dict[str, str] = {}
    for field, label in latency_fields.get(statistic, ()):
        optional_latency_slos[field] = value(
            field,
            f"Optional {label} latency limit in ms (blank or 0 = no limit)",
            "",
        )
    metric_names = {
        "avg_e2e_latency_ms": "mean_e2e_latency_ms",
        "avg_ttft_ms": "mean_ttft_ms",
        "avg_tpot_ms": "mean_tpot_ms",
    }
    slo = {
        metric_names.get(name, name): limit
        for name, raw in optional_latency_slos.items()
        if raw and (limit := parse_nonnegative_number(name, raw)) > 0
    }
    offline_mode = mode == "offline_throughput"
    offline_unbounded = offline_mode and not slo
    if offline_mode:
        # Every offline task starts with an unbounded benchmark. For an SLO
        # task, the observed saturation capacity becomes the upper bracket of
        # the following SLO search; without an SLO it remains unbounded.
        if getattr(args, "max_concurrency", None):
            raise ValueError(
                "offline tasks do not accept --max-concurrency; InferOpt measures unbounded "
                "startup capacity and derives any SLO probes automatically"
            )
        max_concurrency = 1
        if getattr(args, "concurrency_points", None):
            raise ValueError(
                "concurrency points apply only to online tasks; offline tasks first "
                "measure unbounded client concurrency and derive SLO probes automatically"
            )
        concurrency_points: list[int] = []
        if interactive:
            print(
                "Offline mode: the first benchmark leaves client concurrency unbounded "
                "and omits bench_serving --max-concurrency. "
                "Initial request window: at least 40 requests; it expands after startup from the observed capacity."
            )
    else:
        # An online task with an SLO is an admission-capacity search, not a
        # fixed-client-concurrency benchmark.  Do not ask users to invent an
        # initial pressure point: the executor probes SGLang's resolved
        # max_running_requests first, then brackets the SLO boundary.
        requested_points = getattr(args, "concurrency_points", None)
        adaptive_runtime_capacity = bool(slo) and not requested_points
        fallback_max_concurrency: int | None = None
        if adaptive_runtime_capacity:
            raw_fallback = getattr(args, "fallback_max_concurrency", None)
            if raw_fallback is None and getattr(args, "max_concurrency", None) is not None:
                # Preserve old non-interactive usage: --max-concurrency is a
                # fallback only when automatic runtime capacity is unavailable.
                raw_fallback = getattr(args, "max_concurrency")
            if raw_fallback is None and interactive:
                raw_fallback = ask(
                    "Optional fallback concurrency only if SGLang cannot report max_running_requests (blank = fail clearly)",
                    "",
                )
            if raw_fallback not in {None, ""}:
                fallback_max_concurrency = int(raw_fallback)
                if fallback_max_concurrency <= 0:
                    raise ValueError("fallback concurrency must be positive")
            max_concurrency = None
            concurrency_points = []
            if interactive:
                print(
                    "Online SLO calibration will start from SGLang's runtime-resolved "
                    "max_running_requests, then automatically bracket the highest SLO-safe concurrency."
                )
        else:
            default_concurrency = "8" if mode == "online_latency" else "64"
            max_concurrency = int(value(
                "max_concurrency",
                "Fixed client concurrency target (used because no adaptive latency-SLO search is requested)",
                default_concurrency,
            ))
            concurrency_points = parse_concurrency_points(value(
                "concurrency_points", "Concurrency points to measure, comma or space separated (blank = automatic 1,2,4,... sweep)", ""
            ))
            if concurrency_points and concurrency_points[-1] != max_concurrency:
                raise ValueError("explicit concurrency points must include the target concurrency as their largest value")
    shared_prefix_tokens = 0
    if dataset_name == "synthetic":
        shared_prefix_tokens = int(value(
            "shared_prefix_tokens",
            "Shared prefix tokens (synthetic common prefix; 0 uses independent random token IDs)",
            "0",
        ))
    experiment_mode = value(
        "experiment_mode",
        "Experiment intensity: fast (narrow), balanced (default), or max (widest search)",
        "balanced",
    )
    # Keep old task-generation scripts usable without exposing the retired
    # name in the current CLI.
    if experiment_mode == "rigorous":
        experiment_mode = "max"
    visible_gpus = value(
        "cuda_visible_devices",
        "GPUs to use (press Enter or type all for every visible GPU; otherwise use comma-separated indexes/UUIDs, e.g. 0 or 0,1,2; no spaces)",
        "all",
    )
    visibility_env = visibility_environment(visible_gpus)
    inventory = autopilot.parse_nvidia_inventory() or autopilot.parse_amd_inventory()
    detected_gpus = (
        autopilot.selected_gpus({"env": visibility_env}, inventory)
        if isinstance(inventory, dict) else []
    )
    detected_count = len(detected_gpus)
    if interactive and detected_gpus:
        summary = ", ".join(
            f"GPU {gpu.get('index')} {gpu.get('name')} "
            f"({int(gpu.get('memory_mib', 0)) // 1024} GiB, "
            f"{int(gpu.get('memory_used_mib', 0))} MiB used, "
            f"{float(gpu.get('utilization_gpu_pct', 0)):.0f}% util)"
            for gpu in detected_gpus
        )
        print(f"Detected visible accelerators: {summary}")
        busy = [
            str(gpu.get("index")) for gpu in detected_gpus
            if int(gpu.get("memory_used_mib", 0)) >= 512
            or float(gpu.get("utilization_gpu_pct", 0)) >= 10
        ]
        if busy:
            print(
                "Warning: selected GPUs currently appear busy: " + ", ".join(busy)
                + ". Choose a smaller/explicit GPU selection if those jobs are not owned by this run."
            )
    max_gpus = int(value(
        "max_gpus",
        "Maximum GPUs InferOpt may occupy at once (the scheduler derives trial concurrency from each configuration's TP/PP/DP size)",
        str(max(1, detected_count)),
    ))
    if max_gpus < 1:
        raise ValueError("maximum GPUs must be positive")
    if detected_count and max_gpus > detected_count:
        raise ValueError(
            f"maximum GPUs cannot exceed the {detected_count} selected visible GPUs"
        )
    configured_parallel_trials = getattr(args, "parallel_trials", None)
    parallel_trials = configured_parallel_trials if configured_parallel_trials is not None else max_gpus
    if not 1 <= parallel_trials <= 16:
        raise ValueError("parallel trials must be an integer from 1 through 16")
    experiment_profiles = {
        "fast": {
            "search_depth": "evidence_guided", "max_trials": 24, "max_gpu_hours": 1.5,
            "max_wall_time_minutes": 90,
        },
        "balanced": {
            # This is a safety cap rather than prepaid runtime.  Three GPU
            # hours was too small for a long-context two-GPU run to reach the
            # confirmation stage after profiling and 40 accurate candidates.
            "search_depth": "evidence_guided", "max_trials": 40, "max_gpu_hours": 6,
            "max_wall_time_minutes": 360,
        },
        "max": {
            "search_depth": "thorough", "max_trials": 96, "max_gpu_hours": 10,
            "max_wall_time_minutes": 720,
        },
    }
    if experiment_mode not in experiment_profiles:
        raise ValueError("experiment intensity must be fast, balanced, or max")
    profile = experiment_profiles[experiment_mode]

    def positive_override(name: str, default: int | float, cast: type[int] | type[float]) -> int | float:
        raw = getattr(args, name, None)
        if raw is None:
            return default
        parsed = cast(raw)
        if parsed <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
        return parsed

    max_trials = positive_override("max_trials", profile["max_trials"], int)
    max_gpu_hours = positive_override("max_gpu_hours", profile["max_gpu_hours"], float)
    max_wall_time_minutes = positive_override(
        "max_wall_time_minutes", profile["max_wall_time_minutes"], float
    )
    raw_allow_download = getattr(args, "allow_download", None)
    allow_download = (
        raw_allow_download
        if isinstance(raw_allow_download, bool)
        else parse_yes_no(
            "allow-download",
            value(
                "allow_download",
                "Allow downloading official SGLang Cookbook docs when the selected checkout has no matching local recipe (yes/no)",
                "yes",
            ),
        )
    )
    confirmation_repetitions = positive_override(
        "confirmation_repetitions", 3 if experiment_mode == "max" else 2, int
    )
    raw_kv_precision = getattr(args, "allow_kv_cache_precision_tuning", None)
    allow_kv_cache_precision_tuning = (
        raw_kv_precision
        if isinstance(raw_kv_precision, bool)
        else parse_yes_no(
            "allow-kv-cache-precision-tuning",
            value(
                "allow_kv_cache_precision_tuning",
                "Allow FP8 KV-cache candidates (may affect output quality; yes/no)",
                "no",
            ),
        )
    )
    configured_attestation = getattr(args, "quality_attestation", None)
    configured_quality_dataset = getattr(args, "quality_evaluation_dataset", None)
    quality_attestation: str | None = None
    quality_evaluation_dataset: str | None = None
    if allow_kv_cache_precision_tuning:
        raw_quality_dataset = str(value(
            "quality_evaluation_dataset",
            "Optional quality-evaluation dataset used by the external attestation (blank = performance-only candidate)",
            "",
        )).strip()
        if raw_quality_dataset:
            dataset_path = Path(raw_quality_dataset).expanduser().resolve()
            if not dataset_path.is_file():
                raise ValueError("quality evaluation dataset must be an existing file")
            quality_evaluation_dataset = str(dataset_path)
        raw_attestation = str(value(
            "quality_attestation",
            "Optional external quality-attestation JSON for deployable FP8 KV recommendations (blank = performance-only candidate)",
            "",
        )).strip()
        if raw_attestation:
            attestation_path = Path(raw_attestation).expanduser().resolve()
            if not attestation_path.is_file():
                raise ValueError("quality attestation must be an existing JSON file")
            quality_attestation = str(attestation_path)
        if quality_attestation is not None and quality_evaluation_dataset is None:
            raise ValueError(
                "quality attestation requires a quality evaluation dataset for hash verification"
            )
    elif configured_attestation is not None or configured_quality_dataset is not None:
        raise ValueError(
            "--quality-attestation and --quality-evaluation-dataset require "
            "--allow-kv-cache-precision-tuning"
        )
    raw_history = getattr(args, "enable_history", None)
    enable_history = (
        raw_history
        if isinstance(raw_history, bool)
        else parse_yes_no(
            "enable-history",
            value(
                "enable_history",
                "Save private local benchmark history for future exactly compatible runs (yes/no)",
                "yes",
            ),
        )
    )
    configured_history_database = getattr(args, "history_database", None)
    configured_warm_start_limit = getattr(args, "warm_start_limit", None)
    history_database: str | None = None
    warm_start_limit: int | None = None
    if enable_history:
        history_database = value(
            "history_database",
            "Private SQLite benchmark-history database",
            str(Path(output_dir).expanduser() / "inferopt-history.sqlite3"),
        )
        warm_start_limit = int(value(
            "warm_start_limit",
            "Maximum number of exactly compatible historical configurations used only as weak priors",
            "5",
        ))
        if not 0 <= warm_start_limit <= 20:
            raise ValueError("warm-start-limit must be an integer from 0 through 20")
    elif configured_history_database is not None or configured_warm_start_limit is not None:
        raise ValueError(
            "--history-database and --warm-start-limit require --enable-history"
        )
    parameter_evolution_mode = value(
        "parameter_evolution_mode",
        "New SGLang parameter policy: conservative (audit only) or experimental (bounded automatic exploration)",
        "conservative",
    ).strip().lower()
    if parameter_evolution_mode not in {"conservative", "experimental"}:
        raise ValueError("new parameter policy must be conservative or experimental")
    configured_evolution_budget = getattr(args, "parameter_evolution_budget_pct", None)
    configured_provisional_trials = getattr(args, "max_provisional_trials", None)
    if parameter_evolution_mode == "conservative" and (
        configured_evolution_budget is not None
        or configured_provisional_trials is not None
    ):
        raise ValueError(
            "--parameter-evolution-budget-pct and --max-provisional-trials "
            "require --parameter-evolution-mode experimental"
        )
    evolution_budget_pct: float | None = None
    if parameter_evolution_mode == "experimental":
        evolution_budget_pct = float(value(
            "parameter_evolution_budget_pct",
            "Maximum discovery budget percentage for provisional new parameters (0-25; confirmation is never reduced)",
            "10",
        ))
        if not 0 <= evolution_budget_pct <= 25:
            raise ValueError("new parameter exploration budget percentage must be between 0 and 25")
    raw_cost = str(value(
        "cost_per_gpu_hour",
        "Optional cost per GPU-hour for report-only cost/token estimates (blank skips all cost questions)",
        "",
    )).strip()
    cost_per_gpu_hour = float(raw_cost) if raw_cost else None
    configured_currency = getattr(args, "currency", None)
    currency: str | None = None
    if cost_per_gpu_hour is not None:
        if cost_per_gpu_hour <= 0:
            raise ValueError("cost-per-gpu-hour must be positive when provided")
        currency = value("currency", "Currency label for cost reporting", "USD")
    elif configured_currency is not None:
        raise ValueError("--currency requires --cost-per-gpu-hour")
    # Measurement fidelity is deliberately independent of search breadth.
    # Start with five pressure waves; p99 SLOs use ten. Duration-based reruns
    # expand either window only when the completed run is still too short.
    planning_concurrency = max_concurrency or 1
    initial_request_count = 40 if offline_mode else max(40, planning_concurrency * 5)
    initial_warmup_count = min(32, max(8, math.ceil(initial_request_count / 10)))
    confirmation_requests = max(1, math.ceil(initial_request_count / 2))
    if slo:
        confirmation_requests = max(confirmation_requests, planning_concurrency * 5)
        if any(key.startswith("p99_") for key in slo):
            confirmation_requests = max(confirmation_requests, planning_concurrency * 10)

    calibration_steps = 1
    calibration_value = 1
    while calibration_value < planning_concurrency:
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
        "max_gpus": max_gpus,
        "parallel_trials": parallel_trials,
        "search_depth": profile["search_depth"],
        "workload": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            **({} if offline_mode or max_concurrency is None else {"max_concurrency": max_concurrency}),
            "unbounded_client_concurrency": offline_unbounded,
            "unbounded_initial_probe": offline_mode,
            **({"initial_backlog_requests": initial_request_count} if offline_mode else {}),
            "request_rate": "inf",
            "num_prompts": initial_request_count,
        },
        "slo": slo,
        "objective": {
            "metric": "total_throughput_tps" if mode == "offline_throughput" else "request_throughput_rps",
            "direction": "maximize",
            "resource_scope": (
                getattr(args, "resource_scope", None)
                or ("per_gpu" if mode == "offline_throughput" else "per_service")
            ),
            "min_improvement_pct": 1,
            "max_regression_pct": 5,
        },
        "budget": {
            "max_trials": max_trials,
            "max_gpu_hours": max_gpu_hours,
            "max_wall_time_minutes": max_wall_time_minutes,
        },
        "profiling": {"enabled": True},
        "confirmation_repetitions": confirmation_repetitions,
        "measurement": {
            "warmup_requests": initial_warmup_count,
            "min_measurement_requests": initial_request_count,
            "min_measurement_seconds": 15,
            "confirmation_requests": confirmation_requests,
            "p99_request_waves": 10,
            "adaptive_confirmation_cv_pct": 5,
            "adaptive_confirmation_max_repetitions": max(3, confirmation_repetitions),
            "adaptive_confirmation_min_measurement_seconds": 30,
            "bayesian_sequential": True,
            "bayesian_min_blocks": confirmation_repetitions,
            "bayesian_max_blocks": 6,
            "bayesian_accept_probability": 0.95,
            "bayesian_reject_probability": 0.05,
            "bayesian_prior_mean_pct": 0.0,
            "bayesian_prior_strength": 0.01,
        },
        "calibration": {
            "enabled": not offline_unbounded, "min_concurrency": 1,
            **({} if offline_mode or max_concurrency is None else {"max_concurrency": max_concurrency}),
            **({"fallback_max_concurrency": fallback_max_concurrency} if not offline_mode and fallback_max_concurrency is not None else {}),
            "strategy": "adaptive", "max_steps": 8 if offline_mode or max_concurrency is None else calibration_steps, "stop_on_slo_failure": True,
            **({"concurrencies": concurrency_points, "max_steps": len(concurrency_points)} if concurrency_points else {}),
        },
        "offline": True,
        "allow_download": allow_download,
        "deployment": {"allow_model_variant_recommendations": True, "allow_auto_model_switch": False},
        "quality": {
            "allow_kv_cache_precision_tuning": allow_kv_cache_precision_tuning,
            "max_regression_pct": 5,
            **(
                {"attestation_path": quality_attestation}
                if quality_attestation is not None else {}
            ),
            **(
                {"evaluation_dataset": quality_evaluation_dataset}
                if quality_evaluation_dataset is not None else {}
            ),
        },
        "history": (
            {
                "enabled": True,
                "database": str(Path(history_database).expanduser().resolve()),
                "warm_start_limit": warm_start_limit,
            }
            if enable_history else {"enabled": False}
        ),
        "parameter_evolution": (
            {
                "mode": "experimental",
                "exploration_budget_pct": evolution_budget_pct,
                "minimum_confidence": 0.8,
                **(
                    {"max_provisional_trials": int(configured_provisional_trials)}
                    if configured_provisional_trials is not None else {}
                ),
            }
            if parameter_evolution_mode == "experimental"
            else {"mode": "conservative"}
        ),
        **(
            {"economics": {
                "cost_per_gpu_hour": cost_per_gpu_hour,
                "currency": currency,
            }}
            if cost_per_gpu_hour is not None else {}
        ),
        "env": visibility_env,
    }
    if dataset_name in {"custom", "sharegpt"}:
        task["workload"]["dataset"] = {
            "name": dataset_name,
            "path": str(Path(dataset_path).expanduser().resolve()),
            "apply_chat_template": apply_chat_template,
        }
    if shared_prefix_tokens:
        prefix = shared_prefix_tokens
        if not 0 < prefix < input_tokens:
            raise ValueError("--shared-prefix-tokens must be between 1 and input tokens - 1")
        task["workload"]["prefix_reuse_ratio"] = prefix / input_tokens
        task["workload"]["shared_prefix"] = {
            "groups": 8,
            "prompts_per_group": max(1, math.ceil(task["workload"]["num_prompts"] / 8)),
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


def doctor(
    task: dict[str, Any], progress: autopilot.ProgressReporter | None = None,
) -> dict[str, Any]:
    errors = autopilot.validate_task(task)
    if errors:
        return {"status": "invalid_task", "errors": errors}
    task = autopilot.materialize_runtime_task(task)
    if progress:
        progress.emit("doctor", "discovering accelerators", completed=0, total=6)
    hardware = autopilot.parse_nvidia_inventory() or autopilot.parse_amd_inventory()
    if hardware is None:
        return {"status": "no_supported_accelerator", "errors": ["no NVIDIA or AMD accelerator inventory available"]}
    if progress:
        progress.emit("doctor", "reading local model metadata", completed=1, total=6)
    model = autopilot.model_inventory(task["model_path"])
    if progress:
        progress.emit("doctor", "checking the configured SGLang Python and CLI", completed=2, total=6)
    framework = autopilot.framework_evidence(task)
    if progress:
        progress.emit("doctor", "comparing the live ServerArgs contract", completed=3, total=6)
    try:
        parameter_evolution = autopilot.parameter_evolution_preflight(
            task, hardware, model, framework, progress=progress
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parameter_evolution = {
            "status": "unavailable", "reason": str(exc),
            "policy": "doctor continues; plan/run still require a valid current ServerArgs contract",
        }
    single_gpu = autopilot.single_gpu_feasibility(task, hardware, model)
    feasibility = autopilot.deployment_feasibility(task, hardware, model)
    if progress:
        progress.emit("doctor", "checking model/GPU deployment feasibility", completed=4, total=6)
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
    if progress:
        progress.emit("doctor", "checking Nsight tools and final blockers", completed=5, total=6)
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
    result = {
        "schema_version": 1,
        "status": "ready" if not blocking_errors else "attention_required",
        "blocking_errors": blocking_errors,
        "hardware": hardware,
        "model": model,
        "framework": framework,
        "parameter_evolution": parameter_evolution,
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
    if progress:
        progress.emit(
            "doctor", f"completed with status={result['status']}",
            completed=6, total=6,
        )
    return result


def tune_moe(task: dict[str, Any], profile_path: str, result_path: str,
             output_dir: str, timeout_minutes: float, max_batch_sizes: int,
             validate_end_to_end: bool = True,
             topk_ids_dir: str | None = None,
             progress: autopilot.ProgressReporter | None = None) -> dict[str, Any]:
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
    progress = progress or autopilot.ProgressReporter()
    progress.emit(
        "optional-moe",
        "discovering the current SGLang/model/GPU contract for the approved kernel search",
    )
    discovery = autopilot.discover(execution_task, progress=progress)
    plan = autopilot.moe_kernel_optimization_plan(execution_task, discovery, profile)
    progress.emit(
        "optional-moe",
        "starting explicitly approved high-cost kernel search; this is outside inferopt run",
    )
    execution = autopilot.execute_moe_kernel_tuning(
        execution_task, plan, Path(output_dir).expanduser().resolve(),
        progress=progress,
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


def multiline_shell_command(command: list[Any]) -> str:
    """Render a command as a directly copyable multiline shell block."""
    groups: list[str] = []
    index = 0
    while index < len(command):
        token = str(command[index])
        if (
            token.startswith("-")
            and index + 1 < len(command)
            and not str(command[index + 1]).startswith("-")
        ):
            groups.append(
                f"{shlex.quote(token)} {shlex.quote(str(command[index + 1]))}"
            )
            index += 2
        else:
            groups.append(shlex.quote(token))
            index += 1
    if not groups:
        return ""
    return groups[0] + " \\\n  " + " \\\n  ".join(groups[1:])


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
        f"- Experiment intensity: `{final.get('experiment_mode', 'unknown')}`",
        f"- Decision: `{final.get('recommendation_status', 'unknown')}`",
        f"- Deployable: `{final.get('deployable', False)}`",
        f"- Parameter-routing diagnosis: `{routing_primary}`",
        f"- Raw Nsight diagnosis: `{diagnosis.get('primary_bottleneck', 'unavailable')}`",
        f"- Nsight timing comparable to unprofiled baseline: `{diagnosis.get('profiling_run_performance_comparable', 'unknown')}`",
        f"- Nsight comparison source: `{diagnosis.get('profile_comparison_source', 'unavailable')}`; "
        f"throughput regression `{diagnosis.get('profile_throughput_regression_pct', 'unavailable')}%`",
        f"- Current process elapsed: `{final.get('current_process_elapsed_sec', final.get('elapsed_sec', 'unknown'))}` seconds",
        f"- Logical experiment elapsed: `{final.get('logical_experiment_elapsed_sec', final.get('elapsed_sec', 'unknown'))}` seconds",
        "",
    ]
    comparison = recommendation.get("comparison", {}) if isinstance(recommendation, dict) else {}
    objective_metric = comparison.get("objective_metric", "unavailable")
    interval = comparison.get("confidence_interval", {})
    posterior = comparison.get("bayesian_posterior", {})
    lines.extend(["## Executive Summary", ""])
    if recommendation:
        lines.extend([
            f"- Objective: `{objective_metric}`",
            f"- Resource scope: `{comparison.get('resource_scope', 'per_service')}`",
            f"- Baseline: `{comparison.get('baseline', 'unavailable')}`",
            f"- Recommended: `{comparison.get('candidate', 'baseline retained')}`",
            f"- Measured improvement: `{comparison.get('improvement_pct', 0):.3f}%`"
            if isinstance(comparison.get("improvement_pct"), (int, float))
            else "- Measured improvement: `unavailable`",
            (
                f"- 95% confidence interval: `[{interval.get('lower_pct'):.3f}%, "
                f"{interval.get('upper_pct'):.3f}%]`"
                if isinstance(interval, dict)
                and isinstance(interval.get("lower_pct"), (int, float))
                and isinstance(interval.get("upper_pct"), (int, float))
                else "- 95% confidence interval: `unavailable`"
            ),
            (
                f"- P(gain > configured minimum): "
                f"`{posterior.get('probability_improvement_gt_minimum'):.4f}`"
                if isinstance(posterior, dict)
                and isinstance(posterior.get("probability_improvement_gt_minimum"), (int, float))
                else "- P(gain > configured minimum): `unavailable`"
            ),
            f"- Bottleneck: `{search_plan.get('canonical_bottleneck', {}).get('primary', final.get('bottleneck_class', routing_primary))}`",
        ])
        if comparison.get("resource_scope") == "per_gpu":
            lines.extend([
                f"- Raw per-service baseline/recommended: "
                f"`{comparison.get('raw_baseline')}` / `{comparison.get('raw_candidate')}`",
                f"- Baseline/recommended GPU count: "
                f"`{comparison.get('baseline_accelerator_count')}` / "
                f"`{comparison.get('candidate_accelerator_count')}`",
            ])
    else:
        lines.extend([
            "- Recommended deployment: `none`",
            f"- Reason: {final.get('recommendation_reason', 'insufficient evidence')}",
        ])
    summary_command = final.get("deployment_command_minimal")
    if isinstance(summary_command, list):
        lines.extend(["", "```bash", shlex.join(str(item) for item in summary_command), "```"])
    host_plan = final.get("host_deployment_plan", {})
    resume_evidence = final.get("resume_evidence", {})
    if isinstance(resume_evidence, dict) and resume_evidence.get("resumed"):
        lines.extend([
            "", "## Resume Evidence", "",
            f"- Raw profile reused: `{resume_evidence.get('profile_reused', False)}`",
            f"- Screening reused: `{resume_evidence.get('screening_reused', False)}`",
            f"- Search policy changed: `{resume_evidence.get('search_policy_changed', False)}`",
            f"- Policy: {resume_evidence.get('policy', 'unavailable')}",
        ])
    if isinstance(host_plan, dict) and host_plan:
        lines.extend([
            "",
            f"- Confirmed deployment scope: `{host_plan.get('confirmed_scope', 'unknown')}`",
            f"- Host aggregate measured: `{host_plan.get('measured_host_aggregate', False)}`",
            f"- Possible replica layout: `{host_plan.get('possible_replica_count', 0)}` "
            f"replicas × `{host_plan.get('gpus_per_service', 'unknown')}` GPUs/service",
            f"- Scope policy: {host_plan.get('policy', 'unavailable')}",
        ])
    lines.append("")
    discovery = final.get("discovery", {})
    evolution = final.get("parameter_evolution") or search_plan.get(
        "parameter_evolution", {}
    )
    if isinstance(evolution, dict) and evolution:
        exploration = evolution.get("exploration_budget", {})
        added_parameters = evolution.get("added", [])
        removed_parameters = evolution.get("removed", [])
        changed_parameters = evolution.get("changed_parameters", [])
        lines.extend([
            "## SGLang Parameter Evolution",
            "",
            f"- Current contract hash: `{evolution.get('contract_hash', 'unavailable')}`",
            f"- Contract diff status: `{evolution.get('diff_status', 'unavailable')}`",
            f"- Added parameters: `{len(added_parameters)}`; sample `{added_parameters[:20]}`",
            f"- Removed parameters: `{len(removed_parameters)}`; sample `{removed_parameters[:20]}`",
            f"- Changed parameters: `{len(changed_parameters)}`; sample `{changed_parameters[:20]}`",
            f"- State counts: `{evolution.get('state_counts', {})}`",
            f"- Policy: `{evolution.get('policy', {}).get('mode', 'conservative')}`",
            f"- Provisional exploration slots: `{exploration.get('slots', 0)}`",
            f"- Selected provisional parameters: `{evolution.get('selected_for_exploration', [])}`",
            f"- Contract/evidence persistence: `{evolution.get('persistence', {})}`",
            "- Safety policy: every parameter is audited. Newly added provisional flags require "
            "explicit experimental mode; existing uncovered flags still require bounded high-confidence "
            "semantics and all common context/dependency/risk gates.",
            "- Confirmation budget is never reduced by provisional exploration.",
            "",
        ])
    capability_registry = search_plan.get("parameter_capability_registry", {})
    if isinstance(capability_registry, dict) and capability_registry:
        semantic = capability_registry.get("semantic_selection", {})
        configurations = semantic.get("configurations", []) if isinstance(semantic, dict) else []
        decisions = semantic.get("decisions", []) if isinstance(semantic, dict) else []
        lines.extend([
            "## Parameter Capability Registry",
            "",
            f"- Audited current ServerArgs parameters: `"
            f"{capability_registry.get('audited_parameter_count', 0)}`",
            f"- Classification counts: `{capability_registry.get('state_counts', {})}`",
            "- Selection policy: every parameter uses the same semantic, applicability, "
            "dependency/conflict, risk, bottleneck/workload, and measured-result gates.",
            f"- Context: `{semantic.get('context', {}) if isinstance(semantic, dict) else {}}`",
        ])
        if configurations:
            lines.append("- Semantically selected configurations:")
            for item in configurations:
                lines.append(
                    f"  - `{item.get('name', 'unknown')}`: `{item.get('config', {})}` "
                    f"(mechanism `{item.get('mechanism', 'unknown')}`, confidence "
                    f"`{item.get('confidence', 'unknown')}`)"
                )
        else:
            lines.append("- Semantically selected configurations: `none`")
        rejected = [
            item for item in decisions
            if isinstance(item, dict) and not item.get("relevant")
        ]
        if rejected:
            lines.append("- Context-rejected high-confidence parameters:")
            for item in rejected[:20]:
                lines.append(
                    f"  - `{item.get('parameter', 'unknown')}`: "
                    f"{'; '.join(str(value) for value in item.get('reasons', []))}"
                )
        lines.append("")
    startup_acceleration = final.get(
        "startup_acceleration", search_plan.get("startup_acceleration", {})
    )
    if isinstance(startup_acceleration, dict) and startup_acceleration:
        profiled_startup = startup_acceleration.get("profiled_startup", {})
        lines.extend([
            "## Startup Acceleration",
            "",
            f"- Feature: `{startup_acceleration.get('feature', 'unknown')}`",
            f"- SGLang version: `{startup_acceleration.get('sglang_version', 'unknown')}`",
            f"- Eligible for an owned persistent daemon: `{startup_acceleration.get('eligible', False)}`",
            f"- Automatically enabled for this run: `{startup_acceleration.get('automatic_execution_enabled', False)}`",
            f"- Decision: `{startup_acceleration.get('decision', 'unknown')}`",
            f"- Reasons: `{startup_acceleration.get('reasons', [])}`",
            f"- Profiled startup telemetry: `{profiled_startup}`",
            f"- Required capacity pins: `{startup_acceleration.get('required_capacity_pins', {})}`",
            f"- Safety: {startup_acceleration.get('observed_safety_rule', 'unavailable')}",
            f"- Lifecycle: {startup_acceleration.get('lifecycle_policy', 'unavailable')}",
            "",
        ])
    cookbook_knowledge = (
        discovery.get("cookbook", {}) if isinstance(discovery, dict) else {}
    )
    local_cookbook = cookbook_knowledge.get("local_checkout", {}) if isinstance(cookbook_knowledge, dict) else {}
    if isinstance(local_cookbook, dict) and local_cookbook.get("status") == "available":
        documents = local_cookbook.get("documents", [])
        recipes = local_cookbook.get("recipes", [])
        lines.extend(["## Cookbook Knowledge", ""])
        lines.append("- Source: local SGLang checkout; no external page was required for this run.")
        for document in documents:
            lines.append(
                f"- Matched page: `{document.get('path', 'unknown')}` "
                f"(SGLang commit `{document.get('commit') or 'unavailable'}`, "
                f"SHA-256 `{document.get('sha256', 'unavailable')}`)"
            )
        if recipes:
            lines.append(
                "- Parsed launch recipes: `" + "`, `".join(
                    str(recipe.get("name", "unknown")) for recipe in recipes
                ) + "`"
            )
        for excluded in local_cookbook.get("excluded_recipes", []):
            lines.append(
                f"- Rejected documented variant `{excluded.get('name', 'unknown')}` "
                f"({excluded.get('documented_model') or 'unspecified checkpoint'}): "
                f"{excluded.get('reason', 'no reason recorded')}"
            )
        for behavior in local_cookbook.get("automatic_behaviors", []):
            lines.append(
                f"- Documented automatic behavior `{behavior.get('parameter', 'unknown')}`: "
                f"{behavior.get('evidence', 'no evidence text')} "
                f"(policy: {behavior.get('policy', 'record only')})"
            )
        lines.append(
            "- Hardware policy: exact verified cells retain their atomic topology; same-architecture "
            "cells are locally adapted and remeasured; cross-architecture cells are evidence-only."
        )
        lines.append("")
    cookbook_preflight = final.get("cookbook_preflight", {})
    if isinstance(cookbook_preflight, dict):
        candidates = cookbook_preflight.get("candidate_bundles", [])
        exclusions = cookbook_preflight.get("excluded_bundles", [])
        hardware_priors = cookbook_preflight.get("hardware_prior_candidates", [])
        if candidates or exclusions or hardware_priors:
            lines.extend(["## Cookbook Qualification", ""])
            if candidates:
                for bundle in candidates:
                    evidence = bundle.get("selection_evidence", {})
                    lines.append(
                        f"- Candidate `{bundle.get('name', 'unknown')}`: qualification "
                        f"`{evidence.get('hardware_qualification', 'generic')}`, documented "
                        f"hardware `{evidence.get('documented_hardware', [])}`, detected "
                        f"hardware `{evidence.get('detected_hardware', [])}`, adaptations "
                        f"`{evidence.get('adaptation_changes', [])}`"
                    )
            if hardware_priors:
                lines.append(
                    f"- Evidence-only one-factor priors: `{hardware_priors}`"
                )
            for excluded in exclusions:
                lines.append(
                    f"- Excluded `{excluded.get('name', 'unknown')}`: "
                    f"{excluded.get('reason', 'no reason recorded')}"
                )
            lines.append("")
    cookbook_snapshot = final.get("cookbook_snapshot", {})
    if isinstance(cookbook_snapshot, dict) and cookbook_snapshot:
        status = cookbook_snapshot.get("status", "unknown")
        if status != "available":
            lines.extend([
                "## Cookbook Availability", "",
                f"- Snapshot status: `{status}`",
                f"- Reason: `{cookbook_snapshot.get('reason', 'unavailable')}`",
                "- Model-specific Cookbook bundles were not eligible for this run; this is not a full recipe search.",
                "",
            ])
    automatic_behaviors = final.get("cookbook_automatic_behaviors", [])
    if isinstance(automatic_behaviors, list) and automatic_behaviors:
        lines.extend(["## Cookbook Automatic Behaviors", ""])
        for behavior in automatic_behaviors:
            lines.append(
                f"- `{behavior.get('parameter', 'unknown')}`: expected "
                f"`{behavior.get('expected_enabled')}`, observed "
                f"`{behavior.get('observed_resolved_value')}`, status "
                f"`{behavior.get('status')}` — {behavior.get('evidence', '')}"
            )
        lines.append("")
    if diagnosis:
        shares = diagnosis.get("shares_pct", {})
        kernel_shares = diagnosis.get("gpu_kernel_shares_pct", {
            key.removesuffix("_kernels"): value
            for key, value in shares.items() if key.endswith("_kernels")
        })
        api_shares = diagnosis.get("cuda_api_time_shares_pct", {
            key.removeprefix("cuda_").removesuffix("_apis"): value
            for key, value in shares.items() if key.endswith("_apis")
        })
        activity_shares = diagnosis.get("gpu_activity_shares_pct", {
            "memory_operations": shares.get("gpu_memops_within_activity")
        })
        top_kernels = diagnosis.get("top_kernels", [])
        top_kernel_families = diagnosis.get("top_kernel_families", [])
        top_apis = diagnosis.get("top_cuda_apis", [])
        timing_comparable = diagnosis.get("profiling_run_performance_comparable") is True
        lines.extend([
            "## Nsight Systems Evidence",
            "",
            "Kernel percentages below are shares of total GPU kernel time, not shares of the full profile wall-clock timeline.",
            f"- GPU timeline active/gap: `{diagnosis.get('gpu_timeline_active_pct', 'unknown')}%` / `{diagnosis.get('gpu_timeline_gap_pct', 'unknown')}%`",
            f"- GPU kernel-time shares: `{json.dumps(kernel_shares, sort_keys=True)}`",
            f"- CPU CUDA API-time shares: `{json.dumps(api_shares, sort_keys=True)}`",
            f"- GPU activity shares: `{json.dumps(activity_shares, sort_keys=True)}`",
            f"- Average CUDA launch/queue latency: `{diagnosis.get('avg_launch_latency_ns', 'unknown')} ns` / `{diagnosis.get('avg_kernel_queue_latency_ns', 'unknown')} ns`",
            f"- Top GPU kernels: `{json.dumps([{**item, 'name': str(item.get('name', ''))[:160]} for item in top_kernels[:5]], sort_keys=True)}`",
            f"- Top GPU operator families: `{json.dumps([{'name': item.get('name'), 'time_pct': item.get('time_pct'), 'instances': item.get('instances')} for item in top_kernel_families[:5]], sort_keys=True)}`",
            f"- Top CUDA APIs: `{json.dumps(top_apis[:5], sort_keys=True)}`",
            "- Routing policy: " + (
                "kernel, timeline-gap, and CUDA API timing evidence may all influence parameter priority."
                if timing_comparable
                else "the profiled run was timing-distorted; host-gap and CUDA API timing are excluded, while relative GPU kernel-time shares remain eligible for backend routing."
            ),
            "- Limit: Nsys does not establish occupancy, memory-bandwidth saturation, or instruction stalls; those require a bounded NCU follow-up on a trace-proven hotspot.",
            "",
        ])
    roofline = profile.get("roofline", {}) if isinstance(profile, dict) else {}
    if isinstance(roofline, dict):
        lines.extend(["## Roofline Direction", ""])
        status = roofline.get("status", "roofline_unavailable")
        lines.append(f"- Status: `{status}`")
        if status == "available":
            lines.extend([
                f"- Classification: `{roofline.get('classification')}`",
                f"- Arithmetic intensity: `{roofline.get('arithmetic_intensity_flops_per_byte')}` FLOPs/byte",
                f"- Compute utilization: `{roofline.get('compute_utilization')}`",
                f"- Memory-bandwidth utilization: `{roofline.get('memory_bandwidth_utilization')}`",
                f"- Routing: {roofline.get('routing')}",
            ])
        else:
            lines.extend([
                f"- Reason: {roofline.get('reason', 'shape-matched NCU metrics unavailable')}",
                f"- Next step: {roofline.get('next_step', 'collect NCU metrics for the trace-proven hotspot')}",
            ])
        lines.append("")
    workload = final.get("analysis_workload", {})
    if isinstance(workload, dict):
        deployment = final.get("deployment_policy", {})
        if not isinstance(deployment, dict):
            deployment = {}
        deployment_mode = final.get("deployment_mode", deployment.get("mode"))
        offline_unbounded = deployment_mode == "offline_throughput" and not final.get("requested_slo")
        concurrency_lines = (
            [
                "- Client concurrency: `unbounded` (`bench_serving --max-concurrency` was omitted)",
                f"- Initial request window: `{workload.get('initial_backlog_requests', 'unknown')}` requests (expanded from SGLang runtime capacity after startup)",
            ]
            if offline_unbounded else [
                f"- Requested concurrency: `{final.get('calibration', {}).get('target_concurrency', workload.get('max_concurrency', 'unknown'))}`",
                f"- Selected SLO-safe execution concurrency: `{final.get('calibration', {}).get('selected_analysis_concurrency', workload.get('max_concurrency', 'unknown'))}`",
            ]
        )
        dataset = workload.get("dataset", {"name": "synthetic"})
        if not isinstance(dataset, dict):
            dataset = {"name": "unknown"}
        source = dataset.get("name", "synthetic")
        if source == "synthetic" and workload.get("shared_prefix"):
            source = "generated-shared-prefix"
        lines.extend([
            "## Workload Evidence",
            "",
            f"- Data source: `{source}`",
            f"- Dataset path: `{dataset.get('path', 'not applicable')}`",
            f"- Planning token shape: input `{workload.get('input_tokens', 'unknown')}`, output `{workload.get('output_tokens', 'unknown')}`",
            *concurrency_lines,
            "",
        ])
        p99_slos = [
            key for key in final.get("requested_slo", {})
            if isinstance(key, str) and key.startswith("p99_")
        ]
        if p99_slos:
            measurement = final.get("measurement_policy", {})
            selected_concurrency = final.get("calibration", {}).get(
                "selected_analysis_concurrency", workload.get("max_concurrency")
            )
            waves = measurement.get("p99_request_waves", 10)
            minimum_requests = (
                selected_concurrency * waves
                if isinstance(selected_concurrency, int) and isinstance(waves, int)
                else "see per-trial measurement_validity"
            )
            lines.extend([
                "### P99 Sample Policy",
                "",
                f"- Concurrency waves per measured window: `{waves}`",
                f"- Selected-concurrency request floor: `{minimum_requests}`",
                "- This concurrency-scaled policy limits experiment cost. At low concurrency it provides fewer than 100 observations, so empirical p99 behaves like a near-maximum and has lower statistical confidence; inspect repeated-window stability before deployment.",
                "",
            ])
    confirmation = final.get("confirmation")
    if isinstance(confirmation, dict):
        adaptive = confirmation.get("adaptive_confirmation", {})
        resident = bool(confirmation.get("resident_ab", False))
        session_policy = (
            "Baseline and candidate services remain resident across repeated windows."
            if resident else
            "Each measurement window used an independent server session; no resident reuse is claimed."
        )
        lines.extend([
            "## Confirmation Cost",
            "",
            f"- Independent measurement windows: `{confirmation.get('planned_trials', 'unknown')}`",
            f"- Model-loading server sessions: `{confirmation.get('planned_server_sessions', confirmation.get('planned_trials', 'unknown'))}`",
            f"- Resident multi-GPU A/B: `{confirmation.get('resident_ab', False)}`",
            f"- Parallel same-configuration replicas: `{confirmation.get('confirmation_replica_parallel', False)}`",
            f"- Replica policy: `{confirmation.get('confirmation_replica_policy', 'not enabled')}`",
            f"- Measurement order: `{confirmation.get('measurement_order', 'sequential resident sessions')}`",
            f"- Adaptive noise extension triggered: `{adaptive.get('triggered', False)}`",
            f"- Adaptive confirmation evidence: `{json.dumps(adaptive, sort_keys=True)}`",
            f"- Session policy: {session_policy} Nsight profiling is separate and is not counted as a performance baseline.",
            "",
        ])
        confirmation_attempts = final.get("confirmation_attempts", [])
        if isinstance(confirmation_attempts, list) and confirmation_attempts:
            lines.extend([
                "### Confirmation Tournament", "",
                "The first nominee is compared with the original baseline. Later nominees "
                "challenge the current confirmed champion directly, avoiding repeated loads of an inferior baseline.",
            ])
            for index, attempt in enumerate(confirmation_attempts, 1):
                lines.append(
                    f"- Attempt {index}: anchor `{attempt.get('comparison_anchor', 'original_baseline')}`, "
                    f"status `{attempt.get('recommendation_status', 'unknown')}`, windows "
                    f"`{attempt.get('completed_trials', 'unknown')}`."
                )
            lines.append("")
        if (
            final.get("deployment_policy", {}).get("mode") == "offline_throughput"
            and not final.get("requested_slo")
        ):
            execution_workload = final.get("execution_workload", {})
            capacity = execution_workload.get(
                "observed_practical_capacity",
                execution_workload.get("observed_admission_capacity"),
            )
            waves = autopilot.offline_saturation_waves(
                {"experiment_mode": final.get("experiment_mode", "balanced")},
                confirmation=True,
            )
            request_floor = (
                int(capacity) * waves
                if isinstance(capacity, int) and not isinstance(capacity, bool)
                else "unavailable"
            )
            lines.extend([
                f"- Offline practical capacity: `{capacity}`",
                f"- Initial capacity waves per confirmation window: `{waves}`",
                f"- Initial request floor per confirmation window: `{request_floor}`",
                f"- Initial request floor per configuration across the two paired blocks: "
                f"`{int(request_floor) * 2 if isinstance(request_floor, int) else 'unavailable'}` "
                f"({waves * 2} saturated waves).",
                f"- Adaptive policy: each window starts with {waves} saturated waves; paired Bayesian "
                "blocks provide the initial evidence per configuration, "
                "and additional complete blocks are reserved only when the posterior remains unresolved.",
                "",
            ])
        confidence_rows = [
            item for item in confirmation.get("aggregates", [])
            if item.get("kind") == "candidate"
            and isinstance(item.get("comparison", {}).get("confidence_interval"), dict)
        ]
        if confidence_rows:
            lines.extend(["### Statistical Decision", ""])
            for item in confidence_rows:
                comparison = item["comparison"]
                interval = comparison["confidence_interval"]
                lines.append(
                    f"- `{item.get('configuration_name')}`: point improvement "
                    f"`{comparison.get('improvement_pct', 0):.3f}%`, 95% CI "
                    f"`[{interval.get('lower_pct', 0):.3f}%, {interval.get('upper_pct', 0):.3f}%]`, "
                    f"statistically positive `{comparison.get('statistically_positive', False)}`"
                )
            lines.append("")
        sequential = confirmation.get("bayesian_sequential")
        if isinstance(sequential, dict):
            lines.extend([
                "### Bayesian Sequential Decision", "",
                f"- Action: `{sequential.get('action')}` after `{sequential.get('blocks')}` paired blocks",
                f"- P(gain > 0): `{sequential.get('probability_improvement_gt_zero', 0):.4f}`",
                f"- P(gain > configured minimum): `{sequential.get('probability_improvement_gt_minimum', 0):.4f}`",
                f"- Posterior mean improvement: `{sequential.get('posterior_mean_improvement_pct', 0):.3f}%`",
                "",
            ])
    calibration = final.get("calibration")
    if isinstance(calibration, dict) and calibration.get("points"):
        lines.extend([
            "## Capacity Calibration Cost",
            "",
            f"- Concurrency windows: `{len(calibration.get('points', []))}`",
            f"- Model-loading server sessions: `{calibration.get('server_sessions', 'unknown')}`",
            "- Adaptive maximum/halving/binary-search points reuse the same loaded baseline service.",
            "",
        ])
    if recommendation:
        lines.extend([
            "## Recommended Configuration",
            "",
            "```json",
            json.dumps(recommendation.get("config", recommendation), indent=2, sort_keys=True),
            "```",
        ])
        provisional = final.get("provisional_configuration")
        if isinstance(provisional, dict):
            comparison = provisional.get("comparison", {})
            lines.extend([
                "", "### Best Measured Unconfirmed Candidate", "",
                "A separate candidate remains promising but did not clear every confirmation gate:",
                "```json",
                json.dumps(provisional.get("config", provisional), indent=2, sort_keys=True),
                "```",
                f"- Point improvement: `{comparison.get('improvement_pct', 'unavailable')}%`",
                f"- Evidence state: `{provisional.get('evidence_state', 'unconfirmed')}`",
                f"- Remaining gates: `{provisional.get('rejection_reasons', [])}`",
            ])
    else:
        lines.extend([
            "## Recommendation",
            "",
            "No deployment command is recommended.",
            f"Status: `{final.get('recommendation_status', 'unknown')}`.",
            f"Reason: {final.get('recommendation_reason', 'insufficient deployable evidence')}",
        ])
        provisional = final.get("provisional_configuration")
        if isinstance(provisional, dict):
            comparison = provisional.get("comparison", {})
            lines.extend([
                "", "### Best Measured Unconfirmed Candidate", "",
                "The following non-baseline configuration has positive measured evidence but is not authorized for deployment:",
                "```json",
                json.dumps(provisional.get("config", provisional), indent=2, sort_keys=True),
                "```",
                f"- Point improvement: `{comparison.get('improvement_pct', 'unavailable')}%`",
                f"- Evidence state: `{provisional.get('evidence_state', 'unconfirmed')}`",
                f"- Remaining gates: `{provisional.get('rejection_reasons', [])}`",
            ])
        safe_baseline = final.get("safe_baseline_configuration")
        safe_baseline_command = final.get(
            "safe_baseline_deployment_command_reproducible"
        )
        if isinstance(safe_baseline, dict):
            lines.extend([
                "", "### Safe Measured Baseline", "",
                "This is the measured fallback, not an optimized winner:",
                "```json",
                json.dumps(safe_baseline.get("config", safe_baseline), indent=2, sort_keys=True),
                "```",
            ])
            if isinstance(safe_baseline_command, list):
                lines.extend([
                    "```bash",
                    multiline_shell_command(safe_baseline_command),
                    "```",
                ])
    command = final.get("deployment_command")
    if isinstance(command, list):
        lines.extend([
            "", "## Copy-Paste Deployment Command", "",
            "Copy the complete block below and run it directly:",
            "", "```bash", multiline_shell_command(command), "```",
        ])
        minimal_command = final.get("deployment_command_minimal")
        if isinstance(minimal_command, list) and minimal_command != command:
            lines.extend([
                "", "### Minimal Command", "",
                "This shorter command depends on the tested SGLang version retaining the same defaults.",
                "```bash", multiline_shell_command(minimal_command), "```",
            ])
    model = discovery.get("model", {}) if isinstance(discovery, dict) else {}
    if isinstance(model, dict) and (model.get("weight_quantization") or model.get("checkpoint_dtype")):
        lines.extend([
            "", "## Model Precision", "",
            f"- Weight format: `{model.get('weight_quantization') or model.get('quantization') or 'unquantized'}`",
            f"- Checkpoint/activation dtype: `{model.get('checkpoint_dtype') or model.get('dtype') or 'auto'}`",
            "- Launch policy: SGLang reads checkpoint metadata automatically; no dtype or quantization flag is injected unless the task explicitly requests one.",
        ])
    quality_gate = final.get("quality_gate", {})
    if isinstance(quality_gate, dict) and quality_gate.get("required"):
        lines.extend([
            "", "## Quality Gate", "",
            f"- State: `{quality_gate.get('state', 'unknown')}`",
            f"- Parameter: `{quality_gate.get('parameter', 'unknown')}={quality_gate.get('value', 'unknown')}`",
            f"- Evaluation dataset: `{quality_gate.get('evaluation_dataset') or 'not evaluated'}`",
            f"- Deployable: `{quality_gate.get('passed', False)}`",
            f"- Reason: {quality_gate.get('reason', 'quality evidence is required')}",
        ])
    economics = final.get("economics", {})
    if isinstance(economics, dict):
        lines.extend(["", "## Cost Per Token", ""])
        if economics.get("available"):
            currency = economics.get("currency", "USD")
            lines.extend([
                f"- Cost per GPU-hour: `{currency} {economics.get('cost_per_gpu_hour')}`",
                f"- Baseline/winner GPU count: `{economics.get('baseline_gpu_count')}` / "
                f"`{economics.get('winner_gpu_count')}`",
                f"- Experiment cost: `{currency} {economics.get('experiment_cost', 0):.4f}`",
                f"- Baseline $/M total tokens: `{economics.get('baseline', {}).get('cost_per_million_total_tokens')}`",
                f"- Winner $/M total tokens: `{economics.get('winner', {}).get('cost_per_million_total_tokens')}`",
                f"- Savings $/M total tokens: `{economics.get('savings', {}).get('cost_per_million_total_tokens')}`",
                f"- Relative cost reduction: `{economics.get('savings', {}).get('relative_pct')}`%",
                f"- Policy: {economics.get('policy')}",
            ])
        else:
            lines.append(f"- Unavailable: {economics.get('reason')}")
    history = search_plan.get("history", {})
    if isinstance(history, dict):
        priors = history.get("priors", {}) if isinstance(history.get("priors"), dict) else {}
        lines.extend([
            "", "## Trial History", "",
            f"- Enabled: `{history.get('enabled', False)}`",
            f"- Private SQLite database: `{history.get('database', 'unavailable')}`",
            f"- Strict compatibility fingerprint: `{history.get('compatibility_fingerprint', 'unavailable')}`",
            f"- Historical candidate trials created: `{priors.get('candidate_trials_created', 0)}`",
            f"- Parameters with compatible priors: `{sorted(priors.get('parameter_priors', {}))}`",
            f"- Policy: {priors.get('policy', history.get('policy', 'strict compatibility only'))}",
        ])
    cookbook_screen = final.get("cookbook_initial_screen", {})
    if isinstance(cookbook_screen, dict):
        aggregates = cookbook_screen.get("aggregates", [])
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
            best_was_confirmed = bool(
                recommendation
                and final.get("recommendation_status") == "confirmed_candidate"
                and recommendation.get("config", recommendation) == best_observed.get("config", {})
                and recommendation.get("env", {}) == best_observed.get("env", {})
            )
            lines.extend([
                "", "## Best One-Factor Screening Delta", "",
                "This section reports the one-factor screen; deployment status comes from final confirmation.",
                "```json",
                json.dumps(best_observed.get("config", {}), indent=2, sort_keys=True),
                "```",
                f"- Screening change: `{best_observed['comparison']['improvement_pct']:.3f}%`",
                (
                    "- Final confirmation: `confirmed_candidate`"
                    if best_was_confirmed
                    else f"- Screening-only rejection reasons: `{', '.join(best_observed.get('rejection_reasons', [])) or 'none'}`"
                ),
            ])
        mtp_rows = [
            row for row in screening.get("results", [])
            if "speculative_algorithm" in (row.get("config") or {})
            and isinstance(row.get("runtime_observations"), dict)
        ]
        if mtp_rows:
            lines.extend(["", "## MTP Runtime Evidence", ""])
            for row in mtp_rows:
                speculative = row["runtime_observations"].get("speculative", {})
                direct = speculative.get("acceptance_rate_pct", {})
                acceptance = (
                    direct.get("p50")
                    if isinstance(direct, dict) and direct.get("p50") is not None
                    else speculative.get("inferred_acceptance_rate_pct")
                )
                acceptance_text = (
                    f"{float(acceptance):.3f}%"
                    if isinstance(acceptance, (int, float)) else "unavailable"
                )
                lines.append(
                    f"- `{row.get('configuration_name')}`: acceptance telemetry "
                    f"`{acceptance_text}`, "
                    f"available `{speculative.get('telemetry_available', False)}`"
                )
    bottleneck = final.get("bottleneck", {}) if isinstance(final.get("bottleneck"), dict) else {}
    mechanism = bottleneck.get("screening_mechanism", {}) if isinstance(bottleneck.get("screening_mechanism"), dict) else {}
    if mechanism:
        lines.extend(["", "## Evidence", "", f"- Screening classification: `{mechanism.get('classification', 'unavailable')}`"])
    escalation = bottleneck.get("operator_escalation", {}) if isinstance(bottleneck, dict) else {}
    if isinstance(escalation, dict):
        lines.extend(["", "## Kernel Optimization Direction", ""])
        if escalation.get("required"):
            kernel = escalation.get("top_kernel_family", escalation.get("top_kernel", {}))
            evidence = escalation.get("evidence", {})
            lines.extend([
                f"- Top GPU operator family: `{kernel.get('name', 'unknown')}`",
                f"- Share of GPU-active kernel time: `{evidence.get('family_share_of_gpu_active_pct', evidence.get('kernel_share_of_gpu_active_pct', 'unknown'))}%`",
                "- Amdahl upper bound for a 2x speedup of that operator family within GPU execution: "
                f"`{escalation.get('two_x_kernel_speedup_gpu_execution_upper_bound_pct', 'unknown')}%`",
                "- This is a GPU-execution bound, not a claimed end-to-end gain.",
                f"- Next step: {escalation.get('next_step', 'run a shape-matched kernel profile')}",
            ])
        else:
            lines.append(
                f"- No automatic kernel escalation: {escalation.get('reason', 'insufficient kernel concentration evidence')}"
            )
    parameter_search = final.get("parameter_search", {})
    bottleneck_classification = search_plan.get("bottleneck_classification", {})
    canonical_bottleneck = search_plan.get("canonical_bottleneck", {})
    if isinstance(canonical_bottleneck, dict) and canonical_bottleneck:
        quality = canonical_bottleneck.get("evidence_quality", {})
        lines.extend([
            "", "## Canonical Bottleneck", "",
            f"- Primary class: `{canonical_bottleneck.get('primary')}`",
            f"- Co-primary classes: `{canonical_bottleneck.get('co_primary', [canonical_bottleneck.get('primary')])}`",
            f"- Prior class: `{canonical_bottleneck.get('prior_primary', canonical_bottleneck.get('primary'))}`",
            f"- Updated by interventions: `{canonical_bottleneck.get('posterior_updated', False)}`",
            f"- Dominant intervention: `{canonical_bottleneck.get('dominant_intervention_mechanism')}` at `{canonical_bottleneck.get('dominant_intervention_gain_pct')}`%",
            f"- Secondary classes: `{canonical_bottleneck.get('secondary', [])}`",
            f"- Evidence quality: `{json.dumps(quality, sort_keys=True)}`",
            f"- Raw profiler observation: `{json.dumps(canonical_bottleneck.get('raw_profiler_observation', {}), sort_keys=True)}`",
            "- Policy: profiler labels are raw observations; only canonical classes activate rules.",
        ])
    if isinstance(bottleneck_classification, dict) and bottleneck_classification:
        lines.extend([
            "", "## Bottleneck Classifier", "",
            f"- Primary class: `{bottleneck_classification.get('primary')}`",
            f"- Co-primary classes (score margin `{bottleneck_classification.get('co_primary_score_margin', 'n/a')}`): "
            f"`{bottleneck_classification.get('co_primary', [bottleneck_classification.get('primary')])}`",
            f"- Secondary classes: `{bottleneck_classification.get('secondary', [])}`",
            f"- Confidence: `{bottleneck_classification.get('confidence')}`",
            f"- Evidence: `{json.dumps(bottleneck_classification.get('evidence', {}), sort_keys=True)}`",
            f"- Ruleset: `{bottleneck_classification.get('ruleset_version')}`",
        ])
    budget_accounting = final.get("budget_accounting", {})
    if isinstance(budget_accounting, dict) and budget_accounting:
        lines.extend([
            "", "## Trial Budget", "",
            f"- Planned discovery/refinement/confirmation: `{budget_accounting.get('planned', {})}`",
            f"- Used discovery/refinement/confirmation: `{budget_accounting.get('used', {})}`",
            f"- Used percentages: `{budget_accounting.get('used_percentages', {})}`",
            f"- Unused trials: `{budget_accounting.get('unused_trials', 0)}`",
            f"- Reclamation: {budget_accounting.get('reclamation_policy', 'unused earlier tiers flow forward')}",
        ])
    if isinstance(parameter_search, dict):
        lines.extend([
            "", "## Parameter Search", "",
            f"- Attempted parameter candidates: `{parameter_search.get('attempted_parameter_candidates', 'unknown')}`",
            f"- Executed parameter candidates: `{parameter_search.get('executed_parameter_candidates', 'unknown')}`",
            f"- Failed parameter candidates: `{parameter_search.get('failed_parameter_candidates', 'unknown')}`",
            f"- Candidate counts by stage: `{parameter_search.get('stage_candidate_counts', {})}`",
            f"- Measurement tiers: `{parameter_search.get('measurement_tiers', {})}`",
            "- Deferred candidates continued with unused refinement capacity: "
            f"`{parameter_search.get('continued_exploration_candidates', [])}`",
            "- Persisted positive steady-state signals (not deployment-authorized until composed/confirmed): "
            f"`{parameter_search.get('positive_screening_signals', [])}`",
            f"- Champion augmentation: `{parameter_search.get('champion_augmentation', {})}`",
            f"- Refinement idle-slot backfill: `{parameter_search.get('idle_slot_backfill', {})}`",
            "- Distinct serving mechanisms measured: "
            f"`{len(parameter_search.get('measured_distinct_mechanisms', []))}/"
            f"{parameter_search.get('required_distinct_mechanisms', 'unknown')}` "
            f"({parameter_search.get('measured_distinct_mechanisms', [])})",
            f"- Executed mechanisms: `{parameter_search.get('executed_distinct_mechanisms', [])}`",
            f"- Scheduled but not validly measured: `{parameter_search.get('missing_mechanism_classes', [])}`",
            f"- Attempted but unavailable optional mechanisms: `{parameter_search.get('unavailable_mechanism_classes', [])}`",
            f"- Deployment-blocking mechanism gaps: `{parameter_search.get('blocking_missing_mechanism_classes', [])}`",
            f"- Required scalar/bundle breadth: `{parameter_search.get('required_parameter_breadth', 'unknown')}`",
            f"- Evidence sufficient for a deployment recommendation: `{parameter_search.get('sufficient_evidence', False)}`",
        ])
        if final.get("experiment_mode") == "fast":
            lines.append(
                "- Fast-mode scope: the recommendation is statistically confirmed within the measured "
                "mechanism subset; unmeasured applicable Cookbook/model-native mechanisms remain visible "
                "under Candidate Coverage. Use balanced or max for a broader optimum claim."
            )
        for item in parameter_search.get("selection_evidence", []):
            if not isinstance(item, dict):
                continue
            selector = item.get("parameter") or ", ".join(item.get("parameters", []))
            lines.append(
                f"- Selected `{item.get('name')}` from `{selector}` "
                f"(family `{item.get('family')}`, trigger magnitude `{item.get('trigger_magnitude', 'bundle')}`): "
                f"{item.get('reason', 'compatible workload/profile candidate')}"
            )
        mandatory = parameter_search.get("mandatory_capacity_parameters", [])
        missing = parameter_search.get("missing_mandatory_capacity_parameters", [])
        if mandatory:
            lines.append(f"- Mandatory offline capacity controls: `{mandatory}`")
            lines.append(
                f"- Uncovered mandatory controls: `{missing}`. "
                "When nonempty, the result is only best within the tested subset."
            )
    registry = search_plan.get("candidate_registry", {})
    if isinstance(registry, dict) and registry:
        coverage = registry.get("coverage", {})
        lines.extend([
            "", "## Candidate Coverage", "",
            f"- Candidate proposals: `{coverage.get('proposals', 0)}`",
            f"- Unique merged candidates: `{coverage.get('unique_candidates', 0)}`",
            f"- Candidate states: `{coverage.get('by_state', {})}`",
            f"- Mechanism/state coverage: `{coverage.get('by_mechanism', {})}`",
            f"- Mechanism lifecycle: `{coverage.get('mechanism_lifecycle', {})}`",
        ])
        schedule = search_plan.get("mechanism_schedule", {})
        if isinstance(schedule, dict):
            lines.append(
                f"- Discovery schedule: `{schedule.get('policy', 'unavailable')}`; "
                f"scheduled mechanisms `{schedule.get('scheduled_mechanisms', schedule.get('covered_mechanisms', []))}`."
            )
        outcomes = search_plan.get("mechanism_outcomes_after_screen", [])
        if isinstance(outcomes, list) and outcomes:
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue
                lines.append(
                    f"- `{outcome.get('mechanism')}`: `{outcome.get('state')}`; "
                    f"best change `{outcome.get('best_improvement_pct')}`; "
                    f"{outcome.get('reason')}"
                )
    composition_parent_gates = final.get("composition_parent_gates", [])
    if isinstance(composition_parent_gates, list) and composition_parent_gates:
        lines.extend(["", "## Composition Parsimony", ""])
        lines.append(
            "Generated combinations must remain positive against the original baseline and clear a smaller "
            "incremental parsimony threshold against their strongest measured direct parent. The global "
            "task threshold is not reapplied to every added flag."
        )
        for gate in composition_parent_gates:
            if not isinstance(gate, dict):
                continue
            improvement = gate.get("improvement_pct")
            improvement_text = (
                f"{float(improvement):.3f}%"
                if isinstance(improvement, (int, float)) else "unavailable"
            )
            lines.append(
                f"- `{gate.get('configuration_name')}` versus "
                f"`{gate.get('parent_configuration_name', 'unavailable')}`: incremental change "
                f"`{improvement_text}`, required "
                f"`{gate.get('minimum_improvement_pct', 'unknown')}%`, accepted "
                f"`{gate.get('accepted', False)}`; {gate.get('reason', 'no reason recorded')}."
            )
    pipeline = final.get("parallel_pipeline", {})
    if isinstance(pipeline, dict):
        lines.extend([
            "", "## GPU Scheduling", "",
            f"- Nsys/preprofile overlap enabled: `{pipeline.get('enabled', False)}`",
            f"- Profiling GPU: `{pipeline.get('profile_gpu', 'serial/default')}`",
            f"- Screening GPU pool: `{pipeline.get('screening_gpus', [])}`",
            f"- Configured screening-worker cap: `{pipeline.get('screening_parallel_workers', 1)}`",
            f"- Nsys overlap qualification: `{pipeline.get('profile_pipeline_qualification', {})}`",
            "- Actual concurrency is resource-packed per candidate TP/PP/DP size; a TP=2 trial on two GPUs runs one service at a time.",
            f"- GPU allocation: `{pipeline.get('screening_gpu_allocation', 'exclusive')}`",
            f"- Policy: {pipeline.get('policy', 'serial profiling')}",
        ])
        if pipeline.get("error"):
            lines.append(f"- Pipeline fallback error: `{pipeline['error']}`")
    candidate_limit = search_plan.get("screening_candidate_limit")
    if candidate_limit is not None:
        early_stop = search_plan.get("screening_early_stop", {})
        lines.extend([
            f"- Planned high-impact/fallback candidate limit: `{candidate_limit}`",
            f"- Selection policy: {search_plan.get('screening_selection_policy', 'unavailable')}",
            f"- Early-stop policy: `{early_stop}`",
        ])
    resolved = search_plan.get("resolved_baseline", {})
    if recommendation and isinstance(resolved, dict):
        effective_recommendation = {
            **resolved,
            **recommendation.get("config", recommendation),
        }
        lines.extend([
            "", "## Effective Runtime Settings", "",
            "The launch command emits only measured deltas. The resolved values below were active during the benchmark and must be recorded with the tested SGLang version when reproducing this result.",
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
        match_order = search_plan.get("parameter_match_order", [])
        if isinstance(match_order, list) and match_order:
            rendered_matches = [
                f"{item.get('parameter')}:{item.get('magnitude')}:{item.get('rule_ids')}"
                for item in match_order if isinstance(item, dict)
            ]
            lines.append(f"- Trigger-matched parameter order: `{rendered_matches}`")
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
            finally_confirmed = bool(
                recommendation
                and final.get("recommendation_status") == "confirmed_candidate"
                and recommendation.get("config", recommendation) == item.get("config", {})
                and recommendation.get("env", {}) == item.get("env", {})
            )
            lines.append(
                f"- Measured `{item.get('configuration_name')}`: objective change "
                f"`{comparison.get('improvement_pct', 'unavailable')}%`; "
                f"confirmed `{finally_confirmed or item.get('confirmed', False)}`; "
                f"rejections `{'none' if finally_confirmed else ', '.join(item.get('rejection_reasons', [])) or 'none'}`."
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
        profiling = final.get("profiling", {})
        profile_dir = Path(str(profiling.get("run_dir", run_dir / "profile"))) if isinstance(profiling, dict) else run_dir / "profile"
        tune_command = " ".join([
            "inferopt tune-moe",
            "--task", shlex.quote(str(run_dir / "task.json")),
            "--profile", shlex.quote(str(profile_dir / "nsys-diagnosis.json")),
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
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {package_version()}"
    )
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
    init.add_argument("--dataset-name", choices=["synthetic", "custom", "sharegpt"])
    init.add_argument("--dataset-path")
    init.add_argument(
        "--apply-chat-template", action=argparse.BooleanOptionalAction, default=None,
        help="apply the model chat template to custom/sharegpt prompts",
    )
    init.add_argument("--p99-e2e-latency-ms")
    init.add_argument("--p99-ttft-ms")
    init.add_argument("--p99-tpot-ms")
    init.add_argument(
        "--latency-slo-statistic", choices=["p99", "avg"],
        help="use one statistic family for all latency limits",
    )
    init.add_argument("--avg-e2e-latency-ms")
    init.add_argument("--avg-ttft-ms")
    init.add_argument("--avg-tpot-ms")
    init.add_argument(
        "--max-concurrency",
        help=(
            "fixed online client concurrency target; for adaptive online SLO tasks it is a "
            "legacy alias for --fallback-max-concurrency; offline tasks reject this option"
        ),
    )
    init.add_argument(
        "--resource-scope", choices=["per_gpu", "per_service"],
        help=(
            "comparison scope for throughput objectives; offline defaults to per_gpu "
            "so TP/PP/DP candidates pay for every GPU they use, online defaults to per_service"
        ),
    )
    init.add_argument(
        "--fallback-max-concurrency",
        help=(
            "optional client concurrency used only when adaptive online SLO calibration "
            "cannot read SGLang max_running_requests"
        ),
    )
    init.add_argument("--concurrency-points", help="comma-separated capacity/SLO measurement points; must end at max concurrency")
    init.add_argument("--shared-prefix-tokens")
    init.add_argument("--experiment-mode", choices=["fast", "balanced", "max"])
    init.add_argument(
        "--allow-download", action=argparse.BooleanOptionalAction, default=None,
        help="allow downloading official Cookbook docs only when the selected checkout has no matching local recipe (default: enabled)",
    )
    init.add_argument(
        "--allow-kv-cache-precision-tuning",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allow FP8 KV-cache candidates; disabled by default because precision can affect quality",
    )
    init.add_argument(
        "--quality-evaluation-dataset",
        help="dataset file whose SHA-256 must match the external quality attestation",
    )
    init.add_argument(
        "--quality-attestation",
        help=(
            "external quality-attestation JSON; requires FP8 KV-cache tuning and must "
            "record approved/method/dataset hash/metric/baseline/candidate/regression"
        ),
    )
    init.add_argument(
        "--enable-history", action=argparse.BooleanOptionalAction, default=None,
        help="save private local benchmark history for future exactly compatible runs (default: enabled)",
    )
    init.add_argument("--history-database")
    init.add_argument("--warm-start-limit", type=int)
    init.add_argument(
        "--parameter-evolution-mode", choices=["conservative", "experimental"],
        help="audit new SGLang parameters only, or reserve bounded discovery trials for safe provisional parameters",
    )
    init.add_argument(
        "--parameter-evolution-budget-pct", type=float,
        help="maximum percentage of discovery trials for provisional new parameters (0-25; default 10)",
    )
    init.add_argument(
        "--max-provisional-trials", type=int,
        help="optional absolute cap for provisional new-parameter trials",
    )
    init.add_argument(
        "--cost-per-gpu-hour", type=float,
        help="optional report-only GPU-hour cost; omit to disable all cost reporting questions",
    )
    init.add_argument(
        "--currency", help="cost-report currency label; requires --cost-per-gpu-hour"
    )
    init.add_argument(
        "--parallel-trials", type=int,
        help="advanced cap on concurrent trials; defaults to --max-gpus",
    )
    init.add_argument(
        "--max-gpus", type=int,
        help="maximum GPUs InferOpt may occupy concurrently; defaults to detected visible GPUs",
    )
    init.add_argument(
        "--max-trials", type=int,
        help="optional cap on all benchmark trials; overrides the experiment-mode default",
    )
    init.add_argument(
        "--max-gpu-hours", type=float,
        help="optional aggregate GPU-hour budget; overrides the experiment-mode default",
    )
    init.add_argument(
        "--max-wall-time-minutes", type=float,
        help="optional wall-clock budget in minutes; overrides the experiment-mode default",
    )
    init.add_argument(
        "--confirmation-repetitions", type=int,
        help=(
            "independent benchmark windows per baseline/candidate service; defaults to 2 for "
            "fast/balanced and 3 for max, and may add longer windows when CV exceeds 5%%, "
            "while each configuration is loaded only once"
        ),
    )
    init.add_argument(
        "--cuda-visible-devices",
        help="comma-separated GPU indexes/UUIDs, e.g. 0 or 0,1,2 (no spaces), or 'all' (default)",
    )
    for name in ("doctor", "feasibility", "plan", "run", "validate"):
        item = commands.add_parser(name)
        item.add_argument("--task", required=True)
        item.add_argument("--output")
        item.add_argument(
            "--progress", choices=["plain", "json", "none"], default="plain",
            help="progress format written to stderr",
        )
        if name == "run":
            item.add_argument("--yes", action="store_true")
            item.add_argument(
                "--resume-run-dir",
                help="resume compatible completed stages from an existing run directory",
            )
            item.add_argument(
                "--profile-dir",
                help="reuse a completed compatible Nsight profile directory",
            )
    report = commands.add_parser("report", help="render a human-readable completed-run report")
    report.add_argument("--result", required=True)
    report.add_argument("--output", required=True)
    catalog_parser = commands.add_parser(
        "parameter-catalog", help="export the current SGLang ServerArgs contract"
    )
    catalog_parser.add_argument("--repository", required=True)
    catalog_parser.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter used by the selected SGLang checkout",
    )
    catalog_parser.add_argument("--output", required=True)
    catalog_parser.add_argument(
        "--progress", choices=["plain", "json", "none"], default="plain"
    )
    diff_parser = commands.add_parser(
        "parameter-diff", help="compare two exported SGLang parameter contracts"
    )
    diff_parser.add_argument("--baseline", required=True)
    diff_parser.add_argument("--current", required=True)
    diff_parser.add_argument("--output", required=True)
    roofline = commands.add_parser("roofline", help="analyze shape-matched NCU roofline metrics")
    roofline.add_argument("--profile", required=True)
    roofline.add_argument("--metrics-csv")
    roofline.add_argument("--output")
    roofline.add_argument(
        "--progress", choices=["plain", "json", "none"], default="plain"
    )
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
    tune_moe_parser.add_argument(
        "--progress", choices=["plain", "json", "none"], default="plain"
    )
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
    generate_moe_parser.add_argument(
        "--progress", choices=["plain", "json", "none"], default="plain"
    )
    args = parser.parse_args()
    task: dict[str, Any] | None = None
    try:
        if args.command == "init":
            write_json(init_task(args), args.output)
            return 0
        if args.command == "report":
            target = Path(args.output).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            final = inferopt.load_json(args.result)
            target.write_text(markdown_report(final), encoding="utf-8")
            print(f"report: {target}")
            return 0
        if args.command == "parameter-catalog":
            catalog_progress = autopilot.ProgressReporter(args.progress)
            catalog_progress.emit(
                "parameter-catalog", "loading ServerArgs from the selected checkout",
                completed=0, total=3,
            )
            repository = Path(args.repository).expanduser().resolve()
            try:
                catalog = autopilot.parameter_catalog(
                    {
                        "python": str(Path(args.python).expanduser().resolve()),
                        "repository": str(repository), "env": {},
                    },
                    progress=lambda message: catalog_progress.emit(
                        "parameter-catalog", message, completed=0, total=3
                    ),
                )
                catalog_progress.emit(
                    "parameter-catalog", "runtime argparse extraction completed",
                    completed=2, total=3,
                )
            except Exception as exc:
                catalog_progress.emit(
                    "parameter-catalog",
                    f"runtime import unavailable ({type(exc).__name__}); using AST audit fallback",
                    completed=1, total=3,
                )
                catalog = sglang_catalog.export_catalog_static(repository)
                catalog["runtime_extraction_error"] = f"{type(exc).__name__}: {exc}"
                catalog_progress.emit(
                    "parameter-catalog", "static AST extraction completed",
                    completed=2, total=3,
                )
            contract = parameter_evolution.build_parameter_contract(catalog)
            write_json(contract, args.output)
            catalog_progress.emit(
                "parameter-catalog",
                f"wrote {len(contract.get('parameters', {}))} parameters to {args.output}",
                completed=3, total=3,
            )
            return 0
        if args.command == "parameter-diff":
            difference = parameter_evolution.diff_parameter_contract(
                inferopt.load_json(args.baseline), inferopt.load_json(args.current)
            )
            write_json(difference, args.output)
            return 0
        if args.command == "roofline":
            roofline_progress = autopilot.ProgressReporter(args.progress)
            roofline_progress.emit("roofline", "loading and analyzing the Nsys report")
            profile = profile_sglang.diagnose_existing(
                Path(args.profile).expanduser(),
                progress=lambda event: roofline_progress.emit(
                    "roofline " + str(event.get("phase", "stats")),
                    str(event.get("message", "working")),
                    completed=event.get("completed"), total=event.get("total"),
                ),
            )
            metrics = (
                profile_sglang.parse_ncu_roofline_csv(args.metrics_csv)
                if args.metrics_csv else None
            )
            result = profile_sglang.roofline_diagnosis(
                profile.get("tool", {}).get("ncu", {}),
                profile.get("diagnosis", {}).get("top_kernels", [None])[0],
                metrics,
            )
            inferopt.dump_json(result, args.output)
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
                progress=autopilot.ProgressReporter(args.progress),
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
            validate_progress = autopilot.ProgressReporter(args.progress)
            validate_progress.emit("validate", "checking task and dataset structure", completed=0, total=1)
            errors = autopilot.validate_task(task)
            inferopt.dump_json({"valid": not errors, "errors": errors}, args.output)
            validate_progress.emit(
                "validate", "validation completed", completed=1, total=1
            )
            return 0 if not errors else 2
        if args.command in {"doctor", "feasibility"}:
            doctor_progress = autopilot.ProgressReporter(args.progress)
            diagnosis = doctor(task, progress=doctor_progress)
            if args.command == "feasibility":
                result = diagnosis.get("deployment_feasibility", diagnosis)
                success = result.get("status") == "deployable_as_is"
            else:
                result = diagnosis
                success = result.get("status") == "ready"
            inferopt.dump_json(result, args.output)
            return 0 if success else 2
        if args.command == "plan":
            plan_progress = autopilot.ProgressReporter(args.progress)
            inferopt.dump_json(
                autopilot.build_plan(task, progress=plan_progress), args.output
            )
            return 0
        if args.command == "run":
            if not args.yes:
                raise ValueError("run requires --yes after reviewing doctor and plan")
            if args.resume_run_dir:
                task["resume_run_dir"] = str(Path(args.resume_run_dir).expanduser().resolve())
            if args.profile_dir:
                task["profile_dir"] = str(Path(args.profile_dir).expanduser().resolve())
            inferopt.dump_json(
                autopilot.run_autopilot(
                    task, progress_reporter=autopilot.ProgressReporter(args.progress)
                ),
                args.output,
            )
            return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if args.command == "run" and isinstance(task, dict):
            run_dir: Path | None = None
            if task.get("resume_run_dir"):
                run_dir = Path(task["resume_run_dir"]).expanduser()
            else:
                output_dir = Path(task.get("output_dir", ".")).expanduser()
                prefix = f"{task.get('name', '')}-autopilot-"
                candidates = sorted(
                    (
                        path for path in output_dir.glob(prefix + "*")
                        if path.is_dir()
                    ),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    run_dir = candidates[0]
            failure = {
                "schema_version": 1,
                "state": "failed",
                "error_type": type(exc).__name__,
                "detail": str(exc),
                "run_dir": str(run_dir) if run_dir is not None else None,
                "completed_artifacts": (
                    sorted(path.name for path in run_dir.iterdir() if path.is_file())
                    if run_dir is not None and run_dir.is_dir() else []
                ),
            }
            if run_dir is not None and run_dir.is_dir():
                write_json(failure, run_dir / "failure.json")
            if args.output:
                inferopt.dump_json(failure, args.output)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
