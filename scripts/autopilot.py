#!/usr/bin/env python3
"""Run one-shot, hardware-aware SGLang deployment optimization."""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from autotune import command_manifest, execute, execution_errors, write_json
from inferopt import dump_json, load_json
from profile_sglang import diagnose_existing, run_profile
from sglang_catalog import export_catalog


REQUIRED_TOP_LEVEL = {
    "name",
    "repository",
    "python",
    "model_path",
    "workload",
    "slo",
    "objective",
    "budget",
    "output_dir",
}
OPTIONAL_TOP_LEVEL = {
    "offline",
    "allow_download",
    "env",
    "confirmation_repetitions",
    "port",
    "model",
    "profiling",
    "profile_dir",
    "measurement",
    "deployment_mode",
    "experiment_mode",
    "calibration",
    "search_depth",
    "knowledge",
    "capability_overrides",
    "deployment",
    "quality",
    "model_variants",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProgressReporter:
    """Render concise one-shot execution progress without changing artifacts."""

    def __init__(self) -> None:
        self.started = time.monotonic()

    def emit(self, stage: str, message: str) -> None:
        elapsed = int(time.monotonic() - self.started)
        print(f"[inferopt +{elapsed // 60:02d}:{elapsed % 60:02d}] {stage}: {message}", flush=True)

    def trial(self, stage: str, event: dict[str, Any]) -> None:
        index = event["trial_index"]
        total = event["trial_count"]
        if event["event"] == "trial_skipped":
            self.emit(
                stage,
                f"trial {index}/{total} {event['trial_name']}: skipped "
                f"({event['capability']} unavailable: {event['reason']})",
            )
            return
        if event["event"] == "trial_started":
            self.emit(stage, f"trial {index}/{total} {event['trial_name']}: starting server and benchmark")
            return
        if not event.get("ok"):
            self.emit(stage, f"trial {index}/{total} failed: {event.get('detail') or 'unknown error'}")
            return
        metrics = event.get("metrics", {})
        rps = metrics.get("request_throughput_rps")
        p99 = metrics.get("p99_e2e_latency_ms")
        summary = []
        if isinstance(rps, (int, float)):
            summary.append(f"{rps:.3f} RPS")
        if isinstance(p99, (int, float)):
            summary.append(f"p99 E2E {p99:.1f} ms")
        summary.append("SLO pass" if event.get("slo_passed") else "SLO not passed")
        self.emit(stage, f"trial {index}/{total} completed ({', '.join(summary)})")


def execute_with_progress(
    spec: dict[str, Any], reporter: ProgressReporter, stage: str
) -> dict[str, Any]:
    reporter.emit(stage, "preparing experiment")
    report = execute(spec, progress=lambda event: reporter.trial(stage, event))
    reporter.emit(
        stage,
        f"finished: {report['completed_trials']}/{report['planned_trials']} trials, "
        f"{report['stop_reason']}",
    )
    return report


def run_readonly(
    command: list[str], timeout: int = 15, cwd: str | None = None, env: dict[str, str] | None = None
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
            env=env,
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError:
        return {"available": False}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "error": type(exc).__name__}


def validate_task(task: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in sorted(REQUIRED_TOP_LEVEL - set(task)):
        errors.append(f"missing required field: {key}")
    for key in sorted(set(task) - REQUIRED_TOP_LEVEL - OPTIONAL_TOP_LEVEL):
        errors.append(f"unsupported field: {key}")
    for key in ("repository", "python", "model_path", "output_dir"):
        value = task.get(key)
        if not isinstance(value, str) or not Path(value).expanduser().is_absolute():
            errors.append(f"{key} must be an absolute path")
    if isinstance(task.get("repository"), str) and not Path(task["repository"]).is_dir():
        errors.append("repository must exist")
    if isinstance(task.get("python"), str) and not Path(task["python"]).is_file():
        errors.append("python must exist")
    if isinstance(task.get("model_path"), str) and not Path(task["model_path"]).is_dir():
        errors.append("model_path must be an existing local directory")
    for section in ("workload", "slo", "objective", "budget"):
        if not isinstance(task.get(section), dict):
            errors.append(f"{section} must be an object")
    workload = task.get("workload", {})
    for key in ("input_tokens", "output_tokens", "max_concurrency", "num_prompts"):
        value = workload.get(key) if isinstance(workload, dict) else None
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"workload.{key} must be a positive integer")
    request_rate = workload.get("request_rate", "inf") if isinstance(workload, dict) else None
    if request_rate != "inf" and (
        not isinstance(request_rate, (int, float)) or isinstance(request_rate, bool) or request_rate <= 0
    ):
        errors.append("workload.request_rate must be positive or 'inf'")
    if isinstance(workload, dict) and workload.get("shared_prefix") is not None:
        try:
            shared_prefix_benchmark(workload)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    deployment_mode = task.get("deployment_mode", "online_latency")
    if deployment_mode not in {"online_latency", "offline_throughput"}:
        errors.append("deployment_mode must be online_latency or offline_throughput")
    if task.get("search_depth", "thorough") not in {"evidence_guided", "thorough"}:
        errors.append("search_depth must be evidence_guided or thorough")
    if task.get("experiment_mode", "balanced") not in {"fast", "balanced", "rigorous"}:
        errors.append("experiment_mode must be fast, balanced, or rigorous")
    budget = task.get("budget", {})
    for key in ("max_trials", "max_gpu_hours", "max_wall_time_minutes"):
        value = budget.get(key) if isinstance(budget, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            errors.append(f"budget.{key} must be positive")
    if isinstance(budget, dict) and not isinstance(budget.get("max_trials"), int):
        errors.append("budget.max_trials must be an integer")
    repetitions = task.get("confirmation_repetitions", 3)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or not 2 <= repetitions <= 9:
        errors.append("confirmation_repetitions must be an integer from 2 through 9")
    elif isinstance(budget, dict) and isinstance(budget.get("max_trials"), int):
        minimum_trials = repetitions * 2 + 3
        if budget["max_trials"] < minimum_trials:
            errors.append(
                f"budget.max_trials must be at least {minimum_trials} for profiling, screening, and confirmation"
            )
    environment = task.get("env", {})
    if not isinstance(environment, dict):
        errors.append("env must be an object")
    knowledge = task.get("knowledge", {})
    if not isinstance(knowledge, dict):
        errors.append("knowledge must be an object")
    else:
        supported = {"model_cookbook_url", "hardware_reference_urls", "require_cookbook"}
        if any(key not in supported for key in knowledge):
            errors.append("knowledge supports only model_cookbook_url, hardware_reference_urls, and require_cookbook")
        if "model_cookbook_url" in knowledge and not isinstance(knowledge["model_cookbook_url"], str):
            errors.append("knowledge.model_cookbook_url must be a URL string")
        if "require_cookbook" in knowledge and not isinstance(knowledge["require_cookbook"], bool):
            errors.append("knowledge.require_cookbook must be boolean")
        if "hardware_reference_urls" in knowledge and (
            not isinstance(knowledge["hardware_reference_urls"], list)
            or any(not isinstance(url, str) for url in knowledge["hardware_reference_urls"])
        ):
            errors.append("knowledge.hardware_reference_urls must be an array of URL strings")
    capability_overrides = task.get("capability_overrides", {})
    if not isinstance(capability_overrides, dict):
        errors.append("capability_overrides must be an object")
    elif any(key not in {"mtp"} for key in capability_overrides):
        errors.append("capability_overrides supports only mtp")
    elif capability_overrides.get("mtp", "auto") not in {"auto", "disabled"}:
        errors.append("capability_overrides.mtp must be auto or disabled")
    deployment = task.get("deployment", {})
    if not isinstance(deployment, dict):
        errors.append("deployment must be an object")
    elif any(key not in {"allow_model_variant_recommendations", "allow_auto_model_switch"} for key in deployment):
        errors.append("deployment supports only allow_model_variant_recommendations and allow_auto_model_switch")
    elif deployment.get("allow_auto_model_switch", False):
        errors.append("deployment.allow_auto_model_switch is not supported; model changes require an explicit new task")
    quality = task.get("quality", {})
    if not isinstance(quality, dict):
        errors.append("quality must be an object")
    elif any(key not in {"evaluation_dataset", "max_regression_pct"} for key in quality):
        errors.append("quality supports only evaluation_dataset and max_regression_pct")
    elif "evaluation_dataset" in quality and (
        not isinstance(quality["evaluation_dataset"], str)
        or not Path(quality["evaluation_dataset"]).expanduser().is_absolute()
    ):
        errors.append("quality.evaluation_dataset must be an absolute path")
    elif "max_regression_pct" in quality and (
        not isinstance(quality["max_regression_pct"], (int, float))
        or isinstance(quality["max_regression_pct"], bool)
        or quality["max_regression_pct"] < 0
    ):
        errors.append("quality.max_regression_pct must be non-negative")
    variants = task.get("model_variants", [])
    if not isinstance(variants, list):
        errors.append("model_variants must be an array")
    else:
        for index, variant in enumerate(variants):
            if not isinstance(variant, dict):
                errors.append(f"model_variants[{index}] must be an object")
                continue
            if set(variant) - {"name", "model_path", "quantization"}:
                errors.append(f"model_variants[{index}] supports only name, model_path, and quantization")
            if not isinstance(variant.get("name"), str) or not variant["name"]:
                errors.append(f"model_variants[{index}].name must be a non-empty string")
            path = variant.get("model_path")
            if not isinstance(path, str) or not Path(path).expanduser().is_absolute() or not Path(path).is_dir():
                errors.append(f"model_variants[{index}].model_path must be an existing absolute directory")
    if task.get("offline", True) is False and not task.get("allow_download", False):
        errors.append("offline=false requires allow_download=true")
    profiling = task.get("profiling", {})
    if not isinstance(profiling, dict):
        errors.append("profiling must be an object")
    elif profiling.get("enabled", True) is not True:
        errors.append("profiling.enabled must be true for automatic optimization")
    profile_dir = task.get("profile_dir")
    if profile_dir is not None and (
        not isinstance(profile_dir, str) or not Path(profile_dir).expanduser().is_absolute()
    ):
        errors.append("profile_dir must be an absolute path when provided")
    measurement = task.get("measurement") or {}
    if not isinstance(measurement, dict):
        errors.append("measurement must be an object")
    elif any(key not in {"warmup_requests", "min_measurement_requests", "min_measurement_seconds"} for key in measurement):
        errors.append("measurement supports only warmup_requests, min_measurement_requests, and min_measurement_seconds")
    else:
        for key in ("warmup_requests", "min_measurement_requests"):
            if key in measurement and (not isinstance(measurement[key], int) or measurement[key] <= 0):
                errors.append(f"measurement.{key} must be a positive integer")
        if "min_measurement_seconds" in measurement and (
            not isinstance(measurement["min_measurement_seconds"], (int, float))
            or isinstance(measurement["min_measurement_seconds"], bool)
            or measurement["min_measurement_seconds"] <= 0
        ):
            errors.append("measurement.min_measurement_seconds must be positive")
    calibration = task.get("calibration") or {}
    if not isinstance(calibration, dict):
        errors.append("calibration must be an object")
    else:
        supported = {"enabled", "min_concurrency", "max_concurrency", "max_steps", "stop_on_slo_failure", "concurrencies"}
        if any(key not in supported for key in calibration):
            errors.append("calibration supports only enabled, concurrency range, explicit concurrencies, max_steps, and stop_on_slo_failure")
        if "enabled" in calibration and not isinstance(calibration["enabled"], bool):
            errors.append("calibration.enabled must be boolean")
        for key in ("min_concurrency", "max_concurrency", "max_steps"):
            if key in calibration and (
                not isinstance(calibration[key], int) or isinstance(calibration[key], bool) or calibration[key] <= 0
            ):
                errors.append(f"calibration.{key} must be a positive integer")
        if "stop_on_slo_failure" in calibration and not isinstance(calibration["stop_on_slo_failure"], bool):
            errors.append("calibration.stop_on_slo_failure must be boolean")
        points = calibration.get("concurrencies")
        if points is not None:
            if (
                not isinstance(points, list) or not points or len(points) > 16
                or any(not isinstance(point, int) or isinstance(point, bool) or point <= 0 for point in points)
            ):
                errors.append("calibration.concurrencies must contain 1 through 16 positive integers")
            elif points != sorted(set(points)):
                errors.append("calibration.concurrencies must be unique and ascending")
            elif isinstance(workload, dict) and points[-1] != workload.get("max_concurrency"):
                errors.append("calibration.concurrencies must end at workload.max_concurrency")
        if (
            isinstance(calibration.get("min_concurrency"), int)
            and isinstance(calibration.get("max_concurrency"), int)
            and calibration["min_concurrency"] > calibration["max_concurrency"]
        ):
            errors.append("calibration.min_concurrency must not exceed calibration.max_concurrency")
    return errors


def deployment_policy(task: dict[str, Any]) -> dict[str, Any]:
    mode = task.get("deployment_mode", "online_latency")
    if mode == "offline_throughput":
        return {
            "mode": mode,
            "objective": "maximize sustained aggregate throughput at the highest stable batch pressure",
            "latency_slo_required": False,
            "calibration_multiplier": 64,
            "calibration_floor": 32,
            "candidate_focus": ["batching", "KV_capacity", "scheduler", "CUDA_graph", "kernel_backends"],
            "final_acceptance": "declared SLOs plus error/correctness gates; latency is observational unless declared",
        }
    return {
        "mode": "online_latency",
        "objective": "maximize goodput or request throughput while every declared tail-latency SLO passes",
        "latency_slo_required": True,
        "calibration_multiplier": 16,
        "calibration_floor": 8,
        "candidate_focus": ["tail_latency", "scheduler", "CUDA_graph", "cache", "kernel_backends"],
        "final_acceptance": "every target-workload E2E/TTFT/TPOT/ITL and error SLO must pass on every confirmation repetition",
    }


def calibration_concurrencies(task: dict[str, Any]) -> list[int]:
    """Produce a budget-aware geometric load sweep without changing user workload semantics."""
    calibration = task.get("calibration") or {}
    if calibration.get("enabled", True) is False:
        return []
    explicit_points = calibration.get("concurrencies")
    if isinstance(explicit_points, list):
        return list(explicit_points)
    workload = task["workload"]
    policy = deployment_policy(task)
    target = workload["max_concurrency"]
    range_requested = "min_concurrency" in calibration or "max_concurrency" in calibration
    start = int(calibration.get("min_concurrency", target))
    cap = int(calibration.get(
        "max_concurrency",
        target if range_requested else max(target * policy["calibration_multiplier"], policy["calibration_floor"]),
    ))
    requested_steps = int(calibration.get("max_steps", 5))
    repetitions = int(task.get("confirmation_repetitions", 3))
    # Reserve one profile, a baseline/candidate screen, and a full confirmation.
    affordable_steps = max(0, int(task["budget"]["max_trials"]) - (1 + 2 + 2 * repetitions))
    steps = min(requested_steps, affordable_steps)
    values: list[int] = []
    value = start
    while len(values) < steps and value <= cap:
        values.append(value)
        value *= 2
    if values and values[-1] != cap and len(values) < steps:
        values.append(cap)
    return values


def parse_nvidia_inventory() -> dict[str, Any] | None:
    query = run_readonly([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version,pci.bus_id,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if query.get("returncode") != 0 or not query.get("stdout"):
        return None
    gpus = []
    for line in query["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "memory_mib": int(float(parts[2])),
            "driver_version": parts[3],
            "pci_bus_id": parts[4],
            "compute_capability": parts[5],
        })
    if not gpus:
        return None
    topology = run_readonly(["nvidia-smi", "topo", "-m"])
    return {
        "vendor": "nvidia",
        "gpus": gpus,
        "topology": topology.get("stdout", ""),
        "topology_returncode": topology.get("returncode"),
    }


def parse_amd_inventory() -> dict[str, Any] | None:
    query = run_readonly(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if query.get("returncode") != 0 or not query.get("stdout"):
        return None
    try:
        payload = json.loads(query["stdout"])
    except json.JSONDecodeError:
        payload = {"raw": query["stdout"]}
    gpus = []
    if isinstance(payload, dict):
        for index, (card, values) in enumerate(sorted(payload.items())):
            if not isinstance(values, dict):
                continue
            lowered = {str(key).lower(): value for key, value in values.items()}
            name = next(
                (
                    str(value)
                    for key, value in lowered.items()
                    if any(token in key for token in ("card series", "product name", "device name"))
                ),
                str(card),
            )
            memory_bytes = next(
                (
                    int(re.sub(r"[^0-9]", "", str(value)))
                    for key, value in lowered.items()
                    if "vram total memory" in key and re.search(r"[0-9]", str(value))
                ),
                0,
            )
            gpus.append({"index": index, "name": name, "memory_mib": memory_bytes // 1024**2})
    topology = run_readonly(["rocm-smi", "--showtopo", "--json"])
    return {
        "vendor": "amd",
        "gpus": gpus,
        "rocm_smi": payload,
        "topology": topology.get("stdout", ""),
        "topology_returncode": topology.get("returncode"),
    }


def load_hardware_catalog() -> dict[str, Any]:
    try:
        resource = importlib.resources.files("inference_autopilot_data").joinpath("hardware-profiles.json")
        with importlib.resources.as_file(resource) as path:
            return load_json(path)
    except (ModuleNotFoundError, FileNotFoundError):
        # Direct script execution from a source checkout does not install the
        # package-data module, so retain the checked-out reference as fallback.
        path = Path(__file__).resolve().parent.parent / "references" / "hardware-profiles.json"
        return load_json(path)


def match_hardware_profile(inventory: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any] | None:
    names = " ".join(str(gpu.get("name", "")) for gpu in inventory.get("gpus", []))
    if not names.strip():
        names = json.dumps(inventory, sort_keys=True)
    for profile in catalog.get("profiles", []):
        if profile.get("vendor") != inventory.get("vendor"):
            continue
        if any(pattern.lower() in names.lower() for pattern in profile.get("name_patterns", [])):
            matched = deepcopy(profile)
            matched["matched_variant"] = next(
                (
                    variant
                    for variant in profile.get("known_variants", [])
                    if any(
                        pattern.lower() in names.lower()
                        for pattern in variant.get("name_patterns", [variant.get("name", "")])
                        if pattern
                    )
                ),
                None,
            )
            return matched
    return None


def topology_class(inventory: dict[str, Any]) -> str:
    if len(inventory.get("gpus", [])) <= 1:
        return "single-gpu"
    topology = str(inventory.get("topology", ""))
    if inventory.get("vendor") == "nvidia":
        if re.search(r"\bNV\d+\b", topology):
            return "nvlink-or-nvswitch"
        if any(token in topology for token in ("PIX", "PXB", "PHB", "SYS")):
            return "pcie"
    if inventory.get("vendor") == "amd" and topology:
        return "infinity-fabric-or-runtime-reported"
    return "unknown"


def framework_evidence(task: dict[str, Any]) -> dict[str, Any]:
    repository = Path(task["repository"])
    server_args = repository / "python" / "sglang" / "srt" / "server_args.py"
    source = server_args.read_text(encoding="utf-8") if server_args.is_file() else ""
    commit = run_readonly(["git", "rev-parse", "HEAD"], cwd=str(repository))
    runtime_env = os.environ.copy()
    repo_python = str(repository / "python")
    runtime_env["PYTHONPATH"] = repo_python + (
        os.pathsep + runtime_env["PYTHONPATH"] if runtime_env.get("PYTHONPATH") else ""
    )
    cli_help = run_readonly(
        [task["python"], "-m", "sglang.launch_server", "--help"],
        timeout=60,
        cwd=str(repository),
        env=runtime_env,
    )
    cli_text = cli_help.get("stdout", "") + "\n" + cli_help.get("stderr", "")
    cli_flags = sorted(set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", cli_text)))
    return {
        "repository": str(repository),
        "git_commit": commit.get("stdout") if commit.get("returncode") == 0 else None,
        "server_args_path": str(server_args) if server_args.is_file() else None,
        "server_args_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest() if source else None,
        "launch_server_help_sha256": hashlib.sha256(cli_text.encode("utf-8")).hexdigest() if cli_text else None,
        "launch_server_help_available": cli_help.get("returncode") == 0 and bool(cli_flags),
        "launch_server_help_error": cli_help.get("stderr") if cli_help.get("returncode") != 0 else None,
        "launch_server_cli_flags": cli_flags,
        "default_policy": "preserve defaults computed by this installed SGLang version; screen only explicit deltas",
    }


def parameter_catalog(task: dict[str, Any]) -> dict[str, Any]:
    return export_catalog(Path(task["repository"]))


def model_inventory(model_path: str) -> dict[str, Any]:
    root = Path(model_path)
    config_path = root / "config.json"
    config = load_json(config_path) if config_path.is_file() else {}
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
    weight_files = [path for path in root.rglob("*") if path.is_file() and path.suffix in weight_suffixes]
    weight_bytes = sum(path.stat().st_size for path in weight_files)
    architectures = config.get("architectures", [])
    architecture_text = " ".join(str(value) for value in architectures).lower()
    model_type = str(config.get("model_type", "")).lower()
    is_moe = bool(config.get("num_experts") or config.get("num_local_experts") or "moe" in architecture_text)
    is_hybrid = any(token in json.dumps(config).lower() for token in ("mamba", "ssm", "linear_attention"))
    index_path = root / "model.safetensors.index.json"
    index = load_json(index_path) if index_path.is_file() else {}
    weight_map = index.get("weight_map", {}) if isinstance(index, dict) else {}
    mtp_weight_keys = [
        key for key in weight_map
        if "mtp" in key.lower() or "nextn" in key.lower() or "multi_token" in key.lower()
    ]
    mtp_files = sorted(path.name for path in root.glob("*mtp*.safetensors"))
    text_config = config.get("text_config", config)
    mtp_layers = (
        text_config.get("num_nextn_predict_layers", text_config.get("num_mtp_layers", 0))
        if isinstance(text_config, dict) else 0
    )
    has_mtp_weights = bool(mtp_files or mtp_weight_keys)
    quantization_config = config.get("quantization_config", {})
    if not isinstance(quantization_config, dict):
        quantization_config = {}
    return {
        "config_path": str(config_path) if config_path.is_file() else None,
        "architectures": architectures,
        "model_type": model_type,
        "dtype": config.get("torch_dtype", config.get("dtype")),
        "quantization": config.get("quantization", quantization_config.get("quant_method")),
        "context_length": config.get("max_position_embeddings"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "num_experts": config.get("num_experts", config.get("num_local_experts")),
        "is_moe": is_moe,
        "is_hybrid": is_hybrid,
        "num_mtp_layers": mtp_layers,
        "mtp_weight_files": mtp_files,
        "mtp_weight_key_count": len(mtp_weight_keys),
        "has_mtp_weights": has_mtp_weights,
        "weight_files": len(weight_files),
        "weight_bytes": weight_bytes,
        "weight_gib": weight_bytes / 1024**3,
    }


def inferred_cookbook_url(model: dict[str, Any]) -> str | None:
    architecture_text = " ".join(str(item) for item in model.get("architectures", [])).lower()
    if "qwen3_5" in architecture_text and model.get("has_mtp_weights"):
        return "https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6"
    return None


def fetch_reference(url: str, task: dict[str, Any]) -> dict[str, Any]:
    """Fetch an explicitly supplied official reference and retain only evidence metadata."""
    if not url.startswith("https://"):
        return {"url": url, "status": "rejected", "reason": "only HTTPS references are allowed"}
    environment = task.get("env", {}) if isinstance(task.get("env"), dict) else {}
    proxies = {
        scheme: environment[key]
        for scheme, key in (("http", "http_proxy"), ("https", "https_proxy"))
        if isinstance(environment.get(key), str) and environment[key]
    }
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
        request = urllib.request.Request(url, headers={"User-Agent": "inference-autopilot/1.0"})
        with opener.open(request, timeout=20) as response:
            body = response.read(2_000_000)
        text = body.decode("utf-8", errors="replace")
        return {
            "url": url,
            "status": "fetched",
            "retrieved_at": utc_now(),
            "sha256": hashlib.sha256(body).hexdigest(),
            "text": text,
        }
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return {"url": url, "status": "unavailable", "reason": type(exc).__name__}


def cookbook_evidence(task: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    knowledge = task.get("knowledge", {}) if isinstance(task.get("knowledge"), dict) else {}
    url = knowledge.get("model_cookbook_url") or inferred_cookbook_url(model)
    required = bool(knowledge.get("require_cookbook", False))
    if url is None:
        return {"status": "not_matched", "required": required, "model_profile": None}
    fetched = fetch_reference(url, task) if task.get("allow_download", False) else {
        "url": url, "status": "not_fetched", "reason": "allow_download=false"
    }
    text = str(fetched.pop("text", ""))
    normalized = re.sub(r"\s+", " ", text).lower()
    qwen36 = "qwen3.6" in url.lower() or "qwen3.6" in normalized
    claims = {
        "mtp_eagle": "speculative-algorithm eagle" in normalized,
        "mamba_extra_buffer": "extra_buffer" in normalized and "mamba" in normalized,
        "page_size_64": "page-size 64" in normalized or "page size 64" in normalized,
        "spec_v2": "sglang_enable_spec_v2" in normalized,
    }
    profile = None
    if qwen36:
        profile = {
            "name": "qwen3.6-hybrid-mtp",
            "requires_mtp_weights": True,
            "initial_bundles": [
                {
                    "name": "qwen36-prefix-cache-lpm",
                    "config": {
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "schedule_policy": "lpm",
                        "mem_fraction_static": 0.8,
                    },
                },
                {
                    "name": "qwen36-mtp-eagle-3-1-4",
                    "config": {
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mem_fraction_static": 0.8,
                    },
                },
                {
                    "name": "qwen36-mtp-prefix-cache-lpm",
                    "config": {
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "page_size": 64,
                        "mem_fraction_static": 0.8,
                    },
                },
                {
                    "name": "qwen36-mtp-prefix-memory-075",
                    "config": {
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "page_size": 64,
                        "schedule_policy": "lpm",
                        "mem_fraction_static": 0.75,
                    },
                },
                {
                    "name": "qwen36-mtp-prefix-admission-graph",
                    "config": {
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "page_size": 64,
                        "schedule_policy": "lpm",
                        "mem_fraction_static": 0.8,
                        "max_running_requests": 16,
                        "cuda_graph_max_bs_decode": 8,
                    },
                },
                {
                    "name": "qwen36-mtp-prefix-continuous-decode-2",
                    "config": {
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "page_size": 64,
                        "schedule_policy": "lpm",
                        "mem_fraction_static": 0.8,
                        "num_continuous_decode_steps": 2,
                    },
                },
                {
                    "name": "qwen36-mtp-prefix-moe-triton",
                    "config": {
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "page_size": 64,
                        "schedule_policy": "lpm",
                        "mem_fraction_static": 0.8,
                        "moe_runner_backend": "triton",
                    },
                },
            ],
            "claims": claims,
        }
        if task.get("capability_overrides", {}).get("mtp") == "disabled":
            profile["initial_bundles"] = [
                bundle for bundle in profile["initial_bundles"]
                if "speculative_algorithm" not in bundle["config"]
            ]
            profile["mtp_status"] = "disabled_by_task"
    return {
        **fetched,
        "required": required,
        "model_profile": profile,
        "claims": claims,
        "checkpoint_verification": {
            "has_mtp_weights": bool(model.get("has_mtp_weights")),
            "mtp_weight_key_count": model.get("mtp_weight_key_count", 0),
        },
    }


def selected_gpus(task: dict[str, Any], inventory: dict[str, Any]) -> list[dict[str, Any]]:
    gpus = inventory.get("gpus", [])
    visible = task.get("env", {}).get("CUDA_VISIBLE_DEVICES", task.get("env", {}).get("HIP_VISIBLE_DEVICES"))
    if visible is None:
        return gpus
    identifiers = [item.strip() for item in str(visible).split(",") if item.strip() and item.strip() != "-1"]
    if all(identifier.isdigit() for identifier in identifiers):
        indexes = {int(identifier) for identifier in identifiers}
        return [gpu for gpu in gpus if gpu.get("index") in indexes]
    # UUID visibility cannot always be correlated across vendor tools. Use the
    # lowest-memory discovered devices conservatively for feasibility checks.
    return sorted(gpus, key=lambda gpu: gpu.get("memory_mib", 0))[: len(identifiers)]


def visible_gpu_count(task: dict[str, Any], inventory: dict[str, Any]) -> int:
    return len(selected_gpus(task, inventory))


def minimum_tp(task: dict[str, Any], inventory: dict[str, Any], model: dict[str, Any]) -> int:
    gpus = selected_gpus(task, inventory)
    count = len(gpus)
    if not count:
        raise ValueError("no visible GPUs remain after applying the visibility environment")
    if all(gpu.get("memory_mib", 0) > 0 for gpu in gpus):
        memory_bytes = min(gpu["memory_mib"] for gpu in gpus) * 1024**2
    else:
        raise ValueError("GPU memory could not be discovered; refusing to guess tensor parallelism")
    required = max(model.get("weight_bytes", 0) * 1.12, model.get("weight_bytes", 0) + 4 * 1024**3)
    for tp in range(1, count + 1):
        if count % tp == 0 and required / tp <= memory_bytes * 0.88:
            return tp
    raise ValueError("model weights do not fit the visible GPU memory with a supported tensor-parallel divisor")


def estimate_kv_cache_bytes(model: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    """Return a transparent BF16 KV upper estimate for feasibility routing."""
    layers = model.get("num_hidden_layers")
    hidden = model.get("hidden_size")
    heads = model.get("num_attention_heads")
    kv_heads = model.get("num_key_value_heads") or heads
    if not all(isinstance(value, int) and value > 0 for value in (layers, hidden, heads, kv_heads)):
        return {"available": False, "reason": "model config lacks attention dimensions"}
    head_dim = hidden // heads
    if head_dim <= 0:
        return {"available": False, "reason": "invalid attention dimensions"}
    # K and V, BF16/FP16 storage. Hybrid models may use less; this is an
    # admission estimate, not a claimed allocator allocation.
    bytes_per_token = layers * kv_heads * head_dim * 2 * 2
    tokens = (int(workload["input_tokens"]) + int(workload["output_tokens"])) * int(workload["max_concurrency"])
    return {
        "available": True,
        "assumption": "BF16/FP16 K/V upper estimate before prefix reuse and allocator effects",
        "bytes_per_token": bytes_per_token,
        "target_tokens": tokens,
        "estimated_bytes": bytes_per_token * tokens,
    }


def single_gpu_feasibility(task: dict[str, Any], inventory: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Assess one visible GPU without launching SGLang or changing a model."""
    gpus = selected_gpus(task, inventory)
    if len(gpus) != 1 or not isinstance(gpus[0].get("memory_mib"), int) or gpus[0]["memory_mib"] <= 0:
        return {
            "status": "insufficient_inventory",
            "reason": "single-GPU feasibility requires exactly one visible GPU with discovered memory",
        }
    gpu_bytes = gpus[0]["memory_mib"] * 1024**2
    usable_bytes = int(gpu_bytes * 0.88)
    weights = int(model.get("weight_bytes", 0))
    kv = estimate_kv_cache_bytes(model, task["workload"])
    kv_bytes = int(kv.get("estimated_bytes", 0))
    runtime_reserve = max(4 * 1024**3, int(gpu_bytes * 0.05))
    estimated_peak = int(weights * 1.12) + kv_bytes + runtime_reserve
    headroom = usable_bytes - estimated_peak
    status = "deployable_as_is" if headroom >= 0 else "requires_parallel_or_variant"
    required_gpus = max(1, math.ceil(estimated_peak / max(1, usable_bytes)))
    profile = match_hardware_profile(inventory, load_hardware_catalog()) or {}
    precisions = profile.get("precision", [])
    suggestions = []
    if status != "deployable_as_is" and task.get("deployment", {}).get("allow_model_variant_recommendations", True):
        if "fp8" in precisions:
            suggestions.append({"target": "fp8 checkpoint", "reason": "hardware catalog reports FP8 capability; verify the exact checkpoint/backend and quality before use"})
        if any(item in precisions for item in ("mxfp4", "mxfp6", "mxfp8")):
            suggestions.append({"target": "FP4/MXFP4 checkpoint", "reason": "hardware catalog reports low-precision capability; requires an explicit compatible checkpoint and quality validation"})
    return {
        "status": status,
        "scope": "single_gpu_estimate",
        "gpu": {"name": gpus[0].get("name"), "memory_gib": round(gpu_bytes / 1024**3, 2), "usable_gib": round(usable_bytes / 1024**3, 2)},
        "model_weight_gib": round(weights / 1024**3, 2),
        "weight_reserve_gib": round(weights * 1.12 / 1024**3, 2),
        "runtime_reserve_gib": round(runtime_reserve / 1024**3, 2),
        "kv_cache_estimate": {**kv, "estimated_gib": round(kv_bytes / 1024**3, 2)},
        "estimated_peak_gib": round(estimated_peak / 1024**3, 2),
        "estimated_headroom_gib": round(headroom / 1024**3, 2),
        "minimum_gpu_count_estimate": required_gpus,
        "recommendations": suggestions,
        "quality_gate": {
            "required_for_model_variant": bool(suggestions),
            "evaluation_dataset": task.get("quality", {}).get("evaluation_dataset"),
            "max_regression_pct": task.get("quality", {}).get("max_regression_pct"),
            "policy": "never switch, download, or claim a quantized model is acceptable without an explicit quality evaluation",
        },
    }


def power_of_two_floor(value: int) -> int:
    return 1 << max(0, int(math.log2(max(1, value))))


def power_of_two_ceil(value: int) -> int:
    floor = power_of_two_floor(value)
    return floor if floor == value else floor * 2


def expected_prefill_tokens(workload: dict[str, Any]) -> int:
    """Estimate uncached prefill work per request for candidate routing.

    This is a prior only. The benchmark and cache metrics remain the source of
    truth because a declared prefix-reuse ratio need not equal the observed hit
    rate.
    """
    input_tokens = int(workload["input_tokens"])
    reuse_ratio = float(workload.get("prefix_reuse_ratio", 0.0))
    return max(1, int(round(input_tokens * max(0.0, 1.0 - reuse_ratio))))


def chunk_candidates(
    task: dict[str, Any], framework_default: int | None = None
) -> list[int]:
    """Return a workload-centred chunk sweep, including the resolved default.

    `chunked_prefill_size` is a prefill-batch budget, not a per-request value.
    The relevant first-order boundary is therefore concurrent uncached prefill
    tokens. SGLang's current-version resolved default is deliberately retained
    as an anchor instead of embedding a GPU-specific default in this tool.
    """
    workload = task["workload"]
    boundary = expected_prefill_tokens(workload) * int(workload["max_concurrency"])
    if boundary < 256:
        return []
    lower = max(256, power_of_two_floor(boundary) // 2)
    centre = max(256, power_of_two_floor(boundary))
    upper = max(256, power_of_two_ceil(boundary))
    resolved_default = framework_default if isinstance(framework_default, int) and framework_default > 0 else None
    # Do not grow activation allocations without a bounded local reason. The
    # default is still included as an anchor even if it falls outside this cap.
    cap = max(16384, (resolved_default or upper) * 4)
    sweep_end = min(cap, max(upper * 2, (resolved_default or upper) // 2))
    values: set[int] = set()
    value = lower
    while value <= sweep_end:
        values.add(value)
        value *= 2
    values.update({centre, upper})
    if resolved_default is not None:
        values.add(resolved_default)
    return sorted(value for value in values if 256 <= value <= cap)


def shared_prefix_benchmark(workload: dict[str, Any]) -> dict[str, Any] | None:
    """Translate task-level shared-prefix intent to SGLang's native dataset."""
    config = workload.get("shared_prefix")
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("workload.shared_prefix must be an object")
    groups = int(config.get("groups", 1))
    prompts_per_group = int(config.get("prompts_per_group", math.ceil(workload["num_prompts"] / groups)))
    system_tokens = int(config.get("system_prompt_tokens", max(1, workload["input_tokens"] * 3 // 4)))
    question_tokens = int(config.get("question_tokens", workload["input_tokens"] - system_tokens))
    if groups <= 0 or prompts_per_group <= 0 or system_tokens <= 0 or question_tokens <= 0:
        raise ValueError("workload.shared_prefix groups, prompts_per_group, and token lengths must be positive")
    if system_tokens + question_tokens != workload["input_tokens"]:
        raise ValueError("workload.shared_prefix system_prompt_tokens + question_tokens must equal workload.input_tokens")
    return {
        "dataset_name": "generated-shared-prefix",
        "num_prompts": groups * prompts_per_group,
        "gsp_num_groups": groups,
        "gsp_prompts_per_group": prompts_per_group,
        "gsp_system_prompt_len": system_tokens,
        "gsp_question_len": question_tokens,
        "gsp_output_len": workload["output_tokens"],
        "gsp_range_ratio": float(config.get("range_ratio", 1.0)),
        "gsp_ordered": bool(config.get("ordered", False)),
    }


def build_execution_spec(
    task: dict[str, Any],
    discovery: dict[str, Any],
    *,
    stage_name: str,
    baseline: dict[str, Any],
    space: dict[str, list[Any]],
    max_trials: int,
    repetitions: int,
    remaining_gpu_hours: float,
    remaining_wall_minutes: float,
) -> dict[str, Any]:
    workload = task["workload"]
    measurement = task.get("measurement") or {}
    min_requests = int(measurement.get("min_measurement_requests", max(256, workload["max_concurrency"] * 64)))
    warmup_requests = int(measurement.get("warmup_requests", max(32, workload["max_concurrency"] * 8)))
    model = discovery["model"]
    inventory = discovery["hardware"]
    gpu_count = visible_gpu_count(task, inventory)
    gpu_model = "unknown"
    selected = selected_gpus(task, inventory)
    if selected:
        gpu_model = selected[0]["name"]
    shared_prefix = shared_prefix_benchmark(workload)
    benchmark = {
        "dataset_name": "random-ids",
        "num_prompts": max(workload["num_prompts"], min_requests),
        "random_input_len": workload["input_tokens"],
        "random_output_len": workload["output_tokens"],
        "random_range_ratio": 1.0,
        "request_rate": workload.get("request_rate", "inf"),
        "max_concurrency": workload["max_concurrency"],
        "warmup_requests": warmup_requests,
        "min_measurement_seconds": float(measurement.get("min_measurement_seconds", 30)),
        "seed": 1,
        "output_details": True,
    }
    if shared_prefix is not None:
        benchmark.update(shared_prefix)
        benchmark["num_prompts"] = max(benchmark["num_prompts"], min_requests)
        groups = benchmark["gsp_num_groups"]
        benchmark["gsp_prompts_per_group"] = math.ceil(benchmark["num_prompts"] / groups)
        benchmark["num_prompts"] = groups * benchmark["gsp_prompts_per_group"]
    return {
        "name": f"{task['name']}-{stage_name}"[:64],
        "mode": "execute",
        "framework": "sglang",
        "repository": task["repository"],
        "model": {
            "path": task["model_path"],
            "dtype": task.get("model", {}).get("dtype", model.get("dtype")),
            "architecture": "moe" if model.get("is_moe") else "dense",
            "context_length": task.get("model", {}).get("context_length", model.get("context_length")),
        },
        "hardware": {
            "hosts": 1,
            "gpus_per_host": gpu_count,
            "gpu_model": gpu_model,
            "interconnect": "runtime-discovered",
        },
        "workload": {
            "arrival": "closed_loop" if workload.get("request_rate", "inf") == "inf" else "poisson",
            "request_rate": workload.get("request_rate", "inf"),
            "max_concurrency": workload["max_concurrency"],
            "num_prompts": workload["num_prompts"],
            "input_tokens": {"p50": workload["input_tokens"], "p95": workload["input_tokens"]},
            "output_tokens": {"p50": workload["output_tokens"], "p95": workload["output_tokens"]},
            "prefix_reuse_ratio": workload.get("prefix_reuse_ratio", 0.0),
        },
        "slo": task["slo"],
        "objective": task["objective"],
        "budget": {
            "max_trials": max_trials,
            "max_gpu_hours": max(0.01, remaining_gpu_hours),
            "max_wall_time_minutes": max(1, remaining_wall_minutes),
            "max_consecutive_failures": 3,
        },
        "scope": {
            "allow_launch": True,
            "allow_download": task.get("allow_download", False),
            "allow_profiling": False,
            "allow_parameter_changes": True,
            "allow_code_changes": False,
            "allow_kernel_changes": False,
            "production": False,
            "output_dir": str(Path(task["output_dir"]) / "stages"),
        },
        "execution": {
            "python": task["python"],
            "host": "127.0.0.1",
            "port": task.get("port", 31000),
            "offline": task.get("offline", True),
            "require_accelerator": True,
            "startup_timeout_sec": 1200,
            "benchmark_timeout_sec": 1800,
            "shutdown_timeout_sec": 60,
            "env": task.get("env", {}),
            "parameter_bindings": execution_parameter_bindings(discovery),
        },
        "benchmark": benchmark,
        "search": {
            "strategy": "one_factor",
            "repetitions": repetitions,
            "order": "interleaved",
            "max_cv_pct": 10,
            "min_confirm_repetitions": task.get("confirmation_repetitions", 3),
            "require_all_slo_pass": True,
            "baseline": baseline,
            "space": space,
            "parameter_order": list(space),
        },
    }


def calibration_spec(
    task: dict[str, Any], discovery: dict[str, Any], concurrency: int,
    remaining_gpu_hours: float, remaining_wall_minutes: float,
) -> dict[str, Any]:
    calibrated_task = deepcopy(task)
    calibrated_task["workload"]["max_concurrency"] = concurrency
    # Capacity points below the target do not need the target's warmup and
    # request count. Keep a meaningful floor while scaling with actual load.
    target_concurrency = max(1, task["workload"]["max_concurrency"])
    scale = min(1.0, concurrency / target_concurrency)
    measurement = calibrated_task.get("measurement") or {}
    target_requests = int(measurement.get("min_measurement_requests", max(64, concurrency * 16)))
    calibrated_task["measurement"] = {
        **measurement,
        "warmup_requests": max(16, math.ceil(int(measurement.get("warmup_requests", 32)) * scale)),
        "min_measurement_requests": max(32, math.ceil(target_requests * scale)),
    }
    calibrated_task["workload"]["num_prompts"] = max(
        calibrated_task["measurement"]["min_measurement_requests"], concurrency * 8
    )
    return build_execution_spec(
        calibrated_task,
        discovery,
        stage_name=f"calibrate-c{concurrency}",
        baseline={"tp_size": discovery["derived"]["minimum_tp_size"]},
        space={},
        max_trials=1,
        repetitions=1,
        remaining_gpu_hours=remaining_gpu_hours,
        remaining_wall_minutes=remaining_wall_minutes,
    )


def run_calibration(
    task: dict[str, Any], discovery: dict[str, Any], root: Path,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Measure baseline capacity without conflating it with the target-workload winner."""
    policy = deployment_policy(task)
    points: list[dict[str, Any]] = []
    started = time.monotonic()
    used_gpu_hours = 0.0
    concurrencies = calibration_concurrencies(task)
    for point_index, concurrency in enumerate(concurrencies, start=1):
        remaining_gpu_hours = float(task["budget"]["max_gpu_hours"]) - used_gpu_hours
        remaining_wall_minutes = float(task["budget"]["max_wall_time_minutes"]) - (time.monotonic() - started) / 60
        if remaining_gpu_hours <= 0 or remaining_wall_minutes <= 0:
            break
        spec = calibration_spec(task, discovery, concurrency, remaining_gpu_hours, remaining_wall_minutes)
        stage = f"capacity {point_index}/{len(concurrencies)} (concurrency={concurrency})"
        report = execute_with_progress(spec, progress, stage) if progress else execute(spec)
        aggregate = next((item for item in report.get("aggregates", []) if item.get("kind") == "baseline"), None)
        slo_passed = bool(aggregate and aggregate.get("slo", {}).get("passed"))
        completed = bool(aggregate and aggregate.get("completed_repetitions") == 1)
        valid = completed and (slo_passed or (policy["mode"] == "offline_throughput" and not task["slo"]))
        points.append({
            "concurrency": concurrency,
            "run_dir": report.get("run_dir"),
            "stop_reason": report.get("stop_reason"),
            "metrics": aggregate.get("metrics", {}) if aggregate else {},
            "slo_passed": slo_passed,
            "valid_for_analysis": valid,
        })
        used_gpu_hours += float(report.get("approx_gpu_hours", 0))
        if (
            policy["mode"] == "online_latency"
            and not valid
            and (task.get("calibration") or {}).get("stop_on_slo_failure", True)
        ):
            break
    valid_points = [point for point in points if point["valid_for_analysis"]]
    target = task["workload"]["max_concurrency"]
    selected = valid_points[-1]["concurrency"] if valid_points else target
    return {
        "policy": policy,
        "target_concurrency": target,
        "points": points,
        "selected_analysis_concurrency": selected,
        "stopped_before_requested_cap": bool(points) and len(points) < len(calibration_concurrencies(task)),
        "approx_gpu_hours": used_gpu_hours,
        "completed_trials": len(points),
    }


def discover(task: dict[str, Any]) -> dict[str, Any]:
    hardware = parse_nvidia_inventory() or parse_amd_inventory()
    if hardware is None:
        raise RuntimeError("no supported NVIDIA or AMD accelerator inventory available")
    catalog = load_hardware_catalog()
    model = model_inventory(task["model_path"])
    cookbook = cookbook_evidence(task, model)
    if cookbook.get("required") and cookbook.get("status") != "fetched":
        raise RuntimeError(
            "required model cookbook could not be fetched before optimization: "
            + str(cookbook.get("reason") or cookbook.get("status"))
        )
    profile = match_hardware_profile(hardware, catalog)
    tp_size = minimum_tp(task, hardware, model)
    framework = framework_evidence(task)
    if not framework["launch_server_help_available"]:
        raise RuntimeError(
            "could not obtain the current sglang.launch_server CLI surface: "
            + str(framework.get("launch_server_help_error") or "unknown error")
        )
    parameters = parameter_catalog(task)
    cli_flags = set(framework["launch_server_cli_flags"])
    compatible_parameters = []
    for item in parameters["parameters"]:
        item = deepcopy(item)
        item["cli_visible"] = bool(set(item["flags"]) & cli_flags)
        compatible_parameters.append(item)
    parameters["parameters"] = compatible_parameters
    parameters["cli_visible_parameter_count"] = sum(
        1 for item in compatible_parameters if item["cli_visible"]
    )
    parameters["parameter_contract"] = {
        "source": "ServerArgs.add_cli_args + current sglang.launch_server --help",
        "server_args_sha256": framework["server_args_sha256"],
        "launch_server_help_sha256": framework["launch_server_help_sha256"],
    }
    return {
        "collected_at": utc_now(),
        "hardware": hardware,
        "hardware_profile": profile,
        "topology_class": topology_class(hardware),
        "hardware_catalog_retrieved_at": catalog.get("retrieved_at"),
        "framework": framework,
        "parameter_catalog": parameters,
        "model": model,
        "cookbook": cookbook,
        "derived": {
            "visible_gpu_count": visible_gpu_count(task, hardware),
            "visible_gpus": selected_gpus(task, hardware),
            "minimum_tp_size": tp_size,
            "typical_prefill_batch_tokens": expected_prefill_tokens(task["workload"])
            * task["workload"]["max_concurrency"],
            "expected_uncached_prefill_tokens_per_request": expected_prefill_tokens(task["workload"]),
            "chunked_prefill_candidates": chunk_candidates(task),
        },
    }


def catalog_index(discovery: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["dest"]: item
        for item in discovery["parameter_catalog"]["parameters"]
        if not item.get("deprecated") and item.get("cli_visible", True)
    }


def execution_parameter_bindings(discovery: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Freeze the discovered ServerArgs bindings into reproducible trial specs."""
    return {
        item["dest"]: {
            "primary_flag": item["primary_flag"],
            "action": item.get("action"),
            "value_type": item.get("value_type"),
            "choices": item.get("choices"),
        }
        for item in discovery["parameter_catalog"]["parameters"]
        if item.get("cli_visible", True) and not item.get("deprecated")
    }


def add_ranked_candidate(
    ranked: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    parameter: str,
    values: list[Any],
    reason: str,
    evidence: list[str],
    tier: str = "evidence",
) -> None:
    metadata = catalog.get(parameter)
    if metadata is None:
        return
    choices = metadata.get("choices")
    filtered = []
    for value in values:
        if choices is not None and value not in choices:
            continue
        if value == metadata.get("default") or value in filtered:
            continue
        filtered.append(value)
    if filtered:
        existing = next((item for item in ranked if item["parameter"] == parameter), None)
        if existing is not None:
            existing["values"] = list(dict.fromkeys([*existing["values"], *filtered]))
            existing["evidence"] = list(dict.fromkeys([*existing["evidence"], *evidence]))
            if tier == "sensitivity":
                existing["tiers"] = list(dict.fromkeys([*existing.get("tiers", ["evidence"]), tier]))
            return
        ranked.append({
            "parameter": parameter,
            "values": filtered,
            "family": metadata["family"],
            "flag": metadata["primary_flag"],
            "installed_default": metadata.get("default"),
            "reason": reason,
            "evidence": evidence,
            "semantics": metadata.get("help"),
            "tiers": [tier],
        })


def hardware_backends(discovery: dict[str, Any]) -> dict[str, list[str]]:
    profile = discovery.get("hardware_profile") or {}
    architecture = profile.get("architecture")
    vendor = discovery["hardware"].get("vendor")
    if vendor == "amd":
        return {"attention": ["aiter", "triton"], "moe": ["aiter", "triton"]}
    if architecture == "blackwell":
        return {
            "attention": ["trtllm_mha", "fa4", "flashinfer", "triton"],
            "moe": ["flashinfer_trtllm", "flashinfer_cutedsl", "deep_gemm", "triton"],
        }
    if architecture == "hopper":
        return {
            "attention": ["fa3", "flashinfer", "triton"],
            "moe": ["deep_gemm", "flashinfer_trtllm", "triton"],
        }
    return {"attention": ["flashinfer", "triton"], "moe": ["triton"]}


def prometheus_value(lines: list[str], metric: str) -> float | None:
    pattern = re.compile(rf"^{re.escape(metric)}(?:\{{.*\}})?\s+([-+0-9.eE]+)$")
    values = []
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            try:
                values.append(float(match.group(1)))
            except ValueError:
                pass
    return values[-1] if values else None


def known_failed_candidates(task: dict[str, Any]) -> set[tuple[str, str]]:
    """Reuse only local, same-task evidence for one-factor backend/config failures."""
    failed: set[tuple[str, str]] = set()
    output_dir = task.get("output_dir")
    if not isinstance(output_dir, str):
        return failed
    root = Path(output_dir) / "stages"
    if not root.is_dir():
        return failed
    for result_path in root.glob("*/results.json"):
        spec_path = result_path.parent / "spec.json"
        if not spec_path.is_file():
            continue
        try:
            baseline = load_json(spec_path).get("search", {}).get("baseline", {})
            rows = load_json(result_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(baseline, dict) or not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("ok") or row.get("kind") != "candidate":
                continue
            config = row.get("config", {})
            if not isinstance(config, dict):
                continue
            changed = [(key, value) for key, value in config.items() if baseline.get(key) != value]
            if len(changed) == 1:
                key, value = changed[0]
                failed.add((key, json.dumps(value, sort_keys=True)))
    return failed


def parameter_audit(
    catalog: dict[str, dict[str, Any]],
    ranked: list[dict[str, Any]],
    discovery: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    """Account for every current-version CLI parameter in a search plan.

    A serving binary exposes many control-plane, topology, and diagnostic
    options. Treating all of them as independent performance dials would make
    an unbounded and invalid experiment. This audit makes every exclusion
    explicit instead of silently hiding it behind a small candidate list.
    """
    selected = {item["parameter"] for item in ranked}
    cookbook = discovery.get("cookbook", {})
    profile = cookbook.get("model_profile", {}) if isinstance(cookbook, dict) else {}
    for bundle in profile.get("initial_bundles", []) if isinstance(profile, dict) else []:
        if isinstance(bundle, dict) and isinstance(bundle.get("config"), dict):
            selected.update(bundle["config"])
    gpu_count = discovery["derived"]["visible_gpu_count"]
    is_moe = bool(discovery["model"].get("is_moe"))
    has_draft_model = (
        bool(task.get("speculative"))
        or bool(task.get("draft_model_path"))
        or (
            bool(discovery["model"].get("has_mtp_weights"))
            and task.get("capability_overrides", {}).get("mtp") != "disabled"
        )
    )
    static_identity = {
        "model_path", "served_model_name", "host", "port", "api_key",
        "tokenizer_path", "chat_template", "revision", "trust_remote_code",
    }
    entries: list[dict[str, Any]] = []
    summary: dict[str, int] = {}
    for parameter, metadata in sorted(catalog.items()):
        state = "excluded"
        reason = "not selected by the measured bottleneck and workload routing"
        if metadata["family"] == "speculative" and task.get("capability_overrides", {}).get("mtp") == "disabled":
            state = "inapplicable"
            reason = "MTP/speculative decoding is explicitly disabled for this SGLang-model compatibility combination"
        elif parameter in selected:
            state = "selected"
            reason = "scheduled for an isolated, SLO-gated measurement"
        elif metadata["family"] == "observability":
            reason = "diagnostic or logging control; changing it would perturb the measurement"
        elif parameter in static_identity:
            reason = "model, endpoint, or compatibility identity; not a serving-path tuning dial"
        elif metadata["family"] == "speculative" and not has_draft_model:
            state = "inapplicable"
            reason = "speculative decoding requires an explicitly supplied compatible draft model and acceptance workload"
        elif metadata["family"] == "parallelism" and gpu_count == 1:
            state = "inapplicable"
            reason = "multi-GPU parallelism setting is not meaningful on the discovered single-GPU topology"
        elif metadata["family"] == "communication" and gpu_count == 1:
            state = "inapplicable"
            reason = "collective-communication setting is not meaningful on the discovered single-GPU topology"
        elif metadata["family"] == "moe" and not is_moe:
            state = "inapplicable"
            reason = "MoE-specific setting is not applicable to the detected dense model"
        elif metadata.get("required") or metadata.get("action") == "append":
            reason = "structured deployment/control-plane input; it is not safely enumerable as a scalar startup candidate"
        entries.append({
            "parameter": parameter,
            "flag": metadata["primary_flag"],
            "family": metadata["family"],
            "state": state,
            "reason": reason,
        })
        summary[state] = summary.get(state, 0) + 1
    return {
        "policy": "all CLI-visible, non-deprecated SGLang ServerArgs are accounted for; only compatible serving-path dials enter isolated SLO-gated experiments",
        "summary": summary,
        "parameters": entries,
    }


def cookbook_candidate_bundles(
    discovery: dict[str, Any], catalog: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate cookbook configuration bundles against the local CLI and checkpoint."""
    cookbook = discovery.get("cookbook", {})
    profile = cookbook.get("model_profile") if isinstance(cookbook, dict) else None
    if not isinstance(profile, dict):
        return [], []
    excluded: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    if profile.get("requires_mtp_weights") and not discovery["model"].get("has_mtp_weights"):
        excluded.append({
            "profile": profile.get("name"),
            "reason": "cookbook MTP bundle requires checkpoint MTP weights, but none were found locally",
        })
        return bundles, excluded
    for bundle in profile.get("initial_bundles", []):
        config = bundle.get("config", {})
        missing = [name for name in config if name not in catalog]
        invalid = [
            name for name, value in config.items()
            if name in catalog
            and catalog[name].get("choices") is not None
            and value not in catalog[name]["choices"]
        ]
        if missing or invalid:
            excluded.append({
                "name": bundle.get("name"),
                "reason": "bundle is absent from or incompatible with the installed SGLang CLI",
                "missing_parameters": missing,
                "invalid_parameters": invalid,
            })
            continue
        bundles.append(deepcopy(bundle))
    return bundles, excluded


def cookbook_initial_search_plan(task: dict[str, Any], discovery: dict[str, Any]) -> dict[str, Any]:
    catalog = catalog_index(discovery)
    bundles, exclusions = cookbook_candidate_bundles(discovery, catalog)
    required_environment: dict[str, Any] = {}
    for bundle in bundles:
        environment = bundle.get("env", {})
        if isinstance(environment, dict):
            required_environment.update(environment)
    return {
        "schema_version": 1,
        "phase": "cookbook_initialization",
        "ranked_parameter_groups": [],
        "cookbook_candidate_bundles": bundles,
        "cookbook_bundle_exclusions": exclusions,
        "required_environment": required_environment,
        "parameter_audit": parameter_audit(catalog, [], discovery, task),
        "policy": "benchmark locally compatible model-cookbook configuration bundles across capability, prefix cache, scheduler, memory, CUDA Graph, and MoE families before profiling; retain only SLO-valid evidence",
    }


def diagnosed_search_plan(
    task: dict[str, Any], discovery: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    diagnosis = profile["diagnosis"]
    primary = diagnosis["primary_bottleneck"]
    secondary = set(diagnosis.get("secondary_bottlenecks", []))
    shares = diagnosis.get("shares_pct", {})
    catalog = catalog_index(discovery)
    ranked: list[dict[str, Any]] = []
    cookbook_bundles, cookbook_bundle_exclusions = cookbook_candidate_bundles(discovery, catalog)
    workload = task["workload"]
    concurrency = workload["max_concurrency"]
    boundary = discovery["derived"]["typical_prefill_batch_tokens"]
    backends = hardware_backends(discovery)
    evidence = [f"nsys.primary_bottleneck={primary}"]
    effective = profile.get("effective_server_config", {})
    prometheus_lines = profile.get("prometheus", {}).get("selected_samples", [])
    runtime = profile.get("runtime_observations", {})
    runtime_decode = runtime.get("decode", {}) if isinstance(runtime, dict) else {}
    runtime_prefill = runtime.get("prefill", {}) if isinstance(runtime, dict) else {}
    decode_graph_coverage = runtime_decode.get("cuda_graph_coverage_pct")
    cached_token_share = runtime_prefill.get("cached_token_share_pct")
    decode_graph_active = (
        any("mode=\"decode_cuda_graph\"" in line for line in prometheus_lines)
        or (isinstance(decode_graph_coverage, (int, float)) and decode_graph_coverage >= 95.0)
    )
    known_failures = known_failed_candidates(task)
    queue_reqs = prometheus_value(prometheus_lines, "sglang:num_queue_reqs")
    token_usage = prometheus_value(prometheus_lines, "sglang:token_usage")
    retractions = prometheus_value(prometheus_lines, "sglang:num_retracted_reqs")
    # A closed-loop client normally has no waiting queue while it keeps all
    # target requests in flight. A post-run queue sample cannot suppress
    # scheduler tuning for that workload shape.
    underdriven = (
        workload.get("request_rate", "inf") != "inf"
        and queue_reqs == 0
        and token_usage is not None
        and token_usage < 0.5
    )
    mode = deployment_policy(task)["mode"]
    evidence.extend([
        f"prometheus.queue_reqs={queue_reqs}",
        f"prometheus.token_usage={token_usage}",
        f"prometheus.retractions={retractions}",
        f"scheduler_log.decode_cuda_graph_coverage_pct={decode_graph_coverage}",
        f"scheduler_log.prefill_cached_token_share_pct={cached_token_share}",
    ])

    # Offline optimization deliberately runs at calibrated batch pressure. It
    # must explore capacity controls even when a short profiler range happens
    # not to contain a queue sample. Online mode keeps these controls behind
    # the tail-latency and queue evidence below.
    if mode == "offline_throughput":
        base_mem_fraction = effective.get("mem_fraction_static", catalog.get("mem_fraction_static", {}).get("default"))
        if isinstance(base_mem_fraction, (int, float)) and not isinstance(base_mem_fraction, bool):
            add_ranked_candidate(
                ranked, catalog, "mem_fraction_static",
                [round(max(0.60, float(base_mem_fraction) - 0.03), 3), round(min(0.97, float(base_mem_fraction) + 0.02), 3)],
                "sweep KV allocation around the resolved SGLang default at sustained batch pressure",
                evidence + [f"resolved_mem_fraction_static={base_mem_fraction}"],
            )
        add_ranked_candidate(
            ranked, catalog, "max_running_requests", [concurrency, concurrency * 2, concurrency * 4],
            "sweep admission ceiling after capacity calibration rather than assuming the interactive default", evidence,
        )
        add_ranked_candidate(
            ranked, catalog, "schedule_conservativeness", [0.3, 0.6, 1.1],
            "sweep admission aggressiveness under calibrated offline queue pressure", evidence,
        )
        add_ranked_candidate(
            ranked, catalog, "num_continuous_decode_steps", [2, 4, 8],
            "amortize scheduler work for throughput-oriented continuous batching", evidence,
        )
        add_ranked_candidate(
            ranked, catalog, "enable_mixed_chunk", [True],
            "test mixed prefill/decode batching only for throughput-oriented execution", evidence,
        )

    if task.get("search_depth", "thorough") == "thorough":
        # Sensitivity candidates provide coverage when the target workload is
        # low-load or a short trace cannot expose every capacity knob. They are
        # deliberately one-factor and remain subject to the same SLO gates.
        base_mem_fraction = effective.get("mem_fraction_static", catalog.get("mem_fraction_static", {}).get("default"))
        if isinstance(base_mem_fraction, (int, float)) and not isinstance(base_mem_fraction, bool):
            add_ranked_candidate(
                ranked, catalog, "mem_fraction_static",
                [round(max(0.60, float(base_mem_fraction) - 0.03), 3), round(min(0.97, float(base_mem_fraction) + 0.02), 3)],
                "sensitivity screen KV allocation around the resolved current-version default",
                evidence + [f"resolved_mem_fraction_static={base_mem_fraction}"], tier="sensitivity",
            )
        add_ranked_candidate(
            ranked, catalog, "max_running_requests", [max(1, concurrency * 2), max(2, concurrency * 4)],
            "sensitivity screen request-admission ceiling at calibrated service pressure", evidence, tier="sensitivity",
        )
        add_ranked_candidate(
            ranked, catalog, "num_continuous_decode_steps", [2, 4],
            "sensitivity screen scheduler amortization while retaining tail-latency gates", evidence, tier="sensitivity",
        )
        graph_bs = 1 << math.ceil(math.log2(max(1, concurrency)))
        add_ranked_candidate(
            ranked, catalog, "cuda_graph_max_bs_decode", [graph_bs, graph_bs * 2],
            "sensitivity screen CUDA Graph decode coverage around calibrated concurrency", evidence, tier="sensitivity",
        )
        add_ranked_candidate(
            ranked, catalog, "page_size", [1, 16, 32],
            "sensitivity screen KV page granularity with full SLO regression checks", evidence, tier="sensitivity",
        )
        if workload["input_tokens"] >= 1024:
            add_ranked_candidate(
                ranked, catalog, "enable_mixed_chunk", [True],
                "sensitivity screen mixed prefill/decode batching for non-trivial prompts", evidence, tier="sensitivity",
            )

    dependent_bundles: list[dict[str, Any]] = []
    if discovery["model"].get("is_hybrid") and workload.get("prefix_reuse_ratio", 0) > 0:
        if "mamba_radix_cache_strategy" in catalog and "page_size" in catalog:
            dependent_bundles.append({
                "name": "hybrid-mamba-extra-buffer-page-64",
                "config": {"mamba_radix_cache_strategy": "extra_buffer", "page_size": 64},
                "reason": "the cookbook requires page_size=64 when testing Mamba extra_buffer cache strategy",
                "evidence": evidence + [
                    "cookbook.mamba_radix_cache_strategy=extra_buffer",
                    "cookbook.page_size=64",
                ],
            })

    if primary in {"host_or_scheduler_stall", "cpu_gpu_synchronization"} and not underdriven:
        add_ranked_candidate(
            ranked, catalog, "num_continuous_decode_steps", [2, 4],
            "amortize scheduler and launch overhead; TTFT remains a hard gate",
            evidence + [f"gpu_active_pct={diagnosis.get('gpu_timeline_active_pct')}"]
        )
        add_ranked_candidate(
            ranked, catalog, "scheduler_recv_interval", [2, 4],
            "reduce scheduler polling overhead when GPU timeline contains host-side gaps", evidence
        )
        if concurrency >= 16:
            add_ranked_candidate(
                ranked, catalog, "tokenizer_worker_num", [2, 4],
                "parallelize tokenizer work at high request concurrency", evidence
            )

    if primary == "attention" or "attention" in secondary or shares.get("attention_kernels", 0) >= 20:
        phase = "prefill_attention_backend" if workload["input_tokens"] >= workload["output_tokens"] else "decode_attention_backend"
        add_ranked_candidate(
            ranked, catalog, phase, backends["attention"],
            "compare installed, hardware-compatible attention implementations for the dominant phase",
            evidence + [f"attention_kernel_pct={shares.get('attention_kernels', 0):.3f}"]
        )

    if discovery["model"].get("is_moe") and (
        primary in {"moe_compute", "gemm_compute", "mixed_gpu_compute"}
        or "moe_compute" in secondary
        or shares.get("moe_kernels", 0) >= 15
    ):
        add_ranked_candidate(
            ranked, catalog, "moe_runner_backend", backends["moe"],
            "compare MoE runners because expert/GEMM kernels dominate or the model is MoE under mixed compute",
            evidence + [f"moe_kernel_pct={shares.get('moe_kernels', 0):.3f}"]
        )
        if discovery["derived"]["visible_gpu_count"] > 1 and discovery["derived"]["minimum_tp_size"] > 1:
            add_ranked_candidate(
                ranked, catalog, "enable_dp_attention", [True],
                "test data-parallel attention for a tensor-parallel MoE deployment; retain it only when trace and SLO gates improve",
                evidence + [f"topology={discovery['topology_class']}"]
            )

    if primary == "gemm_compute" and not discovery["model"].get("is_moe"):
        add_ranked_candidate(
            ranked, catalog, "bf16_gemm_backend", ["cutedsl"],
            "compare the installed BF16 GEMM backend on supported Blackwell hardware", evidence
        )

    if (primary == "communication" or "communication" in secondary) and discovery["derived"]["visible_gpu_count"] > 1:
        topology = discovery["topology_class"]
        add_ranked_candidate(
            ranked, catalog, "enable_mscclpp", [True],
            "test small-message collective optimization after nsys attributes material GPU time to communication",
            evidence + [f"topology={topology}", f"communication_pct={shares.get('communication_kernels', 0):.3f}"]
        )
        add_ranked_candidate(
            ranked, catalog, "disable_custom_all_reduce", [True],
            "measure NCCL fallback against SGLang custom all-reduce on the discovered topology", evidence
        )

    if primary == "memory_transfer" or "memory_transfer" in secondary:
        add_ranked_candidate(
            ranked, catalog, "page_size", [1, 16, 32, 64],
            "measure KV page granularity when memory operations are material", evidence
        )

    if (
        primary in {"host_or_scheduler_stall", "mixed_gpu_compute"}
        or "cuda_synchronization" in secondary
    ) and not decode_graph_active:
        graph_bs = max(1, 1 << math.ceil(math.log2(max(1, concurrency))))
        add_ranked_candidate(
            ranked, catalog, "cuda_graph_max_bs_decode", [graph_bs, graph_bs * 2],
            "match decode graph coverage to the observed serving concurrency", evidence
        )

    if workload.get("prefix_reuse_ratio", 0.0) >= 0.2:
        add_ranked_candidate(
            ranked, catalog, "schedule_policy", ["lpm"],
            "exploit declared prefix reuse with longest-prefix-match scheduling",
            [f"prefix_reuse_ratio={workload.get('prefix_reuse_ratio')}"]
        )
    elif workload.get("prefix_reuse_ratio", 0.0) == 0:
        add_ranked_candidate(
            ranked, catalog, "disable_radix_cache", [True],
            "measure radix-cache bookkeeping cost because the workload declares no prefix reuse",
            ["prefix_reuse_ratio=0"]
        )

    default_chunk = effective.get("chunked_prefill_size")
    chunks = chunk_candidates(task, default_chunk)
    nondefault_chunks = [value for value in chunks if value != default_chunk]
    if nondefault_chunks:
        add_ranked_candidate(
            ranked, catalog, "chunked_prefill_size", nondefault_chunks,
            "screen values below, at, and above the concurrent uncached-prefill boundary; retain the resolved SGLang default as the baseline",
            [
                f"expected_prefill_tokens_per_request={expected_prefill_tokens(workload)}",
                f"concurrent_prefill_boundary={boundary}",
                f"resolved_sglang_default={default_chunk}",
            ]
        )
    if queue_reqs is not None and queue_reqs > 0:
        if token_usage is not None and token_usage < 0.8:
            add_ranked_candidate(
                ranked, catalog, "schedule_conservativeness", [0.3, 0.6],
                "queue is nonempty while KV use is low, indicating conservative admission",
                evidence,
            )
            add_ranked_candidate(
                ranked, catalog, "max_running_requests", [concurrency * 2, concurrency * 4],
                "queue exists; test whether admission is constraining throughput", evidence,
            )
        elif retractions is not None and retractions > 0:
            add_ranked_candidate(
                ranked, catalog, "schedule_conservativeness", [1.1, 1.3],
                "retractions indicate KV pressure; reduce unsafe admission", evidence,
            )

    for item in ranked:
        original = list(item["values"])
        item["values"] = [
            value for value in original
            if (item["parameter"], json.dumps(value, sort_keys=True)) not in known_failures
        ]
        if len(item["values"]) != len(original):
            item["evidence"].append("excluded because the identical one-factor candidate failed in a prior local run")
    ranked = [item for item in ranked if item["values"]]
    family_coverage: dict[str, dict[str, Any]] = {}
    for metadata in catalog.values():
        family = metadata["family"]
        family_coverage.setdefault(family, {"available_parameters": 0, "selected_parameters": []})
        family_coverage[family]["available_parameters"] += 1
    for item in ranked:
        family_coverage[item["family"]]["selected_parameters"].append(item["parameter"])
    return {
        "schema_version": 4,
        "profiler_evidence": diagnosis,
        "ranked_parameter_groups": ranked,
        "cookbook_candidate_bundles": cookbook_bundles,
        "cookbook_bundle_exclusions": cookbook_bundle_exclusions,
        "policy": "one conceptual parameter per candidate; rank compatible parameter families using calibration, measured queue, KV, cache, CUDA-graph, topology, and kernel evidence",
        "deployment_mode": mode,
        "parameter_family_coverage": family_coverage,
        "parameter_audit": parameter_audit(catalog, ranked, discovery, task),
        "workload_assessment": {
            "underdriven": underdriven,
            "reason": (
                "queue is empty and KV use is below 50%; run a closed-loop load calibration before interpreting scheduler or memory tuning"
                if underdriven else "workload produced sufficient queue or KV pressure for parameter routing"
            ),
        },
        "excluded_prior_failures": [
            {"parameter": key, "value": json.loads(value)}
            for key, value in sorted(known_failures)
        ],
        "ranked_configuration_bundles": dependent_bundles,
        "resolved_baseline": {
            key: effective.get(key)
            for key in (
                "chunked_prefill_size", "mem_fraction_static", "max_running_requests",
                "max_total_tokens", "kv_cache_dtype", "moe_runner_backend",
                "cuda_graph_max_bs_decode", "cuda_graph_max_bs_prefill",
                "attention_backend", "prefill_attention_backend", "decode_attention_backend",
            )
            if key in effective
        },
    }


def profile_spec(
    task: dict[str, Any], discovery: dict[str, Any], baseline: dict[str, Any] | None = None
) -> dict[str, Any]:
    spec = build_execution_spec(
        task, discovery, stage_name="profile", baseline={
            "tp_size": discovery["derived"]["minimum_tp_size"],
            "enable_metrics": True, **(baseline or {}),
        },
        space={}, max_trials=1, repetitions=1,
        remaining_gpu_hours=float(task["budget"]["max_gpu_hours"]),
        remaining_wall_minutes=float(task["budget"]["max_wall_time_minutes"]),
    )
    spec["scope"]["allow_profiling"] = True
    profile_prompts = min(
        task["workload"]["num_prompts"],
        max(32, task["workload"]["max_concurrency"] * 2),
    )
    spec["benchmark"]["num_prompts"] = profile_prompts
    return spec


def profile_matches_task(profile: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    path = Path(str(profile.get("run_dir", ""))) / "spec.json"
    if not path.is_file():
        return ["profile artifact is missing its immutable spec.json"]
    observed = load_json(path)
    expected_pairs = {
        "repository": (observed.get("repository"), expected.get("repository")),
        "model.path": (observed.get("model", {}).get("path"), expected.get("model", {}).get("path")),
        "input_tokens": (observed.get("benchmark", {}).get("random_input_len"), expected.get("benchmark", {}).get("random_input_len")),
        "output_tokens": (observed.get("benchmark", {}).get("random_output_len"), expected.get("benchmark", {}).get("random_output_len")),
        "max_concurrency": (observed.get("benchmark", {}).get("max_concurrency"), expected.get("benchmark", {}).get("max_concurrency")),
    }
    return [
        f"profile does not match current {name}: {actual!r} != {wanted!r}"
        for name, (actual, wanted) in expected_pairs.items()
        if actual != wanted
    ]


def screening_spec(
    task: dict[str, Any], discovery: dict[str, Any], search_plan: dict[str, Any],
    remaining_gpu_hours: float | None = None, remaining_wall_minutes: float | None = None,
    remaining_trials: int | None = None, baseline: dict[str, Any] | None = None,
    confirmation_reserve_trials: int | None = None,
) -> dict[str, Any]:
    repetitions = task.get("confirmation_repetitions", 3)
    confirmation_reserve = (
        repetitions * 2 if confirmation_reserve_trials is None else confirmation_reserve_trials
    )
    total_trials = int(task["budget"]["max_trials"]) if remaining_trials is None else remaining_trials
    screening_trials = max(1, total_trials - confirmation_reserve)
    tp_size = discovery["derived"]["minimum_tp_size"]
    baseline_config = {"tp_size": tp_size, **(baseline or {})}
    candidate_budget = max(0, screening_trials - 1)
    bundles = [
        *search_plan.get("cookbook_candidate_bundles", []),
        *search_plan.get("ranked_configuration_bundles", []),
    ]
    valid_bundles = [
        bundle for bundle in bundles
        if isinstance(bundle, dict) and isinstance(bundle.get("config"), dict)
    ]
    one_factor_budget = max(0, candidate_budget - len(valid_bundles))
    selected: list[tuple[str, Any]] = []
    used_families: set[str] = set()
    ranked = search_plan["ranked_parameter_groups"]
    for item in ranked:
        if len(selected) >= one_factor_budget:
            break
        if item["family"] in used_families:
            continue
        selected.append((item["parameter"], item["values"][0]))
        used_families.add(item["family"])
    for item in ranked:
        for value in item["values"]:
            if len(selected) >= one_factor_budget:
                break
            pair = (item["parameter"], value)
            if pair not in selected:
                selected.append(pair)
        if len(selected) >= one_factor_budget:
            break
    if valid_bundles:
        configurations = [
            {"name": bundle["name"], "config": {**baseline_config, **bundle["config"]}}
            for bundle in valid_bundles[:candidate_budget]
        ]
        for parameter, value in selected:
            if len(configurations) >= candidate_budget:
                break
            configurations.append({
                "name": f"{parameter}-{str(value).lower()}"[:96],
                "config": {**baseline_config, parameter: value},
            })
        spec = explicit_configuration_spec(
            task, discovery,
            stage_name="screen",
            baseline=baseline_config,
            configurations=configurations,
            max_trials=1 + len(configurations),
            repetitions=1,
            remaining_gpu_hours=float(task["budget"]["max_gpu_hours"]) if remaining_gpu_hours is None else remaining_gpu_hours,
            remaining_wall_minutes=float(task["budget"]["max_wall_time_minutes"]) if remaining_wall_minutes is None else remaining_wall_minutes,
        )
        for bundle in valid_bundles:
            environment = bundle.get("env", {})
            if isinstance(environment, dict):
                spec["execution"]["env"].update(environment)
        return spec
    space: dict[str, list[Any]] = {}
    for parameter, value in selected:
        space.setdefault(parameter, []).append(value)
    candidate_count = sum(len(values) for values in space.values())
    max_trials = min(screening_trials, 1 + candidate_count)
    return build_execution_spec(
        task,
        discovery,
        stage_name="screen",
        baseline=baseline_config,
        space=space,
        max_trials=max_trials,
        repetitions=1,
        remaining_gpu_hours=float(task["budget"]["max_gpu_hours"]) if remaining_gpu_hours is None else remaining_gpu_hours,
        remaining_wall_minutes=float(task["budget"]["max_wall_time_minutes"]) if remaining_wall_minutes is None else remaining_wall_minutes,
    )


def initial_cookbook_trial_budget(task: dict[str, Any], completed_calibration_trials: int) -> int:
    """Limit exploratory Cookbook trials so profiling and real parameter tuning remain possible."""
    confirmation_reserve = int(task.get("confirmation_repetitions", 3)) * 2
    profile_reserve = 1
    # A baseline plus three independent parameter families is the minimum
    # evidence required to call the post-profile pass an optimization search.
    minimum_parameter_screen_reserve = 4
    return max(
        0,
        int(task["budget"]["max_trials"])
        - completed_calibration_trials
        - profile_reserve
        - minimum_parameter_screen_reserve
        - confirmation_reserve,
    )


def explicit_configuration_spec(
    task: dict[str, Any],
    discovery: dict[str, Any],
    *,
    stage_name: str,
    baseline: dict[str, Any],
    configurations: list[dict[str, Any]],
    max_trials: int,
    repetitions: int,
    remaining_gpu_hours: float,
    remaining_wall_minutes: float,
) -> dict[str, Any]:
    spec = build_execution_spec(
        task,
        discovery,
        stage_name=stage_name,
        baseline=baseline,
        space={},
        max_trials=max_trials,
        repetitions=repetitions,
        remaining_gpu_hours=remaining_gpu_hours,
        remaining_wall_minutes=remaining_wall_minutes,
    )
    spec["search"].update({
        "strategy": "explicit_configurations",
        "explicit_configurations": configurations,
        "parameter_order": [],
    })
    return spec


def fastest_slo_valid_configuration(result: dict[str, Any], objective: dict[str, Any]) -> dict[str, Any] | None:
    """Select a representative profile baseline without applying final decision gates.

    The pre-profile run is deliberately an exploratory coarse screen. It must
    profile the best observed SLO-valid configuration even when a one-sample
    comparison cannot yet clear the final minimum-improvement threshold.
    """
    metric = objective["metric"]
    maximize = objective["direction"] == "maximize"
    eligible = [
        item for item in result.get("aggregates", [])
        if item.get("all_repetitions_slo_passed")
        and isinstance(item.get("metrics", {}).get(metric), (int, float))
    ]
    if not eligible:
        return None
    return deepcopy(sorted(
        eligible,
        key=lambda item: float(item["metrics"][metric]),
        reverse=maximize,
    )[0]["config"])


def interaction_spec(
    task: dict[str, Any],
    discovery: dict[str, Any],
    screen: dict[str, Any],
    remaining_trials: int,
    remaining_gpu_hours: float,
    remaining_wall_minutes: float,
) -> dict[str, Any] | None:
    """Combine screened winners without a combinatorial parameter-product search."""
    baseline = deepcopy(screen["aggregates"][0]["config"])
    accepted = [
        item for item in screen["aggregates"][1:]
        if item.get("screening_accepted") and item.get("comparison", {}).get("improvement_pct") is not None
    ]
    accepted.sort(key=lambda item: item["comparison"]["improvement_pct"], reverse=True)
    if len(accepted) < 2:
        return None
    repetitions = int(task.get("confirmation_repetitions", 3))
    confirmation_reserve = repetitions * 2
    candidate_slots = max(0, remaining_trials - confirmation_reserve - 1)
    primary = accepted[0]
    primary_changes = {
        key: value for key, value in primary["config"].items()
        if baseline.get(key) != value
    }
    configurations: list[dict[str, Any]] = []
    for contender in accepted[1:]:
        contender_changes = {
            key: value for key, value in contender["config"].items()
            if baseline.get(key) != value
        }
        if not contender_changes or set(contender_changes) & set(primary_changes):
            continue
        combined = deepcopy(baseline)
        combined.update(primary_changes)
        combined.update(contender_changes)
        configurations.append({
            "name": f"combine-{primary['configuration_name']}-and-{contender['configuration_name']}"[:96],
            "config": combined,
        })
        if len(configurations) >= candidate_slots:
            break
    if not configurations:
        return None
    return explicit_configuration_spec(
        task, discovery,
        stage_name="interact",
        baseline=baseline,
        configurations=configurations,
        max_trials=1 + len(configurations),
        repetitions=1,
        remaining_gpu_hours=remaining_gpu_hours,
        remaining_wall_minutes=remaining_wall_minutes,
    )


def confirmation_spec(
    task: dict[str, Any],
    discovery: dict[str, Any],
    screen: dict[str, Any],
    remaining_trials: int,
    remaining_gpu_hours: float,
    remaining_wall_minutes: float,
) -> dict[str, Any]:
    repetitions = int(task.get("confirmation_repetitions", 3))
    baseline = deepcopy(screen["aggregates"][0]["config"])
    winner = screen.get("screening_winner")
    configurations: list[dict[str, Any]] = []
    if winner is not None:
        candidate_config = winner["config"]
        if candidate_config != baseline:
            configurations.append({"name": "selected-candidate", "config": candidate_config})
    required = repetitions * (2 if configurations else 1)
    if remaining_trials < required:
        raise ValueError(f"insufficient remaining trial budget for confirmation: need {required}, have {remaining_trials}")
    if remaining_gpu_hours <= 0:
        raise ValueError("GPU-hour budget exhausted before confirmation")
    if remaining_wall_minutes <= 0:
        raise ValueError("wall-time budget exhausted before confirmation")
    return explicit_configuration_spec(
        task, discovery,
        stage_name="confirm",
        baseline=baseline,
        configurations=configurations or [{"name": "baseline-repeat", "config": baseline}],
        max_trials=required,
        repetitions=repetitions,
        remaining_gpu_hours=remaining_gpu_hours,
        remaining_wall_minutes=remaining_wall_minutes,
    )


def bottleneck_summary(screen: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    aggregates = screen.get("aggregates", [])
    if not aggregates:
        return {
            "classification": "screening_unavailable",
            "typical_prefill_batch_tokens": expected_prefill_tokens(task["workload"]) * task["workload"]["max_concurrency"],
            "baseline_metrics": {},
            "chunked_prefill_evidence": [],
            "explanation": "parameter-screening baseline did not complete; inspect the recorded trial failure before interpreting bottlenecks",
        }
    baseline = aggregates[0]
    candidates = screen["aggregates"][1:]
    chunk_rows = [row for row in candidates if row["configuration_name"].startswith("chunked_prefill_size-")]
    boundary = expected_prefill_tokens(task["workload"]) * task["workload"]["max_concurrency"]
    evidence = []
    for row in chunk_rows:
        value = row["config"].get("chunked_prefill_size")
        improvement = row.get("comparison", {}).get("improvement_pct")
        evidence.append({"chunked_prefill_size": value, "improvement_pct": improvement})
    below = [item for item in evidence if item["chunked_prefill_size"] < boundary]
    at_or_above = [item for item in evidence if item["chunked_prefill_size"] >= boundary]
    fragmentation = bool(
        below
        and at_or_above
        and min((item["improvement_pct"] or 0) for item in below) < -3
        and max((item["improvement_pct"] or 0) for item in at_or_above) > -2
    )
    return {
        "classification": "excessive_prefill_fragmentation" if fragmentation else "no_single_dominant_bottleneck",
        "typical_prefill_batch_tokens": boundary,
        "baseline_metrics": baseline["metrics"],
        "chunked_prefill_evidence": evidence,
        "explanation": (
            "chunk sizes below the typical concurrent prefill batch boundary regress throughput; keep the chunk at or above the boundary"
            if fragmentation
            else "screening did not isolate a dominant bottleneck; preserve the confirmed recommendation and profile only if SLO or optimization targets remain unmet"
        ),
    }


def operator_escalation_plan(profile: dict[str, Any]) -> dict[str, Any]:
    """Bound kernel work to a trace-proven hotspot and state its possible payoff."""
    diagnosis = profile["diagnosis"]
    top = diagnosis.get("top_kernels", [])
    if not top:
        return {"required": False, "reason": "nsys did not identify a CUDA kernel"}
    kernel = top[0]
    kernel_gpu_share = float(kernel.get("time_pct") or 0) / 100
    if kernel_gpu_share < 0.25:
        return {
            "required": False,
            "reason": "no single CUDA kernel accounts for at least 25% of GPU-active time",
            "top_kernel": kernel,
        }
    # Amdahl's law applies only to the measured GPU execution slice here.
    gpu_execution_upper_bound = (1 / (1 - kernel_gpu_share + kernel_gpu_share / 2) - 1) * 100
    ncu = profile.get("tool", {}).get("ncu", {})
    return {
        "required": True,
        "top_kernel": kernel,
        "evidence": {
            "kernel_share_of_gpu_active_pct": round(kernel_gpu_share * 100, 3),
            "scope": "CUDA kernel execution time only; excludes model loading and must not be converted to end-to-end gain without request-level attribution",
        },
        "two_x_kernel_speedup_gpu_execution_upper_bound_pct": round(gpu_execution_upper_bound, 3),
        "end_to_end_upper_bound_pct": None,
        "next_step": (
            "run a shape-matched Nsight Compute capture and the corresponding SGLang kernel microbenchmark before proposing a code change"
            if ncu.get("performance_counter_access") else
            "Nsight Compute is installed but GPU counters are not accessible. Enable driver-level profiler permissions, then run a shape-matched microbenchmark; do not auto-edit a kernel."
        ),
        "ncu": ncu,
    }


def final_server_command(spec: dict[str, Any], recommendation: dict[str, Any]) -> list[str]:
    trial = {"config": recommendation["config"], "name": "recommended"}
    placeholder = Path(spec["scope"]["output_dir"]) / "DEPLOYMENT_COMMAND"
    return command_manifest(spec, trial, placeholder)["server"]


def build_plan(task: dict[str, Any]) -> dict[str, Any]:
    errors = validate_task(task)
    if errors:
        raise ValueError("; ".join(errors))
    discovery = discover(task)
    baseline_profile_spec = profile_spec(task, discovery)
    errors = execution_errors(baseline_profile_spec)
    if errors:
        raise ValueError("generated profiling spec is invalid: " + "; ".join(errors))
    initial_plan = cookbook_initial_search_plan(task, discovery)
    return {
        "schema_version": 4,
        "execution_enabled": False,
        "discovery": discovery,
        "deployment_policy": deployment_policy(task),
        "knowledge_preflight": {
            "order": "hardware and model inventory -> official cookbook and hardware references -> local CLI/checkpoint compatibility -> initial bundle benchmark",
            "cookbook": discovery.get("cookbook"),
            "hardware_reference_urls": (task.get("knowledge") or {}).get("hardware_reference_urls", []),
        },
        "calibration": {
            "required": (task.get("calibration") or {}).get("enabled", True),
            "target_workload": deepcopy(task["workload"]),
            "concurrency_points": calibration_concurrencies(task),
            "policy": "calibrate capacity only for diagnosis; final recommendation is always revalidated against target_workload",
        },
        "profiling": {
            "required": True,
            "tool": "nsight_systems" if discovery["hardware"]["vendor"] == "nvidia" else "pytorch_or_rpd",
            "capture_policy": "warm up first, then capture benchmark only with framework profiling control",
            "spec": baseline_profile_spec,
        },
        "search": {
            "initial_bundles_before_profiling": True,
            "generated_after_profiling": True,
            "parameter_catalog_count": discovery["parameter_catalog"]["parameter_count"],
            "initial_plan": initial_plan,
            "policy": "benchmark cookbook-derived, locally compatible startup bundles first; profile the best SLO-valid initial configuration at calibrated analysis load; then screen profiler-driven deltas and revalidate on the target workload",
        },
        "confirmation": {
            "automatic": True,
            "repetitions": task.get("confirmation_repetitions", 3),
        "policy": "confirm the best screened or interaction candidate against baseline; confirm baseline alone when no candidate clears screening gates",
        },
    }


def run_autopilot(task: dict[str, Any]) -> dict[str, Any]:
    progress = ProgressReporter()
    progress.emit("setup", "validating task, hardware, model, and installed SGLang parameters")
    plan = build_plan(task)
    root = Path(task["output_dir"]).expanduser() / (
        f"{task['name']}-autopilot-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    root.mkdir(parents=True, exist_ok=False)
    os.chmod(root, 0o700)
    write_json(root / "task.json", task)
    write_json(root / "plan.json", plan)
    progress.emit("setup", f"ready; artifacts: {root}")
    started = time.monotonic()
    if plan["discovery"]["hardware"]["vendor"] != "nvidia":
        raise RuntimeError("automatic AMD profiling requires the RPD/PyTorch executor, which is not yet implemented")
    progress.emit("capacity", "measuring the baseline SLO-safe concurrency curve")
    calibration = run_calibration(task, plan["discovery"], root, progress)
    write_json(root / "calibration.json", calibration)
    if not any(point.get("valid_for_analysis") for point in calibration["points"]):
        progress.emit("capacity", "no valid baseline point; stopping before Cookbook, profiling, and parameter search")
        raise RuntimeError(
            "baseline capacity calibration failed; inspect " + str(root / "calibration.json")
        )
    gpu_count = plan["discovery"]["derived"]["visible_gpu_count"]
    used_trials_before_profile = calibration["completed_trials"]
    used_gpu_hours_before_profile = calibration["approx_gpu_hours"]
    cookbook_initial = cookbook_initial_search_plan(task, plan["discovery"])
    write_json(root / "cookbook-initial-plan.json", cookbook_initial)
    execution_task = deepcopy(task)
    execution_task["env"] = {
        **task.get("env", {}),
        **cookbook_initial.get("required_environment", {}),
    }
    initial_screen: dict[str, Any] | None = None
    initial_candidate = {"tp_size": plan["discovery"]["derived"]["minimum_tp_size"]}
    if cookbook_initial["cookbook_candidate_bundles"]:
        cookbook_trial_budget = initial_cookbook_trial_budget(task, used_trials_before_profile)
        cookbook_initial["allocated_trial_budget"] = cookbook_trial_budget
        if cookbook_trial_budget < 2:
            cookbook_initial["deferred_reason"] = (
                "insufficient trial budget after reserving Nsight profiling, three parameter candidates, "
                "and confirmation"
            )
            write_json(root / "cookbook-initial-plan.json", cookbook_initial)
        else:
            progress.emit(
                "cookbook",
                f"screening Cookbook bundles with {cookbook_trial_budget} trials reserved for this exploratory stage",
            )
        elapsed_minutes = (time.monotonic() - started) / 60
        if cookbook_trial_budget >= 2:
            initial_spec = screening_spec(
                execution_task, plan["discovery"], cookbook_initial,
                remaining_gpu_hours=float(task["budget"]["max_gpu_hours"]) - used_gpu_hours_before_profile,
                remaining_wall_minutes=float(task["budget"]["max_wall_time_minutes"]) - elapsed_minutes,
                remaining_trials=cookbook_trial_budget,
                confirmation_reserve_trials=0,
            )
            errors = execution_errors(initial_spec)
            if errors:
                raise ValueError("generated cookbook initial spec is invalid: " + "; ".join(errors))
            write_json(root / "cookbook-initial-spec.json", initial_spec)
            initial_screen = execute_with_progress(initial_spec, progress, "cookbook screening")
            write_json(root / "cookbook-initial.json", initial_screen)
            used_trials_before_profile += initial_screen["completed_trials"]
            used_gpu_hours_before_profile += initial_screen["approx_gpu_hours"]
            initial_candidate = fastest_slo_valid_configuration(initial_screen, execution_task["objective"])
            if initial_candidate is None:
                initial_baseline = next(
                    (item for item in initial_screen.get("aggregates", []) if item.get("kind") == "baseline"), None
                )
                if initial_baseline is not None:
                    initial_candidate = deepcopy(initial_baseline["config"])
    analysis_task = deepcopy(execution_task)
    analysis_task["workload"]["max_concurrency"] = calibration["selected_analysis_concurrency"]
    analysis_profile_spec = profile_spec(analysis_task, plan["discovery"], initial_candidate)
    write_json(root / "analysis-profile-spec.json", analysis_profile_spec)
    reused_profile = (
        task.get("profile_dir") is not None
        and calibration["selected_analysis_concurrency"] == task["workload"]["max_concurrency"]
        and initial_screen is None
    )
    if reused_profile:
        progress.emit("nsys", "reusing the requested compatible Nsight Systems profile")
        profiling = diagnose_existing(Path(task["profile_dir"]).expanduser())
        mismatches = profile_matches_task(profiling, analysis_profile_spec)
        if mismatches:
            raise RuntimeError("cannot reuse profile: " + "; ".join(mismatches))
    else:
        progress.emit("nsys", "capturing and analyzing a bounded serving-only Nsight Systems trace")
        profiling = run_profile(analysis_profile_spec, root / "profile")
    write_json(root / "nsys-diagnosis.json", profiling)
    if profiling["status"].get("state") != "completed":
        raise RuntimeError("required baseline profiling did not complete")
    if not profiling["diagnosis"].get("top_kernels"):
        raise RuntimeError("required nsys trace contains no parsed CUDA kernels")
    search_plan = diagnosed_search_plan(analysis_task, plan["discovery"], profiling)
    # Cookbook bundles were already measured before profiling. The second pass
    # is reserved for profiler-driven deltas and must not re-run that stage.
    search_plan["cookbook_candidate_bundles"] = []
    write_json(root / "search-plan.json", search_plan)
    progress.emit("search", "screening profiler- and workload-selected parameter changes")
    elapsed_minutes = (time.monotonic() - started) / 60
    profile_gpu_hours = 0.0 if reused_profile else profiling["elapsed_sec"] / 3600 * gpu_count
    used_before_screen = used_gpu_hours_before_profile + profile_gpu_hours
    remaining_gpu_hours = float(task["budget"]["max_gpu_hours"]) - used_before_screen
    remaining_wall_minutes = float(task["budget"]["max_wall_time_minutes"]) - elapsed_minutes
    if remaining_gpu_hours <= 0 or remaining_wall_minutes <= 0:
        raise RuntimeError("profiling exhausted the experiment budget before parameter screening")
    screen_spec = screening_spec(
        execution_task, plan["discovery"], search_plan,
        remaining_gpu_hours=remaining_gpu_hours,
        remaining_wall_minutes=remaining_wall_minutes,
        remaining_trials=int(task["budget"]["max_trials"]) - used_trials_before_profile - (0 if reused_profile else 1),
        baseline=initial_candidate,
    )
    errors = execution_errors(screen_spec)
    if errors:
        raise ValueError("generated screening spec is invalid: " + "; ".join(errors))
    write_json(root / "screening-spec.json", screen_spec)
    screen = execute_with_progress(screen_spec, progress, "parameter screening")
    write_json(root / "screening.json", screen)
    used_trials = screen["completed_trials"]
    used_gpu_hours = used_before_screen + screen["approx_gpu_hours"]
    elapsed_minutes = (time.monotonic() - started) / 60
    remaining_trials = int(task["budget"]["max_trials"]) - used_trials_before_profile - (0 if reused_profile else 1) - used_trials
    remaining_gpu_hours = float(task["budget"]["max_gpu_hours"]) - used_gpu_hours
    remaining_wall_minutes = float(task["budget"]["max_wall_time_minutes"]) - elapsed_minutes
    interaction: dict[str, Any] | None = None
    interaction_error: str | None = None
    interaction_plan: dict[str, Any] | None = None
    attempted_parameter_candidates = [
        row for row in screen.get("results", []) if row.get("kind") == "candidate"
    ]
    executed_parameter_candidates = [row for row in attempted_parameter_candidates if row.get("ok")]
    parameter_search = {
        "planned_trials": screen.get("planned_trials", 0),
        "executed_trials": screen.get("completed_trials", 0),
        "attempted_parameter_candidates": len(attempted_parameter_candidates),
        "executed_parameter_candidates": len(executed_parameter_candidates),
        "failed_parameter_candidates": len(attempted_parameter_candidates) - len(executed_parameter_candidates),
        "sufficient_evidence": bool(executed_parameter_candidates),
    }
    if not executed_parameter_candidates:
        interaction_error = (
            "no deployment parameter candidate completed with comparable benchmark evidence"
        )
    elif screen["stop_reason"] == "completed_search":
        try:
            interaction_plan = interaction_spec(
                execution_task, plan["discovery"], screen,
                remaining_trials, remaining_gpu_hours, remaining_wall_minutes,
            )
            if interaction_plan is not None:
                progress.emit("interaction", "testing compatible combinations of accepted parameter changes")
                errors = execution_errors(interaction_plan)
                if errors:
                    raise ValueError("generated interaction spec is invalid: " + "; ".join(errors))
                write_json(root / "interaction-spec.json", interaction_plan)
                interaction = execute_with_progress(interaction_plan, progress, "interaction screening")
                write_json(root / "interaction.json", interaction)
                used_trials += interaction["completed_trials"]
                used_gpu_hours += interaction["approx_gpu_hours"]
                elapsed_minutes = (time.monotonic() - started) / 60
                remaining_trials -= interaction["completed_trials"]
                remaining_gpu_hours = float(task["budget"]["max_gpu_hours"]) - used_gpu_hours
                remaining_wall_minutes = float(task["budget"]["max_wall_time_minutes"]) - elapsed_minutes
        except (ValueError, RuntimeError) as exc:
            interaction_error = str(exc)
    confirmation: dict[str, Any] | None = None
    confirmation_error: str | None = None
    decision_input = interaction or screen
    if executed_parameter_candidates and decision_input["stop_reason"] == "completed_search":
        try:
            confirm_spec = confirmation_spec(
                execution_task,
                plan["discovery"],
                decision_input,
                remaining_trials,
                remaining_gpu_hours,
                remaining_wall_minutes,
            )
            write_json(root / "confirmation-spec.json", confirm_spec)
            progress.emit("confirmation", "repeating baseline and selected candidate to reject measurement noise")
            confirmation = execute_with_progress(confirm_spec, progress, "confirmation")
            write_json(root / "confirmation.json", confirmation)
        except (ValueError, RuntimeError) as exc:
            confirmation_error = str(exc)
    if executed_parameter_candidates:
        decision = confirmation or decision_input
    else:
        decision = {
            **decision_input,
            "recommended_configuration": None,
            "recommendation_status": "insufficient_parameter_evidence",
            "recommendation_reason": (
                "no deployment parameter candidate completed; inspect failed candidates or increase the trial budget"
            ),
        }
    recommendation = decision.get("recommended_configuration")
    deploy_command = final_server_command(
        (load_json(root / "confirmation-spec.json") if confirmation else screen_spec), recommendation
    ) if recommendation is not None else None
    final = {
        "schema_version": 3,
        "run_dir": str(root),
        "completed_at": utc_now(),
        "elapsed_sec": time.monotonic() - started,
        "discovery": plan["discovery"],
        "deployment_policy": plan["deployment_policy"],
        "calibration": calibration,
        "cookbook_initial_screen": initial_screen,
        "profiled_initial_configuration": initial_candidate,
        "analysis_workload": analysis_task["workload"],
        "profiling": profiling,
        "profiling_reused": reused_profile,
        "search_plan": search_plan,
        "parameter_search": parameter_search,
        "screening": screen,
        "interaction": interaction,
        "interaction_error": interaction_error,
        "confirmation": confirmation,
        "confirmation_error": confirmation_error,
        "bottleneck": {
            "nsys": profiling["diagnosis"],
            "screening_mechanism": bottleneck_summary(screen, execution_task),
            "operator_escalation": operator_escalation_plan(profiling),
        },
        "recommendation_status": decision.get("recommendation_status"),
        "recommendation_reason": decision.get("recommendation_reason"),
        "recommended_configuration": recommendation,
        "deployment_command": deploy_command,
        "deployable": recommendation is not None and decision.get("recommendation_status") in {
            "confirmed_candidate", "retain_confirmed_baseline"
        },
        "total_completed_trials": used_trials_before_profile + (0 if reused_profile else 1) + used_trials + (confirmation.get("completed_trials", 0) if confirmation else 0),
        "total_approx_gpu_hours": used_gpu_hours + (confirmation.get("approx_gpu_hours", 0) if confirmation else 0),
    }
    write_json(root / "final.json", final)
    progress.emit(
        "complete",
        f"{final['recommendation_status']}; completed {final['total_completed_trials']} trials. "
        f"Result: {root / 'final.json'}",
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--task", required=True)
    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--task", required=True)
    discover_parser.add_argument("--output")
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--task", required=True)
    plan_parser.add_argument("--output")
    search_parser = subparsers.add_parser("search-plan")
    search_parser.add_argument("--task", required=True)
    search_parser.add_argument("--profile", required=True)
    search_parser.add_argument("--output")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--task", required=True)
    run_parser.add_argument("--yes", action="store_true")
    run_parser.add_argument("--output")
    args = parser.parse_args()
    try:
        task = load_json(args.task)
        if args.command == "validate":
            errors = validate_task(task)
            dump_json({"valid": not errors, "errors": errors}, None)
            return 0 if not errors else 2
        if args.command == "discover":
            errors = validate_task(task)
            if errors:
                raise ValueError("; ".join(errors))
            dump_json(discover(task), args.output)
            return 0
        if args.command == "plan":
            dump_json(build_plan(task), args.output)
            return 0
        if args.command == "search-plan":
            errors = validate_task(task)
            if errors:
                raise ValueError("; ".join(errors))
            profile = load_json(args.profile)
            if not isinstance(profile.get("diagnosis"), dict):
                raise ValueError("profile must contain a completed nsys diagnosis")
            dump_json(diagnosed_search_plan(task, discover(task), profile), args.output)
            return 0
        if args.command == "run":
            if not args.yes:
                raise ValueError("run requires --yes after reviewing the one-shot plan")
            dump_json(run_autopilot(task), args.output)
            return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
