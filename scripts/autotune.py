#!/usr/bin/env python3
"""Run bounded, local SGLang parameter trials from a structured specification."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable

from sglang_runtime import summarize_sglang_log

from inferopt import compare, dump_json, inventory, load_json, slo_results, summarize, validate_spec


VALUE_FLAGS: dict[str, tuple[str, type, float | None, float | None]] = {
    "tp_size": ("--tp-size", int, 1, 1024),
    "dp_size": ("--dp-size", int, 1, 1024),
    "pp_size": ("--pp-size", int, 1, 1024),
    "ep_size": ("--ep-size", int, 1, 1024),
    "moe_dp_size": ("--moe-dp-size", int, 1, 1024),
    "mem_fraction_static": ("--mem-fraction-static", float, 0.05, 0.99),
    "max_running_requests": ("--max-running-requests", int, 1, 1048576),
    "max_total_tokens": ("--max-total-tokens", int, 1, None),
    "chunked_prefill_size": ("--chunked-prefill-size", int, -1, None),
    "max_prefill_tokens": ("--max-prefill-tokens", int, 1, None),
    "prefill_max_requests": ("--prefill-max-requests", int, 1, None),
    "num_continuous_decode_steps": ("--num-continuous-decode-steps", int, 1, 64),
    "scheduler_recv_interval": ("--scheduler-recv-interval", int, 1, 64),
    "tokenizer_worker_num": ("--tokenizer-worker-num", int, 1, 256),
    "detokenizer_worker_num": ("--detokenizer-worker-num", int, 1, 256),
    "schedule_conservativeness": ("--schedule-conservativeness", float, 0.0, 100.0),
    "page_size": ("--page-size", int, 1, 256),
    "cuda_graph_max_bs_decode": ("--cuda-graph-max-bs-decode", int, 1, 4096),
    "cuda_graph_max_bs_prefill": ("--cuda-graph-max-bs-prefill", int, 1, 1048576),
    "cuda_graph_backend_decode": ("--cuda-graph-backend-decode", str, None, None),
    "cuda_graph_backend_prefill": ("--cuda-graph-backend-prefill", str, None, None),
    "cuda_graph_tc_compiler": ("--cuda-graph-tc-compiler", str, None, None),
    "attention_backend": ("--attention-backend", str, None, None),
    "prefill_attention_backend": ("--prefill-attention-backend", str, None, None),
    "decode_attention_backend": ("--decode-attention-backend", str, None, None),
    "sampling_backend": ("--sampling-backend", str, None, None),
    "bf16_gemm_backend": ("--bf16-gemm-backend", str, None, None),
    "fp8_gemm_runner_backend": ("--fp8-gemm-backend", str, None, None),
    "fp4_gemm_runner_backend": ("--fp4-gemm-backend", str, None, None),
    "moe_runner_backend": ("--moe-runner-backend", str, None, None),
    "moe_a2a_backend": ("--moe-a2a-backend", str, None, None),
    "deepep_mode": ("--deepep-mode", str, None, None),
    "load_balance_method": ("--load-balance-method", str, None, None),
    "schedule_policy": ("--schedule-policy", str, None, None),
    "kv_cache_dtype": ("--kv-cache-dtype", str, None, None),
    "speculative_algorithm": ("--speculative-algorithm", str, None, None),
    "speculative_draft_model_path": ("--speculative-draft-model-path", str, None, None),
    "speculative_num_steps": ("--speculative-num-steps", int, 1, 16),
    "speculative_eagle_topk": ("--speculative-eagle-topk", int, 1, 16),
    "speculative_num_draft_tokens": ("--speculative-num-draft-tokens", int, 1, 64),
    "mamba_radix_cache_strategy": ("--mamba-radix-cache-strategy", str, None, None),
}

BOOL_FLAGS = {
    "enable_metrics": "--enable-metrics",
    "disable_cuda_graph": "--disable-cuda-graph",
    "disable_decode_cuda_graph": "--disable-decode-cuda-graph",
    "disable_prefill_cuda_graph": "--disable-prefill-cuda-graph",
    "disable_cuda_graph_padding": "--disable-cuda-graph-padding",
    "disable_radix_cache": "--disable-radix-cache",
    "disable_overlap_schedule": "--disable-overlap-schedule",
    "enable_mixed_chunk": "--enable-mixed-chunk",
    "enable_dp_attention": "--enable-dp-attention",
    "enable_two_batch_overlap": "--enable-two-batch-overlap",
    "enable_single_batch_overlap": "--enable-single-batch-overlap",
    "enable_mscclpp": "--enable-mscclpp",
    "enable_torch_symm_mem": "--enable-torch-symm-mem",
    "enable_nccl_nvls": "--enable-nccl-nvls",
    "pre_warm_nccl": "--pre-warm-nccl",
    "disable_custom_all_reduce": "--disable-custom-all-reduce",
    "enable_torch_compile": "--enable-torch-compile",
}

BENCHMARK_KEYS = {
    "dataset_name",
    "dataset_path",
    "num_prompts",
    "random_input_len",
    "random_output_len",
    "random_range_ratio",
    "request_rate",
    "max_concurrency",
    "warmup_requests",
    "min_measurement_seconds",
    "seed",
    "output_details",
    "gsp_num_groups",
    "gsp_prompts_per_group",
    "gsp_system_prompt_len",
    "gsp_question_len",
    "gsp_output_len",
    "gsp_range_ratio",
    "gsp_ordered",
}

SUPPORTED_DATASETS = {"random", "random-ids", "custom", "sharegpt", "generated-shared-prefix"}
SYNTHETIC_RANDOM_DATASETS = {"random", "random-ids"}
SEARCH_KEYS = {
    "strategy",
    "baseline",
    "space",
    "repetitions",
    "order",
    "max_cv_pct",
    "min_confirm_repetitions",
    "require_all_slo_pass",
    "parameter_order",
    "explicit_configurations",
}

ALLOWED_ENV = {
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "NCCL_DEBUG",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "SGLANG_USE_AITER",
    "SGLANG_TORCH_PROFILER_DIR",
    "SGLANG_ENABLE_SPEC_V2",
    "TOKENIZERS_PARALLELISM",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def enable_child_subreaper() -> bool:
    if sys.platform != "linux":
        return False
    try:
        # SGLang launches worker grandchildren. Becoming a subreaper ensures
        # they return here instead of accumulating under a non-reaping PID 1.
        return ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    except (AttributeError, OSError):
        return False


def reap_exited_children(timeout: float = 2.0) -> list[int]:
    reaped: list[int] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = False
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return reaped
            if pid == 0:
                break
            found = True
            reaped.append(pid)
        if not found:
            time.sleep(0.05)
    return reaped


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def catalog_binding(name: str, bindings: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the recorded current-SGLang binding for a parameter."""
    if not isinstance(bindings, dict):
        return None
    binding = bindings.get(name)
    return binding if isinstance(binding, dict) else None


def validate_parameter(name: str, value: Any, bindings: dict[str, Any] | None = None) -> str | None:
    binding = catalog_binding(name, bindings)
    if binding is not None:
        action = binding.get("action")
        if action is None and name in BOOL_FLAGS:
            return None if isinstance(value, bool) else f"{name} must be boolean"
        if action in {"store_true", "store_false"}:
            return None if isinstance(value, bool) else f"{name} must be boolean"
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            return f"{name} must be a scalar value"
        choices = binding.get("choices")
        if isinstance(choices, list) and value not in choices:
            return f"{name} must be one of {choices}"
        value_type = binding.get("value_type")
        if value_type == "int" and not isinstance(value, int):
            return f"{name} must be integer"
        if value_type == "float" and not isinstance(value, (int, float)):
            return f"{name} must be numeric"
        return None
    if isinstance(bindings, dict):
        return f"unsupported current-SGLang server parameter: {name}"
    if name in BOOL_FLAGS:
        return None if isinstance(value, bool) else f"{name} must be boolean"
    rule = VALUE_FLAGS.get(name)
    if not rule:
        return f"unsupported server parameter: {name}"
    _, expected, minimum, maximum = rule
    if expected is int and (not isinstance(value, int) or isinstance(value, bool)):
        return f"{name} must be integer"
    if expected is float and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        return f"{name} must be numeric"
    if expected is str and (not isinstance(value, str) or not value or value.startswith("-")):
        return f"{name} must be a non-empty value"
    if isinstance(value, (int, float)) and minimum is not None and value < minimum:
        return f"{name} must be >= {minimum}"
    if isinstance(value, (int, float)) and maximum is not None and value > maximum:
        return f"{name} must be <= {maximum}"
    return None


def execution_errors(spec: dict[str, Any]) -> list[str]:
    errors = validate_spec(spec)
    execution = spec.get("execution")
    search = spec.get("search")
    benchmark = spec.get("benchmark")
    name = spec.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
        errors.append("name must be a safe 1-64 character directory name")
    if not isinstance(execution, dict):
        errors.append("execution must be an object")
        execution = {}
    bindings = execution.get("parameter_bindings")
    if bindings is not None and not isinstance(bindings, dict):
        errors.append("execution.parameter_bindings must be an object")
        bindings = None
    if not isinstance(search, dict):
        errors.append("search must be an object")
        search = {}
    if not isinstance(benchmark, dict):
        errors.append("benchmark must be an object")
        benchmark = {}
    if spec.get("mode") != "execute":
        errors.append("autotune run requires mode=execute")
    scope = spec.get("scope", {})
    if not scope.get("allow_launch"):
        errors.append("scope.allow_launch must be true")
    if not scope.get("allow_parameter_changes"):
        errors.append("scope.allow_parameter_changes must be true")
    if scope.get("production"):
        errors.append("autotune does not run against production")
    repository = Path(str(spec.get("repository", ""))).expanduser()
    if not repository.is_absolute() or not repository.is_dir():
        errors.append("repository must be an existing absolute directory")
    output_dir = Path(str(spec.get("scope", {}).get("output_dir", ""))).expanduser()
    if not output_dir.is_absolute() or output_dir == Path("/"):
        errors.append("scope.output_dir must be an absolute non-root directory")
    python = execution.get("python", sys.executable)
    if not isinstance(python, str) or not Path(python).expanduser().is_absolute():
        errors.append("execution.python must be an absolute path")
    elif not Path(python).expanduser().is_file():
        errors.append("execution.python must be an existing file")
    host = execution.get("host", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost"}:
        errors.append("execution.host must be loopback for this single-host executor")
    port = execution.get("port", 30000)
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        errors.append("execution.port must be an integer between 1024 and 65535")
    for key in ("startup_timeout_sec", "benchmark_timeout_sec", "shutdown_timeout_sec"):
        value = execution.get(key, {"startup_timeout_sec": 900, "benchmark_timeout_sec": 1800, "shutdown_timeout_sec": 30}[key])
        if not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"execution.{key} must be positive")
    environment = execution.get("env", {})
    if not isinstance(environment, dict):
        errors.append("execution.env must be an object")
    else:
        for key, value in environment.items():
            if key not in ALLOWED_ENV:
                errors.append(f"execution.env key is not allowed: {key}")
            if not isinstance(value, (str, int, float, bool)):
                errors.append(f"execution.env.{key} must be scalar")
    if execution.get("offline", True) is False and not scope.get("allow_download", False):
        errors.append("execution.offline=false requires scope.allow_download=true")
    baseline = search.get("baseline", {})
    space = search.get("space", {})
    for key in sorted(set(search) - SEARCH_KEYS):
        errors.append(f"unsupported search field: {key}")
    strategy = search.get("strategy", "one_factor")
    if strategy not in {"one_factor", "explicit_configurations"}:
        errors.append("search.strategy must be one_factor or explicit_configurations")
    repetitions = search.get("repetitions", 1)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or not 1 <= repetitions <= 9:
        errors.append("search.repetitions must be an integer from 1 through 9")
        repetitions = 1
    if search.get("order", "interleaved") != "interleaved":
        errors.append("search.order currently supports only interleaved")
    parameter_order = search.get("parameter_order")
    if parameter_order is None:
        parameter_order = sorted(search.get("space", {})) if isinstance(search.get("space", {}), dict) else []
    if not isinstance(parameter_order, list) or any(not isinstance(item, str) for item in parameter_order):
        errors.append("search.parameter_order must be an array of parameter names")
        parameter_order = []
    max_cv_pct = search.get("max_cv_pct", 10.0)
    if not isinstance(max_cv_pct, (int, float)) or isinstance(max_cv_pct, bool) or not 0 <= max_cv_pct <= 100:
        errors.append("search.max_cv_pct must be between 0 and 100")
    min_confirm_repetitions = search.get("min_confirm_repetitions", 3)
    if (
        not isinstance(min_confirm_repetitions, int)
        or isinstance(min_confirm_repetitions, bool)
        or not 2 <= min_confirm_repetitions <= 9
    ):
        errors.append("search.min_confirm_repetitions must be an integer from 2 through 9")
    if not isinstance(search.get("require_all_slo_pass", True), bool):
        errors.append("search.require_all_slo_pass must be boolean")
    budget = spec.get("budget", {})
    max_trials = budget.get("max_trials") if isinstance(budget, dict) else None
    if not isinstance(max_trials, int) or isinstance(max_trials, bool) or max_trials <= 0:
        errors.append("budget.max_trials must be a positive integer for execution")
    elif max_trials < repetitions:
        errors.append("budget.max_trials must be at least search.repetitions")
    if not isinstance(baseline, dict):
        errors.append("search.baseline must be an object")
        baseline = {}
    if not isinstance(space, dict):
        errors.append("search.space must be an object")
        space = {}
    if strategy == "one_factor" and set(parameter_order) != set(space):
        errors.append("search.parameter_order must contain every search.space parameter exactly once")
    for name, value in baseline.items():
        error = validate_parameter(name, value, bindings)
        if error:
            errors.append(f"search.baseline.{error}")
    for name, values in space.items():
        if not isinstance(values, list) or not values:
            errors.append(f"search.space.{name} must be a non-empty array")
            continue
        for value in values:
            error = validate_parameter(name, value, bindings)
            if error:
                errors.append(f"search.space.{error}")
    explicit_configurations = search.get("explicit_configurations", [])
    if strategy == "explicit_configurations":
        if space:
            errors.append("explicit_configurations strategy requires an empty search.space")
        if not isinstance(explicit_configurations, list) or not explicit_configurations:
            errors.append("search.explicit_configurations must be a non-empty array")
            explicit_configurations = []
        for index, item in enumerate(explicit_configurations):
            if not isinstance(item, dict):
                errors.append(f"search.explicit_configurations[{index}] must be an object")
                continue
            name = item.get("name")
            config = item.get("config")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", name):
                errors.append(f"search.explicit_configurations[{index}].name must be a safe name")
            if not isinstance(config, dict) or not config:
                errors.append(f"search.explicit_configurations[{index}].config must be a non-empty object")
                continue
            for parameter, value in config.items():
                error = validate_parameter(parameter, value, bindings)
                if error:
                    errors.append(f"search.explicit_configurations[{index}].{error}")
    available_accelerators = accelerator_count(spec)
    declared_hosts = spec.get("hardware", {}).get("hosts", 1)
    if declared_hosts != 1:
        errors.append("execution supports exactly one host")
    configurations = [baseline]
    if strategy == "explicit_configurations":
        configurations.extend(
            item["config"] for item in explicit_configurations
            if isinstance(item, dict) and isinstance(item.get("config"), dict)
        )
    elif isinstance(space, dict):
        for parameter, values in space.items():
            if isinstance(values, list):
                for value in values:
                    config = deepcopy(baseline)
                    config[parameter] = value
                    configurations.append(config)
    for config in configurations:
        tp_size = config.get("tp_size", 1)
        if isinstance(tp_size, int) and tp_size > available_accelerators:
            errors.append(f"tp_size={tp_size} exceeds {available_accelerators} visible accelerators")
            break
    unknown_benchmark = set(benchmark) - BENCHMARK_KEYS
    for key in sorted(unknown_benchmark):
        errors.append(f"unsupported benchmark field: {key}")
    dataset_name = benchmark.get("dataset_name", "random-ids")
    if dataset_name not in SUPPORTED_DATASETS:
        errors.append("benchmark.dataset_name must be random-ids, random, custom, sharegpt, or generated-shared-prefix")
    if dataset_name in {"custom", "sharegpt"} or (
        dataset_name == "random" and execution.get("offline", True)
    ):
        dataset_path = Path(str(benchmark.get("dataset_path", ""))).expanduser()
        if not dataset_path.is_absolute() or not dataset_path.is_file():
            if dataset_name == "random":
                errors.append(
                    "offline random dataset requires benchmark.dataset_path pointing to a local ShareGPT JSON; "
                    "use random-ids for download-free synthetic input"
                )
            else:
                errors.append("benchmark.dataset_path must be an existing absolute file")
    for key in ("num_prompts", "random_input_len", "random_output_len", "max_concurrency", "warmup_requests", "seed"):
        if key in benchmark and (not isinstance(benchmark[key], int) or isinstance(benchmark[key], bool) or benchmark[key] < 0):
            errors.append(f"benchmark.{key} must be a non-negative integer")
    for key in ("gsp_num_groups", "gsp_prompts_per_group", "gsp_system_prompt_len", "gsp_question_len", "gsp_output_len"):
        if key in benchmark and (not isinstance(benchmark[key], int) or isinstance(benchmark[key], bool) or benchmark[key] <= 0):
            errors.append(f"benchmark.{key} must be a positive integer")
    if "gsp_range_ratio" in benchmark and (
        not isinstance(benchmark["gsp_range_ratio"], (int, float))
        or isinstance(benchmark["gsp_range_ratio"], bool)
        or not 0 < benchmark["gsp_range_ratio"] <= 1
    ):
        errors.append("benchmark.gsp_range_ratio must be in (0, 1]")
    if "gsp_ordered" in benchmark and not isinstance(benchmark["gsp_ordered"], bool):
        errors.append("benchmark.gsp_ordered must be boolean")
    minimum_seconds = benchmark.get("min_measurement_seconds", 0)
    if (
        not isinstance(minimum_seconds, (int, float))
        or isinstance(minimum_seconds, bool)
        or minimum_seconds < 0
    ):
        errors.append("benchmark.min_measurement_seconds must be a non-negative number")
    range_ratio = benchmark.get("random_range_ratio", 1.0)
    if (
        not isinstance(range_ratio, (int, float))
        or isinstance(range_ratio, bool)
        or not 0 <= range_ratio <= 1
    ):
        errors.append("benchmark.random_range_ratio must be between 0 and 1")
    if benchmark.get("num_prompts", 0) <= 0:
        errors.append("benchmark.num_prompts must be positive")
    if benchmark.get("max_concurrency", 0) <= 0:
        errors.append("benchmark.max_concurrency must be positive")
    request_rate = benchmark.get("request_rate", "inf")
    if request_rate != "inf" and (not isinstance(request_rate, (int, float)) or request_rate <= 0):
        errors.append("benchmark.request_rate must be positive or 'inf'")
    concurrency = benchmark.get("max_concurrency", 0)
    if isinstance(concurrency, int) and benchmark.get("num_prompts", 0) < concurrency:
        errors.append("benchmark.num_prompts must be at least max_concurrency")
    return errors


def parameter_args(config: dict[str, Any], bindings: dict[str, Any] | None = None) -> list[str]:
    argv: list[str] = []
    for name in sorted(config):
        value = config[name]
        binding = catalog_binding(name, bindings)
        if binding is not None:
            action = binding.get("action")
            if action is None and name in BOOL_FLAGS:
                if value:
                    argv.append(BOOL_FLAGS[name])
                continue
            if action == "store_true":
                if value:
                    argv.append(str(binding["primary_flag"]))
                continue
            if action == "store_false":
                if not value:
                    argv.append(str(binding["primary_flag"]))
                continue
            argv.extend([str(binding["primary_flag"]), str(value)])
            continue
        if name in BOOL_FLAGS:
            if value:
                argv.append(BOOL_FLAGS[name])
            continue
        flag, _, _, _ = VALUE_FLAGS[name]
        argv.extend([flag, str(value)])
    return argv


def candidate_matrix(spec: dict[str, Any]) -> list[dict[str, Any]]:
    search = spec["search"]
    baseline = deepcopy(search.get("baseline", {}))
    candidates = [{"name": "baseline", "kind": "baseline", "changed": None, "config": baseline}]
    seen = {json.dumps(baseline, sort_keys=True)}
    if search.get("strategy", "one_factor") == "explicit_configurations":
        for item in search.get("explicit_configurations", []):
            config = deepcopy(item["config"])
            signature = json.dumps(config, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append({
                "name": item["name"],
                "kind": "candidate",
                "changed": {"parameters": sorted(config)},
                "config": config,
            })
        repetitions = int(search.get("repetitions", 1))
        configuration_limit = max(1, int(spec["budget"]["max_trials"]) // repetitions)
        return candidates[:configuration_limit]
    space = search.get("space", {})
    parameter_order = search.get("parameter_order", sorted(space))
    for parameter in parameter_order:
        for value in space[parameter]:
            config = deepcopy(baseline)
            config[parameter] = value
            signature = json.dumps(config, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            label = re.sub(r"[^a-z0-9._-]+", "-", str(value).lower()).strip("-.") or "value"
            candidates.append({
                "name": f"{parameter}-{label}"[:96],
                "kind": "candidate",
                "changed": {"parameter": parameter, "value": value},
                "config": config,
            })
    repetitions = int(search.get("repetitions", 1))
    configuration_limit = max(1, int(spec["budget"]["max_trials"]) // repetitions)
    return candidates[:configuration_limit]


def measurement_plan(spec: dict[str, Any]) -> list[dict[str, Any]]:
    configurations = candidate_matrix(spec)
    repetitions = int(spec["search"].get("repetitions", 1))
    trials: list[dict[str, Any]] = []
    for repeat_index in range(repetitions):
        ordered = configurations if repeat_index % 2 == 0 else list(reversed(configurations))
        for configuration in ordered:
            trial = deepcopy(configuration)
            trial["configuration_name"] = configuration["name"]
            trial["repeat_index"] = repeat_index
            if repetitions > 1:
                trial["name"] = f"{configuration['name']}-r{repeat_index + 1:02d}"[:104]
            trials.append(trial)
    return trials


def command_manifest(spec: dict[str, Any], trial: dict[str, Any], trial_dir: Path) -> dict[str, Any]:
    execution = spec["execution"]
    benchmark = spec["benchmark"]
    python = str(Path(execution.get("python", sys.executable)).expanduser())
    host = execution.get("host", "127.0.0.1")
    port = int(execution.get("port", 30000))
    server = [
        python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(spec["model"]["path"]),
        "--host",
        host,
        "--port",
        str(port),
    ]
    model = spec["model"]
    if model.get("revision"):
        server.extend(["--revision", str(model["revision"])])
    if model.get("dtype"):
        server.extend(["--dtype", str(model["dtype"])])
    if model.get("context_length"):
        server.extend(["--context-length", str(model["context_length"])])
    if model.get("quantization"):
        server.extend(["--quantization", str(model["quantization"])])
    server.extend(parameter_args(trial["config"], execution.get("parameter_bindings")))
    bench = [
        python,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--base-url",
        f"http://{host}:{port}",
        "--model",
        str(spec["model"]["path"]),
        "--dataset-name",
        benchmark.get("dataset_name", "random-ids"),
        "--num-prompts",
        str(benchmark["num_prompts"]),
        "--max-concurrency",
        str(benchmark["max_concurrency"]),
        "--ready-check-timeout-sec",
        "5",
        "--warmup-requests",
        str(benchmark.get("warmup_requests", 3)),
        "--seed",
        str(benchmark.get("seed", 1)),
        "--output-file",
        str(trial_dir / "result.jsonl"),
        "--disable-tqdm",
    ]
    dataset_name = benchmark.get("dataset_name", "random-ids")
    if dataset_name in SYNTHETIC_RANDOM_DATASETS:
        bench.extend(["--random-input-len", str(benchmark.get("random_input_len", 1024))])
        bench.extend(["--random-output-len", str(benchmark.get("random_output_len", 256))])
        bench.extend(["--random-range-ratio", str(benchmark.get("random_range_ratio", 1.0))])
    if dataset_name == "random-ids":
        bench.append("--tokenize-prompt")
    if dataset_name == "generated-shared-prefix":
        for key, flag in (
            ("gsp_num_groups", "--gsp-num-groups"),
            ("gsp_prompts_per_group", "--gsp-prompts-per-group"),
            ("gsp_system_prompt_len", "--gsp-system-prompt-len"),
            ("gsp_question_len", "--gsp-question-len"),
            ("gsp_output_len", "--gsp-output-len"),
            ("gsp_range_ratio", "--gsp-range-ratio"),
        ):
            if key in benchmark:
                bench.extend([flag, str(benchmark[key])])
        if benchmark.get("gsp_ordered", False):
            bench.append("--gsp-ordered")
    if benchmark.get("dataset_path"):
        bench.extend(["--dataset-path", str(benchmark["dataset_path"])])
    if benchmark.get("request_rate", "inf") != "inf":
        bench.extend(["--request-rate", str(benchmark["request_rate"])])
    need_details = (
        benchmark.get("output_details", False)
        or spec["objective"]["metric"] == "request_goodput_rps"
        or "max_error_rate" in spec.get("slo", {})
    )
    if need_details:
        bench.append("--output-details")
    return {"server": server, "benchmark": bench}


def sanitized_environment(spec: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in spec["execution"].get("env", {}).items():
        env[key] = str(value)
    if spec["execution"].get("offline", True):
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    for key in ("NO_PROXY", "no_proxy"):
        entries = [item.strip() for item in env.get(key, "").split(",") if item.strip()]
        for loopback in ("127.0.0.1", "localhost"):
            if loopback not in entries:
                entries.append(loopback)
        env[key] = ",".join(entries)
    repo_python = str(Path(spec["repository"]) / "python")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_python if not existing else f"{repo_python}{os.pathsep}{existing}"
    return env


def wait_port_available(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                listening = True
        except (ConnectionRefusedError, TimeoutError, OSError):
            listening = False
        if not listening:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"port {host}:{port} remained in use for {timeout:g} seconds")
        time.sleep(0.5)


def wait_ready(url: str, process: subprocess.Popen[Any], timeout: float) -> tuple[bool, str | None]:
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            return False, f"server exited during startup with code {code}"
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return True, None
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = type(exc).__name__
        time.sleep(1)
    return False, f"health timeout; last error: {last_error}"


def stop_owned_process(process: subprocess.Popen[Any], timeout: float) -> dict[str, Any]:
    if process.poll() is not None:
        return {"method": "already_exited", "returncode": process.returncode}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
        return {"method": "sigterm_process_group", "returncode": process.returncode}
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
        return {"method": "sigkill_process_group", "returncode": process.returncode}
    except (ProcessLookupError, PermissionError) as exc:
        return {"method": "stop_error", "error": type(exc).__name__}


def accelerator_count(spec: dict[str, Any]) -> int:
    env = spec["execution"].get("env", {})
    visible = env.get("CUDA_VISIBLE_DEVICES", env.get("HIP_VISIBLE_DEVICES"))
    if visible is not None:
        items = [item for item in str(visible).split(",") if item.strip() and item.strip() != "-1"]
        return max(1, len(items))
    return max(1, int(spec.get("hardware", {}).get("gpus_per_host", 1)))


def has_accelerator() -> bool:
    for command in (["nvidia-smi", "-L"], ["rocm-smi", "--showproductname"]):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and result.stdout.strip():
            return True
    return False


def classify_failure(server_log: Path, benchmark_log: Path, detail: str) -> str:
    text = detail.lower()
    for path in (server_log, benchmark_log):
        if path.exists():
            text += "\n" + path.read_text(encoding="utf-8", errors="replace")[-20000:].lower()
    if any(pattern in text for pattern in (
        "not enough values to unpack",
        "shape mismatch",
        "invalid shape",
        "unsupported model architecture",
        "backend is not supported",
        "not implemented for",
    )):
        return "backend_incompatible"
    if "out of memory" in text or "cuda oom" in text:
        return "oom"
    if "xid" in text or "ecc" in text:
        return "gpu_health"
    if "address already in use" in text or "remained in use" in text:
        return "port_conflict"
    if any(pattern in text for pattern in (
        "offlinemodeisenabled",
        "localentrynotfounderror",
        "cannot reach huggingface.co",
        "trying to locate the files on the hub",
    )):
        return "dataset_unavailable"
    if "no module named" in text:
        return "dependency_missing"
    if "unrecognized arguments" in text:
        return "configuration"
    if any(pattern in text for pattern in (
        "health timeout",
        "benchmark timeout after",
        "timed out",
        "timeoutexpired",
        "readtimeout",
        "connecttimeout",
    )):
        return "timeout"
    return "runtime"


def capability_family(trial: dict[str, Any]) -> str | None:
    """Return a runtime capability shared by multiple candidate bundles."""
    algorithm = trial.get("config", {}).get("speculative_algorithm")
    if not isinstance(algorithm, str) or not algorithm.strip():
        return None
    normalized = algorithm.strip().lower()
    if normalized == "eagle":
        return "mtp_eagle"
    return f"speculative_{normalized}"


def capability_failure_reason(
    trial: dict[str, Any], status: dict[str, Any], server_log: Path
) -> dict[str, Any] | None:
    """Identify failures that make every remaining candidate in a family unusable."""
    family = capability_family(trial)
    failure_class = status.get("failure_class")
    if family is None or failure_class not in {"dependency_missing", "backend_incompatible"}:
        return None
    log_text = server_log.read_text(encoding="utf-8", errors="replace") if server_log.exists() else ""
    missing_module = re.search(r"No module named ['\"]([^'\"]+)['\"]", log_text)
    reason = status.get("detail") or "startup failed"
    if missing_module is not None:
        reason = f"missing Python module: {missing_module.group(1)}"
    return {
        "family": family,
        "failure_class": failure_class,
        "reason": reason,
        "origin_trial": trial["name"],
        "origin_configuration": trial["configuration_name"],
    }


def set_cli_option(argv: list[str], option: str, value: int) -> None:
    """Replace a single benchmark option while preserving the rendered command."""
    index = argv.index(option)
    argv[index + 1] = str(value)


def increase_benchmark_request_count(argv: list[str], target_prompts: int) -> int:
    """Raise the effective request count for a normal or generated-prefix workload.

    SGLang's generated-shared-prefix dataset derives its actual request count
    from groups times prompts-per-group, so changing only --num-prompts does
    not lengthen the measurement window.
    """
    set_cli_option(argv, "--num-prompts", target_prompts)
    if "--gsp-num-groups" not in argv:
        return target_prompts
    groups = int(argv[argv.index("--gsp-num-groups") + 1])
    effective = max(groups, math.ceil(target_prompts / groups) * groups)
    set_cli_option(argv, "--gsp-prompts-per-group", effective // groups)
    return effective


def run_trial(
    spec: dict[str, Any], trial: dict[str, Any], trial_dir: Path, time_limit_sec: float
) -> dict[str, Any]:
    trial_dir.mkdir(parents=True, exist_ok=False)
    os.chmod(trial_dir, 0o700)
    manifest = command_manifest(spec, trial, trial_dir)
    write_json(trial_dir / "trial.json", trial)
    write_json(trial_dir / "commands.json", manifest)
    status: dict[str, Any] = {"state": "starting", "started_at": now_iso()}
    write_json(trial_dir / "status.json", status)
    server_log_path = trial_dir / "server.log"
    benchmark_log_path = trial_dir / "benchmark.log"
    execution = spec["execution"]
    host = execution.get("host", "127.0.0.1")
    port = int(execution.get("port", 30000))
    env = sanitized_environment(spec)
    process: subprocess.Popen[Any] | None = None
    started = time.monotonic()
    previous_signal_handlers: dict[int, Any] = {}

    def interrupt_trial(_signum: int, _frame: Any) -> None:
        # Raise into the try/finally below so the owned SGLang process group is
        # stopped before the autotune controller exits.
        raise KeyboardInterrupt

    for interrupt_signal in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[interrupt_signal] = signal.getsignal(interrupt_signal)
        signal.signal(interrupt_signal, interrupt_trial)
    try:
        wait_port_available(host, port, float(execution.get("shutdown_timeout_sec", 30)))
        with server_log_path.open("w", encoding="utf-8") as server_log:
            process = subprocess.Popen(
                manifest["server"],
                cwd=spec["repository"],
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            write_json(trial_dir / "process.json", {
                "pid": process.pid,
                "process_group": process.pid,
                "started_at": now_iso(),
                "command": manifest["server"],
                "owned_by": str(trial_dir),
            })
            startup_limit = min(float(execution.get("startup_timeout_sec", 900)), time_limit_sec)
            ready, detail = wait_ready(
                f"http://{host}:{port}/v1/models",
                process,
                startup_limit,
            )
            if not ready:
                raise RuntimeError(detail or "server failed health check")
            status["state"] = "benchmarking"
            status["ready_at"] = now_iso()
            write_json(trial_dir / "status.json", status)
            remaining = time_limit_sec - (time.monotonic() - started)
            if remaining <= 0:
                raise RuntimeError("trial time budget exhausted before benchmark")
            raw_path = trial_dir / "result.jsonl"
            minimum_duration = float(spec["benchmark"].get("min_measurement_seconds", 0))
            benchmark = list(manifest["benchmark"])
            attempts: list[dict[str, Any]] = []
            max_steady_state_attempts = 5
            for attempt_index in range(1, max_steady_state_attempts + 1):
                remaining = time_limit_sec - (time.monotonic() - started)
                if remaining <= 0:
                    raise RuntimeError("trial time budget exhausted during benchmark")
                with benchmark_log_path.open("a", encoding="utf-8") as benchmark_log:
                    benchmark_log.write(f"\n=== benchmark attempt {attempt_index} ===\n")
                    result = subprocess.run(
                        benchmark,
                        cwd=spec["repository"],
                        env=env,
                        stdout=benchmark_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=min(float(execution.get("benchmark_timeout_sec", 1800)), remaining),
                        check=False,
                    )
                if result.returncode != 0:
                    raise RuntimeError(f"benchmark exited with code {result.returncode}")
                summary = summarize_jsonl(raw_path, spec)
                attempts.append({
                    "attempt": attempt_index,
                    "num_prompts": int(benchmark[benchmark.index("--num-prompts") + 1]),
                    "measurement_validity": summary["measurement_validity"],
                    "result_file": raw_path.name,
                })
                if summary["measurement_validity"]["duration_gate_passed"]:
                    break
                if attempt_index == max_steady_state_attempts:
                    raise RuntimeError(
                        "benchmark did not reach the minimum steady-state measurement duration after "
                        f"{max_steady_state_attempts} attempts"
                    )
                short_path = trial_dir / f"result-short-attempt-{attempt_index}.jsonl"
                raw_path.replace(short_path)
                attempts[-1]["result_file"] = short_path.name
                duration = summary["measurement_validity"].get("duration_sec") or 0
                current_prompts = int(benchmark[benchmark.index("--num-prompts") + 1])
                multiplier = max(2.0, (minimum_duration / duration) * 1.2) if duration > 0 else 2.0
                next_prompts = max(current_prompts + 1, math.ceil(current_prompts * multiplier))
                effective_prompts = increase_benchmark_request_count(benchmark, next_prompts)
                attempts[-1]["next_effective_num_prompts"] = effective_prompts
            write_json(trial_dir / "benchmark-attempts.json", attempts)
            runtime_observations = summarize_sglang_log(
                server_log_path.read_text(encoding="utf-8", errors="replace")
            )
            summary["runtime_observations"] = runtime_observations
            write_json(trial_dir / "runtime-observations.json", runtime_observations)
            write_json(trial_dir / "summary.json", summary)
            status.update({
                "state": "completed",
                "completed_at": now_iso(),
                "elapsed_sec": time.monotonic() - started,
                "slo_passed": summary["slo"]["passed"],
            })
            return {"ok": True, "summary": summary, "status": status}
    except subprocess.TimeoutExpired as exc:
        detail = f"benchmark timeout after {exc.timeout} seconds"
        status.update({"state": "failed", "completed_at": now_iso(), "detail": detail})
        status["failure_class"] = classify_failure(server_log_path, benchmark_log_path, detail)
        return {"ok": False, "status": status}
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        detail = str(exc)
        status.update({"state": "failed", "completed_at": now_iso(), "detail": detail})
        status["failure_class"] = classify_failure(server_log_path, benchmark_log_path, detail)
        return {"ok": False, "status": status}
    finally:
        if process is not None:
            status["shutdown"] = stop_owned_process(process, float(execution.get("shutdown_timeout_sec", 30)))
            status["shutdown"]["reaped_descendants"] = reap_exited_children()
        status["elapsed_sec"] = time.monotonic() - started
        write_json(trial_dir / "status.json", status)
        for interrupt_signal, previous in previous_signal_handlers.items():
            signal.signal(interrupt_signal, previous)


def summarize_jsonl(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL record at line {line_number}")
            records.append(value)
    if not records:
        raise ValueError("benchmark produced no JSONL records")
    summary = summarize(records, spec)
    summary["slo"] = slo_results(summary, spec)
    throughput = summary["metrics"].get("request_throughput_rps")
    completed = summary.get("raw_completed")
    recorded_durations = [
        float(record["duration"])
        for record in records
        if isinstance(record.get("duration"), (int, float)) and record["duration"] > 0
    ]
    # Current SGLang writes one aggregate JSONL record, not one record per
    # request. Its duration field is therefore the authoritative measurement
    # window; counting JSONL rows would grossly over-expand a rerun.
    duration_source = "sglang_result_duration" if recorded_durations else "throughput_estimate"
    duration = (
        sum(recorded_durations)
        if recorded_durations
        else completed / throughput
        if isinstance(completed, (int, float)) and isinstance(throughput, (int, float)) and throughput > 0
        else None
    )
    summary["measurement_validity"] = {
        "purpose": "sample-validity gate only; not an SLO or optimization objective",
        "request_count": completed if isinstance(completed, (int, float)) else None,
        "duration_sec": duration,
        "duration_source": duration_source,
        "minimum_duration_sec": spec["benchmark"].get("min_measurement_seconds", 0),
        "duration_gate_passed": duration is not None and duration >= float(spec["benchmark"].get("min_measurement_seconds", 0)),
    }
    return summary


def aggregate_results(rows: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    expected = int(spec["search"].get("repetitions", 1))
    objective_metric = spec["objective"]["metric"]
    max_cv_pct = float(spec["search"].get("max_cv_pct", 10.0))
    require_all_slo = spec["search"].get("require_all_slo_pass", True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["configuration_name"], []).append(row)
    aggregates: list[dict[str, Any]] = []
    for configuration_name, group in grouped.items():
        completed = [row for row in group if row["ok"]]
        metric_names = sorted({key for row in completed for key in row.get("metrics", {})})
        samples = {
            metric: [float(row["metrics"][metric]) for row in completed if metric in row.get("metrics", {})]
            for metric in metric_names
        }
        metrics = {metric: median(values) for metric, values in samples.items() if values}
        objective_samples = samples.get(objective_metric, [])
        objective_cv_pct: float | None = None
        if objective_samples:
            sample_mean = mean(objective_samples)
            objective_cv_pct = 0.0 if len(objective_samples) == 1 else (
                pstdev(objective_samples) / abs(sample_mean) * 100 if sample_mean else None
            )
        summary = {"schema_version": 1, "metrics": metrics}
        summary["slo"] = slo_results(summary, spec)
        all_slo_passed = (
            len(completed) == expected
            and all(row.get("slo", {}).get("passed", False) for row in completed)
        )
        stable = (
            len(completed) == expected
            and objective_cv_pct is not None
            and objective_cv_pct <= max_cv_pct
        )
        first = group[0]
        aggregates.append({
            "configuration_name": configuration_name,
            "kind": first["kind"],
            "config": first["config"],
            "expected_repetitions": expected,
            "completed_repetitions": len(completed),
            "failed_repetitions": len(group) - len(completed),
            "metrics": metrics,
            "metric_samples": samples,
            "slo": summary["slo"],
            "all_repetitions_slo_passed": all_slo_passed,
            "objective_cv_pct": objective_cv_pct,
            "max_cv_pct": max_cv_pct,
            "stable": stable,
            "eligible_for_confirmation": stable and (all_slo_passed or not require_all_slo),
        })
    return aggregates


def evaluate_aggregates(
    aggregates: list[dict[str, Any]], spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    baseline = next((item for item in aggregates if item["kind"] == "baseline"), None)
    screening_winner: dict[str, Any] | None = None
    confirmed_winner: dict[str, Any] | None = None
    min_confirm_repetitions = int(spec["search"].get("min_confirm_repetitions", 3))
    if baseline is not None:
        baseline["confirmed"] = (
            baseline["eligible_for_confirmation"]
            and baseline["completed_repetitions"] >= min_confirm_repetitions
        )
        baseline["rejection_reasons"] = [] if baseline["confirmed"] else [
            "baseline_not_confirmed"
        ]
    for item in aggregates:
        if item["kind"] == "baseline":
            continue
        if baseline is None or not baseline.get("metrics"):
            item.update({
                "screening_accepted": False,
                "confirmed": False,
                "rejection_reasons": ["baseline_unavailable"],
            })
            continue
        comparison = compare(
            {"metrics": baseline["metrics"]},
            {"metrics": item["metrics"]},
            spec,
        )
        item["comparison"] = comparison
        item["screening_accepted"] = comparison["accepted"]
        item["confirmed"] = (
            comparison["accepted"]
            and baseline["eligible_for_confirmation"]
            and item["eligible_for_confirmation"]
            and baseline["completed_repetitions"] >= min_confirm_repetitions
            and item["completed_repetitions"] >= min_confirm_repetitions
        )
        rejection_reasons: list[str] = []
        if not comparison["candidate_slo"]["passed"]:
            rejection_reasons.append("candidate_slo_failed")
        if not comparison["secondary_regressions_passed"]:
            rejection_reasons.append("secondary_metric_regression")
        improvement = comparison.get("improvement_pct")
        minimum = comparison["minimum_improvement_pct"]
        if improvement is None or improvement < minimum:
            rejection_reasons.append("objective_improvement_below_minimum")
        if item["completed_repetitions"] < item["expected_repetitions"]:
            rejection_reasons.append("incomplete_repetitions")
        if not item["stable"]:
            rejection_reasons.append("objective_variation_too_high")
        if not item["all_repetitions_slo_passed"]:
            rejection_reasons.append("not_all_repetitions_passed_slo")
        if not baseline.get("confirmed", False):
            rejection_reasons.append("baseline_not_confirmed")
        elif item["completed_repetitions"] < min_confirm_repetitions:
            rejection_reasons.append("insufficient_confirmation_repetitions")
        item["rejection_reasons"] = rejection_reasons
        if comparison["accepted"] and (
            screening_winner is None
            or comparison["improvement_pct"] > screening_winner["comparison"]["improvement_pct"]
        ):
            screening_winner = item
        if item["confirmed"] and (
            confirmed_winner is None
            or comparison["improvement_pct"] > confirmed_winner["comparison"]["improvement_pct"]
        ):
            confirmed_winner = item
    return aggregates, screening_winner, confirmed_winner


def deployment_recommendation(
    aggregates: list[dict[str, Any]],
    screening_winner: dict[str, Any] | None,
    confirmed_winner: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str, str]:
    if confirmed_winner is not None:
        return (
            confirmed_winner,
            "confirmed_candidate",
            "candidate passed improvement, SLO, repetition, and stability gates",
        )
    baseline = next((item for item in aggregates if item["kind"] == "baseline"), None)
    if baseline is not None and baseline.get("confirmed", False):
        return (
            baseline,
            "retain_confirmed_baseline",
            "no candidate cleared all gates; baseline is stable and passed every SLO repetition",
        )
    if screening_winner is not None:
        return (
            None,
            "screening_only",
            "candidate improved in screening but lacks confirmation evidence",
        )
    return None, "no_valid_configuration", "no configuration has sufficient deployable evidence"


def decision_report(spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregates = aggregate_results(rows, spec)
    aggregates, screening_winner, confirmed_winner = evaluate_aggregates(aggregates, spec)
    recommended, recommendation_status, recommendation_reason = deployment_recommendation(
        aggregates, screening_winner, confirmed_winner
    )
    return {
        "schema_version": 2,
        "winner": confirmed_winner,
        "screening_winner": screening_winner,
        "winner_status": "confirmed" if confirmed_winner else (
            "screening_only" if screening_winner else "none"
        ),
        "recommended_configuration": recommended,
        "recommendation_status": recommendation_status,
        "recommendation_reason": recommendation_reason,
        "aggregates": aggregates,
    }


def report_existing_run(run_dir: str | Path) -> dict[str, Any]:
    directory = Path(run_dir).expanduser()
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError("run directory must be an existing absolute directory")
    spec = load_json(directory / "spec.json")
    rows_value = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    if not isinstance(rows_value, list) or not rows_value:
        raise ValueError("run results.json must contain a non-empty array")
    for row in rows_value:
        if not isinstance(row, dict):
            raise ValueError("run results.json contains a non-object row")
        if "configuration_name" not in row:
            row["configuration_name"] = row.get("name", "unknown")
        if "repeat_index" not in row:
            row["repeat_index"] = 0
    report = decision_report(spec, rows_value)
    report["run_dir"] = str(directory)
    report["source_results"] = str(directory / "results.json")
    return report


def prepare_run(spec: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    root = Path(spec["scope"]["output_dir"]).expanduser()
    if not root.is_absolute():
        raise ValueError("scope.output_dir must be absolute")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root / f"{spec['name']}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    os.chmod(run_dir, 0o700)
    configurations = candidate_matrix(spec)
    trials = measurement_plan(spec)
    write_json(run_dir / "spec.json", spec)
    write_json(run_dir / "inventory.json", inventory())
    write_json(run_dir / "manifest.json", {
        "created_at": now_iso(),
        "run_dir": str(run_dir),
        "trial_count": len(trials),
        "configuration_count": len(configurations),
        "repetitions": int(spec["search"].get("repetitions", 1)),
        "trials": [
            {**trial, "commands": command_manifest(spec, trial, run_dir / f"trial-{index:03d}-{trial['name']}")}
            for index, trial in enumerate(trials)
        ],
    })
    return run_dir, trials


def execute(
    spec: dict[str, Any], progress: Callable[[dict[str, Any]], None] | None = None
) -> dict[str, Any]:
    child_subreaper_enabled = enable_child_subreaper()
    errors = execution_errors(spec)
    if errors:
        raise ValueError("; ".join(errors))
    if spec["execution"].get("require_accelerator", True) and not has_accelerator():
        raise RuntimeError("no NVIDIA or AMD accelerator detected")
    run_dir, trials = prepare_run(spec)
    started = time.monotonic()
    gpu_count = accelerator_count(spec)
    max_wall = float(spec["budget"]["max_wall_time_minutes"]) * 60
    max_gpu_hours = float(spec["budget"]["max_gpu_hours"])
    max_gpu_elapsed = max_gpu_hours * 3600 / gpu_count
    max_failures = int(spec["budget"].get("max_consecutive_failures", 3))
    rows: list[dict[str, Any]] = []
    skipped_capability_trials: list[dict[str, Any]] = []
    disabled_capabilities: dict[str, dict[str, Any]] = {}
    failures = 0
    stop_reason: str | None = None
    for index, trial in enumerate(trials):
        elapsed = time.monotonic() - started
        if elapsed >= max_wall:
            stop_reason = "wall_time_budget_exhausted"
            break
        if elapsed >= max_gpu_elapsed:
            stop_reason = "gpu_hour_budget_exhausted"
            break
        remaining_time = min(max_wall - elapsed, max_gpu_elapsed - elapsed)
        trial_dir = run_dir / f"trial-{index:03d}-{trial['name']}"
        family = capability_family(trial)
        if family is not None and family in disabled_capabilities:
            skipped = {
                "index": index,
                "name": trial["name"],
                "configuration_name": trial["configuration_name"],
                "repeat_index": trial["repeat_index"],
                "kind": trial["kind"],
                "config": trial["config"],
                "capability": family,
                "reason": disabled_capabilities[family]["reason"],
                "disabled_by": disabled_capabilities[family]["origin_trial"],
            }
            skipped_capability_trials.append(skipped)
            if progress is not None:
                progress({
                    "event": "trial_skipped",
                    "trial_index": index + 1,
                    "trial_count": len(trials),
                    "trial_name": trial["name"],
                    "capability": family,
                    "reason": skipped["reason"],
                })
            continue
        if progress is not None:
            progress({
                "event": "trial_started",
                "trial_index": index + 1,
                "trial_count": len(trials),
                "trial_name": trial["name"],
                "configuration_name": trial["configuration_name"],
                "kind": trial["kind"],
            })
        result = run_trial(spec, trial, trial_dir, remaining_time)
        row: dict[str, Any] = {
            "index": index,
            "name": trial["name"],
            "configuration_name": trial["configuration_name"],
            "repeat_index": trial["repeat_index"],
            "kind": trial["kind"],
            "config": trial["config"],
            "directory": str(trial_dir),
            "ok": result["ok"],
            "status": result["status"],
        }
        if not result["ok"]:
            rows.append(row)
            write_json(run_dir / "results.json", rows)
            disabled = capability_failure_reason(
                trial, result["status"], trial_dir / "server.log"
            )
            if disabled is not None:
                disabled_capabilities[disabled["family"]] = disabled
                row["status"]["capability_disabled"] = disabled
                write_json(run_dir / "results.json", rows)
            else:
                failures += 1
            if progress is not None:
                progress({
                    "event": "trial_finished",
                    "trial_index": index + 1,
                    "trial_count": len(trials),
                    "trial_name": trial["name"],
                    "ok": False,
                    "detail": result["status"].get("detail"),
                })
            if trial["kind"] == "baseline":
                stop_reason = "baseline_failed"
                break
            if result["status"].get("failure_class") == "gpu_health":
                stop_reason = "gpu_health_failure"
                break
            if disabled is None and failures >= max_failures:
                stop_reason = "consecutive_failure_budget_exhausted"
                break
            continue
        failures = 0
        summary = result["summary"]
        row["metrics"] = summary["metrics"]
        row["slo"] = summary["slo"]
        rows.append(row)
        write_json(run_dir / "results.json", rows)
        if progress is not None:
            progress({
                "event": "trial_finished",
                "trial_index": index + 1,
                "trial_count": len(trials),
                "trial_name": trial["name"],
                "ok": True,
                "metrics": summary["metrics"],
                "slo_passed": summary["slo"].get("passed"),
            })
    decision = decision_report(spec, rows)
    aggregates = decision["aggregates"]
    write_json(run_dir / "aggregates.json", aggregates)
    final = {
        "schema_version": 2,
        "child_subreaper_enabled": child_subreaper_enabled,
        "run_dir": str(run_dir),
        "completed_at": now_iso(),
        "elapsed_sec": time.monotonic() - started,
        "approx_gpu_hours": (time.monotonic() - started) / 3600 * gpu_count,
        "planned_trials": len(trials),
        "completed_trials": len(rows),
        "skipped_capability_trials": skipped_capability_trials,
        "disabled_capabilities": list(disabled_capabilities.values()),
        "stop_reason": stop_reason or "completed_search",
        **decision,
        "results": rows,
    }
    write_json(run_dir / "final.json", final)
    return final


def render_plan(spec: dict[str, Any]) -> dict[str, Any]:
    errors = execution_errors(spec)
    if errors:
        raise ValueError("; ".join(errors))
    output_root = Path(spec["scope"]["output_dir"]) / "PLAN_ONLY"
    configurations = candidate_matrix(spec)
    trials = measurement_plan(spec)
    return {
        "valid": True,
        "execution_enabled": False,
        "trial_count": len(trials),
        "configuration_count": len(configurations),
        "repetitions": int(spec["search"].get("repetitions", 1)),
        "budget": spec["budget"],
        "trials": [
            {**trial, "commands": command_manifest(spec, trial, output_root / f"trial-{index:03d}-{trial['name']}")}
            for index, trial in enumerate(trials)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--spec", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--spec", required=True)
    plan_parser.add_argument("--output")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--spec", required=True)
    run_parser.add_argument("--yes", action="store_true")
    run_parser.add_argument("--output")
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-dir", required=True)
    report_parser.add_argument("--output")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            spec = load_json(args.spec)
            errors = execution_errors(spec)
            dump_json({"valid": not errors, "errors": errors}, None)
            return 0 if not errors else 2
        if args.command == "plan":
            spec = load_json(args.spec)
            dump_json(render_plan(spec), args.output)
            return 0
        if args.command == "run":
            spec = load_json(args.spec)
            if not args.yes:
                raise ValueError("run requires --yes after reviewing the generated plan")
            dump_json(execute(spec), args.output)
            return 0
        if args.command == "report":
            dump_json(report_existing_run(args.run_dir), args.output)
            return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
