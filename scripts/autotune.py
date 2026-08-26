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
import threading
import time
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable

from sglang_runtime import summarize_sglang_log
from bayesian import sequential_decision, sequential_decision_from_samples

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
    "torch_compile_max_bs": ("--torch-compile-max-bs", int, 1, 4096),
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
    "max_mamba_cache_size": ("--max-mamba-cache-size", int, 1, None),
    "mamba_ssm_dtype": ("--mamba-ssm-dtype", str, None, None),
    "mamba_full_memory_ratio": ("--mamba-full-memory-ratio", float, 0.01, 1.0),
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
    "unbounded_concurrency",
    "auto_max_concurrency",
    "warmup_requests",
    "min_measurement_seconds",
    "min_tail_samples",
    "near_slo_tail_samples",
    "near_slo_margin_pct",
    "p99_request_waves",
    "seed",
    "output_details",
    "flush_cache",
    "gsp_num_groups",
    "gsp_prompts_per_group",
    "gsp_system_prompt_len",
    "gsp_question_len",
    "gsp_output_len",
    "gsp_range_ratio",
    "gsp_ordered",
    "apply_chat_template",
    "sharegpt_context_len",
    "baseline_reference_num_prompts",
    "baseline_reference_min_measurement_seconds",
    "saturation_capacity",
    "saturation_waves",
    "calibration_session",
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
    "include_baseline",
    "reference_baseline",
    "interaction_policy",
    "interaction_phase",
    "adaptive_refinement_parents",
    "adaptive_refinement_candidates",
    "mechanism_refinement_candidates",
    "mechanism_followup",
    "threshold_seed_names",
    "optional_positive_seed_names",
    "candidate_slots",
    "generated_combinations",
    "compatible_combinations",
    "budget_omitted_combinations",
    "candidate_limit",
    "selection_policy",
    "selected_parameter_candidates",
    "selection_evidence",
    "mechanism_schedule",
    "required_mechanism_coverage",
    "history_candidate_quota",
    "history_candidates_selected",
    "mandatory_mechanism_parameters",
    "mechanism_coverage_target",
    "covered_submechanisms",
    "high_magnitude_rule_parameter_floor",
    "high_magnitude_rule_coverage",
    "deferred_triggered_parameters",
    "budget_allocation",
    "reclaimed_discovery_trials",
    "min_successful_candidates_before_early_stop",
    "early_stop_coverage_floor",
    "early_stop_improvement_pct",
    "reuse_server_across_repetitions",
    "adaptive_confirmation_cv_pct",
    "adaptive_confirmation_max_repetitions",
    "adaptive_confirmation_min_measurement_seconds",
    "bayesian_sequential",
    "bayesian_min_blocks",
    "bayesian_max_blocks",
    "bayesian_accept_probability",
    "bayesian_reject_probability",
    "bayesian_prior_mean_pct",
    "bayesian_prior_strength",
    "history_prior",
    "compatibility_baseline",
    "compatibility_evidence",
    "sibling_refinement_candidates",
    "sibling_refinement_policy",
    "provisional_parameter_candidates",
    "provisional_parameter_names",
    "provisional_exploration_budget",
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
    "SGLANG_MOE_CONFIG_DIR",
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
    min_tail_samples = benchmark.get("min_tail_samples", 0)
    valid_min_tail_samples = (
        isinstance(min_tail_samples, int)
        and not isinstance(min_tail_samples, bool)
        and min_tail_samples >= 0
    )
    if not valid_min_tail_samples:
        errors.append("benchmark.min_tail_samples must be a non-negative integer")
    near_slo_tail_samples = benchmark.get("near_slo_tail_samples", min_tail_samples)
    if (
        not isinstance(near_slo_tail_samples, int)
        or isinstance(near_slo_tail_samples, bool)
        or near_slo_tail_samples < 0
        or valid_min_tail_samples and near_slo_tail_samples < min_tail_samples
    ):
        errors.append("benchmark.near_slo_tail_samples must be an integer at least min_tail_samples")
    near_slo_margin_pct = benchmark.get("near_slo_margin_pct", 10)
    if (
        not isinstance(near_slo_margin_pct, (int, float))
        or isinstance(near_slo_margin_pct, bool)
        or not 0 <= near_slo_margin_pct <= 100
    ):
        errors.append("benchmark.near_slo_margin_pct must be between 0 and 100")
    p99_request_waves = benchmark.get("p99_request_waves", 0)
    if (
        not isinstance(p99_request_waves, int)
        or isinstance(p99_request_waves, bool)
        or p99_request_waves < 0
    ):
        errors.append("benchmark.p99_request_waves must be a non-negative integer")
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
    parallel_trials = execution.get("parallel_trials", 1)
    if not isinstance(parallel_trials, int) or isinstance(parallel_trials, bool) or parallel_trials < 1:
        errors.append("execution.parallel_trials must be a positive integer")
    elif parallel_trials > 16:
        errors.append("execution.parallel_trials must not exceed 16")
    elif port + parallel_trials - 1 > 65535:
        errors.append("execution.port plus parallel_trials must not exceed 65535")
    if execution.get("gpu_allocation", "exclusive") not in {"exclusive"}:
        errors.append("execution.gpu_allocation must be 'exclusive'")
    benchmark_module = execution.get("benchmark_module", "sglang.bench_serving")
    if benchmark_module not in {"sglang.benchmark.serving", "sglang.bench_serving"}:
        errors.append("execution.benchmark_module must be a supported SGLang benchmark module")
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
    calibration_session = benchmark.get("calibration_session")
    if calibration_session is not None:
        if not isinstance(calibration_session, dict):
            errors.append("benchmark.calibration_session must be an object")
        else:
            allowed_calibration = {
                "strategy", "concurrencies", "min_concurrency", "fallback_max_concurrency", "max_steps",
                "request_waves", "requested_concurrency", "stop_on_slo_failure",
                "initial_unbounded_probe",
            }
            if any(key not in allowed_calibration for key in calibration_session):
                errors.append("benchmark.calibration_session contains unsupported fields")
            if calibration_session.get("strategy") not in {"adaptive_slo", "fixed_curve"}:
                errors.append("benchmark.calibration_session.strategy must be adaptive_slo or fixed_curve")
            if "initial_unbounded_probe" in calibration_session and not isinstance(
                calibration_session["initial_unbounded_probe"], bool
            ):
                errors.append("benchmark.calibration_session.initial_unbounded_probe must be boolean")
            for key in ("min_concurrency", "fallback_max_concurrency", "max_steps", "request_waves"):
                value = calibration_session.get(key)
                if value is not None and (
                    not isinstance(value, int) or isinstance(value, bool) or value <= 0
                ):
                    errors.append(f"benchmark.calibration_session.{key} must be a positive integer")
            requested = calibration_session.get("requested_concurrency")
            if requested is not None and requested != "runtime_resolved" and (
                not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0
            ):
                errors.append("benchmark.calibration_session.requested_concurrency must be a positive integer or runtime_resolved")
            concurrency_values = calibration_session.get("concurrencies", [])
            if (
                not isinstance(concurrency_values, list)
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in concurrency_values
                )
            ):
                errors.append("benchmark.calibration_session.concurrencies must contain positive integers")
            if not isinstance(calibration_session.get("stop_on_slo_failure", False), bool):
                errors.append("benchmark.calibration_session.stop_on_slo_failure must be boolean")
    baseline = search.get("baseline", {})
    space = search.get("space", {})
    for key in sorted(set(search) - SEARCH_KEYS):
        errors.append(f"unsupported search field: {key}")
    strategy = search.get("strategy", "one_factor")
    if strategy not in {"one_factor", "explicit_configurations"}:
        errors.append("search.strategy must be one_factor or explicit_configurations")
    provisional_names = search.get("provisional_parameter_candidates", [])
    if (
        not isinstance(provisional_names, list)
        or any(not isinstance(name, str) or not name for name in provisional_names)
    ):
        errors.append("search.provisional_parameter_candidates must be an array of names")
    provisional_parameters = search.get("provisional_parameter_names", [])
    if (
        not isinstance(provisional_parameters, list)
        or any(not isinstance(name, str) or not name for name in provisional_parameters)
    ):
        errors.append("search.provisional_parameter_names must be an array of parameter names")
    provisional_budget = search.get("provisional_exploration_budget")
    if provisional_budget is not None and not isinstance(provisional_budget, dict):
        errors.append("search.provisional_exploration_budget must be an object")
    repetitions = search.get("repetitions", 1)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or not 1 <= repetitions <= 9:
        errors.append("search.repetitions must be an integer from 1 through 9")
        repetitions = 1
    if not isinstance(search.get("reuse_server_across_repetitions", False), bool):
        errors.append("search.reuse_server_across_repetitions must be boolean")
    adaptive_cv = search.get("adaptive_confirmation_cv_pct")
    if adaptive_cv is not None and (
        not isinstance(adaptive_cv, (int, float))
        or isinstance(adaptive_cv, bool)
        or not 0 <= adaptive_cv <= 100
    ):
        errors.append("search.adaptive_confirmation_cv_pct must be between 0 and 100")
    adaptive_max_repetitions = search.get("adaptive_confirmation_max_repetitions")
    if adaptive_max_repetitions is not None and (
        not isinstance(adaptive_max_repetitions, int)
        or isinstance(adaptive_max_repetitions, bool)
        or not repetitions <= adaptive_max_repetitions <= 9
    ):
        errors.append(
            "search.adaptive_confirmation_max_repetitions must be between repetitions and 9"
        )
    adaptive_min_seconds = search.get("adaptive_confirmation_min_measurement_seconds")
    if adaptive_min_seconds is not None and (
        not isinstance(adaptive_min_seconds, (int, float))
        or isinstance(adaptive_min_seconds, bool)
        or adaptive_min_seconds <= 0
    ):
        errors.append(
            "search.adaptive_confirmation_min_measurement_seconds must be positive"
        )
    if not isinstance(search.get("bayesian_sequential", False), bool):
        errors.append("search.bayesian_sequential must be boolean")
    bayesian_min = search.get("bayesian_min_blocks", 2)
    bayesian_max = search.get("bayesian_max_blocks", 6)
    if not isinstance(bayesian_min, int) or isinstance(bayesian_min, bool) or bayesian_min < 1:
        errors.append("search.bayesian_min_blocks must be a positive integer")
    if (
        not isinstance(bayesian_max, int) or isinstance(bayesian_max, bool)
        or bayesian_max < bayesian_min or bayesian_max > 20
    ):
        errors.append("search.bayesian_max_blocks must be between bayesian_min_blocks and 20")
    accept_probability = search.get("bayesian_accept_probability", 0.95)
    reject_probability = search.get("bayesian_reject_probability", 0.05)
    for key, value in (
        ("bayesian_accept_probability", accept_probability),
        ("bayesian_reject_probability", reject_probability),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < 1:
            errors.append(f"search.{key} must be between 0 and 1")
    if (
        isinstance(accept_probability, (int, float))
        and isinstance(reject_probability, (int, float))
        and reject_probability >= accept_probability
    ):
        errors.append("search.bayesian_reject_probability must be below accept probability")
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
    # The controller's default confirmation contract is two independent
    # windows. Keep the standalone executor fallback identical so hand-written
    # specs cannot silently require a third repetition that was never planned.
    min_confirm_repetitions = search.get("min_confirm_repetitions", 2)
    reference_only_confirmation = (
        search.get("include_baseline", True) is False
        and isinstance(search.get("reference_baseline"), dict)
    )
    minimum_confirm_repetitions = 1 if reference_only_confirmation else 2
    if (
        not isinstance(min_confirm_repetitions, int)
        or isinstance(min_confirm_repetitions, bool)
        or not minimum_confirm_repetitions <= min_confirm_repetitions <= 9
    ):
        errors.append(
            "search.min_confirm_repetitions must be an integer from "
            f"{minimum_confirm_repetitions} through 9"
        )
    if not isinstance(search.get("require_all_slo_pass", True), bool):
        errors.append("search.require_all_slo_pass must be boolean")
    early_stop_count = search.get("min_successful_candidates_before_early_stop")
    if early_stop_count is not None and (
        not isinstance(early_stop_count, int) or isinstance(early_stop_count, bool)
        or early_stop_count <= 0
    ):
        errors.append("search.min_successful_candidates_before_early_stop must be positive")
    early_stop_gain = search.get("early_stop_improvement_pct")
    if early_stop_gain is not None and (
        not isinstance(early_stop_gain, (int, float)) or isinstance(early_stop_gain, bool)
        or early_stop_gain <= 0
    ):
        errors.append("search.early_stop_improvement_pct must be positive")
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
    include_baseline = search.get("include_baseline", True)
    if not isinstance(include_baseline, bool):
        errors.append("search.include_baseline must be a boolean")
        include_baseline = True
    reference_baseline = search.get("reference_baseline")
    if not include_baseline:
        if not isinstance(reference_baseline, dict):
            errors.append("search.reference_baseline must be an object when include_baseline=false")
        else:
            if not isinstance(reference_baseline.get("config"), dict):
                errors.append("search.reference_baseline.config must be an object")
            if not isinstance(reference_baseline.get("metrics"), dict) or not reference_baseline.get("metrics"):
                errors.append("search.reference_baseline.metrics must be a non-empty object")
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
            candidate_env = item.get("env", {})
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", name):
                errors.append(f"search.explicit_configurations[{index}].name must be a safe name")
            if not isinstance(config, dict) or not config:
                errors.append(f"search.explicit_configurations[{index}].config must be a non-empty object")
                continue
            if not isinstance(candidate_env, dict):
                errors.append(f"search.explicit_configurations[{index}].env must be an object")
            else:
                for key, value in candidate_env.items():
                    if key not in ALLOWED_ENV:
                        errors.append(f"search.explicit_configurations[{index}].env key is not allowed: {key}")
                    if not isinstance(value, (str, int, float, bool)):
                        errors.append(f"search.explicit_configurations[{index}].env.{key} must be scalar")
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
        degrees = {
            name: config.get(name, 1)
            for name in ("tp_size", "pp_size", "dp_size")
        }
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in degrees.values()):
            continue
        requested = math.prod(degrees.values())
        if requested > available_accelerators:
            errors.append(
                "requested parallel ranks "
                f"tp_size={degrees['tp_size']} * pp_size={degrees['pp_size']} * "
                f"dp_size={degrees['dp_size']} = {requested}, exceeding "
                f"{available_accelerators} visible accelerators"
            )
            break
    unknown_benchmark = set(benchmark) - BENCHMARK_KEYS
    for key in sorted(unknown_benchmark):
        errors.append(f"unsupported benchmark field: {key}")
    if not isinstance(benchmark.get("flush_cache", False), bool):
        errors.append("benchmark.flush_cache must be boolean")
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
    saturation_capacity = benchmark.get("saturation_capacity")
    saturation_waves = benchmark.get("saturation_waves")
    if saturation_capacity is not None or saturation_waves is not None:
        if (
            not isinstance(saturation_capacity, int)
            or isinstance(saturation_capacity, bool)
            or saturation_capacity <= 0
        ):
            errors.append("benchmark.saturation_capacity must be a positive integer")
        if (
            not isinstance(saturation_waves, int)
            or isinstance(saturation_waves, bool)
            or saturation_waves <= 0
        ):
            errors.append("benchmark.saturation_waves must be a positive integer")
        if (
            isinstance(saturation_capacity, int)
            and not isinstance(saturation_capacity, bool)
            and isinstance(saturation_waves, int)
            and not isinstance(saturation_waves, bool)
            and isinstance(benchmark.get("num_prompts"), int)
            and benchmark["num_prompts"] < saturation_capacity * saturation_waves
        ):
            errors.append(
                "benchmark.num_prompts must cover saturation_capacity * saturation_waves"
            )
    for key in ("unbounded_concurrency", "auto_max_concurrency"):
        if key in benchmark and not isinstance(benchmark[key], bool):
            errors.append(f"benchmark.{key} must be boolean")
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
    if "apply_chat_template" in benchmark and not isinstance(benchmark["apply_chat_template"], bool):
        errors.append("benchmark.apply_chat_template must be boolean")
    for key in ("sharegpt_context_len", "baseline_reference_num_prompts"):
        if key in benchmark and (
            not isinstance(benchmark[key], int) or isinstance(benchmark[key], bool) or benchmark[key] <= 0
        ):
            errors.append(f"benchmark.{key} must be a positive integer")
    if "baseline_reference_min_measurement_seconds" in benchmark and (
        not isinstance(benchmark["baseline_reference_min_measurement_seconds"], (int, float))
        or isinstance(benchmark["baseline_reference_min_measurement_seconds"], bool)
        or benchmark["baseline_reference_min_measurement_seconds"] <= 0
    ):
        errors.append("benchmark.baseline_reference_min_measurement_seconds must be positive")
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
    unbounded_concurrency = benchmark.get("unbounded_concurrency", False)
    auto_max_concurrency = benchmark.get("auto_max_concurrency", False)
    if not (unbounded_concurrency or auto_max_concurrency) and benchmark.get("max_concurrency", 0) <= 0:
        errors.append("benchmark.max_concurrency must be positive")
    request_rate = benchmark.get("request_rate", "inf")
    if request_rate != "inf" and (not isinstance(request_rate, (int, float)) or request_rate <= 0):
        errors.append("benchmark.request_rate must be positive or 'inf'")
    concurrency = benchmark.get("max_concurrency", 0)
    if not (unbounded_concurrency or auto_max_concurrency) and isinstance(concurrency, int) and benchmark.get("num_prompts", 0) < concurrency:
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
    candidates = []
    if search.get("include_baseline", True):
        candidates.append({"name": "baseline", "kind": "baseline", "changed": None, "config": baseline, "env": {}})
    seen = {json.dumps({"config": baseline, "env": {}}, sort_keys=True)}
    if search.get("strategy", "one_factor") == "explicit_configurations":
        for item in search.get("explicit_configurations", []):
            config = deepcopy(item["config"])
            candidate_env = deepcopy(item.get("env", {}))
            signature = json.dumps({"config": config, "env": candidate_env}, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append({
                "name": item["name"],
                "kind": "candidate",
                "changed": {"parameters": sorted(config)},
                "config": config,
                "env": candidate_env,
                **(
                    {"provisional_parameter": item["provisional_parameter"]}
                    if isinstance(item.get("provisional_parameter"), str) else {}
                ),
                **(
                    {"provisional_state": item["provisional_state"]}
                    if isinstance(item.get("provisional_state"), str) else {}
                ),
                **(
                    {"provisional_atomic_config": deepcopy(item["provisional_atomic_config"])}
                    if isinstance(item.get("provisional_atomic_config"), dict) else {}
                ),
                **(
                    {"registry_candidate_id": item["registry_candidate_id"]}
                    if isinstance(item.get("registry_candidate_id"), str) else {}
                ),
                **(
                    {"registry_mechanism": item["registry_mechanism"]}
                    if isinstance(item.get("registry_mechanism"), str) else {}
                ),
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
            signature = json.dumps({"config": config, "env": {}}, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            label = re.sub(r"[^a-z0-9._-]+", "-", str(value).lower()).strip("-.") or "value"
            candidates.append({
                "name": f"{parameter}-{label}"[:96],
                "kind": "candidate",
                "changed": {"parameter": parameter, "value": value},
                "config": config,
                "env": {},
            })
    repetitions = int(search.get("repetitions", 1))
    configuration_limit = max(1, int(spec["budget"]["max_trials"]) // repetitions)
    return candidates[:configuration_limit]


def measurement_plan(spec: dict[str, Any]) -> list[dict[str, Any]]:
    configurations = candidate_matrix(spec)
    repetitions = int(spec["search"].get("repetitions", 1))
    planned_repetitions = (
        int(spec["search"].get("bayesian_max_blocks", repetitions))
        if spec["search"].get("bayesian_sequential", False)
        and len(configurations) == 2
        and {item["kind"] for item in configurations} == {"baseline", "candidate"}
        else repetitions
    )
    planned_repetitions = min(
        planned_repetitions,
        max(1, int(spec["budget"]["max_trials"]) // max(1, len(configurations))),
    )
    if (
        repetitions > 1
        and spec["search"].get("reuse_server_across_repetitions", False)
        and resident_ab_eligible(spec)
    ):
        sessions: list[dict[str, Any]] = []
        for configuration in configurations:
            trial = deepcopy(configuration)
            trial["configuration_name"] = configuration["name"]
            trial["repeat_index"] = 0
            trial["repeat_indices"] = list(range(repetitions))
            trial["name"] = f"{configuration['name']}-resident"[:104]
            sessions.append(trial)
        return sessions
    trials: list[dict[str, Any]] = []
    for repeat_index in range(planned_repetitions):
        ordered = configurations if repeat_index % 2 == 0 else list(reversed(configurations))
        for configuration in ordered:
            trial = deepcopy(configuration)
            trial["configuration_name"] = configuration["name"]
            trial["repeat_index"] = repeat_index
            if repetitions > 1:
                trial["name"] = f"{configuration['name']}-r{repeat_index + 1:02d}"[:104]
            trials.append(trial)
    return trials


def bayesian_block_decision(
    spec: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    search = spec.get("search", {})
    if not search.get("bayesian_sequential", False):
        return None
    configurations = candidate_matrix(spec)
    if len(configurations) != 2 or {item["kind"] for item in configurations} != {"baseline", "candidate"}:
        return None
    return sequential_decision(
        rows,
        objective_metric=spec["objective"]["metric"],
        minimum_improvement_pct=float(spec["objective"].get("min_improvement_pct", 0)),
        direction=spec["objective"]["direction"],
        min_blocks=int(search.get("bayesian_min_blocks", 2)),
        max_blocks=int(search.get("bayesian_max_blocks", 6)),
        accept_probability=float(search.get("bayesian_accept_probability", 0.95)),
        reject_probability=float(search.get("bayesian_reject_probability", 0.05)),
        prior_mean_pct=float(search.get("bayesian_prior_mean_pct", 0.0)),
        prior_strength=float(search.get("bayesian_prior_strength", 0.01)),
    )


def command_manifest(spec: dict[str, Any], trial: dict[str, Any], trial_dir: Path) -> dict[str, Any]:
    execution = spec["execution"]
    benchmark = spec["benchmark"]
    python = str(Path(execution.get("python", sys.executable)).expanduser())
    host = execution.get("host", "127.0.0.1")
    port = int(trial.get("_port", execution.get("port", 30000)))
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
        execution.get("benchmark_module", "sglang.bench_serving"),
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
    if not benchmark.get("unbounded_concurrency", False) and not benchmark.get("auto_max_concurrency", False):
        bench.extend(["--max-concurrency", str(benchmark["max_concurrency"])])
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
    if benchmark.get("apply_chat_template", False):
        bench.append("--apply-chat-template")
    if benchmark.get("sharegpt_context_len"):
        bench.extend(["--sharegpt-context-len", str(benchmark["sharegpt_context_len"])])
    if benchmark.get("request_rate", "inf") != "inf":
        bench.extend(["--request-rate", str(benchmark["request_rate"])])
    need_details = (
        benchmark.get("output_details", False)
        or spec["objective"]["metric"] == "request_goodput_rps"
        or "max_error_rate" in spec.get("slo", {})
    )
    if need_details:
        bench.append("--output-details")
    if benchmark.get("flush_cache", False):
        bench.append("--flush-cache")
    return {"server": server, "benchmark": bench}


def sanitized_environment(spec: dict[str, Any], trial: dict[str, Any] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in spec["execution"].get("env", {}).items():
        env[key] = str(value)
    for key, value in (trial or {}).get("env", {}).items():
        if key not in ALLOWED_ENV:
            raise ValueError(f"trial environment key is not allowed: {key}")
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


def available_gpu_identifiers(spec: dict[str, Any]) -> list[str]:
    """Return stable GPU identifiers suitable for per-trial visibility."""
    configured = spec.get("execution", {}).get("env", {})
    visible = configured.get("CUDA_VISIBLE_DEVICES", configured.get("HIP_VISIBLE_DEVICES"))
    if visible is not None:
        return [item.strip() for item in str(visible).split(",") if item.strip() and item.strip() != "-1"]
    count = accelerator_count(spec)
    return [str(index) for index in range(count)]


def port_available_now(host: str, port: int) -> bool:
    """Return whether no listener currently owns this loopback port."""
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return False
    except (ConnectionRefusedError, TimeoutError, OSError):
        return True


def reserve_worker_ports(host: str, base_port: int, workers: int) -> list[int]:
    """Find a contiguous idle local port range before launching a worker batch.

    This is a preflight reservation, not a lock: every worker also performs
    the normal wait_port_available check immediately before `Popen` to cover
    a competing process that appears after this probe.
    """
    final_start = 65535 - workers + 1
    for start in range(base_port, final_start + 1):
        ports = list(range(start, start + workers))
        if all(port_available_now(host, port) for port in ports):
            return ports
    raise RuntimeError(
        f"could not find {workers} consecutive available loopback ports at or above {base_port}"
    )


def parallel_screening_batch(
    spec: dict[str, Any], trials: list[dict[str, Any]], start_index: int,
    max_devices: int, run_dir: Path, max_wall: float, max_gpu_seconds: float,
    started: float, used_gpu_seconds: float,
    progress: Callable[[dict[str, Any]], None] | None = None,
    total_trials: int | None = None,
) -> list[tuple[dict[str, Any], Path, float, dict[str, Any]]]:
    """Run a resource-packed queue, backfilling GPUs as trials finish."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    identifiers = available_gpu_identifiers(spec)
    slots = identifiers[:max_devices]
    if not slots:
        return []
    host = str(spec.get("execution", {}).get("host", "127.0.0.1"))
    ports = reserve_worker_ports(
        host, int(spec["execution"].get("port", 30000)), len(trials)
    )
    execution_env = spec.get("execution", {}).get("env", {})
    visibility_key = (
        "HIP_VISIBLE_DEVICES"
        if "HIP_VISIBLE_DEVICES" in execution_env and "CUDA_VISIBLE_DEVICES" not in execution_env
        else "CUDA_VISIBLE_DEVICES"
    )
    jobs: list[tuple[int, dict[str, Any], Path, int]] = []
    for offset, trial in enumerate(trials):
        required_devices = configuration_accelerator_count(spec, trial.get("config", {}))
        assigned = deepcopy(trial)
        assigned["_parallel_offset"] = offset
        assigned["_candidate_env"] = deepcopy(assigned.get("env", {}))
        assigned["_port"] = ports[offset]
        assigned["_progress_trial_index"] = start_index + offset + 1
        assigned["_progress_trial_count"] = total_trials or start_index + len(trials)
        trial_dir = run_dir / f"trial-{start_index + offset:03d}-{assigned['name']}"
        jobs.append((offset, assigned, trial_dir, required_devices))

    def run_one(
        assigned: dict[str, Any], trial_dir: Path, remaining: float,
    ) -> tuple[dict[str, Any], Path, float, dict[str, Any]]:
        trial_started = time.monotonic()
        def phase_progress(event: dict[str, Any]) -> None:
            if progress is None:
                return
            progress({
                "trial_index": assigned["_progress_trial_index"],
                "trial_count": assigned["_progress_trial_count"],
                "trial_name": assigned["name"],
                **event,
            })
        result = (
            run_trial(
                spec, assigned, trial_dir, remaining,
                window_progress=phase_progress,
            )
            if progress is not None
            else run_trial(spec, assigned, trial_dir, remaining)
        )
        return assigned, trial_dir, time.monotonic() - trial_started, result

    output: list[tuple[dict[str, Any], Path, float, dict[str, Any]]] = []
    available = list(slots)
    slot_rank = {identifier: index for index, identifier in enumerate(slots)}
    pending = list(jobs)
    running: dict[Any, list[str]] = {}
    trial_limit = min(
        int(spec.get("execution", {}).get("parallel_trials", 1)), len(jobs)
    )
    with ThreadPoolExecutor(max_workers=trial_limit, thread_name_prefix="inferopt-gpu") as pool:
        while pending or running:
            scheduled = False
            while pending and len(running) < trial_limit:
                fit_index = next(
                    (index for index, item in enumerate(pending) if item[3] <= len(available)),
                    None,
                )
                if fit_index is None:
                    break
                offset, assigned, trial_dir, required_devices = pending.pop(fit_index)
                assigned_slots = available[:required_devices]
                del available[:required_devices]
                assigned["env"] = {
                    **assigned.get("env", {}), visibility_key: ",".join(assigned_slots)
                }
                remaining = min(
                    max_wall - (time.monotonic() - started),
                    (max_gpu_seconds - used_gpu_seconds) / max(1, max_devices),
                )
                if progress is not None:
                    progress({
                        "event": "trial_started",
                        "trial_index": start_index + offset + 1,
                        "trial_count": total_trials or start_index + len(trials),
                        "trial_name": assigned["name"],
                        "configuration_name": assigned["configuration_name"],
                        "kind": assigned["kind"],
                        "parallel_workers": min(
                            trial_limit, max(1, len(slots) // max(1, required_devices))
                        ),
                        "assigned_gpus": assigned_slots,
                    })
                future = pool.submit(run_one, assigned, trial_dir, max(1.0, remaining))
                running[future] = assigned_slots
                scheduled = True
            if not running:
                raise RuntimeError("no pending screening trial fits the available GPU resource pool")
            if scheduled and pending and len(running) < trial_limit:
                continue
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                released = running.pop(future)
                available.extend(released)
                available.sort(key=slot_rank.__getitem__)
                output.append(future.result())
    return sorted(output, key=lambda item: int(item[0].get("_parallel_offset", 0)))


def parallel_trial_eligible(spec: dict[str, Any], trial: dict[str, Any]) -> bool:
    """Allow an independent screening trial to share a resource batch.

    A one-repetition baseline can run beside its first candidates on identical
    GPUs because comparisons are computed only after every result is collected.
    Repeated confirmations remain serial. TP/PP/DP trials reserve their full
    topology and can share the host only with trials assigned disjoint devices.
    """
    return (
        int(spec.get("execution", {}).get("parallel_trials", 1)) > 1
        and int(spec.get("search", {}).get("repetitions", 1)) == 1
        and trial.get("kind") in {"baseline", "candidate"}
        and configuration_accelerator_count(spec, trial.get("config", {}))
        <= len(available_gpu_identifiers(spec))
    )


def parallel_candidate_batch(
    spec: dict[str, Any], trials: list[dict[str, Any]], start: int,
    disabled_capabilities: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the next compatible screening batch.

    The first batch may contain the baseline. Optional backend candidates from
    the same capability family remain serialized so one incompatibility can
    trip the circuit breaker before consuming every worker.
    """
    workers = min(
        int(spec.get("execution", {}).get("parallel_trials", 1)),
        len(available_gpu_identifiers(spec)),
    )
    queue_limit = max(workers, workers * 3)
    batch: list[dict[str, Any]] = []
    families: set[str] = set()
    for trial in trials[start:]:
        if not parallel_trial_eligible(spec, trial):
            break
        family = capability_family(trial)
        if family is not None and family in disabled_capabilities:
            break
        # One failing optional backend should not consume every GPU before its
        # capability circuit breaker can take effect.
        if family is not None and family in families:
            break
        batch.append(trial)
        if family is not None:
            families.add(family)
        if len(batch) >= queue_limit:
            break
    return batch


def wait_port_available(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if port_available_now(host, port):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"port {host}:{port} remained in use for {timeout:g} seconds")
        time.sleep(0.5)


def wait_ready(
    url: str, process: subprocess.Popen[Any], timeout: float,
    heartbeat: Callable[[float], None] | None = None,
) -> tuple[bool, str | None]:
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    last_heartbeat = started
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
        now = time.monotonic()
        if heartbeat is not None and now - last_heartbeat >= 30:
            heartbeat(now - started)
            last_heartbeat = now
        time.sleep(1)
    return False, f"health timeout; last error: {last_error}"


def run_logged_subprocess(
    command: list[str], *, cwd: str, env: dict[str, str], log_handle: Any,
    timeout: float, heartbeat: Callable[[float], None] | None = None,
) -> int:
    """Run a logged subprocess with truthful elapsed-time heartbeats."""
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=log_handle,
        stderr=subprocess.STDOUT, text=True,
    )
    started = time.monotonic()
    last_heartbeat = started
    while process.poll() is None:
        now = time.monotonic()
        elapsed = now - started
        if elapsed >= timeout:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            raise subprocess.TimeoutExpired(command, timeout)
        if heartbeat is not None and now - last_heartbeat >= 30:
            heartbeat(elapsed)
            last_heartbeat = now
        time.sleep(1)
    return int(process.returncode or 0)


def startup_failure_detail(detail: str | None, server_log: Path) -> str:
    """Attach the precise terminal SGLang exception to a parent exit code."""
    generic = detail or "server failed health check"
    if not server_log.exists():
        return generic
    lines = server_log.read_text(encoding="utf-8", errors="replace").splitlines()
    markers = (
        "notimplementederror:", "runtimeerror:", "valueerror:",
        "torch.outofmemoryerror:", "cuda out of memory", "unsupported moe_runner_backend",
    )
    precise = next(
        (line.strip() for line in reversed(lines) if any(marker in line.lower() for marker in markers)),
        None,
    )
    return f"{generic}; root cause: {precise}" if precise else generic


def latest_log_message(path: Path, fallback: str) -> str:
    """Return one bounded, sanitized status line from a growing process log."""
    if not path.exists():
        return fallback
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 16384))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return fallback
    for line in reversed(lines):
        normalized = line.strip()
        if normalized and not normalized.startswith("INFO:     127.0.0.1"):
            return normalized[-500:]
    return fallback


def stop_owned_process(
    process: subprocess.Popen[Any], timeout: float,
    heartbeat: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    if process.poll() is not None:
        return {"method": "already_exited", "returncode": process.returncode}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        started = time.monotonic()
        last_heartbeat = started
        while process.poll() is None and time.monotonic() - started < timeout:
            now = time.monotonic()
            if heartbeat is not None and now - last_heartbeat >= 30:
                heartbeat(now - started)
                last_heartbeat = now
            time.sleep(0.1)
        if process.poll() is not None:
            return {"method": "sigterm_process_group", "returncode": process.returncode}
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


def configuration_accelerator_count(spec: dict[str, Any], config: dict[str, Any]) -> int:
    """Return the accelerator count actually requested by one server trial.

    A host can expose four GPUs while a baseline intentionally launches TP=1.
    Charging that baseline four GPU-hours makes a mixed TP search exhaust its
    budget before testing the requested topologies.  TP, PP, and DP each
    create independent ranks; EP/DCP partition those ranks and must not be
    multiplied again.
    """
    visible = accelerator_count(spec)
    product = 1
    for name in ("tp_size", "pp_size", "dp_size"):
        value = config.get(name, 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            return visible
        product *= value
    return min(visible, max(1, product))


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
    # Prefer the precise SGLang capacity error over unrelated warnings such
    # as optional missing modules that may appear earlier in the same log.
    if any(pattern in text for pattern in (
        "loaded weights leave no gpu memory for kv cache",
        "raise --mem-fraction-static above",
        "minimum viable",
        "insufficient memory for kv cache",
    )):
        return "memory_infeasible"
    if any(pattern in text for pattern in (
        "not enough values to unpack",
        "shape mismatch",
        "invalid shape",
        "unsupported model architecture",
        "backend is not supported",
        "not implemented for",
        "notimplementederror",
        "unsupported moe_runner_backend",
        "use --moe-runner-backend",
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
    if "no module named" in text and (
        "modulenotfounderror" in text or "traceback (most recent call last)" in text
    ):
        return "dependency_missing"
    # SIGKILL is only the fallback after the owned process logs have been
    # inspected for a precise dependency, backend, or memory root cause.
    if "server exited during startup with code -9" in text:
        return "process_killed"
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
    declared_provisional = trial.get("provisional_parameter")
    if isinstance(declared_provisional, str) and declared_provisional:
        return f"provisional_parameter:{declared_provisional}"
    name = str(trial.get("configuration_name") or trial.get("name", ""))
    if name.startswith("provisional-"):
        parts = name[len("provisional-"):].rsplit("-", 1)
        parameter = parts[0] if parts else name[len("provisional-"):]
        return f"provisional_parameter:{parameter}"
    if str(trial.get("name", "")).startswith("long-context-prefill-"):
        return "long_context_prefill_capacity"
    config = trial.get("config", {})
    if config.get("enable_torch_compile") is True:
        return "torch_compile"
    algorithm = config.get("speculative_algorithm")
    if isinstance(algorithm, str) and algorithm.strip():
        normalized = algorithm.strip().lower()
        if normalized == "eagle":
            return "mtp_eagle"
        return f"speculative_{normalized}"
    for parameter in (
        "prefill_attention_backend", "decode_attention_backend",
        "attention_backend", "moe_runner_backend",
    ):
        backend = config.get(parameter)
        if isinstance(backend, str) and backend.strip().lower() not in {"", "auto"}:
            return f"{parameter}:{backend.strip().lower()}"
    return None


def capability_failure_reason(
    trial: dict[str, Any], status: dict[str, Any], server_log: Path
) -> dict[str, Any] | None:
    """Identify failures that make every remaining candidate in a family unusable."""
    family = capability_family(trial)
    failure_class = status.get("failure_class")
    shared_backend_failure = failure_class in {"dependency_missing", "backend_incompatible"}
    shared_capacity_failure = (
        family == "long_context_prefill_capacity"
        and failure_class in {"memory_infeasible", "oom", "process_killed"}
    )
    provisional_failure = bool(
        isinstance(family, str) and family.startswith("provisional_parameter:")
        and failure_class not in {"gpu_health", "port_conflict"}
    )
    if family is None or not (
        shared_backend_failure or shared_capacity_failure or provisional_failure
    ):
        return None
    log_text = server_log.read_text(encoding="utf-8", errors="replace") if server_log.exists() else ""
    missing_module = re.search(r"No module named ['\"]([^'\"]+)['\"]", log_text)
    reason = status.get("detail") or "startup failed"
    if missing_module is not None:
        reason = f"missing Python module: {missing_module.group(1)}"
    elif shared_capacity_failure:
        reason = (
            "a coupled long-context prefill bundle exhausted capacity; skip more aggressive "
            "bundles in this run and retain the recorded startup evidence"
        )
    elif provisional_failure:
        reason = (
            "provisional parameter failed its startup/smoke/full screen; disable remaining "
            "values for this parameter in the current framework/model fingerprint"
        )
    return {
        "family": family,
        "failure_class": failure_class,
        "reason": reason,
        "origin_trial": trial["name"],
        "origin_configuration": trial["configuration_name"],
    }


def set_cli_option(argv: list[str], option: str, value: int) -> None:
    """Replace or add one benchmark option while preserving the rendered command."""
    if option not in argv:
        argv.extend([option, str(value)])
        return
    index = argv.index(option)
    argv[index + 1] = str(value)


def remove_cli_option(argv: list[str], option: str) -> None:
    """Remove a two-token CLI option when an unbounded benchmark is required."""
    while option in argv:
        index = argv.index(option)
        del argv[index:index + 2]


def increase_benchmark_request_count(argv: list[str], target_prompts: int) -> int:
    """Raise the effective request count for a normal or generated-prefix workload.

    SGLang's generated-shared-prefix dataset derives its actual request count
    from groups times prompts-per-group, so changing only --num-prompts does
    not lengthen the measurement window.
    """
    if "--gsp-num-groups" not in argv:
        set_cli_option(argv, "--num-prompts", target_prompts)
        return target_prompts
    groups = int(argv[argv.index("--gsp-num-groups") + 1])
    effective = max(groups, math.ceil(target_prompts / groups) * groups)
    set_cli_option(argv, "--num-prompts", effective)
    set_cli_option(argv, "--gsp-prompts-per-group", effective // groups)
    return effective


def provisional_smoke_plan(
    spec: dict[str, Any], trial: dict[str, Any], benchmark: list[str],
) -> dict[str, Any] | None:
    names = set(spec.get("search", {}).get("provisional_parameter_candidates", []))
    if trial.get("configuration_name") not in names:
        return None
    command = list(benchmark)
    full_prompts = int(command[command.index("--num-prompts") + 1])
    requested = min(
        16, max(4, int(spec.get("hardware", {}).get("gpus_per_host", 1)) * 4)
    )
    effective = increase_benchmark_request_count(command, requested)
    set_cli_option(command, "--warmup-requests", min(2, effective))
    return {
        "command": command,
        "requested_num_prompts": requested,
        "effective_num_prompts": effective,
        "full_benchmark_num_prompts": full_prompts,
        "policy": "same resident server; bounded smoke precedes the full evidence window",
    }


def _find_positive_integer(value: Any, key: str) -> int | None:
    """Find a positive integer key in SGLang's version-dependent JSON shape."""
    if isinstance(value, dict):
        candidate = value.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            return candidate
        for nested in value.values():
            found = _find_positive_integer(nested, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_positive_integer(nested, key)
            if found is not None:
                return found
    return None


def resolved_server_capacity(host: str, port: int, server_log_path: Path) -> dict[str, Any]:
    """Read SGLang's resolved admission limit after startup.

    ``max_running_requests`` is calculated from the loaded model and KV pool;
    it cannot be inferred reliably from a task's requested client concurrency.
    Different SGLang revisions expose the value through different endpoints, so
    retain every successful response and use the server log as a final fallback.
    """
    responses: list[dict[str, Any]] = []
    for endpoint in ("/server_info", "/get_server_info"):
        url = f"http://{host}:{port}{endpoint}"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            capacity = _find_positive_integer(payload, "max_running_requests")
            responses.append({
                "endpoint": endpoint,
                "capacity": capacity,
                "response_type": type(payload).__name__,
                "top_level_keys": sorted(payload)[:32] if isinstance(payload, dict) else None,
            })
            if capacity is not None:
                return {
                    "source": endpoint,
                    "max_running_requests": capacity,
                    "responses": responses,
                }
            # Current SGLang exposes the configured ServerArgs here. A null
            # value means auto-admission is in effect; querying its deprecated
            # alias cannot make it concrete, so use the post-init log instead.
            break
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            responses.append({"endpoint": endpoint, "error": type(exc).__name__})
    log_text = server_log_path.read_text(encoding="utf-8", errors="replace")
    patterns = (
        r"max_running_requests\s*[=:]\s*(\d+)",
        r"max running requests\s*[=:]\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, log_text, flags=re.IGNORECASE)
        if match:
            return {
                "source": "server_log",
                "max_running_requests": int(match.group(1)),
                "responses": responses,
            }
    return {"source": None, "max_running_requests": None, "responses": responses}


def run_trial(
    spec: dict[str, Any], trial: dict[str, Any], trial_dir: Path, time_limit_sec: float,
    window_progress: Callable[[dict[str, Any]], None] | None = None,
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
    port = int(trial.get("_port", execution.get("port", 30000)))
    env = sanitized_environment(spec, trial)
    process: subprocess.Popen[Any] | None = None
    started = time.monotonic()
    previous_signal_handlers: dict[int, Any] = {}

    def phase(name: str, message: str) -> None:
        if window_progress is not None:
            window_progress({
                "event": "trial_phase", "phase": name, "message": message,
            })

    def interrupt_trial(_signum: int, _frame: Any) -> None:
        # Raise into the try/finally below so the owned SGLang process group is
        # stopped before the autotune controller exits.
        raise KeyboardInterrupt

    install_signal_handlers = threading.current_thread() is threading.main_thread()
    if install_signal_handlers:
        for interrupt_signal in (signal.SIGINT, signal.SIGTERM):
            previous_signal_handlers[interrupt_signal] = signal.getsignal(interrupt_signal)
            signal.signal(interrupt_signal, interrupt_trial)
    try:
        phase("port_wait", f"waiting for {host}:{port} to become available")
        wait_port_available(host, port, float(execution.get("shutdown_timeout_sec", 30)))
        with server_log_path.open("w", encoding="utf-8") as server_log:
            phase("server_launch", "starting the owned SGLang server process")
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
                heartbeat=lambda elapsed: phase(
                    "server_startup",
                    f"{latest_log_message(server_log_path, 'waiting for model load/KV allocation/CUDA Graph readiness')}; "
                    f"elapsed={elapsed:.0f}s",
                ),
            )
            if not ready:
                raise RuntimeError(startup_failure_detail(detail, server_log_path))
            phase("server_ready", "health check passed; preparing benchmark")
            status["state"] = "benchmarking"
            status["ready_at"] = now_iso()
            benchmark = list(manifest["benchmark"])
            if spec["benchmark"].get("auto_max_concurrency", False):
                capacity = resolved_server_capacity(host, port, server_log_path)
                write_json(trial_dir / "runtime-capacity.json", capacity)
                resolved_concurrency = capacity.get("max_running_requests")
                calibration_session = spec["benchmark"].get("calibration_session") or {}
                if not isinstance(resolved_concurrency, int) or resolved_concurrency <= 0:
                    fallback = calibration_session.get("fallback_max_concurrency")
                    if not isinstance(fallback, int) or isinstance(fallback, bool) or fallback <= 0:
                        raise RuntimeError(
                            "SGLang did not expose max_running_requests; provide "
                            "calibration.fallback_max_concurrency to permit an explicit client-cap probe"
                        )
                    resolved_concurrency = fallback
                    capacity_source = "task.calibration.fallback_max_concurrency"
                else:
                    capacity_source = capacity.get("source")
                initial_unbounded_probe = bool(
                    (spec["benchmark"].get("calibration_session") or {}).get(
                        "initial_unbounded_probe", False
                    )
                )
                if initial_unbounded_probe:
                    remove_cli_option(benchmark, "--max-concurrency")
                else:
                    set_cli_option(benchmark, "--max-concurrency", resolved_concurrency)
                # The initial task may have been written for a small online
                # load.  A capacity probe needs a real backlog at the resolved
                # admission limit before duration-based expansion can judge it.
                request_waves = int(spec["benchmark"].get("p99_request_waves", 0)) or 5
                effective_prompts = increase_benchmark_request_count(
                    benchmark,
                    max(
                        int(spec["benchmark"]["num_prompts"]),
                        resolved_concurrency * request_waves,
                    ),
                )
                status["resolved_server_max_running_requests"] = resolved_concurrency
                status["resolved_client_max_concurrency"] = resolved_concurrency
                status["resolved_effective_num_prompts"] = effective_prompts
                status["resolved_capacity_source"] = capacity_source
            write_json(trial_dir / "resolved-commands.json", {
                "server": manifest["server"], "benchmark": benchmark,
            })
            write_json(trial_dir / "status.json", status)
            remaining = time_limit_sec - (time.monotonic() - started)
            if remaining <= 0:
                raise RuntimeError("trial time budget exhausted before benchmark")
            def run_benchmark_window(
                command: list[str], raw_path: Path, minimum_duration: float, label: str,
                *, require_tail_gate: bool = True,
            ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
                command[command.index("--output-file") + 1] = str(raw_path)
                attempts: list[dict[str, Any]] = []
                max_steady_state_attempts = 5
                for attempt_index in range(1, max_steady_state_attempts + 1):
                    remaining = time_limit_sec - (time.monotonic() - started)
                    if remaining <= 0:
                        raise RuntimeError("trial time budget exhausted during benchmark")
                    with benchmark_log_path.open("a", encoding="utf-8") as benchmark_log:
                        benchmark_log.write(
                            f"\n=== {label} benchmark attempt {attempt_index} ===\n"
                        )
                        prompts = int(command[command.index("--num-prompts") + 1])
                        phase(
                            "benchmark",
                            f"{label} attempt {attempt_index}/{max_steady_state_attempts}; num_prompts={prompts}",
                        )
                        returncode = run_logged_subprocess(
                            command, cwd=spec["repository"], env=env,
                            log_handle=benchmark_log,
                            timeout=min(
                                float(execution.get("benchmark_timeout_sec", 1800)), remaining
                            ),
                            heartbeat=lambda elapsed: phase(
                                "benchmark",
                                f"{label} attempt {attempt_index}/{max_steady_state_attempts} running; "
                                f"num_prompts={prompts}, elapsed={elapsed:.0f}s",
                            ),
                        )
                    if returncode != 0:
                        raise RuntimeError(f"{label} benchmark exited with code {returncode}")
                    effective_concurrency = (
                        int(command[command.index("--max-concurrency") + 1])
                        if "--max-concurrency" in command else None
                    )
                    window_summary = summarize_jsonl(
                        raw_path, spec, effective_concurrency=effective_concurrency
                    )
                    validity = window_summary["measurement_validity"]
                    validity["minimum_duration_sec"] = minimum_duration
                    duration = validity.get("duration_sec")
                    validity["duration_gate_passed"] = (
                        isinstance(duration, (int, float)) and duration >= minimum_duration
                    )
                    attempts.append({
                        "attempt": attempt_index,
                        "num_prompts": int(command[command.index("--num-prompts") + 1]),
                        "measurement_validity": window_summary["measurement_validity"],
                        "result_file": raw_path.name,
                    })
                    if (
                        window_summary["measurement_validity"]["duration_gate_passed"]
                        and (
                            not require_tail_gate
                            or window_summary["measurement_validity"].get(
                                "tail_sample_gate_passed", True
                            )
                        )
                    ):
                        phase(
                            "benchmark_valid",
                            f"{label} reached measurement gates on attempt {attempt_index}; "
                            f"duration={duration:.2f}s, requests={validity.get('request_count')}",
                        )
                        return window_summary, attempts
                    if attempt_index == max_steady_state_attempts:
                        raise RuntimeError(
                            f"{label} benchmark did not reach the duration and p99-wave validity "
                            f"gates after {max_steady_state_attempts} attempts"
                        )
                    short_path = trial_dir / (
                        f"{raw_path.stem}-short-attempt-{attempt_index}{raw_path.suffix}"
                    )
                    raw_path.replace(short_path)
                    attempts[-1]["result_file"] = short_path.name
                    duration = validity.get("duration_sec") or 0
                    current_prompts = int(command[command.index("--num-prompts") + 1])
                    multiplier = (
                        max(2.0, (minimum_duration / duration) * 1.2)
                        if duration > 0 else 2.0
                    )
                    next_prompts = max(
                        current_prompts + 1,
                        math.ceil(current_prompts * multiplier),
                        int(validity.get("minimum_request_count_for_tail") or 0),
                    )
                    effective_prompts = increase_benchmark_request_count(command, next_prompts)
                    attempts[-1]["next_effective_num_prompts"] = effective_prompts
                    phase(
                        "benchmark_expand",
                        f"{label} did not reach measurement gates; expanding requests "
                        f"from {current_prompts} to {effective_prompts}",
                    )
                raise RuntimeError(f"{label} benchmark did not complete")

            smoke_plan = provisional_smoke_plan(spec, trial, benchmark)
            if smoke_plan is not None:
                phase(
                    "provisional_smoke",
                    f"running {smoke_plan['effective_num_prompts']} error-gate requests before the full benchmark",
                )
                smoke_summary, smoke_attempts = run_benchmark_window(
                    smoke_plan["command"],
                    trial_dir / "provisional-smoke.jsonl",
                    0.0,
                    "provisional smoke",
                    require_tail_gate=False,
                )
                smoke_metrics = smoke_summary.get("metrics", {})
                if (
                    not smoke_metrics
                    or float(smoke_metrics.get("error_rate", 1.0) or 0.0) > 0
                    or int(smoke_summary.get("measurement_validity", {}).get("request_count", 0) or 0) <= 0
                ):
                    raise RuntimeError(
                        "provisional parameter smoke test did not complete error-free requests"
                    )
                status["provisional_smoke"] = {
                    "passed": True,
                    "num_prompts": smoke_plan["effective_num_prompts"],
                    "full_benchmark_num_prompts": smoke_plan["full_benchmark_num_prompts"],
                    "metrics": smoke_metrics,
                    "attempts": smoke_attempts,
                    "policy": smoke_plan["policy"],
                }
                write_json(trial_dir / "provisional-smoke-summary.json", smoke_summary)
                write_json(trial_dir / "status.json", status)
                phase("provisional_smoke", "smoke passed; starting the full evidence window")

            minimum_duration = float(spec["benchmark"].get("min_measurement_seconds", 0))
            summaries: list[dict[str, Any]] = []
            attempts_by_repeat: list[dict[str, Any]] = []
            calibration_session = spec["benchmark"].get("calibration_session")
            if isinstance(calibration_session, dict):
                strategy = calibration_session["strategy"]
                request_waves = int(calibration_session.get("request_waves", 5))
                floor = int(calibration_session.get("min_concurrency", 1))
                configured_steps = calibration_session.get("max_steps")
                initial_unbounded_probe = bool(
                    calibration_session.get("initial_unbounded_probe", False)
                )
                resolved = status.get("resolved_client_max_concurrency")
                configured_points = calibration_session.get("concurrencies", [])
                if strategy == "adaptive_slo":
                    if not isinstance(resolved, int) or resolved <= 0:
                        raise RuntimeError("adaptive SLO calibration requires resolved server capacity")
                    pending_concurrencies = [resolved]
                    max_steps = int(configured_steps or max(3, math.ceil(math.log2(resolved)) + 2))
                else:
                    pending_concurrencies = [
                        int(value) for value in configured_points
                        if isinstance(value, int) and not isinstance(value, bool) and value > 0
                    ]
                    if not pending_concurrencies:
                        raise RuntimeError("fixed calibration curve has no positive concurrency points")
                    max_steps = int(configured_steps or len(pending_concurrencies))
                attempted_concurrencies: set[int] = set()
                highest_passing: int | None = None
                lowest_failing: int | None = None
                base_prompts = int(spec["benchmark"]["num_prompts"])
                while pending_concurrencies and len(summaries) < max_steps:
                    concurrency = pending_concurrencies.pop(0)
                    if concurrency in attempted_concurrencies:
                        continue
                    attempted_concurrencies.add(concurrency)
                    window_index = len(summaries) + 1
                    if window_progress is not None:
                        window_progress({
                            "event": "trial_started", "trial_index": window_index,
                            "trial_count": max_steps,
                            "trial_name": f"capacity-c{concurrency}",
                            "configuration_name": trial["configuration_name"],
                            "kind": trial["kind"],
                        })
                    calibration_benchmark = list(benchmark)
                    if initial_unbounded_probe and window_index == 1:
                        remove_cli_option(calibration_benchmark, "--max-concurrency")
                    else:
                        set_cli_option(calibration_benchmark, "--max-concurrency", concurrency)
                    effective_prompts = increase_benchmark_request_count(
                        calibration_benchmark, max(base_prompts, concurrency * request_waves)
                    )
                    raw_path = trial_dir / f"calibration-c{concurrency:06d}.jsonl"
                    summary, attempts = run_benchmark_window(
                        calibration_benchmark,
                        raw_path,
                        minimum_duration,
                        f"calibration concurrency={concurrency}",
                    )
                    summary.update({
                        "repeat_index": len(summaries),
                        "calibration_concurrency": concurrency,
                        "effective_num_prompts": effective_prompts,
                    })
                    summaries.append(summary)
                    if window_progress is not None:
                        window_progress({
                            "event": "trial_finished", "trial_index": window_index,
                            "trial_count": max_steps,
                            "trial_name": f"capacity-c{concurrency}", "ok": True,
                            "metrics": summary["metrics"],
                            "slo_passed": summary["slo"].get("passed"),
                        })
                    attempts_by_repeat.append({
                        "repeat_index": len(summaries) - 1,
                        "calibration_concurrency": concurrency,
                        "attempts": attempts,
                    })
                    if strategy != "adaptive_slo":
                        if (
                            calibration_session.get("stop_on_slo_failure", False)
                            and not summary.get("slo", {}).get("passed", False)
                        ):
                            break
                        continue
                    passed = bool(summary.get("slo", {}).get("passed"))
                    if passed:
                        highest_passing = max(highest_passing or concurrency, concurrency)
                    else:
                        lowest_failing = min(lowest_failing or concurrency, concurrency)
                    next_concurrency: int | None = None
                    if highest_passing is None and not passed:
                        fallback = max(floor, concurrency // 2)
                        if fallback < concurrency:
                            next_concurrency = fallback
                    elif highest_passing is not None and lowest_failing is not None:
                        if lowest_failing - highest_passing > 1:
                            next_concurrency = (lowest_failing + highest_passing) // 2
                    if (
                        next_concurrency is not None
                        and next_concurrency not in attempted_concurrencies
                        and next_concurrency not in pending_concurrencies
                    ):
                        pending_concurrencies.insert(0, next_concurrency)
            else:
                repeat_indices = trial.get("repeat_indices")
                if not isinstance(repeat_indices, list) or not repeat_indices:
                    repeat_indices = [int(trial.get("repeat_index", 0))]
                repeat_indices = list(repeat_indices)
                initial_repeat_count = len(repeat_indices)
                search = spec.get("search", {})
                adaptive_cv_threshold = search.get("adaptive_confirmation_cv_pct")
                adaptive_max_repetitions = int(
                    search.get(
                        "adaptive_confirmation_max_repetitions", initial_repeat_count
                    )
                )
                adaptive_minimum_duration = float(
                    search.get(
                        "adaptive_confirmation_min_measurement_seconds", minimum_duration
                    )
                )
                adaptive_initial_cv: float | None = None
                repeat_position = 0
                while repeat_position < len(repeat_indices):
                    repeat_index = repeat_indices[repeat_position]
                    adaptive_window = repeat_position >= initial_repeat_count
                    repeated_benchmark = list(benchmark)
                    raw_path = (
                        trial_dir / f"result-r{int(repeat_index) + 1:02d}.jsonl"
                        if len(repeat_indices) > 1 else trial_dir / "result.jsonl"
                    )
                    summary, attempts = run_benchmark_window(
                        repeated_benchmark,
                        raw_path,
                        adaptive_minimum_duration if adaptive_window else minimum_duration,
                        f"measurement r{int(repeat_index) + 1:02d}",
                    )
                    summary["repeat_index"] = int(repeat_index)
                    summary["adaptive_confirmation_window"] = adaptive_window
                    summaries.append(summary)
                    attempts_by_repeat.append({
                        "repeat_index": int(repeat_index),
                        "attempts": attempts,
                    })
                    # A slow first window may have expanded its request count to
                    # satisfy the duration gate. Start later windows at that same
                    # floor instead of rediscovering it through another short run.
                    effective_prompts = int(
                        repeated_benchmark[repeated_benchmark.index("--num-prompts") + 1]
                    )
                    increase_benchmark_request_count(benchmark, effective_prompts)
                    repeat_position += 1
                    if (
                        repeat_position == initial_repeat_count
                        and isinstance(adaptive_cv_threshold, (int, float))
                        and initial_repeat_count >= 2
                        and adaptive_max_repetitions > initial_repeat_count
                    ):
                        adaptive_initial_cv = objective_cv_for_summaries(
                            summaries, spec["objective"]["metric"]
                        )
                        if (
                            adaptive_initial_cv is not None
                            and adaptive_initial_cv > float(adaptive_cv_threshold)
                        ):
                            next_repeat = max(int(value) for value in repeat_indices) + 1
                            repeat_indices.extend(
                                range(
                                    next_repeat,
                                    next_repeat + adaptive_max_repetitions - initial_repeat_count,
                                )
                            )
            summary = summaries[0]
            if not isinstance(calibration_session, dict):
                adaptive_triggered = len(summaries) > initial_repeat_count
                adaptive_evidence = {
                    "enabled": isinstance(adaptive_cv_threshold, (int, float)),
                    "triggered": adaptive_triggered,
                    "initial_objective_cv_pct": adaptive_initial_cv,
                    "trigger_cv_pct": adaptive_cv_threshold,
                    "initial_repetitions": initial_repeat_count,
                    "completed_repetitions": len(summaries),
                    "extended_min_measurement_seconds": (
                        adaptive_minimum_duration if adaptive_triggered else None
                    ),
                }
                for repeated_summary in summaries:
                    repeated_summary["adaptive_confirmation"] = adaptive_evidence
            write_json(trial_dir / "benchmark-attempts.json", attempts_by_repeat)

            reference_prompts = spec["benchmark"].get("baseline_reference_num_prompts")
            if trial["kind"] == "baseline" and isinstance(reference_prompts, int):
                reference_duration = float(
                    spec["benchmark"].get(
                        "baseline_reference_min_measurement_seconds", minimum_duration
                    )
                )
                effective_screening_prompts = int(
                    benchmark[benchmark.index("--num-prompts") + 1]
                )
                screening_duration = summary["measurement_validity"].get("duration_sec")
                reuse_screening_window = (
                    effective_screening_prompts == reference_prompts
                    and isinstance(screening_duration, (int, float))
                    and screening_duration >= reference_duration
                    and "--flush-cache" in benchmark
                )
                if reuse_screening_window:
                    reference_summary = summary
                    reference_attempts = attempts
                    effective_reference_prompts = effective_screening_prompts
                else:
                    reference_benchmark = list(benchmark)
                    increase_benchmark_request_count(reference_benchmark, reference_prompts)
                    if "--flush-cache" not in reference_benchmark:
                        reference_benchmark.append("--flush-cache")
                    reference_summary, reference_attempts = run_benchmark_window(
                        reference_benchmark,
                        trial_dir / "confirmation-reference.jsonl",
                        reference_duration,
                        "confirmation-reference",
                    )
                    write_json(
                        trial_dir / "confirmation-reference-command.json",
                        {"benchmark": reference_benchmark},
                    )
                    effective_reference_prompts = int(
                        reference_benchmark[reference_benchmark.index("--num-prompts") + 1]
                    )
                summary["confirmation_reference"] = {
                    "metrics": reference_summary["metrics"],
                    "slo": reference_summary["slo"],
                    "measurement_validity": reference_summary["measurement_validity"],
                    "num_prompts": effective_reference_prompts,
                    "dataset_name": spec["benchmark"].get("dataset_name", "random-ids"),
                    "reused_screening_window": reuse_screening_window,
                }
                write_json(trial_dir / "confirmation-reference-attempts.json", reference_attempts)
                write_json(trial_dir / "confirmation-reference-summary.json", reference_summary)
            runtime_observations = summarize_sglang_log(
                server_log_path.read_text(encoding="utf-8", errors="replace")
            )
            for repeated_summary in summaries:
                repeated_summary["runtime_observations"] = runtime_observations
            write_json(trial_dir / "runtime-observations.json", runtime_observations)
            write_json(trial_dir / "summary.json", summary)
            if len(summaries) > 1:
                write_json(trial_dir / "summaries.json", summaries)
            status.update({
                "state": "completed",
                "completed_at": now_iso(),
                "elapsed_sec": time.monotonic() - started,
                "slo_passed": all(
                    repeated_summary["slo"]["passed"]
                    for repeated_summary in summaries
                ),
                "measurement_windows": len(summaries),
            })
            if not isinstance(calibration_session, dict):
                status["adaptive_confirmation"] = adaptive_evidence
            return {"ok": True, "summary": summary, "summaries": summaries, "status": status}
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
            phase("cleanup", "stopping the owned server and releasing accelerator resources")
            status["shutdown"] = stop_owned_process(
                process, float(execution.get("shutdown_timeout_sec", 30)),
                heartbeat=lambda elapsed: phase(
                    "cleanup", f"waiting for server exit/resource release; elapsed={elapsed:.0f}s"
                ),
            )
            # waitpid(-1) is process-global. A worker thread must not reap a
            # sibling worker's SGLang descendants while parallel trials run.
            status["shutdown"]["reaped_descendants"] = (
                reap_exited_children()
                if threading.current_thread() is threading.main_thread()
                else []
            )
            phase(
                "cleanup",
                f"server cleanup finished via {status['shutdown'].get('method')}",
            )
        status["elapsed_sec"] = time.monotonic() - started
        write_json(trial_dir / "status.json", status)
        if install_signal_handlers:
            for interrupt_signal, previous in previous_signal_handlers.items():
                signal.signal(interrupt_signal, previous)


def summarize_jsonl(
    path: Path, spec: dict[str, Any], *, effective_concurrency: int | None = None,
) -> dict[str, Any]:
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
    p99_slos = sorted(
        key for key in spec.get("slo", {})
        if isinstance(key, str) and key.startswith("p99_")
    )
    p99_request_waves = int(spec["benchmark"].get("p99_request_waves", 0))
    measurement_concurrency = (
        effective_concurrency
        if isinstance(effective_concurrency, int) and effective_concurrency > 0
        else int(spec["benchmark"].get("max_concurrency", 1))
    )
    if p99_slos and p99_request_waves > 0:
        minimum_tail_requests = measurement_concurrency * p99_request_waves
        tail_requirement_reason = "concurrency_waves"
    else:
        # Read older execution specs without changing their archived contract.
        min_tail_samples = int(spec["benchmark"].get("min_tail_samples", 0))
        boundary_tail_samples = int(
            spec["benchmark"].get("near_slo_tail_samples", min_tail_samples)
        )
        boundary_margin_pct = float(spec["benchmark"].get("near_slo_margin_pct", 10))
        near_boundary = any(
            isinstance(summary["metrics"].get(key), (int, float))
            and isinstance(spec.get("slo", {}).get(key), (int, float))
            and spec["slo"][key] > 0
            and abs(float(summary["metrics"][key]) - float(spec["slo"][key]))
            / float(spec["slo"][key]) * 100 <= boundary_margin_pct
            for key in p99_slos
        )
        required_tail_samples = boundary_tail_samples if near_boundary else min_tail_samples
        minimum_tail_requests = required_tail_samples * 100 if p99_slos else 0
        tail_requirement_reason = "legacy_tail_samples" if p99_slos else "not_applicable"
    request_count = completed if isinstance(completed, (int, float)) else None
    summary["measurement_validity"] = {
        "purpose": "sample-validity gate only; not an SLO or optimization objective",
        "request_count": request_count,
        "duration_sec": duration,
        "duration_source": duration_source,
        "minimum_duration_sec": spec["benchmark"].get("min_measurement_seconds", 0),
        "duration_gate_passed": duration is not None and duration >= float(spec["benchmark"].get("min_measurement_seconds", 0)),
        "p99_slos": p99_slos,
        "p99_request_waves": p99_request_waves if p99_slos else 0,
        "measurement_concurrency": measurement_concurrency if p99_slos else None,
        "minimum_tail_samples": (
            math.ceil(minimum_tail_requests / 100) if p99_slos else 0
        ),
        "minimum_request_count_for_tail": minimum_tail_requests,
        "tail_requirement_reason": tail_requirement_reason,
        "tail_sample_gate_passed": (
            not p99_slos
            or request_count is not None and request_count >= minimum_tail_requests
        ),
    }
    return summary


def objective_cv_for_summaries(
    summaries: list[dict[str, Any]], objective_metric: str,
) -> float | None:
    """Return population CV for completed benchmark-window objective values."""
    values = [
        float(summary["metrics"][objective_metric])
        for summary in summaries
        if isinstance(summary.get("metrics", {}).get(objective_metric), (int, float))
    ]
    if len(values) < 2:
        return None
    sample_mean = mean(values)
    return pstdev(values) / abs(sample_mean) * 100 if sample_mean else None


def outlier_retry_required(
    spec: dict[str, Any], trial: dict[str, Any], baseline_value: Any,
    candidate_value: Any, *, pending_parallel_results: bool = False,
) -> bool:
    """Return whether an extreme one-pass screen deserves one bounded retry."""
    if spec.get("search", {}).get("bayesian_sequential", False):
        return False
    if int(spec.get("search", {}).get("repetitions", 1)) != 1:
        return False
    if trial.get("_outlier_retry") or pending_parallel_results:
        return False
    if not (
        isinstance(baseline_value, (int, float)) and not isinstance(baseline_value, bool)
        and baseline_value != 0
        and isinstance(candidate_value, (int, float)) and not isinstance(candidate_value, bool)
    ):
        return False
    threshold = float(spec.get("search", {}).get("outlier_retry_pct", 15.0))
    return abs(
        (float(candidate_value) - float(baseline_value)) / float(baseline_value) * 100
    ) >= threshold


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
            len(completed) >= expected
            and all(row.get("slo", {}).get("passed", False) for row in completed)
        )
        stable = (
            len(completed) >= expected
            and objective_cv_pct is not None
            and objective_cv_pct <= max_cv_pct
        )
        first = group[0]
        aggregate = {
            "configuration_name": configuration_name,
            "kind": first["kind"],
            "config": first["config"],
            "env": first.get("env", {}),
            **(
                {"registry_candidate_id": first["registry_candidate_id"]}
                if isinstance(first.get("registry_candidate_id"), str) else {}
            ),
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
        }
        reference = next(
            (
                row.get("confirmation_reference") for row in completed
                if isinstance(row.get("confirmation_reference"), dict)
            ),
            None,
        )
        if reference is not None:
            aggregate["confirmation_reference"] = deepcopy(reference)
        aggregates.append(aggregate)
    return aggregates


def evaluate_aggregates(
    aggregates: list[dict[str, Any]], spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    baseline = next((item for item in aggregates if item["kind"] == "baseline"), None)
    screening_winner: dict[str, Any] | None = None
    confirmed_winner: dict[str, Any] | None = None
    min_confirm_repetitions = int(spec["search"].get("min_confirm_repetitions", 2))
    if baseline is not None:
        baseline["confirmed"] = (
            baseline["eligible_for_confirmation"]
            and baseline["completed_repetitions"] >= min_confirm_repetitions
        )
        baseline["evidence_state"] = (
            "confirmed" if baseline["confirmed"] else "screening_evidence_only"
        )
        baseline["rejection_reasons"] = [] if baseline["confirmed"] else [
            "confirmation_pending"
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
        objective_metric = spec["objective"]["metric"]
        confidence = objective_improvement_confidence_interval(
            baseline.get("metric_samples", {}).get(objective_metric, []),
            item.get("metric_samples", {}).get(objective_metric, []),
            spec["objective"]["direction"],
        )
        comparison["confidence_interval"] = confidence
        bayesian_enabled = bool(spec["search"].get("bayesian_sequential", False))
        bayesian = (
            sequential_decision_from_samples(
                baseline.get("metric_samples", {}).get(objective_metric, []),
                item.get("metric_samples", {}).get(objective_metric, []),
                objective_metric=objective_metric,
                minimum_improvement_pct=float(comparison["minimum_improvement_pct"]),
                direction=spec["objective"]["direction"],
                candidate_slo_passes=[item.get("all_repetitions_slo_passed", False)]
                * int(item.get("completed_repetitions", 0)),
                min_blocks=int(spec["search"].get("bayesian_min_blocks", 2)),
                max_blocks=int(spec["search"].get("bayesian_max_blocks", 6)),
                accept_probability=float(
                    spec["search"].get("bayesian_accept_probability", 0.95)
                ),
                reject_probability=float(
                    spec["search"].get("bayesian_reject_probability", 0.05)
                ),
                prior_mean_pct=float(
                    spec["search"].get("bayesian_prior_mean_pct", 0.0)
                ),
                prior_strength=float(
                    spec["search"].get("bayesian_prior_strength", 0.01)
                ),
            )
            if bayesian_enabled else None
        )
        comparison["bayesian_posterior"] = bayesian
        statistical_gate_required = (
            baseline["completed_repetitions"] >= 2
            and item["completed_repetitions"] >= 2
        )
        statistically_positive = (
            bayesian["probability_improvement_gt_zero"] >= 0.99
            if bayesian_enabled and bayesian is not None
            else confidence is not None and confidence["lower_pct"] > 0
        )
        practically_significant = (
            bayesian is not None and bayesian["action"] == "accept"
            if bayesian_enabled
            else confidence is not None
            and confidence["lower_pct"] >= float(comparison["minimum_improvement_pct"])
        )
        comparison["statistical_gate_required"] = statistical_gate_required
        comparison["statistically_positive"] = statistically_positive
        comparison["practically_significant"] = practically_significant
        comparison["noise_limited"] = bool(
            bayesian is not None and bayesian["action"] == "inconclusive"
            if bayesian_enabled
            else confidence is not None
            and confidence["lower_pct"] <= 0 <= confidence["upper_pct"]
        )
        item["comparison"] = comparison
        item["screening_accepted"] = comparison["accepted"]
        item["confirmed"] = (
            comparison["accepted"]
            and baseline["eligible_for_confirmation"]
            and item["eligible_for_confirmation"]
            and baseline["completed_repetitions"] >= min_confirm_repetitions
            and item["completed_repetitions"] >= min_confirm_repetitions
            and (not statistical_gate_required or practically_significant)
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
            rejection_reasons.append("confirmation_pending")
        elif item["completed_repetitions"] < min_confirm_repetitions:
            rejection_reasons.append("insufficient_confirmation_repetitions")
        if statistical_gate_required and not statistically_positive:
            rejection_reasons.append("objective_difference_not_statistically_resolved")
        elif statistical_gate_required and not practically_significant:
            rejection_reasons.append("minimum_improvement_not_statistically_established")
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


def objective_improvement_confidence_interval(
    baseline_samples: list[float], candidate_samples: list[float], direction: str,
) -> dict[str, Any] | None:
    """Return a conservative Welch-style 95% CI for relative improvement."""
    if len(baseline_samples) < 2 or len(candidate_samples) < 2:
        return None
    base_mean = mean(baseline_samples)
    candidate_mean = mean(candidate_samples)
    if base_mean == 0:
        return None

    def sample_variance(values: list[float]) -> float:
        center = mean(values)
        return sum((value - center) ** 2 for value in values) / (len(values) - 1)

    base_term = sample_variance(baseline_samples) / len(baseline_samples)
    candidate_term = sample_variance(candidate_samples) / len(candidate_samples)
    standard_error = math.sqrt(base_term + candidate_term)
    denominator = (
        base_term**2 / (len(baseline_samples) - 1)
        + candidate_term**2 / (len(candidate_samples) - 1)
    )
    degrees_of_freedom = (
        (base_term + candidate_term) ** 2 / denominator if denominator > 0 else math.inf
    )
    critical_values = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042,
    }
    if math.isinf(degrees_of_freedom):
        critical = 1.960
    else:
        conservative_df = max(1, math.floor(degrees_of_freedom))
        eligible_df = [value for value in critical_values if value <= conservative_df]
        critical = critical_values[max(eligible_df)] if eligible_df else critical_values[1]
    raw_delta = candidate_mean - base_mean
    oriented_delta = raw_delta if direction == "maximize" else -raw_delta
    scale = abs(base_mean)
    margin = critical * standard_error
    return {
        "method": "welch_t_approximation",
        "confidence_level": 0.95,
        "baseline_samples": len(baseline_samples),
        "candidate_samples": len(candidate_samples),
        "degrees_of_freedom": degrees_of_freedom,
        "critical_value": critical,
        "point_pct": oriented_delta / scale * 100,
        "lower_pct": (oriented_delta - margin) / scale * 100,
        "upper_pct": (oriented_delta + margin) / scale * 100,
    }


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
    noise_limited = [
        item for item in aggregates
        if item.get("kind") == "candidate"
        and item.get("comparison", {}).get("noise_limited")
        and isinstance(item.get("comparison", {}).get("improvement_pct"), (int, float))
        and item["comparison"]["improvement_pct"] > 0
    ]
    if noise_limited:
        best = max(
            noise_limited,
            key=lambda item: float(item["comparison"]["improvement_pct"]),
        )
        interval = best["comparison"]["confidence_interval"]
        return (
            None,
            "noise_limited",
            "best measured candidate is not statistically distinguishable from baseline; "
            f"95% CI [{interval['lower_pct']:.3f}%, {interval['upper_pct']:.3f}%] crosses zero",
        )
    effect_size_uncertain = [
        item for item in aggregates
        if item.get("kind") == "candidate"
        and isinstance(item.get("comparison", {}).get("confidence_interval"), dict)
        and item.get("comparison", {}).get("statistically_positive")
        and not item.get("comparison", {}).get("practically_significant")
        and isinstance(item.get("comparison", {}).get("improvement_pct"), (int, float))
        and item["comparison"]["improvement_pct"] > 0
    ]
    if effect_size_uncertain:
        best = max(
            effect_size_uncertain,
            key=lambda item: float(item["comparison"]["improvement_pct"]),
        )
        interval = best["comparison"]["confidence_interval"]
        return (
            None,
            "effect_size_uncertain",
            "candidate is statistically faster, but the confidence interval does not "
            "establish the configured minimum improvement; "
            f"95% CI [{interval['lower_pct']:.3f}%, {interval['upper_pct']:.3f}%]",
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
    reference = spec["search"].get("reference_baseline")
    if not any(item["kind"] == "baseline" for item in aggregates) and isinstance(reference, dict):
        metrics = deepcopy(reference["metrics"])
        summary = {"schema_version": 1, "metrics": metrics}
        summary["slo"] = slo_results(summary, spec)
        aggregates.insert(0, {
            "configuration_name": "reference-baseline",
            "kind": "baseline",
            "config": deepcopy(reference.get("config", spec["search"].get("baseline", {}))),
            "env": deepcopy(reference.get("env", {})),
            "expected_repetitions": 1,
            "completed_repetitions": 1,
            "failed_repetitions": 0,
            "metrics": metrics,
            "metric_samples": {key: [float(value)] for key, value in metrics.items() if isinstance(value, (int, float))},
            "slo": summary["slo"],
            "all_repetitions_slo_passed": summary["slo"]["passed"],
            "objective_cv_pct": 0.0,
            "max_cv_pct": float(spec["search"].get("max_cv_pct", 10.0)),
            "stable": True,
            "eligible_for_confirmation": summary["slo"]["passed"] or not spec["search"].get("require_all_slo_pass", True),
            "source": "previously_measured_reference",
        })
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


def resident_ab_eligible(spec: dict[str, Any]) -> bool:
    """Return whether two confirmation services fit on disjoint local GPUs."""
    search = spec.get("search", {})
    if not search.get("reuse_server_across_repetitions", False):
        return False
    if int(search.get("repetitions", 1)) < 2:
        return False
    configurations = candidate_matrix(spec)
    if len(configurations) != 2 or {item["kind"] for item in configurations} != {"baseline", "candidate"}:
        return False
    required = sum(
        configuration_accelerator_count(spec, item["config"])
        for item in configurations
    )
    return required <= len(available_gpu_identifiers(spec))


def execute_resident_ab(
    spec: dict[str, Any], progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Load baseline and winner once, then alternate isolated benchmark windows."""
    errors = execution_errors(spec)
    if errors:
        raise ValueError("; ".join(errors))
    if spec["execution"].get("require_accelerator", True) and not has_accelerator():
        raise RuntimeError("no NVIDIA or AMD accelerator detected")
    run_dir, sessions = prepare_run(spec)
    if len(sessions) != 2:
        raise RuntimeError("resident A/B requires exactly one baseline and one candidate session")
    execution = spec["execution"]
    host = execution.get("host", "127.0.0.1")
    ports = reserve_worker_ports(host, int(execution.get("port", 30000)), 2)
    gpu_slots = available_gpu_identifiers(spec)
    visibility_key = (
        "HIP_VISIBLE_DEVICES"
        if "HIP_VISIBLE_DEVICES" in execution.get("env", {}) else "CUDA_VISIBLE_DEVICES"
    )
    assigned_sessions: list[dict[str, Any]] = []
    slot_offset = 0
    for index, original in enumerate(sessions):
        trial = deepcopy(original)
        needed = configuration_accelerator_count(spec, trial["config"])
        assigned = gpu_slots[slot_offset:slot_offset + needed]
        slot_offset += needed
        trial["_candidate_env"] = deepcopy(trial.get("env", {}))
        trial["env"] = {**trial.get("env", {}), visibility_key: ",".join(assigned)}
        trial["_port"] = ports[index]
        assigned_sessions.append({
            "trial": trial,
            "directory": run_dir / f"session-{index:02d}-{trial['configuration_name']}",
            "assigned_gpus": assigned,
            "accelerator_count": needed,
        })

    started = time.monotonic()
    max_wall = float(spec["budget"]["max_wall_time_minutes"]) * 60
    total_accelerators = sum(session["accelerator_count"] for session in assigned_sessions)
    max_session_seconds = min(
        max_wall,
        float(spec["budget"]["max_gpu_hours"]) * 3600 / max(1, total_accelerators),
    )
    processes: list[subprocess.Popen[Any]] = []
    log_handles: list[Any] = []
    rows: list[dict[str, Any]] = []
    ready_sessions = 0
    try:
        # Start both servers before any measurement. Loading may overlap, but
        # benchmark windows remain serial to avoid host/network contention.
        for session in assigned_sessions:
            trial = session["trial"]
            trial_dir = session["directory"]
            trial_dir.mkdir(parents=True, exist_ok=False)
            os.chmod(trial_dir, 0o700)
            manifest = command_manifest(spec, trial, trial_dir)
            session["manifest"] = manifest
            session["env"] = sanitized_environment(spec, trial)
            write_json(trial_dir / "trial.json", trial)
            write_json(trial_dir / "commands.json", manifest)
            if progress is not None:
                progress({
                    "event": "trial_phase", "trial_index": len(processes) + 1,
                    "trial_count": int(spec["budget"]["max_trials"]),
                    "trial_name": trial["configuration_name"],
                    "phase": "server_launch",
                    "message": f"launching resident service on GPUs {session['assigned_gpus']}",
                })
            wait_port_available(host, int(trial["_port"]), float(execution.get("shutdown_timeout_sec", 30)))
            log_handle = (trial_dir / "server.log").open("w", encoding="utf-8")
            log_handles.append(log_handle)
            process = subprocess.Popen(
                manifest["server"], cwd=spec["repository"], env=session["env"],
                stdout=log_handle, stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
            processes.append(process)
            session["process"] = process
            write_json(trial_dir / "process.json", {
                "pid": process.pid, "process_group": process.pid, "started_at": now_iso(),
                "command": manifest["server"], "owned_by": str(trial_dir),
                "assigned_gpus": session["assigned_gpus"],
            })
        for session in assigned_sessions:
            remaining = max_session_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise RuntimeError("resident A/B startup exhausted the wall-time budget")
            ready, detail = wait_ready(
                f"http://{host}:{session['trial']['_port']}/v1/models",
                session["process"],
                min(float(execution.get("startup_timeout_sec", 900)), remaining),
                heartbeat=(
                    lambda elapsed, session=session: progress({
                        "event": "trial_phase", "trial_index": 1,
                        "trial_count": int(spec["budget"]["max_trials"]),
                        "trial_name": session["trial"]["configuration_name"],
                        "phase": "server_startup",
                        "message": f"resident model loading/CUDA Graph capture; elapsed={elapsed:.0f}s",
                    })
                    if progress is not None else None
                ),
            )
            if not ready:
                raise RuntimeError(startup_failure_detail(
                    detail, session["directory"] / "server.log"
                ))
            ready_sessions += 1
            if progress is not None:
                progress({
                    "event": "trial_phase", "trial_index": ready_sessions,
                    "trial_count": int(spec["budget"]["max_trials"]),
                    "trial_name": session["trial"]["configuration_name"],
                    "phase": "server_ready",
                    "message": f"resident service ready ({ready_sessions}/{len(assigned_sessions)})",
                })

        repetitions = int(spec["search"]["repetitions"])
        adaptive_cv_threshold = spec["search"].get("adaptive_confirmation_cv_pct")
        adaptive_max_repetitions = int(
            spec["search"].get("adaptive_confirmation_max_repetitions", repetitions)
        )
        adaptive_minimum_duration = float(
            spec["search"].get(
                "adaptive_confirmation_min_measurement_seconds",
                spec["benchmark"].get("min_measurement_seconds", 0),
            )
        )
        initial_order: list[tuple[dict[str, Any], int]] = []
        for repeat_index in range(repetitions):
            ordered = assigned_sessions if repeat_index % 2 == 0 else list(reversed(assigned_sessions))
            initial_order.extend((session, repeat_index) for session in ordered)
        potential_trial_count = len(initial_order) + (
            len(assigned_sessions) * (adaptive_max_repetitions - repetitions)
        )

        def measure_window(
            session: dict[str, Any], repeat_index: int, measurement_index: int,
            trial_count: int, minimum_duration: float, adaptive_window: bool,
        ) -> dict[str, Any]:
            trial = session["trial"]
            trial_dir = session["directory"]
            if progress is not None:
                progress({
                    "event": "trial_started", "trial_index": measurement_index,
                    "trial_count": trial_count,
                    "trial_name": f"{trial['configuration_name']}-r{repeat_index + 1:02d}",
                    "configuration_name": trial["configuration_name"], "kind": trial["kind"],
                    "assigned_gpus": session["assigned_gpus"],
                })
            command = list(session.get("benchmark_template", session["manifest"]["benchmark"]))
            raw_path = trial_dir / f"result-r{repeat_index + 1:02d}.jsonl"
            command[command.index("--output-file") + 1] = str(raw_path)
            attempts: list[dict[str, Any]] = []
            for attempt_index in range(1, 6):
                remaining = max_session_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise RuntimeError("resident A/B benchmark exhausted the wall-time budget")
                with (trial_dir / "benchmark.log").open("a", encoding="utf-8") as benchmark_log:
                    benchmark_log.write(
                        f"\n=== measurement r{repeat_index + 1:02d} attempt {attempt_index} ===\n"
                    )
                    prompts = int(command[command.index("--num-prompts") + 1])
                    if progress is not None:
                        progress({
                            "event": "trial_phase", "trial_index": measurement_index,
                            "trial_count": trial_count, "trial_name": f"{trial['configuration_name']}-r{repeat_index + 1:02d}",
                            "phase": "benchmark",
                            "message": f"attempt {attempt_index}/5; num_prompts={prompts}",
                        })
                    returncode = run_logged_subprocess(
                        command, cwd=spec["repository"], env=session["env"],
                        log_handle=benchmark_log,
                        timeout=min(float(execution.get("benchmark_timeout_sec", 1800)), remaining),
                        heartbeat=(
                            lambda elapsed: progress({
                                "event": "trial_phase", "trial_index": measurement_index,
                                "trial_count": trial_count,
                                "trial_name": f"{trial['configuration_name']}-r{repeat_index + 1:02d}",
                                "phase": "benchmark",
                                "message": f"attempt {attempt_index}/5 running; num_prompts={prompts}, elapsed={elapsed:.0f}s",
                            }) if progress is not None else None
                        ),
                    )
                if returncode != 0:
                    raise RuntimeError(
                        f"{trial['configuration_name']} benchmark exited with code {returncode}"
                    )
                effective_concurrency = (
                    int(command[command.index("--max-concurrency") + 1])
                    if "--max-concurrency" in command else None
                )
                summary = summarize_jsonl(
                    raw_path, spec, effective_concurrency=effective_concurrency
                )
                validity = summary["measurement_validity"]
                duration = validity.get("duration_sec")
                validity["minimum_duration_sec"] = minimum_duration
                if isinstance(duration, (int, float)):
                    validity["duration_gate_passed"] = duration >= minimum_duration
                attempts.append({
                    "attempt": attempt_index,
                    "num_prompts": int(command[command.index("--num-prompts") + 1]),
                    "measurement_validity": validity,
                })
                if validity["duration_gate_passed"] and validity.get("tail_sample_gate_passed", True):
                    break
                if attempt_index == 5:
                    raise RuntimeError(
                        f"{trial['configuration_name']} did not reach duration and p99-wave gates"
                    )
                short_path = trial_dir / f"result-r{repeat_index + 1:02d}-short-{attempt_index}.jsonl"
                raw_path.replace(short_path)
                duration = validity.get("duration_sec") or 0
                current = int(command[command.index("--num-prompts") + 1])
                multiplier = (
                    max(2.0, (minimum_duration / duration) * 1.2)
                    if duration > 0 else 2.0
                )
                increase_benchmark_request_count(command, max(
                    current + 1, math.ceil(current * multiplier),
                    int(validity.get("minimum_request_count_for_tail") or 0),
                ))
                if progress is not None:
                    progress({
                        "event": "trial_phase", "trial_index": measurement_index,
                        "trial_count": trial_count,
                        "trial_name": f"{trial['configuration_name']}-r{repeat_index + 1:02d}",
                        "phase": "benchmark_expand",
                        "message": "measurement gates not met; expanded request window for the next attempt",
                    })
            write_json(trial_dir / f"attempts-r{repeat_index + 1:02d}.json", attempts)
            session["benchmark_template"] = list(command)
            row: dict[str, Any] = {
                "index": measurement_index - 1,
                "name": f"{trial['configuration_name']}-r{repeat_index + 1:02d}",
                "configuration_name": trial["configuration_name"],
                "repeat_index": repeat_index, "kind": trial["kind"],
                "config": trial["config"], "env": trial.get("_candidate_env", {}),
                **(
                    {"registry_candidate_id": trial["registry_candidate_id"]}
                    if isinstance(trial.get("registry_candidate_id"), str) else {}
                ),
                **(
                    {"registry_mechanism": trial["registry_mechanism"]}
                    if isinstance(trial.get("registry_mechanism"), str) else {}
                ),
                "directory": str(trial_dir), "ok": True,
                "status": {
                    "state": "completed", "resident_ab": True,
                    "adaptive_confirmation_window": adaptive_window,
                },
                "resources": {
                    "accelerator_count": session["accelerator_count"],
                    "shared_resident_server_session": True,
                },
                "metrics": summary["metrics"], "slo": summary["slo"],
                "measurement_validity": summary["measurement_validity"],
            }
            rows.append(row)
            write_json(run_dir / "results.json", rows)
            if progress is not None:
                progress({
                    "event": "trial_finished", "trial_index": measurement_index,
                    "trial_count": trial_count, "trial_name": row["name"],
                    "ok": True, "metrics": row["metrics"],
                    "slo_passed": row["slo"].get("passed"),
                })
            return row

        for measurement_index, (session, repeat_index) in enumerate(initial_order, 1):
            measure_window(
                session, repeat_index, measurement_index, potential_trial_count,
                float(spec["benchmark"].get("min_measurement_seconds", 0)), False,
            )

        initial_cvs = {
            session["trial"]["configuration_name"]: objective_cv_for_summaries(
                [
                    {"metrics": row["metrics"]}
                    for row in rows
                    if row["configuration_name"] == session["trial"]["configuration_name"]
                ],
                spec["objective"]["metric"],
            )
            for session in assigned_sessions
        }
        bayesian_enabled = bool(spec["search"].get("bayesian_sequential", False))
        posterior = bayesian_block_decision(spec, rows) if bayesian_enabled else None
        adaptive_triggered = False
        if bayesian_enabled:
            for repeat_index in range(repetitions, adaptive_max_repetitions):
                if posterior is not None and posterior["action"] != "continue":
                    break
                ordered = (
                    assigned_sessions
                    if repeat_index % 2 == 0 else list(reversed(assigned_sessions))
                )
                for session in ordered:
                    measure_window(
                        session, repeat_index, len(rows) + 1, potential_trial_count,
                        adaptive_minimum_duration, True,
                    )
                adaptive_triggered = True
                posterior = bayesian_block_decision(spec, rows)
        else:
            adaptive_triggered = (
                isinstance(adaptive_cv_threshold, (int, float))
                and adaptive_max_repetitions > repetitions
                and any(
                    value is not None and value > float(adaptive_cv_threshold)
                    for value in initial_cvs.values()
                )
            )
            if adaptive_triggered:
                for repeat_index in range(repetitions, adaptive_max_repetitions):
                    ordered = (
                        assigned_sessions
                        if repeat_index % 2 == 0 else list(reversed(assigned_sessions))
                    )
                    for session in ordered:
                        measure_window(
                            session, repeat_index, len(rows) + 1, potential_trial_count,
                            adaptive_minimum_duration, True,
                        )
        adaptive_evidence = {
            "enabled": isinstance(adaptive_cv_threshold, (int, float)),
            "triggered": adaptive_triggered,
            "initial_objective_cv_pct": initial_cvs,
            "trigger_cv_pct": adaptive_cv_threshold,
            "initial_repetitions": repetitions,
            "completed_repetitions": (
                adaptive_max_repetitions if adaptive_triggered else repetitions
            ),
            "extended_min_measurement_seconds": (
                adaptive_minimum_duration if adaptive_triggered else None
            ),
            "bayesian_sequential": posterior,
        }
        for row in rows:
            row["status"]["adaptive_confirmation"] = adaptive_evidence
        write_json(run_dir / "results.json", rows)
    finally:
        for session in assigned_sessions:
            process = session.get("process")
            if process is not None:
                if progress is not None:
                    progress({
                        "event": "trial_phase", "trial_index": 1,
                        "trial_count": int(spec["budget"]["max_trials"]),
                        "trial_name": session["trial"]["configuration_name"],
                        "phase": "cleanup", "message": "stopping resident service",
                    })
                shutdown = stop_owned_process(
                    process, float(execution.get("shutdown_timeout_sec", 30)),
                    heartbeat=(
                        lambda elapsed, session=session: progress({
                            "event": "trial_phase", "trial_index": 1,
                            "trial_count": int(spec["budget"]["max_trials"]),
                            "trial_name": session["trial"]["configuration_name"],
                            "phase": "cleanup",
                            "message": f"waiting for resident server exit; elapsed={elapsed:.0f}s",
                        }) if progress is not None else None
                    ),
                )
                if progress is not None:
                    progress({
                        "event": "trial_phase", "trial_index": 1,
                        "trial_count": int(spec["budget"]["max_trials"]),
                        "trial_name": session["trial"]["configuration_name"],
                        "phase": "cleanup",
                        "message": f"resident service stopped via {shutdown.get('method')}",
                    })
        for handle in log_handles:
            handle.close()

    elapsed = time.monotonic() - started
    decision = decision_report(spec, rows)
    aggregates = decision["aggregates"]
    write_json(run_dir / "aggregates.json", aggregates)
    final = {
        "schema_version": 2, "run_dir": str(run_dir), "completed_at": now_iso(),
        "elapsed_sec": elapsed, "approx_gpu_hours": elapsed * total_accelerators / 3600,
        "planned_trials": len(rows),
        "completed_trials": len(rows), "planned_server_sessions": 2,
        "completed_server_sessions": ready_sessions,
        "skipped_capability_trials": [], "disabled_capabilities": [],
        "stop_reason": "completed_search", "resident_ab": True,
        "measurement_order": [row["configuration_name"] for row in rows],
        "adaptive_confirmation": adaptive_evidence,
        "bayesian_sequential": posterior,
        **decision, "results": rows,
    }
    write_json(run_dir / "final.json", final)
    return final


def execute(
    spec: dict[str, Any], progress: Callable[[dict[str, Any]], None] | None = None
) -> dict[str, Any]:
    if resident_ab_eligible(spec):
        return execute_resident_ab(spec, progress)
    child_subreaper_enabled = enable_child_subreaper()
    errors = execution_errors(spec)
    if errors:
        raise ValueError("; ".join(errors))
    if spec["execution"].get("require_accelerator", True) and not has_accelerator():
        raise RuntimeError("no NVIDIA or AMD accelerator detected")
    run_dir, trials = prepare_run(spec)
    started = time.monotonic()
    max_wall = float(spec["budget"]["max_wall_time_minutes"]) * 60
    max_gpu_hours = float(spec["budget"]["max_gpu_hours"])
    max_gpu_seconds = max_gpu_hours * 3600
    used_gpu_seconds = 0.0
    max_failures = int(spec["budget"].get("max_consecutive_failures", 3))
    rows: list[dict[str, Any]] = []
    skipped_capability_trials: list[dict[str, Any]] = []
    disabled_capabilities: dict[str, dict[str, Any]] = {}
    failures = 0
    stop_reason: str | None = None
    baseline_metrics: dict[str, Any] | None = None
    successful_candidate_rows: list[dict[str, Any]] = []
    precomputed_parallel: dict[int, tuple[dict[str, Any], Path, float, dict[str, Any]]] = {}
    for index, trial in enumerate(trials):
        elapsed = time.monotonic() - started
        if elapsed >= max_wall:
            stop_reason = "wall_time_budget_exhausted"
            break
        if used_gpu_seconds >= max_gpu_seconds:
            stop_reason = "gpu_hour_budget_exhausted"
            break
        if index in precomputed_parallel:
            trial, trial_dir, trial_elapsed, result = precomputed_parallel.pop(index)
            trial_gpu_count = configuration_accelerator_count(spec, trial["config"])
            remaining_time = 0.0
        else:
            trial_gpu_count = configuration_accelerator_count(spec, trial["config"])
            remaining_time = min(
                max_wall - elapsed,
                (max_gpu_seconds - used_gpu_seconds) / trial_gpu_count,
            )
            trial_dir = run_dir / f"trial-{index:03d}-{trial['name']}"
            result = None
            trial_elapsed = 0.0
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
        if result is None and parallel_trial_eligible(spec, trial):
            batch = parallel_candidate_batch(spec, trials, index, disabled_capabilities)
            if len(batch) > 1:
                completed = parallel_screening_batch(
                    spec, batch, index,
                    len(available_gpu_identifiers(spec)),
                    run_dir, max_wall,
                    max_gpu_seconds, started, used_gpu_seconds, progress, len(trials),
                )
                for offset, completed_result in enumerate(completed):
                    precomputed_parallel[index + offset] = completed_result
                trial, trial_dir, trial_elapsed, result = precomputed_parallel.pop(index)
                trial_gpu_count = configuration_accelerator_count(spec, trial["config"])
            else:
                result = None
        calibration_progress = isinstance(
            spec.get("benchmark", {}).get("calibration_session"), dict
        )
        if result is None:
            if progress is not None and not calibration_progress:
                progress({
                    "event": "trial_started",
                    "trial_index": index + 1,
                    "trial_count": len(trials),
                    "trial_name": trial["name"],
                    "configuration_name": trial["configuration_name"],
                    "kind": trial["kind"],
                })
            def trial_progress(event: dict[str, Any]) -> None:
                if progress is None:
                    return
                if event.get("event") == "trial_phase":
                    event = {
                        "trial_index": index + 1,
                        "trial_count": len(trials),
                        "trial_name": trial["name"],
                        **event,
                    }
                progress(event)
            trial_started = time.monotonic()
            result = (
                run_trial(
                    spec, trial, trial_dir, remaining_time,
                    window_progress=trial_progress,
                )
                if progress is not None
                else run_trial(spec, trial, trial_dir, remaining_time)
            )
            trial_elapsed = time.monotonic() - trial_started
        used_gpu_seconds += trial_elapsed * trial_gpu_count
        row: dict[str, Any] = {
            "index": index,
            "name": trial["name"],
            "configuration_name": trial["configuration_name"],
            "repeat_index": trial["repeat_index"],
            "kind": trial["kind"],
            "config": trial["config"],
            **(
                {"registry_candidate_id": trial["registry_candidate_id"]}
                if isinstance(trial.get("registry_candidate_id"), str) else {}
            ),
            **(
                {"registry_mechanism": trial["registry_mechanism"]}
                if isinstance(trial.get("registry_mechanism"), str) else {}
            ),
            # Worker GPU visibility is execution placement, not part of the
            # candidate. Persist only user/configuration environment deltas so
            # a winner is not accidentally pinned to its screening device.
            "env": trial.get("_candidate_env", trial.get("env", {})),
            "directory": str(trial_dir),
            "ok": result["ok"],
            "status": result["status"],
            "resources": {
                "accelerator_count": trial_gpu_count,
                "approx_gpu_hours": trial_elapsed * trial_gpu_count / 3600,
            },
            **(
                {"provisional_parameter": trial["provisional_parameter"]}
                if isinstance(trial.get("provisional_parameter"), str) else {}
            ),
            **(
                {"provisional_atomic_config": deepcopy(trial["provisional_atomic_config"])}
                if isinstance(trial.get("provisional_atomic_config"), dict) else {}
            ),
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
            if progress is not None and not calibration_progress:
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
        repeated_summaries = result.get("summaries", [summary])
        if len(repeated_summaries) > 1:
            row["resources"]["approx_gpu_hours"] /= len(repeated_summaries)
            row["resources"]["shared_resident_server_session"] = True
        row["metrics"] = summary["metrics"]
        row["slo"] = summary["slo"]
        row["measurement_validity"] = summary.get("measurement_validity")
        if isinstance(summary.get("runtime_observations"), dict):
            row["runtime_observations"] = summary["runtime_observations"]
        for evidence_key in ("calibration_concurrency", "effective_num_prompts"):
            if evidence_key in summary:
                row[evidence_key] = summary[evidence_key]
        if isinstance(summary.get("confirmation_reference"), dict):
            row["confirmation_reference"] = summary["confirmation_reference"]
        rows.append(row)
        for repeated_summary in repeated_summaries[1:]:
            repeated_row = deepcopy(row)
            repeated_row["repeat_index"] = int(
                repeated_summary.get("repeat_index", repeated_row["repeat_index"])
            )
            repeated_row["name"] = (
                f"{trial['configuration_name']}-r{repeated_row['repeat_index'] + 1:02d}"
            )[:104]
            repeated_row["metrics"] = repeated_summary["metrics"]
            repeated_row["slo"] = repeated_summary["slo"]
            repeated_row["measurement_validity"] = repeated_summary.get("measurement_validity")
            if isinstance(repeated_summary.get("runtime_observations"), dict):
                repeated_row["runtime_observations"] = repeated_summary["runtime_observations"]
            for evidence_key in ("calibration_concurrency", "effective_num_prompts"):
                if evidence_key in repeated_summary:
                    repeated_row[evidence_key] = repeated_summary[evidence_key]
            repeated_row["resources"] = {
                **repeated_row["resources"],
                "shared_resident_server_session": True,
            }
            rows.append(repeated_row)
        write_json(run_dir / "results.json", rows)
        if progress is not None and not calibration_progress:
            progress({
                "event": "trial_finished",
                "trial_index": index + 1,
                "trial_count": len(trials),
                "trial_name": trial["name"],
                "ok": True,
                "metrics": summary["metrics"],
                "slo_passed": summary["slo"].get("passed"),
            })
        if trial["kind"] == "baseline":
            baseline_metrics = summary["metrics"]
        elif baseline_metrics is not None:
            successful_candidate_rows.append(row)
            objective = spec["objective"]["metric"]
            baseline_value = baseline_metrics.get(objective)
            candidate_value = summary["metrics"].get(objective)
            outlier_threshold = float(spec["search"].get("outlier_retry_pct", 15.0))
            if outlier_retry_required(
                spec, trial, baseline_value, candidate_value,
                pending_parallel_results=bool(precomputed_parallel),
            ):
                retry_trial = deepcopy(trial)
                retry_trial["name"] = f"{trial['name']}-outlier-retry"[:104]
                retry_trial["repeat_index"] = int(trial.get("repeat_index", 0)) + 1
                retry_trial["_outlier_retry"] = True
                max_trials = int(spec["budget"]["max_trials"])
                mechanism_coverage_protected = bool(
                    spec.get("search", {}).get("required_mechanism_coverage")
                )
                if len(trials) >= max_trials and mechanism_coverage_protected:
                    row["outlier_retry"] = {
                        "scheduled": False,
                        "threshold_pct": outlier_threshold,
                        "reason": (
                            "mechanism coverage is protected; final confirmation will "
                            "remeasure this strong candidate"
                        ),
                    }
                    write_json(run_dir / "results.json", rows)
                elif len(trials) >= max_trials and index + 1 < len(trials):
                    displaced = trials.pop()
                    row["outlier_retry_displaced_trial"] = displaced["name"]
                if len(trials) < max_trials and not (
                    mechanism_coverage_protected and row.get("outlier_retry", {}).get("scheduled") is False
                ):
                    trials.insert(index + 1, retry_trial)
                    row["outlier_retry"] = {
                        "scheduled": True,
                        "threshold_pct": outlier_threshold,
                        "screening_delta_pct": (
                            (float(candidate_value) - float(baseline_value)) / float(baseline_value) * 100
                        ),
                        "policy": "repeat one extreme one-pass screen without exceeding max_trials",
                    }
                    write_json(run_dir / "results.json", rows)
                    if progress is not None:
                        displaced_note = (
                            f"; displaced lower-priority {row['outlier_retry_displaced_trial']}"
                            if row.get("outlier_retry_displaced_trial") else ""
                        )
                        progress({
                            "event": "trial_plan_updated",
                            "trial_index": index + 1,
                            "trial_count": len(trials),
                            "trial_name": trial["name"],
                            "message": (
                                f"added bounded outlier retry for {trial['configuration_name']}"
                                f"{displaced_note}; max_trials={max_trials}"
                            ),
                        })
        posterior = bayesian_block_decision(spec, rows)
        if posterior is not None:
            completed_blocks = posterior["blocks"]
            if progress is not None and completed_blocks >= int(
                spec["search"].get("bayesian_min_blocks", 2)
            ):
                progress({
                    "event": "bayesian_update",
                    "trial_index": index + 1,
                    "trial_count": len(trials),
                    "trial_name": trial["name"],
                    "posterior": posterior,
                })
            if posterior["action"] in {"accept", "reject", "inconclusive"}:
                stop_reason = "bayesian_" + posterior["action"]
                break
        if trial["kind"] != "baseline" and baseline_metrics is not None:
            minimum_successes = spec["search"].get(
                "min_successful_candidates_before_early_stop"
            )
            early_stop_gain = spec["search"].get("early_stop_improvement_pct")
            if (
                isinstance(minimum_successes, int)
                and isinstance(early_stop_gain, (int, float))
                and len(successful_candidate_rows) >= minimum_successes
                and not precomputed_parallel
            ):
                required_mechanisms = set(
                    spec.get("search", {}).get("required_mechanism_coverage", [])
                )
                measured_mechanisms = {
                    row.get("registry_mechanism")
                    for row in successful_candidate_rows
                    if isinstance(row.get("registry_mechanism"), str)
                }
                if required_mechanisms and not required_mechanisms.issubset(
                    measured_mechanisms
                ):
                    continue
                comparisons = [
                    compare(
                        {"metrics": baseline_metrics},
                        {"metrics": candidate["metrics"]},
                        spec,
                    )
                    for candidate in successful_candidate_rows
                ]
                if any(
                    comparison.get("accepted")
                    and isinstance(comparison.get("improvement_pct"), (int, float))
                    and comparison["improvement_pct"] >= float(early_stop_gain)
                    for comparison in comparisons
                ):
                    stop_reason = "strong_candidate_early_stop"
                    break
    decision = decision_report(spec, rows)
    aggregates = decision["aggregates"]
    write_json(run_dir / "aggregates.json", aggregates)
    planned_measurements = max(len(rows), sum(
        len(trial.get("repeat_indices", [trial.get("repeat_index", 0)]))
        for trial in trials
    ))
    adaptive_by_configuration = {
        row["configuration_name"]: row.get("status", {}).get("adaptive_confirmation")
        for row in rows
        if isinstance(row.get("status", {}).get("adaptive_confirmation"), dict)
    }
    final = {
        "schema_version": 2,
        "child_subreaper_enabled": child_subreaper_enabled,
        "run_dir": str(run_dir),
        "completed_at": now_iso(),
        "elapsed_sec": time.monotonic() - started,
        "approx_gpu_hours": used_gpu_seconds / 3600,
        "planned_trials": planned_measurements,
        "planned_server_sessions": len(trials),
        "completed_server_sessions": sum(
            1 for trial in trials
            if any(
                row.get("configuration_name") == trial.get("configuration_name")
                and row.get("ok")
                for row in rows
            )
        ),
        "completed_trials": len(rows),
        "skipped_capability_trials": skipped_capability_trials,
        "disabled_capabilities": list(disabled_capabilities.values()),
        "stop_reason": stop_reason or "completed_search",
        "adaptive_confirmation": {
            "enabled": bool(adaptive_by_configuration),
            "triggered": any(
                bool(item.get("triggered"))
                for item in adaptive_by_configuration.values()
            ),
            "by_configuration": adaptive_by_configuration,
        },
        "bayesian_sequential": bayesian_block_decision(spec, rows),
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
