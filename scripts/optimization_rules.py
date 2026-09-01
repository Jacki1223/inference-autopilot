#!/usr/bin/env python3
"""Declarative bottleneck matching and value generation for InferOpt.

This module deliberately contains no execution code.  It reduces evidence to a
stable bottleneck vocabulary, matches versioned parameter rules, derives
workload/hardware-sensitive values, and allocates a bounded experiment budget.
The controller remains responsible for validating every emitted flag against
the installed SGLang ServerArgs catalog and measuring every recommendation.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any


RULESET_VERSION = "2026.08.31.1"
CO_PRIMARY_SCORE_MARGIN = 0.03
BOTTLENECK_CLASSES = (
    "prefill_attention_bound",
    "decode_attention_bound",
    "moe_compute_bound",
    "gdn_state_compute_bound",
    "kv_memory_capacity_bound",
    "communication_bound",
    "host_scheduler_bound",
    "mixed_or_unknown",
)
MAGNITUDE_ORDER = {"high": 3, "medium": 2, "low": 1}


# ServerArgs families are useful for documentation, but they are too coarse
# for experiment coverage.  For example, ``memory_cache`` contains capacity,
# allocation-layout and admission controls that are not substitutes for one
# another.  These sub-mechanisms are the unit used by the budget allocator.
PARAMETER_SUBMECHANISMS: dict[str, str] = {
    "mem_fraction_static": "kv_capacity",
    "max_total_tokens": "kv_capacity",
    "page_size": "kv_layout",
    "kv_cache_dtype": "kv_precision",
    "max_prefill_tokens": "prefill_admission",
    "chunked_prefill_size": "prefill_chunking",
    "enable_mixed_chunk": "prefill_decode_overlap",
    "schedule_policy": "request_ordering",
    "schedule_conservativeness": "request_admission",
    "scheduler_recv_interval": "scheduler_cadence",
    "num_continuous_decode_steps": "scheduler_cadence",
    "disable_overlap_schedule": "scheduler_overlap",
    "cuda_graph_max_bs_decode": "decode_cuda_graph",
    "cuda_graph_max_bs_prefill": "prefill_cuda_graph",
    "attention_backend": "attention_backend",
    "prefill_attention_backend": "prefill_attention_backend",
    "decode_attention_backend": "decode_attention_backend",
    "moe_runner_backend": "moe_kernel_backend",
    "moe_a2a_backend": "moe_communication_backend",
    "ep_size": "expert_parallel_topology",
    "moe_dp_size": "expert_parallel_topology",
    "tp_size": "tensor_parallel_topology",
    "pp_size": "pipeline_parallel_topology",
    "dp_size": "data_parallel_topology",
    "enable_dp_attention": "data_parallel_attention",
    "enable_mscclpp": "collective_backend",
    "disable_custom_all_reduce": "collective_backend",
    "mamba_full_memory_ratio": "hybrid_state_capacity",
    "max_mamba_cache_size": "hybrid_state_capacity",
    "mamba_radix_cache_strategy": "hybrid_state_layout",
    "mamba_ssm_dtype": "hybrid_state_precision",
    "linear_attn_prefill_backend": "hybrid_linear_attention_prefill",
    "linear_attn_decode_backend": "hybrid_linear_attention_decode",
    "ple_offload_embedding": "ple_memory_placement",
    "speculative_algorithm": "speculative_algorithm",
    "speculative_num_steps": "speculative_depth",
    "speculative_num_draft_tokens": "speculative_width",
    "disable_radix_cache": "prefix_cache",
}


def parameter_submechanism(parameter: str, family: str | None = None) -> str:
    """Return a causal tuning unit finer than the catalog family.

    Unknown future SGLang flags retain a stable family-derived fallback rather
    than disappearing from coverage accounting when the installed CLI evolves.
    """
    if parameter in PARAMETER_SUBMECHANISMS:
        return PARAMETER_SUBMECHANISMS[parameter]
    return f"catalog:{family}" if family else f"parameter:{parameter}"


def _number(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _summary(value: Any, key: str = "p95") -> float | None:
    if isinstance(value, dict):
        candidate = value.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    return None


def _kernel_family_share(diagnosis: dict[str, Any], family: str) -> float:
    return sum(
        _number(item.get("time_pct"))
        for item in diagnosis.get("top_kernel_families", [])
        if isinstance(item, dict) and item.get("name") == family
    )


def classify_bottleneck(
    task: dict[str, Any], discovery: dict[str, Any], profile: dict[str, Any],
) -> dict[str, Any]:
    """Return a confidence-bearing, multi-label serving bottleneck diagnosis."""
    diagnosis = profile.get("diagnosis", {}) if isinstance(profile, dict) else {}
    shares = diagnosis.get("shares_pct", {}) if isinstance(diagnosis, dict) else {}
    runtime = profile.get("runtime_observations", {}) if isinstance(profile, dict) else {}
    prefill = runtime.get("prefill", {}) if isinstance(runtime, dict) else {}
    decode = runtime.get("decode", {}) if isinstance(runtime, dict) else {}
    moe = runtime.get("moe", {}) if isinstance(runtime, dict) else {}
    model = discovery.get("model", {})
    workload = task.get("workload", {})
    input_tokens = max(1, int(workload.get("input_tokens", 1)))
    output_tokens = max(1, int(workload.get("output_tokens", 1)))
    profile_metrics = profile.get("benchmark", {}).get("metrics", {}) if isinstance(profile, dict) else {}
    e2e = _number(profile_metrics.get("mean_e2e_latency_ms"))
    ttft = _number(profile_metrics.get("mean_ttft_ms"))
    decode_latency_share = max(0.0, e2e - ttft) / e2e if e2e > 0 else (
        output_tokens / (input_tokens + output_tokens)
    )

    attention_share = _number(shares.get("attention_kernels"))
    moe_share = _number(shares.get("moe_kernels"))
    communication_share = _number(shares.get("communication_kernels"))
    gdn_share = _kernel_family_share(diagnosis, "gdn_delta_rule")
    token_usage = max(
        _summary(prefill.get("token_usage_ratio")) or 0.0,
        _summary(decode.get("token_usage_ratio")) or 0.0,
        _summary(prefill.get("mamba_usage_ratio")) or 0.0,
        _summary(decode.get("mamba_usage_ratio")) or 0.0,
    )
    queue_pressure = max(
        _number(prefill.get("queue_nonempty_batch_pct")) / 100.0,
        _number(decode.get("queue_nonempty_batch_pct")) / 100.0,
    )
    timing_comparable = (
        diagnosis.get("profiling_run_performance_comparable") is True
        or "profiling_run_performance_comparable" not in diagnosis
    )
    gpu_active = diagnosis.get("gpu_timeline_active_pct")
    primary_hint = str(diagnosis.get("primary_bottleneck", ""))

    attention_score = min(1.0, attention_share / 35.0)
    scores = {
        "prefill_attention_bound": attention_score * (
            1.0 if input_tokens >= output_tokens * 2 and decode_latency_share < 0.5 else 0.55
        ),
        "decode_attention_bound": attention_score * (
            1.0 if decode_latency_share >= 0.5 else 0.55
        ),
        "moe_compute_bound": max(
            min(1.0, moe_share / 30.0),
            0.45 if model.get("is_moe") and moe.get("missing_tuned_config") else 0.15 if model.get("is_moe") else 0.0,
        ),
        "gdn_state_compute_bound": max(
            min(1.0, gdn_share / 40.0),
            0.30 if model.get("is_hybrid") and gdn_share > 0 else 0.0,
        ),
        "kv_memory_capacity_bound": min(1.0, max(token_usage, queue_pressure * 0.75)),
        "communication_bound": min(1.0, communication_share / 20.0),
        "host_scheduler_bound": (
            max(
                0.75 if primary_hint in {"host_or_scheduler_stall", "cpu_gpu_synchronization"} else 0.0,
                max(0.0, (65.0 - _number(gpu_active, 65.0)) / 40.0),
            ) if timing_comparable else 0.0
        ),
        "mixed_or_unknown": 0.25,
    }
    ordered = sorted(scores, key=lambda name: scores[name], reverse=True)
    primary = ordered[0]
    if scores[primary] < 0.35:
        primary = "mixed_or_unknown"
    co_primary = [
        name for name in ordered
        if name != "mixed_or_unknown"
        and scores[name] >= 0.35
        and scores[name] >= scores[primary] - CO_PRIMARY_SCORE_MARGIN
    ]
    if primary not in co_primary:
        co_primary.insert(0, primary)
    secondary = [
        name for name in ordered
        if name != primary and name != "mixed_or_unknown"
        and scores[name] >= 0.35 and scores[name] >= scores[primary] - 0.25
    ][:3]
    second_score = max((scores[name] for name in scores if name != primary), default=0.0)
    confidence = min(0.99, max(0.25, scores[primary] * 0.75 + max(0.0, scores[primary] - second_score) * 0.25))
    return {
        "schema_version": 1,
        "ruleset_version": RULESET_VERSION,
        "primary": primary,
        "co_primary": co_primary,
        "co_primary_score_margin": CO_PRIMARY_SCORE_MARGIN,
        "secondary": secondary,
        "confidence": round(confidence, 4),
        "scores": {name: round(scores[name], 4) for name in BOTTLENECK_CLASSES},
        "evidence": {
            "attention_kernel_pct": attention_share,
            "moe_kernel_pct": moe_share,
            "gdn_kernel_family_pct": gdn_share,
            "communication_kernel_pct": communication_share,
            "token_usage_p95": token_usage,
            "queue_pressure": queue_pressure,
            "decode_latency_share": decode_latency_share,
            "profile_timing_comparable": timing_comparable,
            "model_is_moe": bool(model.get("is_moe")),
            "model_is_hybrid": bool(model.get("is_hybrid")),
        },
        "policy": "multi-label evidence reduction; unknown is retained when no class clears the confidence floor",
    }


# These presets are versioned knowledge, not claims of universal superiority.
# Each emitted value is still checked against current ServerArgs and benchmarked.
PARAMETER_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "moe_model_backend_sweep", "parameters": ["moe_runner_backend", "ep_size", "enable_dp_attention"],
        "model_all": {"is_moe": True},
        "evidence_any": ["moe_backend_material", "missing_moe_config"],
        "magnitude": "high", "stage": "discovery",
        "source": "trace-material MoE kernels or an explicit missing tuned-kernel configuration",
    },
    {
        "id": "gdn_state_strong_preset",
        "parameters": ["mamba_full_memory_ratio", "mamba_radix_cache_strategy", "page_size", "mem_fraction_static"],
        "bottleneck_any": ["gdn_state_compute_bound"], "model_all": {"is_hybrid": True},
        "magnitude": "high", "stage": "discovery",
        "source": "SGLang hybrid GDN/Mamba state-cache controls",
    },
    {
        "id": "hybrid_model_cache_layout",
        "parameters": ["mamba_full_memory_ratio", "mamba_radix_cache_strategy", "page_size", "mem_fraction_static"],
        "model_all": {"is_hybrid": True}, "magnitude": "high", "stage": "discovery",
        "source": "model-native hybrid state/cache layout",
    },
    {
        "id": "hybrid_linear_attention_backends",
        "parameters": [
            "linear_attn_prefill_backend", "linear_attn_decode_backend",
            "mamba_ssm_dtype",
        ],
        "model_all": {"is_hybrid": True}, "magnitude": "high", "stage": "discovery",
        "source": "phase-specific SGLang GDN/KDA linear-attention backends",
    },
    {
        "id": "prefill_attention_backend_sweep",
        "parameters": ["prefill_attention_backend", "attention_backend"],
        "bottleneck_any": ["prefill_attention_bound"], "magnitude": "high", "stage": "discovery",
        "source": "SGLang phase-specific attention backends",
    },
    {
        "id": "decode_attention_backend_sweep",
        "parameters": ["decode_attention_backend", "attention_backend"],
        "bottleneck_any": ["decode_attention_bound"], "magnitude": "high", "stage": "discovery",
        "source": "SGLang phase-specific attention backends",
    },
    {
        "id": "kv_capacity_sweep",
        "parameters": ["mem_fraction_static", "page_size", "max_prefill_tokens", "chunked_prefill_size"],
        "bottleneck_any": ["kv_memory_capacity_bound"], "magnitude": "high", "stage": "discovery",
        "source": "SGLang KV allocator and prefill admission controls",
    },
    {
        "id": "long_context_offline_sweep",
        "parameters": ["enable_mixed_chunk", "chunked_prefill_size", "max_prefill_tokens", "mem_fraction_static", "page_size"],
        "modes": ["offline_throughput"], "min_input_tokens": 1024,
        "magnitude": "high", "stage": "discovery",
        "source": "long-context throughput amortization",
    },
    {
        "id": "long_context_cache_layout",
        "parameters": ["chunked_prefill_size", "max_prefill_tokens", "page_size", "mem_fraction_static"],
        "min_input_tokens": 8192, "magnitude": "high", "stage": "discovery",
        "source": "long-context prefill and KV page layout",
    },
    {
        "id": "offline_throughput_baseline",
        "parameters": ["enable_mixed_chunk", "mem_fraction_static", "max_prefill_tokens", "schedule_conservativeness", "num_continuous_decode_steps"],
        "modes": ["offline_throughput"], "magnitude": "medium", "stage": "discovery",
        "source": "offline backlog capacity and scheduler controls",
    },
    {
        "id": "online_sensitivity_controls",
        "parameters": ["mem_fraction_static", "num_continuous_decode_steps", "schedule_conservativeness", "page_size", "cuda_graph_max_bs_decode"],
        "modes": ["online_latency"], "magnitude": "low", "stage": "discovery",
        "source": "bounded online sensitivity controls; declared SLOs remain hard gates",
    },
    {
        "id": "communication_topology_sweep",
        "parameters": ["tp_size", "ep_size", "enable_mscclpp", "disable_custom_all_reduce"],
        "bottleneck_any": ["communication_bound"], "min_gpu_count": 2,
        "magnitude": "high", "stage": "discovery",
        "source": "SGLang collective and parallel topology controls",
    },
    {
        "id": "parallel_topology_fit",
        "parameters": ["tp_size"], "min_gpu_count": 2,
        "magnitude": "medium", "stage": "discovery",
        "source": "legal model/head topology on the visible GPU pool",
    },
    {
        "id": "scheduler_stall_sweep",
        "parameters": ["num_continuous_decode_steps", "schedule_conservativeness", "scheduler_recv_interval", "disable_overlap_schedule", "chunked_prefill_size"],
        "bottleneck_any": ["host_scheduler_bound"], "magnitude": "medium", "stage": "discovery",
        "source": "SGLang scheduler amortization controls",
    },
    {
        "id": "decode_heavy_mtp_sweep",
        "parameters": ["speculative_algorithm", "speculative_num_steps", "speculative_num_draft_tokens"],
        "model_all": {"has_mtp_weights": True}, "min_decode_share": 0.20,
        "magnitude": "high", "stage": "discovery",
        "source": "model-native SGLang speculative decoding",
    },
    {
        "id": "shared_prefix_cache_sweep",
        "parameters": ["schedule_policy", "page_size", "mem_fraction_static"],
        "min_prefix_reuse": 0.20, "magnitude": "high", "stage": "discovery",
        "source": "radix-cache locality and LPM scheduling",
    },
    {
        "id": "cuda_graph_coverage_sweep",
        "parameters": ["cuda_graph_max_bs_decode", "cuda_graph_max_bs_prefill"],
        "evidence_any": ["decode_graph_incomplete", "prefill_graph_incomplete"],
        "magnitude": "medium", "stage": "discovery",
        "source": "observed SGLang CUDA Graph coverage",
    },
    {
        "id": "no_prefix_radix_control",
        "parameters": ["disable_radix_cache"], "max_prefix_reuse": 0.0,
        "magnitude": "low", "stage": "discovery",
        "source": "radix-cache bookkeeping sensitivity control",
    },
)


STRONG_CANDIDATES_BY_CLASS = {
    "gdn_state_compute_bound": ["mamba_full_memory_ratio", "mamba_radix_cache_strategy", "page_size", "mem_fraction_static"],
    "moe_compute_bound": ["moe_runner_backend", "ep_size"],
    "prefill_attention_bound": ["prefill_attention_backend", "attention_backend"],
    "decode_attention_bound": ["decode_attention_backend", "attention_backend", "speculative_algorithm"],
    "kv_memory_capacity_bound": ["mem_fraction_static", "page_size", "max_prefill_tokens", "chunked_prefill_size"],
    "communication_bound": ["tp_size", "ep_size", "enable_mscclpp", "disable_custom_all_reduce"],
    "host_scheduler_bound": ["num_continuous_decode_steps", "schedule_conservativeness", "scheduler_recv_interval"],
    "mixed_or_unknown": ["enable_mixed_chunk", "mem_fraction_static"],
}


def validate_rule_catalog() -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, rule in enumerate(PARAMETER_RULES):
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            errors.append(f"rule[{index}] has no stable id")
        elif rule_id in seen:
            errors.append(f"duplicate rule id: {rule_id}")
        else:
            seen.add(rule_id)
        if rule.get("magnitude") not in MAGNITUDE_ORDER:
            errors.append(f"{rule_id}: invalid magnitude")
        if not isinstance(rule.get("parameters"), list) or not rule["parameters"]:
            errors.append(f"{rule_id}: parameters must be a nonempty list")
        unknown = set(rule.get("bottleneck_any", [])) - set(BOTTLENECK_CLASSES)
        if unknown:
            errors.append(f"{rule_id}: unknown bottleneck classes {sorted(unknown)}")
        if not isinstance(rule.get("source"), str) or not rule["source"]:
            errors.append(f"{rule_id}: source is required")
    return errors


def known_rule_parameters() -> set[str]:
    """Return parameters whose performance semantics are versioned in rules."""
    return {
        str(parameter)
        for rule in PARAMETER_RULES
        for parameter in rule.get("parameters", [])
    } | {
        str(parameter)
        for parameters in STRONG_CANDIDATES_BY_CLASS.values()
        for parameter in parameters
    }


def _rule_matches(
    rule: dict[str, Any], task: dict[str, Any], discovery: dict[str, Any],
    profile: dict[str, Any], classification: dict[str, Any], evidence_flags: set[str],
) -> bool:
    labels = {
        classification["primary"],
        *classification.get("co_primary", []),
        *classification.get("secondary", []),
    }
    if rule.get("bottleneck_any") and not labels.intersection(rule["bottleneck_any"]):
        return False
    model = discovery.get("model", {})
    if any(model.get(key) != value for key, value in rule.get("model_all", {}).items()):
        return False
    mode = task.get("deployment_mode") or (
        "offline_throughput" if task.get("offline") else "online_latency"
    )
    if rule.get("modes") and mode not in rule["modes"]:
        return False
    workload = task.get("workload", {})
    if int(workload.get("input_tokens", 0)) < int(rule.get("min_input_tokens", 0)):
        return False
    prefix_reuse = float(workload.get("prefix_reuse_ratio", 0.0))
    if prefix_reuse < float(rule.get("min_prefix_reuse", 0.0)):
        return False
    if "max_prefix_reuse" in rule and prefix_reuse > float(rule["max_prefix_reuse"]):
        return False
    if int(discovery.get("derived", {}).get("visible_gpu_count", 1)) < int(rule.get("min_gpu_count", 1)):
        return False
    vendor = discovery.get("hardware", {}).get("vendor")
    hardware_profile = discovery.get("hardware_profile") or {}
    architecture = hardware_profile.get("architecture")
    if rule.get("vendors") and vendor not in rule["vendors"]:
        return False
    if rule.get("architectures") and architecture not in rule["architectures"]:
        return False
    metrics = profile.get("benchmark", {}).get("metrics", {}) if isinstance(profile, dict) else {}
    e2e = _number(metrics.get("mean_e2e_latency_ms"))
    ttft = _number(metrics.get("mean_ttft_ms"))
    decode_share = max(0.0, e2e - ttft) / e2e if e2e > 0 else (
        int(workload.get("output_tokens", 1)) /
        max(1, int(workload.get("input_tokens", 1)) + int(workload.get("output_tokens", 1)))
    )
    if decode_share < float(rule.get("min_decode_share", 0.0)):
        return False
    if rule.get("evidence_any") and not evidence_flags.intersection(rule["evidence_any"]):
        return False
    return True


def match_parameter_rules(
    task: dict[str, Any], discovery: dict[str, Any], profile: dict[str, Any],
    classification: dict[str, Any], available_parameters: set[str],
) -> dict[str, Any]:
    runtime = profile.get("runtime_observations", {}) if isinstance(profile, dict) else {}
    decode_graph = runtime.get("decode", {}).get("cuda_graph_coverage_pct") if isinstance(runtime, dict) else None
    prefill_graph = runtime.get("prefill", {}).get("cuda_graph_coverage_pct") if isinstance(runtime, dict) else None
    evidence_flags = set()
    if isinstance(decode_graph, (int, float)) and decode_graph < 95:
        evidence_flags.add("decode_graph_incomplete")
    if isinstance(prefill_graph, (int, float)) and prefill_graph < 95:
        evidence_flags.add("prefill_graph_incomplete")
    moe_runtime = runtime.get("moe", {}) if isinstance(runtime, dict) else {}
    if isinstance(moe_runtime, dict) and moe_runtime.get("missing_tuned_config"):
        evidence_flags.add("missing_moe_config")
    moe_share = _number(
        (profile.get("diagnosis", {}).get("shares_pct", {}) if isinstance(profile, dict) else {}).get(
            "moe_kernels"
        )
    )
    if moe_share >= 10.0 or classification.get("primary") == "moe_compute_bound":
        evidence_flags.add("moe_backend_material")
    matches: list[dict[str, Any]] = []
    by_parameter: dict[str, dict[str, Any]] = {}
    for rule in PARAMETER_RULES:
        if not _rule_matches(rule, task, discovery, profile, classification, evidence_flags):
            continue
        parameters = [name for name in rule["parameters"] if name in available_parameters]
        if not parameters:
            continue
        record = {**deepcopy(rule), "parameters": parameters}
        matches.append(record)
        for parameter in parameters:
            current = by_parameter.setdefault(parameter, {
                "parameter": parameter, "rule_ids": [], "sources": [],
                "magnitude": "low", "stage": "discovery",
            })
            current["rule_ids"].append(rule["id"])
            current["sources"].append(rule["source"])
            if MAGNITUDE_ORDER[rule["magnitude"]] > MAGNITUDE_ORDER[current["magnitude"]]:
                current["magnitude"] = rule["magnitude"]
    strong_candidates: list[str] = []
    strong_classes = classification.get("co_primary") or [classification["primary"]]
    for bottleneck_class in strong_classes:
        for parameter in STRONG_CANDIDATES_BY_CLASS.get(bottleneck_class, []):
            if parameter not in strong_candidates:
                strong_candidates.append(parameter)
    return {
        "schema_version": 1,
        "ruleset_version": RULESET_VERSION,
        "matches": matches,
        "parameters": by_parameter,
        "strong_candidates": strong_candidates,
        "match_context": {
            "bottlenecks": list(dict.fromkeys([
                classification["primary"],
                *classification.get("co_primary", []),
                *classification.get("secondary", []),
            ])),
            "deployment_mode": task.get("deployment_mode") or (
                "offline_throughput" if task.get("offline") else "online_latency"
            ),
            "model": {
                "is_moe": bool(discovery.get("model", {}).get("is_moe")),
                "is_hybrid": bool(discovery.get("model", {}).get("is_hybrid")),
                "has_mtp_weights": bool(discovery.get("model", {}).get("has_mtp_weights")),
            },
            "hardware": {
                "vendor": discovery.get("hardware", {}).get("vendor"),
                "architecture": (discovery.get("hardware_profile") or {}).get("architecture"),
                "gpu_count": discovery.get("derived", {}).get("visible_gpu_count"),
            },
            "workload": {
                "input_tokens": task.get("workload", {}).get("input_tokens"),
                "output_tokens": task.get("workload", {}).get("output_tokens"),
                "prefix_reuse_ratio": task.get("workload", {}).get("prefix_reuse_ratio", 0.0),
            },
        },
        "policy": "match (bottleneck, workload, model, hardware) first; order only within matched parameters",
    }


def dynamic_parameter_values(
    parameter: str, task: dict[str, Any], discovery: dict[str, Any],
    profile: dict[str, Any], installed_default: Any = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Return values derived from live hardware, model, workload and defaults."""
    effective = profile.get("effective_server_config", {}) if isinstance(profile, dict) else {}
    workload = task.get("workload", {})
    if parameter == "mem_fraction_static":
        base = effective.get(parameter, installed_default)
        if not isinstance(base, (int, float)) or isinstance(base, bool):
            return [], {"strategy": "unavailable", "reason": "resolved mem_fraction_static is unavailable"}
        gpus = discovery.get("hardware", {}).get("gpus", [])
        memory_mib = min((_number(gpu.get("memory_mib")) for gpu in gpus if _number(gpu.get("memory_mib")) > 0), default=0.0)
        tp = max(1, int(effective.get("tp_size", discovery.get("derived", {}).get("minimum_tp_size", 1)) or 1))
        weight_mib_per_gpu = _number(discovery.get("model", {}).get("weight_bytes")) / 1024**2 / tp
        weight_fraction = weight_mib_per_gpu / memory_mib if memory_mib > 0 else 0.0
        input_tokens = max(1, int(workload.get("input_tokens", 1)))
        context_pressure = min(1.0, input_tokens / 32768.0)
        activation_reserve = max(0.08, min(0.18, 0.06 + 0.08 * context_pressure))
        # Leave an explicit allocator/graph reserve.  A generic 0.95 ceiling
        # produced 0.92 candidates that loaded weights but failed while CUDA
        # Graph pools were allocated.  Cookbook values may still be tested as
        # atomic recipes; the generic ladder must be more conservative.
        allocator_reserve = 0.04
        safe_ceiling = min(
            0.90,
            max(float(base), 1.0 - activation_reserve - allocator_reserve),
        )
        # Never derive a ceiling below the observed model-weight floor.
        static_floor = min(0.90, weight_fraction * 1.08 + 0.02) if weight_fraction else 0.0
        safe_ceiling = max(float(base), safe_ceiling, static_floor)
        headroom = max(0.0, safe_ceiling - float(base))
        values = sorted({
            round(float(base) + headroom * fraction, 3)
            for fraction in (1 / 3, 2 / 3, 1.0)
            if headroom * fraction >= 0.01
        })
        # The lower control is independent of upward allocator headroom.  It
        # measures whether freeing KV/static memory helps runtime workspaces.
        lower = round(max(0.55, float(base) - 0.03), 3)
        if lower != round(float(base), 3):
            values.append(lower)
        return values, {
            "strategy": "vram_headroom_ladder",
            "resolved_base": float(base), "safe_ceiling": round(safe_ceiling, 3),
            "weight_fraction_per_gpu": round(weight_fraction, 4),
            "activation_reserve_fraction": round(activation_reserve, 4),
            "allocator_reserve_fraction": allocator_reserve,
            "gpu_memory_mib": memory_mib, "tp_size": tp,
        }
    if parameter == "chunked_prefill_size":
        base = effective.get(parameter, installed_default)
        input_tokens = max(256, int(workload.get("input_tokens", 256)))
        reuse = float(workload.get("prefix_reuse_ratio", 0.0))
        uncached = max(256, int(round(input_tokens * max(0.0, 1.0 - reuse))))
        boundary = 1 << int(math.ceil(math.log2(uncached)))
        candidates = {max(256, boundary // 2), boundary, boundary * 2}
        if isinstance(base, int) and base > 0:
            candidates.add(base)
            candidates.add(max(256, ((base + boundary) // 2) // 256 * 256))
        values = sorted(value for value in candidates if value > 0 and value != base)
        return values, {
            "strategy": "uncached_workload_boundary",
            "resolved_base": base, "input_tokens": input_tokens,
            "prefix_reuse_ratio": reuse, "uncached_tokens": uncached,
        }
    if parameter == "max_prefill_tokens":
        base = effective.get(parameter, installed_default)
        input_tokens = max(1, int(workload.get("input_tokens", 1)))
        context = discovery.get("model", {}).get("context_length")
        values = {input_tokens, input_tokens * 2, input_tokens * 4}
        if isinstance(base, int) and base > 0:
            values.add(base * 2)
        if isinstance(context, int) and context > 0:
            values = {min(value, context) for value in values}
        return sorted(value for value in values if value > 0 and value != base), {
            "strategy": "workload_prefill_budget", "resolved_base": base,
            "input_tokens": input_tokens, "context_length": context,
        }
    if parameter == "page_size":
        values = [16, 32, 64] if discovery.get("model", {}).get("is_hybrid") else [1, 16, 32]
        return [value for value in values if value != effective.get(parameter, installed_default)], {
            "strategy": "model_cache_layout", "hybrid": bool(discovery.get("model", {}).get("is_hybrid")),
        }
    if parameter == "mamba_full_memory_ratio":
        base = effective.get(parameter, installed_default)
        if isinstance(base, (int, float)) and not isinstance(base, bool):
            return sorted({round(max(0.5, float(base) - 0.15), 2), round(min(1.0, float(base) + 0.05), 2)} - {float(base)}), {
                "strategy": "hybrid_state_memory_neighborhood", "resolved_base": base,
            }
    return [], {"strategy": "preserve_generated_values"}


def tiered_trial_budget(total_trials: int) -> dict[str, Any]:
    """Allocate 60/25/15 with minimum confirmation and forward reclamation."""
    total = max(1, int(total_trials))
    confirmation = max(4 if total >= 8 else 2, int(round(total * 0.15)))
    refinement = max(1, int(round(total * 0.25)))
    if confirmation + refinement >= total:
        refinement = max(0, total - confirmation - 1)
    discovery = max(1, total - confirmation - refinement)
    return {
        "schema_version": 1,
        "total_trials": total,
        "planned": {
            "discovery": discovery,
            "refinement": refinement,
            "confirmation": confirmation,
        },
        "percentages": {
            "discovery": round(discovery / total * 100, 2),
            "refinement": round(refinement / total * 100, 2),
            "confirmation": round(confirmation / total * 100, 2),
        },
        "reclamation_policy": "unused discovery flows to refinement; all unused earlier-tier trials flow to confirmation",
    }


def history_priors(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert exact-compatible historical winners into priors, never trials."""
    by_parameter: dict[str, list[dict[str, Any]]] = {}
    by_configuration: dict[str, dict[str, Any]] = {}
    for item in candidates:
        config = item.get("config", {}) if isinstance(item, dict) else {}
        env = item.get("env", {}) if isinstance(item, dict) else {}
        if not isinstance(config, dict):
            continue
        score = _number(item.get("history_score_pct"))
        samples = max(1, int(item.get("history_samples", 1) or 1))
        signature = json.dumps(
            {"config": config, "env": env},
            sort_keys=True, separators=(",", ":"),
        )
        by_configuration[signature] = {
            "mean_improvement_pct": score,
            "samples": samples,
            "source_runs": deepcopy(item.get("source_runs", [])),
        }
        for parameter, value in config.items():
            if parameter == "tp_size":
                continue
            by_parameter.setdefault(parameter, []).append({
                "value": value, "mean_improvement_pct": score, "samples": samples,
            })
    return {
        "schema_version": 1,
        "parameter_priors": by_parameter,
        "configuration_priors": by_configuration,
        "candidate_trials_created": 0,
        "policy": "history influences matched-rule ordering and Bayesian confirmation priors; it never occupies a discovery slot",
    }


def configuration_history_prior(
    priors: dict[str, Any], config: dict[str, Any], env: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    signature = json.dumps(
        {"config": config, "env": env or {}},
        sort_keys=True, separators=(",", ":"),
    )
    value = priors.get("configuration_priors", {}).get(signature)
    return deepcopy(value) if isinstance(value, dict) else None
