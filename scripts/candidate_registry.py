"""Canonical per-run candidate and bottleneck decision ledger."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


CANONICAL_BOTTLENECKS = {
    "prefill_scheduling_bound",
    "prefill_attention_bound",
    "decode_attention_bound",
    "moe_compute_bound",
    "gdn_state_compute_bound",
    "kv_memory_capacity_bound",
    "communication_bound",
    "host_scheduler_bound",
    "mixed_or_unknown",
}

CANDIDATE_REGISTRY_SCHEMA_VERSION = 2

MECHANISM_ALIASES = {
    "capacity": "kv_capacity",
    "kv": "kv_capacity",
    "prefill": "prefill_chunking",
    "scheduling": "request_admission",
    "scheduler": "scheduler_cadence",
    "moe": "moe_kernel_backend",
    "moe_execution": "moe_kernel_backend",
    "attention": "attention_backend",
    "mamba": "hybrid_state_capacity",
    "mtp": "speculative_algorithm",
    "topology": "parallel_topology",
    "cuda_graph": "prefill_cuda_graph",
}


def normalize_mechanism(value: Any) -> str:
    """Return the single mechanism vocabulary used by planning and reports."""
    name = str(value or "unknown")
    return MECHANISM_ALIASES.get(name, name)


def directional_candidate_reason(group: dict[str, Any], value: Any) -> str | None:
    """Describe the actual direction relative to the resolved runtime value."""
    reason = group.get("reason")
    strategy = group.get("value_strategy", {})
    resolved = strategy.get("resolved_base", strategy.get("resolved_default"))
    if (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and isinstance(resolved, (int, float)) and not isinstance(resolved, bool)
        and value != resolved
    ):
        direction = "increase" if value > resolved else "decrease"
        return (
            f"{direction} {group.get('parameter')} from resolved runtime value "
            f"{resolved} to {value}; {reason or 'measure the workload-specific response'}"
        )
    return reason

LEGACY_BOTTLENECK_MAP = {
    "attention": "prefill_attention_bound",
    "prefill_attention": "prefill_attention_bound",
    "decode_attention": "decode_attention_bound",
    "moe_compute": "moe_compute_bound",
    "gemm_compute": "moe_compute_bound",
    "gdn_state_compute": "gdn_state_compute_bound",
    "memory_transfer": "kv_memory_capacity_bound",
    "communication": "communication_bound",
    "host_or_scheduler_stall": "host_scheduler_bound",
    "cpu_gpu_synchronization": "host_scheduler_bound",
    "mixed_gpu_compute": "mixed_or_unknown",
    "profile_timing_distorted": "mixed_or_unknown",
}

IMPACT_ORDER = {"low": 1, "medium": 2, "high": 3}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def candidate_signature(config_delta: dict[str, Any], env_delta: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json({"config": config_delta, "env": env_delta}).encode("utf-8")
    ).hexdigest()


def canonical_bottleneck_report(
    classification: dict[str, Any], diagnosis: dict[str, Any]
) -> dict[str, Any]:
    primary = str(classification.get("primary", "mixed_or_unknown"))
    if primary not in CANONICAL_BOTTLENECKS:
        primary = LEGACY_BOTTLENECK_MAP.get(primary, "mixed_or_unknown")
    secondary = []
    for value in classification.get("secondary", []):
        normalized = (
            value if value in CANONICAL_BOTTLENECKS
            else LEGACY_BOTTLENECK_MAP.get(str(value), "mixed_or_unknown")
        )
        if normalized != primary and normalized not in secondary:
            secondary.append(normalized)
    timing_comparable = diagnosis.get("profiling_run_performance_comparable") is True
    return {
        "schema_version": CANDIDATE_REGISTRY_SCHEMA_VERSION,
        "primary": primary,
        "secondary": secondary,
        "scores": deepcopy(classification.get("scores", {})),
        "confidence": classification.get("confidence"),
        "supporting_facts": deepcopy(classification.get("evidence", {})),
        "evidence_quality": {
            "profile_timing_comparable": timing_comparable,
            "kernel_shares_usable": True,
            "host_timeline_usable": timing_comparable,
        },
        "raw_profiler_observation": {
            "primary": diagnosis.get("primary_bottleneck"),
            "secondary": deepcopy(diagnosis.get("secondary_bottlenecks", [])),
        },
        "policy": (
            "raw profiler observations are evidence; only canonical bottleneck classes "
            "may activate optimization rules"
        ),
    }


def infer_bundle_mechanism(config: dict[str, Any]) -> str:
    keys = set(config)
    if "speculative_algorithm" in keys:
        return "speculative_decoding"
    if any(key.startswith("mamba_") for key in keys):
        return "state_space_cache"
    if any("torch_compile" in key or key == "piecewise_cuda_graph_compiler" for key in keys):
        return "model_compilation"
    if keys & {"tp_size", "pp_size", "dp_size", "ep_size", "enable_dp_attention"}:
        return "parallel_topology"
    if keys & {"moe_runner_backend", "moe_a2a_backend", "moe_dp_size"}:
        return "moe_kernel_backend"
    if keys & {"chunked_prefill_size", "max_prefill_tokens", "enable_mixed_chunk"}:
        return "prefill_scheduling_bundle"
    if keys & {"mem_fraction_static", "page_size", "kv_cache_dtype", "max_total_tokens"}:
        return "kv_capacity"
    if keys & {"attention_backend", "prefill_attention_backend", "decode_attention_backend"}:
        return "attention_backend"
    if keys & {"linear_attn_prefill_backend", "linear_attn_decode_backend"}:
        return "hybrid_linear_attention_backend"
    if keys & {"cuda_graph_max_bs_decode", "cuda_graph_max_bs_prefill"}:
        return "prefill_cuda_graph"
    return "configuration_bundle"


class CandidateRegistry:
    """Merge proposals, track decisions, and expose one coverage ledger."""

    def __init__(self, *, canonical_bottleneck: dict[str, Any] | None = None) -> None:
        self.candidates: dict[str, dict[str, Any]] = {}
        self.canonical_bottleneck = deepcopy(canonical_bottleneck or {})
        self.events: list[dict[str, Any]] = []

    def propose(
        self,
        *,
        name: str,
        config_delta: dict[str, Any],
        env_delta: dict[str, Any] | None = None,
        mechanism: str,
        source: dict[str, Any],
        expected_impact: str = "medium",
        parameter: str | None = None,
        value: Any = None,
        value_rank: int = 0,
        dependencies: list[dict[str, Any]] | None = None,
        conflicts: list[dict[str, Any]] | None = None,
        risk: dict[str, Any] | None = None,
        value_strategy: dict[str, Any] | None = None,
        state: str = "eligible",
        decision_reason: str | None = None,
    ) -> str:
        env_delta = deepcopy(env_delta or {})
        config_delta = deepcopy(config_delta)
        signature = candidate_signature(config_delta, env_delta)
        candidate_id = signature[:16]
        existing = self.candidates.get(candidate_id)
        if existing is not None:
            if source not in existing["sources"]:
                existing["sources"].append(deepcopy(source))
            existing["aliases"] = list(dict.fromkeys([*existing["aliases"], name]))
            if IMPACT_ORDER.get(expected_impact, 1) > IMPACT_ORDER.get(
                existing.get("expected_impact", "low"), 1
            ):
                existing["expected_impact"] = expected_impact
            existing["value_rank"] = min(
                int(existing.get("value_rank", value_rank)), int(value_rank)
            )
            if state != "eligible" and existing.get("state") == "eligible":
                existing["state"] = state
                existing["decision_reason"] = decision_reason
            self.events.append({
                "event": "proposal_merged", "candidate_id": candidate_id,
                "source": deepcopy(source),
            })
            return candidate_id
        record = {
            "candidate_id": candidate_id,
            "signature": signature,
            "name": name,
            "aliases": [name],
            "config_delta": config_delta,
            "env_delta": env_delta,
            "mechanism": normalize_mechanism(mechanism),
            "parameter": parameter,
            "value": deepcopy(value),
            "value_rank": int(value_rank),
            "sources": [deepcopy(source)],
            "expected_impact": expected_impact,
            "dependencies": deepcopy(dependencies or []),
            "conflicts": deepcopy(conflicts or []),
            "risk": deepcopy(risk or {"level": "safe", "quality_sensitive": False}),
            "value_strategy": deepcopy(value_strategy or {}),
            "state": state,
            "decision_reason": decision_reason,
            "measurements": [],
            "lifecycle": {
                "proposed": True,
                "scheduled_count": 0,
                "executed_count": 0,
                "valid_measurement_count": 0,
                "stages": [],
            },
            "parent_id": None,
            "children": [],
        }
        self.candidates[candidate_id] = record
        self.events.append({
            "event": "proposal_added", "candidate_id": candidate_id,
            "source": deepcopy(source),
        })
        return candidate_id

    def transition(self, candidate_id: str, state: str, reason: str) -> None:
        candidate = self.candidates[candidate_id]
        previous = candidate["state"]
        candidate["state"] = state
        candidate["decision_reason"] = reason
        self.events.append({
            "event": "state_transition", "candidate_id": candidate_id,
            "from": previous, "to": state, "reason": reason,
        })

    def record_measurement(
        self, candidate_id: str, measurement: dict[str, Any]
    ) -> None:
        candidate = self.candidates[candidate_id]
        measurement_key = canonical_json({
            "configuration_name": measurement.get("configuration_name"),
            "metrics": measurement.get("metrics", {}),
            "improvement_pct": measurement.get("improvement_pct"),
        })
        if any(
            canonical_json({
                "configuration_name": item.get("configuration_name"),
                "metrics": item.get("metrics", {}),
                "improvement_pct": item.get("improvement_pct"),
            }) == measurement_key
            for item in candidate.get("measurements", [])
        ):
            return
        lifecycle = candidate.setdefault("lifecycle", {
            "proposed": True, "scheduled_count": 0, "executed_count": 0,
            "valid_measurement_count": 0, "stages": [],
        })
        lifecycle["executed_count"] = int(lifecycle.get("executed_count", 0)) + 1
        if measurement.get("ok", True):
            lifecycle["valid_measurement_count"] = int(
                lifecycle.get("valid_measurement_count", 0)
            ) + 1
        candidate["measurements"].append(deepcopy(measurement))
        if not measurement.get("ok", True):
            self.transition(
                candidate_id, "measurement_failed",
                str(measurement.get("failure_class") or "measurement failed"),
            )
            return
        improvement = measurement.get("improvement_pct")
        minimum = measurement.get("minimum_improvement_pct", 0)
        if not measurement.get("slo_passed", True):
            self.transition(candidate_id, "slo_failed", "candidate violated a declared SLO")
        elif isinstance(improvement, (int, float)) and improvement >= float(minimum):
            self.transition(candidate_id, "measured_positive", "candidate cleared the screening gain threshold")
        elif isinstance(improvement, (int, float)) and improvement > 0:
            self.transition(candidate_id, "measurement_uncertain", "positive effect is below the practical threshold")
        else:
            self.transition(candidate_id, "measured_negative", "candidate did not improve the objective")

    def mark_scheduled(self, candidate_id: str, stage: str) -> None:
        candidate = self.candidates[candidate_id]
        lifecycle = candidate.setdefault("lifecycle", {
            "proposed": True, "scheduled_count": 0, "executed_count": 0,
            "valid_measurement_count": 0, "stages": [],
        })
        lifecycle["scheduled_count"] = int(lifecycle.get("scheduled_count", 0)) + 1
        if stage not in lifecycle.setdefault("stages", []):
            lifecycle["stages"].append(stage)
        self.events.append({
            "event": "candidate_scheduled", "candidate_id": candidate_id,
            "stage": stage,
        })

    def eligible(self) -> list[dict[str, Any]]:
        return [
            deepcopy(item) for item in self.candidates.values()
            if item.get("state") == "eligible"
        ]

    def coverage(self) -> dict[str, Any]:
        by_state: dict[str, int] = {}
        by_mechanism: dict[str, dict[str, int]] = {}
        for item in self.candidates.values():
            state = str(item.get("state", "unknown"))
            mechanism = normalize_mechanism(item.get("mechanism", "unknown"))
            by_state[state] = by_state.get(state, 0) + 1
            bucket = by_mechanism.setdefault(mechanism, {})
            bucket[state] = bucket.get(state, 0) + 1
        applicable_states = {"eligible", "measured_negative", "measured_positive", "measurement_uncertain", "measurement_failed", "slo_failed"}
        lifecycle_mechanisms = {
            "applicable": set(), "proposed": set(), "scheduled": set(),
            "executed": set(), "measured": set(),
        }
        for item in self.candidates.values():
            mechanism = normalize_mechanism(item.get("mechanism"))
            lifecycle = item.get("lifecycle", {})
            lifecycle_mechanisms["proposed"].add(mechanism)
            if item.get("state") in applicable_states:
                lifecycle_mechanisms["applicable"].add(mechanism)
            if int(lifecycle.get("scheduled_count", 0)) > 0:
                lifecycle_mechanisms["scheduled"].add(mechanism)
            if int(lifecycle.get("executed_count", 0)) > 0:
                lifecycle_mechanisms["executed"].add(mechanism)
            if int(lifecycle.get("valid_measurement_count", 0)) > 0:
                lifecycle_mechanisms["measured"].add(mechanism)
        return {
            "proposals": sum(len(item.get("sources", [])) for item in self.candidates.values()),
            "unique_candidates": len(self.candidates),
            "by_state": by_state,
            "by_mechanism": by_mechanism,
            "mechanism_lifecycle": {
                key: sorted(values) for key, values in lifecycle_mechanisms.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_REGISTRY_SCHEMA_VERSION,
            "canonical_bottleneck": deepcopy(self.canonical_bottleneck),
            "candidates": sorted(
                (deepcopy(item) for item in self.candidates.values()),
                key=lambda item: (
                    -IMPACT_ORDER.get(item.get("expected_impact", "low"), 1),
                    item.get("mechanism", ""), item.get("name", ""),
                ),
            ),
            "coverage": self.coverage(),
            "events": deepcopy(self.events),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateRegistry":
        registry = cls(canonical_bottleneck=value.get("canonical_bottleneck", {}))
        registry.candidates = {
            str(item["candidate_id"]): deepcopy(item)
            for item in value.get("candidates", [])
            if isinstance(item, dict) and item.get("candidate_id")
        }
        registry.events = deepcopy(value.get("events", []))
        for item in registry.candidates.values():
            item["mechanism"] = normalize_mechanism(item.get("mechanism"))
            item.setdefault("lifecycle", {
                "proposed": True, "scheduled_count": 0, "executed_count": 0,
                "valid_measurement_count": len(item.get("measurements", [])),
                "stages": [],
            })
        return registry


def registry_from_search_plan(
    search_plan: dict[str, Any]
) -> dict[str, Any]:
    registry = CandidateRegistry(
        canonical_bottleneck=search_plan.get("canonical_bottleneck", {})
    )
    for group in search_plan.get("ranked_parameter_groups", []):
        if not isinstance(group, dict):
            continue
        parameter = group.get("parameter")
        if not isinstance(parameter, str):
            continue
        mechanism = str(group.get("submechanism") or group.get("family") or "unknown")
        sources = [
            {
                "type": "trigger_rule", "id": rule_id,
                "evidence": deepcopy(group.get("evidence", [])),
            }
            for rule_id in group.get("trigger", {}).get("rule_ids", [])
        ] or [{
            "type": "legacy_planner_adapter",
            "tiers": deepcopy(group.get("tiers", [])),
            "evidence": deepcopy(group.get("evidence", [])),
        }]
        history_prior = group.get("history_prior_support")
        if isinstance(history_prior, (int, float)) and history_prior > 0:
            sources.append({
                "type": "history_prior",
                "support": float(history_prior),
            })
        for value_rank, value in enumerate(group.get("values", [])):
            atomic_config = deepcopy(
                group.get("parameter_evolution", {}).get("atomic_config", {})
            ) if group.get("provisional") else {}
            config_delta = {
                **atomic_config,
                parameter: deepcopy(value),
            }
            risk = {
                "level": "quality_sensitive" if parameter == "kv_cache_dtype" else
                "experimental" if group.get("provisional") else "safe",
                "quality_sensitive": parameter == "kv_cache_dtype",
                "provisional": bool(group.get("provisional")),
            }
            for source in sources:
                registry.propose(
                    name=f"{parameter}-{str(value).lower()}"[:96],
                    config_delta=config_delta,
                    mechanism=mechanism,
                    source=source,
                    expected_impact=str(group.get("trigger_magnitude", "medium")),
                    parameter=parameter,
                    value=value,
                    value_rank=value_rank,
                    risk=risk,
                    value_strategy=group.get("value_strategy", {}),
                    decision_reason=directional_candidate_reason(group, value),
                )
    for source_name, bundles in (
        ("cookbook", search_plan.get("cookbook_candidate_bundles", [])),
        ("dependent_bundle", search_plan.get("ranked_configuration_bundles", [])),
    ):
        for bundle in bundles:
            if not isinstance(bundle, dict) or not isinstance(bundle.get("config"), dict):
                continue
            registry.propose(
                name=str(bundle.get("name", source_name))[:96],
                config_delta=deepcopy(bundle["config"]),
                env_delta=deepcopy(bundle.get("env", {})),
                mechanism=str(
                    bundle.get("mechanism")
                    or infer_bundle_mechanism(bundle["config"])
                ),
                source={
                    "type": bundle.get("source_type", source_name),
                    "evidence": deepcopy(bundle.get("evidence", [])),
                    "source": deepcopy(bundle.get("source")),
                },
                expected_impact=str(
                    bundle.get("priority")
                    or ("high" if source_name == "cookbook" else "medium")
                ),
                dependencies=deepcopy(
                    bundle.get("dependencies", bundle.get("requirements", []))
                ),
                conflicts=deepcopy(bundle.get("conflicts", [])),
                risk=deepcopy(bundle.get("risk")),
                decision_reason=bundle.get("reason"),
            )
    for item in search_plan.get("quality_gated_candidates", []):
        if not isinstance(item, dict) or item.get("enabled"):
            continue
        parameter = item.get("parameter")
        if not isinstance(parameter, str):
            continue
        for value in item.get("values", []):
            registry.propose(
                name=f"{parameter}-{str(value).lower()}"[:96],
                config_delta={parameter: deepcopy(value)},
                mechanism="quality_sensitive_precision",
                source={"type": "quality_gate", "policy": item.get("policy")},
                expected_impact="low",
                parameter=parameter,
                value=value,
                risk={"level": "quality_sensitive", "quality_sensitive": True},
                state="quality_sensitive",
                decision_reason="explicit quality authorization is required before this candidate can run",
            )
    for item in search_plan.get("runtime_compatibility_exclusions", []):
        if not isinstance(item, dict) or not isinstance(item.get("parameter"), str):
            continue
        parameter = item["parameter"]
        value = item.get("value")
        registry.propose(
            name=f"{parameter}-{str(value).lower()}"[:96],
            config_delta={parameter: deepcopy(value)},
            mechanism="runtime_compatibility",
            source={"type": "runtime_compatibility"},
            expected_impact="low",
            parameter=parameter,
            value=value,
            state="unsupported",
            decision_reason=str(item.get("reason") or "runtime incompatible"),
        )
    for item in search_plan.get("excluded_prior_failures", []):
        if not isinstance(item, dict) or not isinstance(item.get("parameter"), str):
            continue
        parameter = item["parameter"]
        value = item.get("value")
        registry.propose(
            name=f"{parameter}-{str(value).lower()}"[:96],
            config_delta={parameter: deepcopy(value)},
            mechanism="historical_failure",
            source={"type": "historical_failure"},
            expected_impact="low",
            parameter=parameter,
            value=value,
            state="historically_failed",
            decision_reason="same experiment fingerprint has a definitive prior failure",
        )
    return registry.to_dict()


def update_registry_from_aggregates(
    registry_value: dict[str, Any], aggregates: list[dict[str, Any]]
) -> dict[str, Any]:
    registry = CandidateRegistry.from_dict(registry_value)
    by_name = {
        alias: item["candidate_id"]
        for item in registry.candidates.values()
        for alias in item.get("aliases", [item.get("name")])
        if alias
    }
    for aggregate in aggregates:
        if aggregate.get("kind") != "candidate":
            continue
        candidate_id = (
            aggregate.get("registry_candidate_id")
            or by_name.get(aggregate.get("configuration_name"))
        )
        if candidate_id is None:
            config = aggregate.get("config", {})
            env = aggregate.get("env", {})
            signature = candidate_signature(config, env)
            candidate_id = next(
                (
                    item.get("candidate_id") for item in registry.candidates.values()
                    if item.get("signature") == signature
                ),
                None,
            )
        if candidate_id is None:
            continue
        comparison = aggregate.get("comparison", {})
        registry.record_measurement(candidate_id, {
            "configuration_name": aggregate.get("configuration_name"),
            "ok": aggregate.get("completed_repetitions", 0) > 0,
            "stable": aggregate.get("stable"),
            "slo_passed": aggregate.get("all_repetitions_slo_passed"),
            "metrics": deepcopy(aggregate.get("metrics", {})),
            "improvement_pct": comparison.get("improvement_pct"),
            "minimum_improvement_pct": comparison.get("minimum_improvement_pct", 0),
            "rejection_reasons": deepcopy(aggregate.get("rejection_reasons", [])),
        })
    return registry.to_dict()


def mark_registry_scheduled(
    registry_value: dict[str, Any], candidate_ids: list[str], *, stage: str
) -> dict[str, Any]:
    """Record scheduling separately from execution and valid measurement."""
    registry = CandidateRegistry.from_dict(registry_value)
    for candidate_id in candidate_ids:
        if candidate_id in registry.candidates:
            registry.mark_scheduled(candidate_id, stage)
    return registry.to_dict()


MECHANISM_BOTTLENECK_MAP = {
    "prefill_chunking": "prefill_scheduling_bound",
    "prefill_admission": "prefill_scheduling_bound",
    "prefill_scheduling_bundle": "prefill_scheduling_bound",
    "prefill_decode_overlap": "prefill_scheduling_bound",
    "prefill_cuda_graph": "prefill_scheduling_bound",
    "kv_capacity": "kv_memory_capacity_bound",
    "kv_layout": "kv_memory_capacity_bound",
    "kv_precision": "kv_memory_capacity_bound",
    "request_admission": "host_scheduler_bound",
    "request_ordering": "host_scheduler_bound",
    "scheduler_cadence": "host_scheduler_bound",
    "scheduler_overlap": "host_scheduler_bound",
    "attention_backend": "prefill_attention_bound",
    "prefill_attention_backend": "prefill_attention_bound",
    "decode_attention_backend": "decode_attention_bound",
    "moe_kernel_backend": "moe_compute_bound",
    "moe_communication_backend": "communication_bound",
    "collective_backend": "communication_bound",
    "hybrid_state_capacity": "gdn_state_compute_bound",
    "hybrid_state_layout": "gdn_state_compute_bound",
}


def posterior_bottleneck_report(
    prior: dict[str, Any], registry_value: dict[str, Any],
    *, minimum_improvement_pct: float,
) -> dict[str, Any]:
    """Update the routing diagnosis from controlled parameter interventions."""
    mechanism_gains: dict[str, list[float]] = {}
    for item in registry_value.get("candidates", []):
        mechanism = normalize_mechanism(item.get("mechanism"))
        if mechanism in {"interaction", "configuration_bundle"}:
            continue
        for measurement in item.get("measurements", []):
            gain = measurement.get("improvement_pct")
            if (
                isinstance(gain, (int, float))
                and measurement.get("ok", True)
                and measurement.get("slo_passed", True)
            ):
                mechanism_gains.setdefault(mechanism, []).append(float(gain))
    best_by_mechanism = {
        mechanism: max(values) for mechanism, values in mechanism_gains.items()
        if values
    }
    ranked = sorted(best_by_mechanism.items(), key=lambda item: item[1], reverse=True)
    winner_mechanism = ranked[0][0] if ranked else None
    winner_gain = ranked[0][1] if ranked else None
    posterior_primary = str(prior.get("primary", "mixed_or_unknown"))
    updated = False
    if (
        winner_mechanism in MECHANISM_BOTTLENECK_MAP
        and isinstance(winner_gain, float)
        and winner_gain >= minimum_improvement_pct
    ):
        posterior_primary = MECHANISM_BOTTLENECK_MAP[winner_mechanism]
        updated = posterior_primary != prior.get("primary")
    return {
        **deepcopy(prior),
        "schema_version": CANDIDATE_REGISTRY_SCHEMA_VERSION,
        "primary": posterior_primary,
        "prior_primary": prior.get("primary"),
        "posterior_updated": updated,
        "dominant_intervention_mechanism": winner_mechanism,
        "dominant_intervention_gain_pct": winner_gain,
        "intervention_best_gain_pct": best_by_mechanism,
        "policy": (
            "profile/runtime evidence forms the prior; controlled SLO-valid parameter "
            "interventions update the canonical bottleneck after screening"
        ),
    }
