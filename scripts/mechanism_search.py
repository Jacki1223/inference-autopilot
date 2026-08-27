"""Mechanism-level adaptive scheduling for expensive serving trials."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from candidate_registry import IMPACT_ORDER, candidate_signature, normalize_mechanism


MECHANISM_SEARCH_SCHEMA_VERSION = 3


TERMINAL_STATES = {
    "measured_negative", "slo_failed", "measurement_failed", "historically_failed",
    "unsupported", "inapplicable", "blocked", "already_effective",
}


def _candidate_priority(item: dict[str, Any]) -> tuple[int, int, int, str]:
    source_bonus = sum(
        2 if source.get("type") == "cookbook" else
        1 if source.get("type") == "trigger_rule" else 0
        for source in item.get("sources", [])
        if isinstance(source, dict)
    )
    return (
        -IMPACT_ORDER.get(item.get("expected_impact", "low"), 1),
        -source_bonus,
        int(item.get("value_rank", 0)),
        str(item.get("name", "")),
    )


def initial_mechanism_schedule(
    registry: dict[str, Any], *, budget: int,
    mandatory_parameters: list[str] | tuple[str, ...] = (),
    provisional_slots: int = 0,
) -> dict[str, Any]:
    """Cover mechanisms before spending a second slot in one mechanism."""
    eligible = [
        deepcopy(item) for item in registry.get("candidates", [])
        if item.get("state") == "eligible"
    ]
    mandatory = set(mandatory_parameters)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    provisional_used = 0

    def admit(item: dict[str, Any]) -> bool:
        nonlocal provisional_used
        if len(selected) >= budget or item["candidate_id"] in selected_ids:
            return False
        if item.get("risk", {}).get("provisional"):
            if provisional_used >= provisional_slots:
                return False
            provisional_used += 1
        selected.append(item)
        selected_ids.add(item["candidate_id"])
        return True

    mandatory_selected: set[str] = set()
    for item in sorted(eligible, key=_candidate_priority):
        parameter = item.get("parameter")
        if parameter in mandatory and parameter not in mandatory_selected:
            if admit(item):
                mandatory_selected.add(str(parameter))

    mechanisms: dict[str, list[dict[str, Any]]] = {}
    for item in eligible:
        mechanisms.setdefault(normalize_mechanism(item.get("mechanism")), []).append(item)
    for values in mechanisms.values():
        values.sort(key=_candidate_priority)

    # First pass: one representative from every applicable mechanism.
    represented_mechanisms = {
        normalize_mechanism(item.get("mechanism")) for item in selected
    }
    for mechanism in sorted(
        mechanisms,
        key=lambda name: min(_candidate_priority(item) for item in mechanisms[name]),
    ):
        if mechanism in represented_mechanisms:
            continue
        for item in mechanisms[mechanism]:
            if admit(item):
                represented_mechanisms.add(mechanism)
                break
    # Second pass: round-robin additional values. This preserves breadth while
    # using otherwise idle discovery slots without a blind Cartesian product.
    while len(selected) < budget:
        progress = False
        for mechanism in sorted(mechanisms):
            values = mechanisms[mechanism]
            for item in values:
                if item["candidate_id"] in selected_ids:
                    continue
                if admit(item):
                    progress = True
                break
            if len(selected) >= budget:
                break
        if not progress:
            break

    deferred = []
    for item in eligible:
        if item["candidate_id"] in selected_ids:
            continue
        reason = (
            "provisional exploration budget exhausted"
            if item.get("risk", {}).get("provisional") else
            "mechanism discovery budget exhausted"
        )
        deferred.append({
            "candidate_id": item["candidate_id"], "name": item.get("name"),
            "mechanism": normalize_mechanism(item.get("mechanism")), "reason": reason,
        })
    return {
        "schema_version": MECHANISM_SEARCH_SCHEMA_VERSION,
        "policy": "mandatory controls, then one representative per mechanism, then round-robin depth",
        "selected": selected,
        "deferred": deferred,
        "scheduled_mechanisms": sorted({normalize_mechanism(item.get("mechanism")) for item in selected}),
        # Backward-compatible alias. Reports must label this as scheduled, not measured.
        "covered_mechanisms": sorted({normalize_mechanism(item.get("mechanism")) for item in selected}),
        "budget": budget,
        "provisional_slots": provisional_slots,
        "provisional_used": provisional_used,
    }


def mechanism_outcomes(
    registry: dict[str, Any], *, minimum_improvement_pct: float
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in registry.get("candidates", []):
        grouped.setdefault(normalize_mechanism(item.get("mechanism")), []).append(item)
    outcomes = []
    for mechanism, candidates in sorted(grouped.items()):
        measured = [
            item for item in candidates if item.get("measurements")
        ]
        gains = [
            float(measurement["improvement_pct"])
            for item in measured for measurement in item.get("measurements", [])
            if isinstance(measurement.get("improvement_pct"), (int, float))
            and measurement.get("slo_passed", True)
        ]
        best = max(gains) if gains else None
        remaining = [
            item for item in candidates if item.get("state") == "eligible"
        ]
        if best is not None and best >= minimum_improvement_pct:
            state = "promising"
            reason = "at least one candidate cleared the practical gain threshold"
        elif best is not None and best > 0:
            state = "uncertain"
            reason = "positive signal remains below the practical threshold"
        elif measured and not gains:
            state = "blocked_or_failed"
            reason = "no SLO-valid numeric measurement completed"
        elif measured:
            state = "stopped_negative"
            reason = "measured candidates did not improve the objective"
        else:
            state = "unmeasured"
            reason = "no candidate from this mechanism has completed"
        outcomes.append({
            "mechanism": mechanism,
            "state": state,
            "reason": reason,
            "best_improvement_pct": best,
            "measured_candidates": len(measured),
            "remaining_candidates": len(remaining),
            "candidate_ids": [item.get("candidate_id") for item in candidates],
        })
    return outcomes


def adaptive_followup_schedule(
    registry: dict[str, Any], *, budget: int,
    minimum_improvement_pct: float,
    anchor_config: dict[str, Any] | None = None,
    evaluated_configurations: list[dict[str, Any]] | None = None,
    max_values_per_mechanism: int = 1,
) -> dict[str, Any]:
    """Spend refinement only on promising or uncertain mechanisms."""
    anchor_config = deepcopy(anchor_config or {})
    evaluated_signatures = {
        json.dumps(config, sort_keys=True, separators=(",", ":"))
        for config in (evaluated_configurations or []) if isinstance(config, dict)
    }
    outcomes = mechanism_outcomes(
        registry, minimum_improvement_pct=minimum_improvement_pct
    )
    outcome_by_mechanism = {item["mechanism"]: item for item in outcomes}
    measured_parameters = {
        item.get("parameter")
        for item in registry.get("candidates", [])
        if item.get("measurements") and item.get("parameter")
    }

    def followup_priority(item: dict[str, Any]) -> tuple[Any, ...]:
        source_types = {
            source.get("type") for source in item.get("sources", [])
            if isinstance(source, dict)
        }
        is_unmeasured_semantic_sibling = (
            "parameter_capability_registry" in source_types
            and item.get("parameter") not in measured_parameters
        )
        is_new_parameter = item.get("parameter") not in measured_parameters
        return (
            0 if is_unmeasured_semantic_sibling else 1,
            0 if is_new_parameter else 1,
            *_candidate_priority(item),
        )
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in registry.get("candidates", []):
        if item.get("state") != "eligible":
            continue
        outcome = outcome_by_mechanism.get(normalize_mechanism(item.get("mechanism")), {})
        if outcome.get("state") not in {"promising", "uncertain"}:
            continue
        if (
            outcome.get("state") == "uncertain"
            and float(outcome.get("best_improvement_pct") or 0)
            < max(0.25, minimum_improvement_pct * 0.25)
        ):
            continue
        full_config = {**anchor_config, **item.get("config_delta", {})}
        signature = json.dumps(full_config, sort_keys=True, separators=(",", ":"))
        if signature in evaluated_signatures:
            continue
        candidate = deepcopy(item)
        candidate["full_config"] = full_config
        buckets.setdefault(normalize_mechanism(item.get("mechanism")), []).append(candidate)
    for values in buckets.values():
        values.sort(key=followup_priority)

    selected: list[dict[str, Any]] = []
    offset = 0
    ordered_mechanisms = sorted(
        buckets,
        key=lambda name: (
            0 if outcome_by_mechanism[name]["state"] == "promising" else 1,
            -(outcome_by_mechanism[name].get("best_improvement_pct") or 0),
            name,
        ),
    )
    while (
        len(selected) < budget
        and offset < max(1, int(max_values_per_mechanism))
        and any(
        offset < len(buckets[name]) for name in ordered_mechanisms
        )
    ):
        for mechanism in ordered_mechanisms:
            if offset < len(buckets[mechanism]):
                selected.append(buckets[mechanism][offset])
                if len(selected) >= budget:
                    break
        offset += 1

    return {
        "schema_version": MECHANISM_SEARCH_SCHEMA_VERSION,
        "policy": (
            "refine only promising/uncertain mechanisms; within a positive mechanism, "
            "test an unmeasured high-confidence semantic sibling before another value "
            "of an already measured parameter; round-robin across mechanisms"
        ),
        "selected": selected,
        "mechanism_outcomes": outcomes,
        "budget": budget,
        "max_values_per_mechanism": max(1, int(max_values_per_mechanism)),
        "stopped_mechanisms": [
            item for item in outcomes
            if item["state"] in {"stopped_negative", "blocked_or_failed"}
        ],
    }


def link_parent_child(
    registry: dict[str, Any], parent_id: str, child_config: dict[str, Any],
    *, name: str, mechanism: str = "interaction",
) -> dict[str, Any]:
    """Record a derived interaction with an explicit parent edge."""
    candidates = registry.setdefault("candidates", [])
    parent = next(
        (item for item in candidates if item.get("candidate_id") == parent_id), None
    )
    if parent is None:
        raise KeyError(parent_id)
    child_id = candidate_signature(child_config, {})[:16]
    child = next(
        (item for item in candidates if item.get("candidate_id") == child_id), None
    )
    if child is None:
        child = {
            "candidate_id": child_id,
            "signature": candidate_signature(child_config, {}),
            "name": name,
            "aliases": [name],
            "config_delta": deepcopy(child_config),
            "env_delta": {},
            "mechanism": mechanism,
            "sources": [{"type": "adaptive_interaction"}],
            "expected_impact": "medium",
            "dependencies": [], "conflicts": [],
            "risk": {"level": "safe", "quality_sensitive": False},
            "value_strategy": {}, "state": "eligible",
            "decision_reason": "derived from positive measured parent mechanisms",
            "measurements": [], "parent_id": parent_id, "children": [],
            "parameter": None, "value": None,
        }
        candidates.append(child)
    child["parent_id"] = parent_id
    parent.setdefault("children", [])
    if child_id not in parent["children"]:
        parent["children"].append(child_id)
    return child
