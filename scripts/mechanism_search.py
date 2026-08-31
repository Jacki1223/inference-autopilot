"""Mechanism-level adaptive scheduling for expensive serving trials."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from candidate_registry import IMPACT_ORDER, candidate_signature, normalize_mechanism


MECHANISM_SEARCH_SCHEMA_VERSION = 5


TERMINAL_STATES = {
    "measured_negative", "slo_failed", "measurement_failed", "historically_failed",
    "unsupported", "inapplicable", "blocked", "already_effective",
}


def _candidate_priority(item: dict[str, Any]) -> tuple[Any, ...]:
    source_bonus = sum(
        2 if source.get("type") == "cookbook" else
        1 if source.get("type") == "trigger_rule" else 0
        for source in item.get("sources", [])
        if isinstance(source, dict)
    )
    return (
        -float(item.get("selection_score", 0.0)),
        -IMPACT_ORDER.get(item.get("expected_impact", "low"), 1),
        -source_bonus,
        int(item.get("value_rank", 0)),
        str(item.get("name", "")),
    )


def contextual_semantic_mechanisms(
    registry: dict[str, Any], *, limit: int | None = None,
) -> list[str]:
    """Return high-confidence, safe mechanisms activated by live context.

    Parameter evolution may recognize a newly added ServerArgs flag without a
    handwritten tuning rule.  When its semantic evidence is strong, reserve a
    mechanism-level discovery slot rather than letting lower-risk but generic
    high-score knobs consume the entire screen.  The caller supplies a bounded
    limit so semantic evolution cannot make experiment cost unbounded.
    """
    best_by_mechanism: dict[str, dict[str, Any]] = {}
    for item in registry.get("candidates", []):
        if item.get("state") != "eligible":
            continue
        risk = item.get("risk", {})
        if any(risk.get(key) for key in (
            "unsafe", "quality_sensitive", "provisional", "control_plane",
        )):
            continue
        semantic = False
        for source in item.get("sources", []):
            if not isinstance(source, dict) or source.get("type") != "parameter_capability_registry":
                continue
            metadata = source.get("source", {})
            confidence = metadata.get("confidence", 0.0)
            if (
                metadata.get("state") == "semantically_eligible"
                and isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
                and float(confidence) >= 0.9
            ):
                semantic = True
                break
        if not semantic:
            continue
        mechanism = normalize_mechanism(item.get("mechanism"))
        current = best_by_mechanism.get(mechanism)
        if current is None or _candidate_priority(item) < _candidate_priority(current):
            best_by_mechanism[mechanism] = item
    ordered = sorted(
        best_by_mechanism,
        key=lambda mechanism: _candidate_priority(best_by_mechanism[mechanism]),
    )
    if limit is not None:
        ordered = ordered[:max(0, int(limit))]
    return ordered


def initial_mechanism_schedule(
    registry: dict[str, Any], *, budget: int,
    mandatory_parameters: list[str] | tuple[str, ...] = (),
    mandatory_mechanisms: list[str] | tuple[str, ...] = (),
    provisional_slots: int = 0,
    breadth_target: int | None = None,
    max_values_per_parameter: int = 2,
) -> dict[str, Any]:
    """Meet a bounded breadth floor, then spend depth by contextual utility."""
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

    mandatory_mechanism_selected: set[str] = set()
    for mechanism in mandatory_mechanisms:
        normalized = normalize_mechanism(mechanism)
        for item in mechanisms.get(normalized, []):
            if admit(item):
                mandatory_mechanism_selected.add(normalized)
                break

    # First pass: establish only the mode-specific causal breadth floor. The
    # old all-mechanism pass consumed every balanced slot when many mechanisms
    # matched, leaving no budget for nonlinear value response or a highly
    # scenario-specific mechanism such as shared-prefix scheduling.
    represented_mechanisms = {
        normalize_mechanism(item.get("mechanism")) for item in selected
    }
    target = min(
        budget,
        len(mechanisms),
        max(len(represented_mechanisms), int(
            breadth_target if breadth_target is not None else budget
        )),
    )
    for mechanism in sorted(
        mechanisms,
        key=lambda name: min(_candidate_priority(item) for item in mechanisms[name]),
    ):
        if len(represented_mechanisms) >= target:
            break
        if mechanism in represented_mechanisms:
            continue
        for item in mechanisms[mechanism]:
            if admit(item):
                represented_mechanisms.add(mechanism)
                break
    # Second pass: rank every remaining candidate globally by contextual score.
    # A per-parameter cap prevents one continuous knob from monopolizing the
    # screen, while still allowing balanced/max runs to observe value shape.
    parameter_counts: dict[str, int] = {}
    mechanism_counts: dict[str, int] = {}
    for item in selected:
        mechanism = normalize_mechanism(item.get("mechanism"))
        mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
        parameter = item.get("parameter")
        if isinstance(parameter, str):
            parameter_counts[parameter] = parameter_counts.get(parameter, 0) + 1
    def depth_priority(item: dict[str, Any]) -> tuple[Any, ...]:
        mechanism = normalize_mechanism(item.get("mechanism"))
        exploration_bonus = 8.0 if mechanism not in represented_mechanisms else 0.0
        priority = _candidate_priority(item)
        return (
            -(float(item.get("selection_score", 0.0)) + exploration_bonus),
            *priority[1:],
        )

    remaining = [
        item for item in eligible if item["candidate_id"] not in selected_ids
    ]
    cap = max(1, int(max_values_per_parameter))
    cap_deferred: list[dict[str, Any]] = []
    while remaining and len(selected) < budget:
        remaining.sort(key=depth_priority)
        item = remaining.pop(0)
        parameter = item.get("parameter")
        mechanism = normalize_mechanism(item.get("mechanism"))
        if isinstance(parameter, str) and parameter_counts.get(parameter, 0) >= cap:
            cap_deferred.append(item)
            continue
        if mechanism_counts.get(mechanism, 0) >= cap:
            cap_deferred.append(item)
            continue
        if admit(item):
            represented_mechanisms.add(mechanism)
            mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
            if isinstance(parameter, str):
                parameter_counts[parameter] = parameter_counts.get(parameter, 0) + 1
    # If the cap is the only reason budget remains, fill by utility rather than
    # silently returning unused discovery slots.
    for item in sorted([*remaining, *cap_deferred], key=_candidate_priority):
        if len(selected) >= budget:
            break
        admit(item)

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
        "policy": (
            "mandatory controls, bounded contextual breadth, then highest-utility "
            "depth with a per-parameter value cap"
        ),
        "selected": selected,
        "deferred": deferred,
        "scheduled_mechanisms": sorted({normalize_mechanism(item.get("mechanism")) for item in selected}),
        # Backward-compatible alias. Reports must label this as scheduled, not measured.
        "covered_mechanisms": sorted({normalize_mechanism(item.get("mechanism")) for item in selected}),
        "budget": budget,
        "provisional_slots": provisional_slots,
        "provisional_used": provisional_used,
        "breadth_target": target,
        "max_values_per_parameter": cap,
        "mandatory_mechanisms": sorted({
            normalize_mechanism(value) for value in mandatory_mechanisms
        }),
        "mandatory_mechanisms_selected": sorted(mandatory_mechanism_selected),
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
        elif measured and not gains and remaining:
            state = "fallback_required"
            reason = (
                "the first mechanism representative failed or produced no valid metric; "
                "an unevaluated sibling must be tried before blocking the mechanism"
            )
        elif measured and not gains:
            state = "blocked_or_failed"
            reason = "no SLO-valid numeric measurement completed and no sibling remains"
        elif measured and remaining:
            state = "fallback_required"
            reason = (
                "the measured value was negative, but unmeasured categorical or "
                "directional siblings remain; reject the value, not the mechanism"
            )
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
        if outcome.get("state") not in {
            "promising", "uncertain", "fallback_required",
        }:
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
            0 if outcome_by_mechanism[name]["state"] == "fallback_required" else 1,
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

    # Refinement is an elastic search tier, not a use-it-or-lose-it list of
    # positive siblings. If the positive/uncertain mechanisms cannot consume
    # their allocation, continue with the highest-scoring deferred candidates
    # instead of donating those trials directly to expensive confirmation.
    selected_ids = {item["candidate_id"] for item in selected}
    continued_exploration: list[dict[str, Any]] = []
    if len(selected) < budget:
        for item in sorted(
            (
                value for value in registry.get("candidates", [])
                if value.get("state") == "eligible"
                and value.get("candidate_id") not in selected_ids
            ),
            key=_candidate_priority,
        ):
            full_config = {**anchor_config, **item.get("config_delta", {})}
            signature = json.dumps(
                full_config, sort_keys=True, separators=(",", ":")
            )
            if signature in evaluated_signatures:
                continue
            candidate = deepcopy(item)
            candidate["full_config"] = full_config
            candidate["followup_reason"] = (
                "unused refinement capacity continued the highest-scoring "
                "deferred exploration candidate"
            )
            selected.append(candidate)
            continued_exploration.append(candidate)
            selected_ids.add(candidate["candidate_id"])
            evaluated_signatures.add(signature)
            if len(selected) >= budget:
                break

    return {
        "schema_version": MECHANISM_SEARCH_SCHEMA_VERSION,
        "policy": (
            "refine promising/uncertain mechanisms first; a negative value retains "
            "eligible directional or categorical siblings; unused refinement slots "
            "continue the highest-scoring deferred exploration candidates"
        ),
        "selected": selected,
        "continued_exploration": [
            {
                "candidate_id": item.get("candidate_id"),
                "name": item.get("name"),
                "mechanism": normalize_mechanism(item.get("mechanism")),
                "selection_score": item.get("selection_score"),
            }
            for item in continued_exploration
        ],
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
