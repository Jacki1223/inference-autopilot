#!/usr/bin/env python3
"""Track and safely explore SGLang ServerArgs evolution.

The runtime path is deterministic and does not require an LLM.  It combines
the live argparse contract, source references, local Cookbook evidence and a
versioned safety policy. Unknown flags are audited by default; only bounded,
high-confidence performance dials enter an explicitly enabled provisional
exploration quota.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import math
import re
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CONTRACT_SCHEMA_VERSION = 1
STATES = {
    "validated_rule",
    "experimentally_supported",
    "provisional",
    "unclassified",
    "control_plane",
    "quality_sensitive",
    "unsafe",
    "deprecated",
    "removed",
    "inapplicable",
}
SAFE_ACTIONS = {"store_true", "store_false", "_StoreAction"}
SAFE_VALUE_TYPES = {None, "bool", "int", "float", "str"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((json_safe(item) for item in value), key=str)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return repr(value)


def _load_resource(name: str, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        resource = importlib.resources.files("inference_autopilot_data").joinpath(name)
        return json.loads(resource.read_text(encoding="utf-8"))
    except (AttributeError, FileNotFoundError, ModuleNotFoundError, TypeError, json.JSONDecodeError):
        local = Path(__file__).resolve().parents[1] / "references" / name
        try:
            return json.loads(local.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return deepcopy(fallback)


DEFAULT_SEMANTIC_PATTERNS = {
    "help_keywords": {
        "prefill_chunking": ["chunked prefill", "dynamic chunk", "prefill chunk"],
        "prefill_admission": ["max prefill", "prefill token", "prefill request"],
        "scheduler_cadence": ["scheduler interval", "continuous decode", "scheduler step"],
        "scheduler_overlap": ["overlap schedule", "overlap scheduler"],
        "kv_capacity": ["kv cache capacity", "total token", "memory fraction"],
        "kv_layout": ["page size", "cache page", "kv page"],
        "prefix_cache": ["radix cache", "prefix cache"],
        "decode_cuda_graph": ["decode cuda graph"],
        "prefill_cuda_graph": ["prefill cuda graph"],
        "attention_backend": ["attention backend"],
        "moe_kernel_backend": ["moe runner", "moe backend"],
        "collective_backend": ["all reduce", "collective", "msccl", "nccl"],
        "speculative_decoding": ["speculative", "draft token", "eagle", "nextn"],
    },
    "source_path_keywords": {
        "prefill_chunking": ["chunk", "prefill"],
        "scheduler_cadence": ["scheduler"],
        "kv_capacity": ["memory_pool", "token_pool", "kv_cache"],
        "kv_layout": ["radix_cache", "page"],
        "decode_cuda_graph": ["cuda_graph"],
        "attention_backend": ["attention"],
        "moe_kernel_backend": ["/moe/", "fused_moe"],
        "collective_backend": ["all_reduce", "distributed", "nccl"],
        "speculative_decoding": ["speculative", "eagle", "mtp"],
    },
    "submechanism_family": {
        "prefill_chunking": "scheduler",
        "prefill_admission": "scheduler",
        "scheduler_cadence": "scheduler",
        "scheduler_overlap": "scheduler",
        "kv_capacity": "memory_cache",
        "kv_layout": "memory_cache",
        "prefix_cache": "memory_cache",
        "decode_cuda_graph": "cuda_graph",
        "prefill_cuda_graph": "cuda_graph",
        "attention_backend": "kernel_backend",
        "moe_kernel_backend": "moe",
        "collective_backend": "communication",
        "speculative_decoding": "speculative",
    },
}

DEFAULT_SAFETY_POLICY = {
    "control_plane_tokens": [
        "host", "port", "api_key", "admin", "auth", "password", "secret",
        "path", "dir", "folder", "url", "endpoint", "socket", "dist_init",
        "node_rank", "nnodes", "crash", "dump", "log_level", "metrics_port",
    ],
    "unsafe_tokens": [
        "delete", "overwrite", "unsafe", "abort", "kill", "terminate", "debug",
        "test_only", "mock", "dummy",
    ],
    "quality_sensitive_tokens": [
        "dtype", "quant", "precision", "sparse", "approx", "truncate", "accuracy",
    ],
    "performance_tokens": [
        "batch", "cache", "chunk", "prefill", "decode", "scheduler", "graph",
        "backend", "overlap", "memory", "token", "parallel", "attention", "moe",
        "speculative", "all_reduce", "collective",
    ],
    "minimum_provisional_confidence": 0.80,
    "maximum_enum_candidates": 4,
}


def semantic_patterns() -> dict[str, Any]:
    return _load_resource("parameter-semantic-patterns.json", DEFAULT_SEMANTIC_PATTERNS)


def safety_policy() -> dict[str, Any]:
    return _load_resource("parameter-safety-policy.json", DEFAULT_SAFETY_POLICY)


def normalized_parameter(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "dest": item.get("dest"),
        "flags": sorted(str(flag) for flag in item.get("flags", [])),
        "primary_flag": item.get("primary_flag"),
        "default": json_safe(item.get("default")),
        "required": bool(item.get("required", False)),
        "nargs": json_safe(item.get("nargs")),
        "choices": json_safe(item.get("choices")),
        "value_type": item.get("value_type"),
        "action": item.get("action"),
        "help": str(item.get("help") or ""),
        "deprecated": bool(item.get("deprecated", False)),
        "family": item.get("family", "other"),
        "cli_visible": bool(item.get("cli_visible", True)),
    }


def build_parameter_contract(
    parameter_catalog: dict[str, Any], framework: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parameters = {
        str(item["dest"]): normalized_parameter(item)
        for item in parameter_catalog.get("parameters", [])
        if isinstance(item, dict) and item.get("dest")
    }
    payload = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "framework_commit": (framework or {}).get("git_commit"),
        "server_args_sha256": (framework or {}).get("server_args_sha256"),
        "launch_server_help_sha256": (framework or {}).get("launch_server_help_sha256"),
        "catalog_extraction_mode": parameter_catalog.get("extraction_mode"),
        "catalog_source_sha256": parameter_catalog.get("server_args_source_sha256"),
        "parameters": parameters,
    }
    return {**payload, "contract_hash": hash_payload(payload)}


def _changed_fields(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    changed = {}
    for field in (
        "flags", "primary_flag", "default", "required", "nargs", "choices",
        "value_type", "action", "help", "deprecated", "family", "cli_visible",
    ):
        if old.get(field) != new.get(field):
            changed[field] = {"old": deepcopy(old.get(field)), "new": deepcopy(new.get(field))}
    return changed


def _rename_hints(
    removed: dict[str, dict[str, Any]], added: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    hints = []
    for old_name, old in removed.items():
        old_help = str(old.get("help") or "").lower()
        for new_name, new in added.items():
            new_help = str(new.get("help") or "").lower()
            help_similarity = SequenceMatcher(None, old_help, new_help).ratio() if old_help and new_help else 0.0
            name_similarity = SequenceMatcher(None, old_name, new_name).ratio()
            shared_flags = set(old.get("flags", [])) & set(new.get("flags", []))
            score = max(help_similarity, name_similarity * 0.8, 0.98 if shared_flags else 0.0)
            if score >= 0.72:
                hints.append({
                    "removed": old_name, "added": new_name,
                    "confidence": round(score, 4),
                    "policy": "advisory only; never migrate a rule without current-contract validation",
                })
    return sorted(hints, key=lambda item: item["confidence"], reverse=True)[:32]


def diff_parameter_contract(
    previous: dict[str, Any] | None, current: dict[str, Any],
) -> dict[str, Any]:
    current_parameters = current.get("parameters", {})
    if not previous:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "first_observation",
            "previous_contract_hash": None,
            "current_contract_hash": current.get("contract_hash"),
            "added": sorted(current_parameters), "removed": [], "changed": {},
            "rename_hints": [],
        }
    previous_parameters = previous.get("parameters", {})
    added_names = sorted(set(current_parameters) - set(previous_parameters))
    removed_names = sorted(set(previous_parameters) - set(current_parameters))
    changed = {
        name: fields
        for name in sorted(set(current_parameters) & set(previous_parameters))
        if (fields := _changed_fields(previous_parameters[name], current_parameters[name]))
    }
    status = "unchanged" if not added_names and not removed_names and not changed else "changed"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "previous_contract_hash": previous.get("contract_hash"),
        "current_contract_hash": current.get("contract_hash"),
        "added": added_names,
        "removed": removed_names,
        "changed": changed,
        "rename_hints": _rename_hints(
            {name: previous_parameters[name] for name in removed_names},
            {name: current_parameters[name] for name in added_names},
        ),
    }


def source_references(repository: str | Path, parameter: str, limit: int = 32) -> list[str]:
    root = Path(repository) / "python" / "sglang" / "srt"
    if not root.is_dir():
        return []
    flag = "--" + parameter.replace("_", "-")
    needles = (f".{parameter}", f"['{parameter}']", f'\"{parameter}\"', flag)
    matches = []
    for path in root.rglob("*.py"):
        if path.name == "server_args.py":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(needle in text for needle in needles):
            matches.append(str(path.relative_to(Path(repository))))
            if len(matches) >= limit:
                break
    return sorted(matches)


def source_reference_index(
    repository: str | Path, parameters: set[str], limit_per_parameter: int = 32,
) -> dict[str, list[str]]:
    """Scan SGLang once and build a bounded parameter-to-source index."""
    root = Path(repository) / "python" / "sglang" / "srt"
    index: dict[str, list[str]] = {parameter: [] for parameter in parameters}
    if not root.is_dir() or not parameters:
        return index
    attribute_pattern = re.compile(
        r"(?:server_args|args|self\.server_args)\.([a-zA-Z_][a-zA-Z0-9_]*)"
    )
    string_pattern = re.compile(
        r"['\"]((?:--)?[a-zA-Z_][a-zA-Z0-9_-]*)['\"]"
    )
    for path in root.rglob("*.py"):
        if path.name == "server_args.py":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        observed = set(attribute_pattern.findall(text))
        for token in string_pattern.findall(text):
            observed.add(token.lstrip("-").replace("-", "_"))
        relative = str(path.relative_to(Path(repository)))
        for parameter in observed & parameters:
            if len(index[parameter]) < limit_per_parameter:
                index[parameter].append(relative)
    return {parameter: sorted(paths) for parameter, paths in index.items()}


def cookbook_parameter_evidence(cookbook: dict[str, Any], parameter: str) -> list[dict[str, Any]]:
    evidence = []
    for source_name in ("local_checkout", "repository_snapshot"):
        source = cookbook.get(source_name, {})
        if not isinstance(source, dict):
            continue
        for recipe in source.get("recipes", []):
            if isinstance(recipe, dict) and parameter in recipe.get("config", {}):
                evidence.append({
                    "kind": "recipe", "name": recipe.get("name"),
                    "value": deepcopy(recipe["config"][parameter]),
                    "companion_config": {
                        key: deepcopy(value)
                        for key, value in recipe.get("config", {}).items()
                        if key != parameter
                    },
                    "source": deepcopy(recipe.get("source")),
                    "requirements": deepcopy(recipe.get("requirements", [])),
                    "hardware_affinity": deepcopy(recipe.get("hardware_affinity", [])),
                    "documented_model": recipe.get("documented_model"),
                })
            if isinstance(recipe, dict) and parameter in recipe.get("unrecognized_config", {}):
                evidence.append({
                    "kind": "new_recipe_option", "name": recipe.get("name"),
                    "value": deepcopy(recipe["unrecognized_config"][parameter]),
                    "source": deepcopy(recipe.get("source")),
                    "companion_config": deepcopy(recipe.get("config", {})),
                    "requirements": deepcopy(recipe.get("requirements", [])),
                    "hardware_affinity": deepcopy(recipe.get("hardware_affinity", [])),
                    "documented_model": recipe.get("documented_model"),
                    "policy": "semantic evidence only until safety classification passes",
                })
        for tip in source.get("tuning_tips", []):
            if isinstance(tip, dict) and parameter in {
                *tip.get("parameters", []), *tip.get("unrecognized_parameters", [])
            }:
                evidence.append({
                    "kind": "tip", "path": tip.get("path"), "text": tip.get("text"),
                })
    return evidence[:32]


def _tokens_match(text: str, tokens: list[str]) -> list[str]:
    lowered = text.lower()
    return [
        token for token in tokens
        if re.search(
            rf"(?<![a-z0-9]){re.escape(token.lower())}(?![a-z0-9])",
            lowered,
        )
    ]


def _semantic_votes(
    item: dict[str, Any], source_paths: list[str], cookbook_evidence: list[dict[str, Any]],
) -> tuple[str, dict[str, float], list[str]]:
    patterns = semantic_patterns()
    votes: dict[str, float] = {}
    evidence = []
    help_text = str(item.get("help") or "") + " " + str(item.get("dest") or "")
    for mechanism, keywords in patterns.get("help_keywords", {}).items():
        hits = _tokens_match(help_text, keywords)
        if hits:
            votes[mechanism] = votes.get(mechanism, 0.0) + min(0.45, 0.25 + 0.05 * len(hits))
            evidence.append(f"help_keywords.{mechanism}={hits}")
    joined_paths = " ".join(source_paths).lower()
    for mechanism, keywords in patterns.get("source_path_keywords", {}).items():
        hits = _tokens_match(joined_paths, keywords)
        if hits:
            votes[mechanism] = votes.get(mechanism, 0.0) + min(0.30, 0.15 + 0.05 * len(hits))
            evidence.append(f"source_paths.{mechanism}={hits}")
    if cookbook_evidence:
        cookbook_text = " ".join(str(item) for item in cookbook_evidence)
        for mechanism, keywords in patterns.get("help_keywords", {}).items():
            hits = _tokens_match(cookbook_text, keywords)
            if hits:
                votes[mechanism] = votes.get(mechanism, 0.0) + 0.30
                evidence.append(f"cookbook.{mechanism}={hits}")
    family = str(item.get("family", "other"))
    family_fallback = {
        "scheduler": "scheduler_cadence",
        "memory_cache": "kv_capacity",
        "cuda_graph": "decode_cuda_graph",
        "kernel_backend": "attention_backend",
        "moe": "moe_kernel_backend",
        "communication": "collective_backend",
        "speculative": "speculative_decoding",
    }.get(family)
    if family_fallback:
        votes[family_fallback] = votes.get(family_fallback, 0.0) + 0.08
        evidence.append(f"catalog_family={family}")
    mechanism = max(votes, key=votes.get) if votes else f"catalog:{family}"
    return mechanism, votes, evidence


def _range_from_help(help_text: str) -> tuple[float, float] | None:
    patterns = (
        r"between\s+([-+]?[0-9]*\.?[0-9]+)\s+and\s+([-+]?[0-9]*\.?[0-9]+)",
        r"range\s*[\[\(]\s*([-+]?[0-9]*\.?[0-9]+)\s*,\s*([-+]?[0-9]*\.?[0-9]+)",
        r"min(?:imum)?\s*[=:]?\s*([-+]?[0-9]*\.?[0-9]+).*max(?:imum)?\s*[=:]?\s*([-+]?[0-9]*\.?[0-9]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, help_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            low, high = float(match.group(1)), float(match.group(2))
            if low < high:
                return low, high
    return None


def bounded_candidate_values(item: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    policy = safety_policy()
    default = item.get("default")
    action = item.get("action")
    choices = item.get("choices")
    if action == "store_true":
        return ([True] if default is not True else []), {"strategy": "boolean_inverse"}
    if action == "store_false":
        return ([False] if default is not False else []), {"strategy": "boolean_inverse"}
    if isinstance(choices, list) and choices:
        values = [value for value in choices if value != default]
        cap = int(policy.get("maximum_enum_candidates", 4))
        return values[:cap], {"strategy": "current_serverargs_choices", "choice_count": len(choices)}
    value_type = item.get("value_type")
    if value_type not in {"int", "float"}:
        return [], {"strategy": "no_safe_generic_value_set"}
    bounds = _range_from_help(str(item.get("help") or ""))
    if bounds is None or not isinstance(default, (int, float)) or isinstance(default, bool):
        return [], {"strategy": "numeric_range_not_declared"}
    low, high = bounds
    raw = [low, (low + high) / 2, high]
    if value_type == "int":
        raw = [int(round(value)) for value in raw]
    values = []
    for value in raw:
        if value != default and value not in values:
            values.append(value)
    return values, {"strategy": "declared_help_range", "bounds": [low, high]}


def cookbook_bounded_values(
    item: dict[str, Any], evidence: list[dict[str, Any]],
) -> tuple[list[Any], dict[str, Any]]:
    """Use documented scalar values as evidence, never as a trusted bundle."""
    value_type = item.get("value_type")
    choices = item.get("choices")
    values = []
    for record in evidence:
        if record.get("kind") not in {"recipe", "new_recipe_option"}:
            continue
        value = record.get("value")
        if isinstance(value, (dict, list)) or value is None:
            continue
        if value_type == "int" and (not isinstance(value, int) or isinstance(value, bool)):
            continue
        if value_type == "float" and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            continue
        if value_type == "str" and not isinstance(value, str):
            continue
        if isinstance(choices, list) and value not in choices:
            continue
        if value != item.get("default") and value not in values:
            values.append(value)
    cap = int(safety_policy().get("maximum_enum_candidates", 4))
    return values[:cap], {
        "strategy": "current_cookbook_documented_scalars",
        "evidence_values": len(values),
        "policy": "provisional only; current ServerArgs and runtime smoke still gate execution",
    }


def infer_parameter_semantics(
    item: dict[str, Any], repository: str | Path, cookbook: dict[str, Any],
    *, known_parameters: set[str], explicitly_added: bool,
    indexed_source_paths: list[str] | None = None,
    discovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parameter = str(item["dest"])
    policy = safety_policy()
    combined = " ".join((parameter, str(item.get("primary_flag") or ""), str(item.get("help") or ""))).lower()
    control_hits = _tokens_match(combined, policy.get("control_plane_tokens", []))
    unsafe_hits = _tokens_match(combined, policy.get("unsafe_tokens", []))
    quality_hits = _tokens_match(combined, policy.get("quality_sensitive_tokens", []))
    performance_hits = _tokens_match(combined, policy.get("performance_tokens", []))
    sources = (
        indexed_source_paths
        if indexed_source_paths is not None
        else source_references(repository, parameter)
    )
    cookbook_evidence = cookbook_parameter_evidence(cookbook, parameter)
    mechanism, votes, mechanism_evidence = _semantic_votes(item, sources, cookbook_evidence)
    values, value_strategy = bounded_candidate_values(item)
    if not values and cookbook_evidence:
        cookbook_values, cookbook_strategy = cookbook_bounded_values(
            item, cookbook_evidence
        )
        if cookbook_values:
            values, value_strategy = cookbook_values, cookbook_strategy
    confidence = 0.05
    if performance_hits:
        confidence += min(0.25, 0.10 + 0.03 * len(performance_hits))
    if item.get("family") != "other":
        confidence += 0.15
    if sources:
        confidence += min(0.35, 0.20 + 0.02 * len(sources))
    if cookbook_evidence:
        confidence += 0.25
    if values:
        confidence += 0.10
    if votes:
        confidence += min(0.20, max(votes.values()) * 0.25)
    confidence = min(0.99, confidence)
    companion_configs = [
        deepcopy(record.get("companion_config", {}))
        for record in cookbook_evidence
        if isinstance(record.get("companion_config"), dict)
        and record.get("companion_config")
    ]
    requirements = sorted({
        str(requirement)
        for record in cookbook_evidence
        for requirement in record.get("requirements", [])
    })
    hardware_affinity = sorted({
        str(hardware)
        for record in cookbook_evidence
        for hardware in record.get("hardware_affinity", [])
    })
    discovered_vendor = (discovery or {}).get("hardware", {}).get("vendor")
    discovered_model = (discovery or {}).get("model", {})
    applicability_failures = []
    if "nvidia_gpu" in requirements and discovered_vendor != "nvidia":
        applicability_failures.append("requires NVIDIA")
    if "amd_gpu" in requirements and discovered_vendor != "amd":
        applicability_failures.append("requires AMD")
    if "checkpoint.is_hybrid" in requirements and not discovered_model.get("is_hybrid"):
        applicability_failures.append("requires a hybrid checkpoint")
    if "checkpoint.has_mtp_weights" in requirements and not discovered_model.get("has_mtp_weights"):
        applicability_failures.append("requires checkpoint MTP weights")

    state = "unclassified"
    reason = "insufficient bounded performance semantics"
    if item.get("deprecated") or not item.get("cli_visible", True):
        state, reason = "deprecated", "not active in the current SGLang CLI"
    elif applicability_failures:
        state, reason = "inapplicable", "; ".join(applicability_failures)
    elif parameter in known_parameters:
        state, reason = "validated_rule", "covered by a versioned InferOpt rule or known Cookbook contract"
    elif control_hits:
        state, reason = "control_plane", f"control-plane identity/path tokens: {control_hits}"
    elif unsafe_hits:
        state, reason = "unsafe", f"unsafe/debug behavior tokens: {unsafe_hits}"
    elif quality_hits:
        state, reason = "quality_sensitive", f"precision/quality tokens require explicit quality evaluation: {quality_hits}"
    elif (
        explicitly_added
        and confidence >= float(policy.get("minimum_provisional_confidence", 0.8))
        and bool(values)
        and item.get("action") in SAFE_ACTIONS
        and item.get("value_type") in SAFE_VALUE_TYPES
        and performance_hits
    ):
        state, reason = "provisional", "new bounded performance dial with high-confidence local evidence"

    return {
        "parameter": parameter,
        "state": state,
        "reason": reason,
        "family": item.get("family", "other"),
        "submechanism": mechanism,
        "confidence": round(confidence, 4),
        "candidate_values": values if state == "provisional" else [],
        "value_strategy": value_strategy,
        "companion_configs": companion_configs,
        "applicability": {
            "requirements": requirements,
            "hardware_affinity": hardware_affinity,
            "failures": applicability_failures,
        },
        "risk": {
            "control_plane": bool(control_hits),
            "unsafe": bool(unsafe_hits),
            "quality_sensitive": bool(quality_hits),
        },
        "evidence": {
            "performance_tokens": performance_hits,
            "mechanism_votes": {key: round(value, 4) for key, value in votes.items()},
            "mechanism_evidence": mechanism_evidence,
            "source_files": sources,
            "cookbook": cookbook_evidence,
        },
    }


def normalized_evolution_policy(task: dict[str, Any]) -> dict[str, Any]:
    config = task.get("parameter_evolution") or {}
    mode = str(config.get("mode", "conservative"))
    percentage = float(config.get("exploration_budget_pct", 10.0))
    minimum_confidence = float(
        config.get("minimum_confidence", safety_policy().get("minimum_provisional_confidence", 0.8))
    )
    maximum_trials = config.get("max_provisional_trials")
    return {
        "mode": mode,
        "exploration_budget_pct": min(25.0, max(0.0, percentage)),
        "minimum_confidence": min(1.0, max(0.0, minimum_confidence)),
        "max_provisional_trials": int(maximum_trials) if isinstance(maximum_trials, int) else None,
    }


def provisional_trial_budget(task: dict[str, Any], candidate_count: int) -> dict[str, Any]:
    policy = normalized_evolution_policy(task)
    total = int(task.get("budget", {}).get("max_trials", 0) or 0)
    mode = str(task.get("experiment_mode", "balanced"))
    if policy["mode"] != "experimental" or mode == "fast" or candidate_count <= 0:
        slots = 0
    else:
        percentage_slots = max(1, int(math.floor(total * policy["exploration_budget_pct"] / 100.0)))
        mode_cap = 2 if mode == "balanced" else 6
        configured_cap = policy["max_provisional_trials"]
        slots = min(candidate_count, percentage_slots, mode_cap)
        if isinstance(configured_cap, int):
            slots = min(slots, max(0, configured_cap))
    return {
        "slots": slots,
        "candidate_count": candidate_count,
        "policy": policy,
        "allocation": "reserved from discovery candidates; never from confirmation",
    }


def analyze_parameter_evolution(
    task: dict[str, Any], discovery: dict[str, Any],
    previous_contract: dict[str, Any] | None,
    *, known_parameters: set[str],
) -> dict[str, Any]:
    contract = build_parameter_contract(
        discovery["parameter_catalog"], discovery.get("framework")
    )
    difference = diff_parameter_contract(previous_contract, contract)
    added = set(difference["added"])
    version_additions_are_actionable = difference.get("status") == "changed"
    source_index = source_reference_index(
        task["repository"],
        {
            parameter for parameter in added
            if parameter not in known_parameters
        } if version_additions_are_actionable else set(),
    )
    analyses = []
    for parameter, item in contract["parameters"].items():
        analyses.append(infer_parameter_semantics(
            item, task["repository"], discovery.get("cookbook", {}),
            known_parameters=known_parameters,
            explicitly_added=version_additions_are_actionable and parameter in added,
            indexed_source_paths=source_index.get(parameter, []),
            discovery=discovery,
        ))
    analyses.sort(key=lambda item: (
        0 if item["state"] == "provisional" else 1,
        -float(item["confidence"]), item["parameter"],
    ))
    provisional = [
        item for item in analyses
        if item["state"] == "provisional"
        and item["confidence"] >= normalized_evolution_policy(task)["minimum_confidence"]
    ]
    policy = normalized_evolution_policy(task)
    analysis_by_parameter = {item["parameter"]: item for item in analyses}
    for parameter, fields in difference.get("changed", {}).items():
        choices_change = fields.get("choices") if isinstance(fields, dict) else None
        if not isinstance(choices_change, dict):
            continue
        old_choices = choices_change.get("old")
        new_choices = choices_change.get("new")
        if not isinstance(old_choices, list) or not isinstance(new_choices, list):
            continue
        added_choices = [value for value in new_choices if value not in old_choices]
        base = analysis_by_parameter.get(parameter)
        if not added_choices or not isinstance(base, dict):
            continue
        risk = base.get("risk", {})
        if risk.get("control_plane") or risk.get("unsafe"):
            continue
        if risk.get("quality_sensitive"):
            continue
        cap = int(safety_policy().get("maximum_enum_candidates", 4))
        provisional.append({
            **deepcopy(base),
            "state": "provisional",
            "reason": "current ServerArgs added bounded choices to an existing parameter",
            "confidence": max(0.95, float(base.get("confidence", 0))),
            "candidate_values": added_choices[:cap],
            "value_strategy": {
                "strategy": "new_current_serverargs_choices",
                "old_choices": old_choices,
                "added_choices": added_choices,
            },
            "provisional_kind": "choice_extension",
        })
    provisional = sorted(
        {
            (item["parameter"], canonical_json(item.get("candidate_values", []))): item
            for item in provisional
        }.values(),
        key=lambda item: (-float(item.get("confidence", 0)), item["parameter"]),
    )
    budget = provisional_trial_budget(task, len(provisional))
    return {
        "schema_version": SCHEMA_VERSION,
        "current_contract": contract,
        "contract_diff": difference,
        "policy": policy,
        "state_counts": {
            state: sum(1 for item in analyses if item["state"] == state)
            for state in sorted(STATES)
        },
        "parameters": analyses,
        "provisional_candidates": provisional,
        "exploration_budget": budget,
        "previous_contract_available": previous_contract is not None,
        "policy_summary": (
            "new flags are audited by default; only explicit experimental mode may execute "
            "bounded high-confidence provisional candidates"
        ),
    }


def select_provisional_candidates(
    evolution: dict[str, Any], bottleneck_classification: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    slots = int(evolution.get("exploration_budget", {}).get("slots", 0) or 0)
    if slots <= 0:
        return []
    labels = {
        (bottleneck_classification or {}).get("primary"),
        *((bottleneck_classification or {}).get("secondary") or []),
    }
    label_mechanisms = {
        "kv_memory_capacity_bound": {"kv_capacity", "kv_layout", "prefill_admission", "prefill_chunking"},
        "host_scheduler_bound": {"scheduler_cadence", "scheduler_overlap", "prefill_chunking"},
        "prefill_attention_bound": {"attention_backend", "prefill_chunking", "prefill_cuda_graph"},
        "decode_attention_bound": {"attention_backend", "decode_cuda_graph", "speculative_decoding"},
        "moe_compute_bound": {"moe_kernel_backend"},
        "communication_bound": {"collective_backend"},
    }
    preferred = set().union(*(label_mechanisms.get(label, set()) for label in labels))
    ranked = sorted(
        evolution.get("provisional_candidates", []),
        key=lambda item: (
            0 if item.get("submechanism") in preferred else 1,
            -float(item.get("confidence", 0)),
            item.get("parameter", ""),
        ),
    )
    return deepcopy(ranked[:slots])


def compact_evolution_summary(evolution: dict[str, Any]) -> dict[str, Any]:
    difference = evolution.get("contract_diff", {})
    return {
        "contract_hash": evolution.get("current_contract", {}).get("contract_hash"),
        "diff_status": difference.get("status"),
        "added": difference.get("added", []),
        "removed": difference.get("removed", []),
        "changed_parameters": sorted(difference.get("changed", {})),
        "rename_hints": difference.get("rename_hints", []),
        "state_counts": evolution.get("state_counts", {}),
        "provisional_candidates": [
            {
                "parameter": item.get("parameter"), "state": item.get("state"),
                "submechanism": item.get("submechanism"),
                "confidence": item.get("confidence"),
                "candidate_values": item.get("candidate_values", []),
            }
            for item in evolution.get("provisional_candidates", [])
        ],
        "exploration_budget": evolution.get("exploration_budget", {}),
        "policy": evolution.get("policy", {}),
        "previous_contract_available": evolution.get(
            "previous_contract_available", False
        ),
        "persistence": deepcopy(evolution.get("persistence", {})),
    }
