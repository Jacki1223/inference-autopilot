"""Version-tolerant extraction of scheduler observations from SGLang logs."""

from __future__ import annotations

import re
from typing import Any


def _number(line: str, pattern: str, cast: type[int] | type[float]) -> int | float | None:
    match = re.search(pattern, line, flags=re.IGNORECASE)
    return cast(match.group(1)) if match else None


def _flag(line: str, pattern: str) -> bool | None:
    match = re.search(pattern, line, flags=re.IGNORECASE)
    return match.group(1).lower() == "true" if match else None


def _summary(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(values)

    def percentile(percent: float) -> int | float:
        return ordered[round((len(ordered) - 1) * percent)]

    return {
        "count": len(values), "min": ordered[0], "p50": percentile(0.50),
        "p95": percentile(0.95), "max": ordered[-1], "mean": sum(values) / len(values),
    }


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator <= 0 else 100.0 * numerator / denominator


def _batch_summary(records: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    def values(key: str) -> list[int | float]:
        return [record[key] for record in records if record.get(key) is not None]

    graphs = [record["cuda_graph"] for record in records if record.get("cuda_graph") is not None]
    queues = values("queue_requests")
    result: dict[str, Any] = {
        "batch_count": len(records), "running_requests": _summary(values("running_requests")),
        "queue_requests": _summary(queues),
        "queue_nonempty_batch_pct": _ratio(sum(value > 0 for value in queues), len(queues)),
        "cuda_graph_coverage_pct": _ratio(sum(graphs), len(graphs)),
        "mamba_usage_ratio": _summary(values("mamba_usage_ratio")),
    }
    if phase == "decode":
        result.update({
            "full_tokens": _summary(values("full_tokens")),
            "token_usage_ratio": _summary(values("token_usage_ratio")),
            "mamba_requests": _summary(values("mamba_requests")),
            "generation_tokens_per_sec": _summary(values("tokens_per_sec")),
        })
    else:
        new_tokens, cached_tokens = values("new_tokens"), values("cached_tokens")
        result.update({
            "new_sequences": _summary(values("new_sequences")), "new_tokens": _summary(new_tokens),
            "cached_tokens": _summary(cached_tokens),
            "cached_token_share_pct": _ratio(sum(cached_tokens), sum(cached_tokens) + sum(new_tokens)),
            "pending_tokens": _summary(values("pending_tokens")),
            "token_usage_ratio": _summary(values("token_usage_ratio")),
            "input_tokens_per_sec": _summary(values("tokens_per_sec")),
        })
    return result


def _speculative_summary(lines: list[str]) -> dict[str, Any]:
    """Extract acceptance telemetry when this SGLang revision emits it.

    Log wording changes across speculative backends, so absence is represented
    explicitly instead of being treated as zero acceptance.
    """
    acceptance: list[float] = []
    accepted_tokens: list[int] = []
    drafted_tokens: list[int] = []
    for line in lines:
        acceptance_value = _number(
            line, r"(?:speculative\s+)?acceptance(?:[_\s-]?rate)?\s*[:=]\s*([0-9.]+)", float
        )
        if acceptance_value is not None:
            acceptance.append(float(acceptance_value) * (100.0 if acceptance_value <= 1.0 else 1.0))
        accepted = _number(line, r"accepted(?:\s+draft)?\s+tokens?\s*[:=]\s*(\d+)", int)
        drafted = _number(
            line,
            r"(?:^|[,;]\s*)draft(?:ed)?\s+tokens?\s*[:=]\s*(\d+)",
            int,
        )
        if accepted is not None:
            accepted_tokens.append(int(accepted))
        if drafted is not None:
            drafted_tokens.append(int(drafted))
    inferred_pct = (
        _ratio(sum(accepted_tokens), sum(drafted_tokens))
        if accepted_tokens and drafted_tokens else None
    )
    return {
        "telemetry_available": bool(acceptance or inferred_pct is not None),
        "acceptance_rate_pct": _summary(acceptance),
        "accepted_draft_tokens": _summary(accepted_tokens),
        "drafted_tokens": _summary(drafted_tokens),
        "inferred_acceptance_rate_pct": inferred_pct,
        "policy": "unavailable telemetry is not interpreted as an MTP failure",
    }


def summarize_sglang_log(text: str) -> dict[str, Any]:
    """Parse periodic scheduler lines and optional speculative telemetry."""
    decode: list[dict[str, Any]] = []
    prefill: list[dict[str, Any]] = []
    missing_moe_configs: list[str] = []
    lines = text.splitlines()
    for line in lines:
        config_match = re.search(r"Config file not found at\s+(.+?\.json)(?:,\s+you\s+can|$)", line, flags=re.IGNORECASE)
        if config_match:
            path = config_match.group(1).strip()
            is_moe_config = "moe kernel config" in line.lower() or "fused_moe" in path.lower() or "/layers/moe/" in path.lower()
            if path not in missing_moe_configs and is_moe_config:
                missing_moe_configs.append(path)
        common = {
            "running_requests": _number(line, r"#running-req:\s*(\d+)", int),
            "queue_requests": _number(line, r"#queue-req:\s*(\d+)", int),
            "token_usage_ratio": _number(line, r"(?:full\s+)?token usage:\s*([0-9.]+)", float),
            "mamba_usage_ratio": _number(line, r"mamba usage:\s*([0-9.]+)", float),
            "cuda_graph": _flag(line, r"cuda graph:\s*(true|false)"),
        }
        if "Decode batch" in line:
            decode.append({**common, "full_tokens": _number(line, r"#(?:full\s+)?token:\s*(\d+)", int), "mamba_requests": _number(line, r"mamba num:\s*(\d+)", int), "tokens_per_sec": _number(line, r"gen throughput \(token/s\):\s*([0-9.]+)", float)})
        elif "Prefill batch" in line:
            prefill.append({**common, "new_sequences": _number(line, r"#new-seq:\s*(\d+)", int), "new_tokens": _number(line, r"#new-token:\s*(\d+)", int), "cached_tokens": _number(line, r"#cached-token:\s*(\d+)", int), "pending_tokens": _number(line, r"#pending-token:\s*(\d+)", int), "tokens_per_sec": _number(line, r"input throughput \(token/s\):\s*([0-9.]+)", float)})
    return {
        "schema_version": 2, "parser": "sglang_scheduler_batch_log",
        "decode": _batch_summary(decode, "decode"), "prefill": _batch_summary(prefill, "prefill"),
        "speculative": _speculative_summary(lines),
        "moe": {"missing_tuned_config": bool(missing_moe_configs), "missing_config_count": len(missing_moe_configs), "missing_config_files": missing_moe_configs, "requires_down_kernel_config": any("_down.json" in path for path in missing_moe_configs)},
    }
