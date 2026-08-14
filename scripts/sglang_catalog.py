#!/usr/bin/env python3
"""Export the installed SGLang server argparse surface as structured JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def json_value(value: Any) -> Any:
    if value is argparse.SUPPRESS:
        return "__SUPPRESS__"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return repr(value)


def action_kind(action: argparse.Action) -> str:
    name = type(action).__name__
    if isinstance(action, argparse._StoreTrueAction):
        return "store_true"
    if isinstance(action, argparse._StoreFalseAction):
        return "store_false"
    if isinstance(action, argparse._AppendAction):
        return "append"
    return name


def parameter_family(dest: str) -> str:
    if dest in {"tokenizer_worker_num", "detokenizer_worker_num", "scheduler_recv_interval"}:
        return "cpu_frontend"
    if dest.startswith("speculative_"):
        return "speculative"
    if dest.startswith("mamba_") or dest in {"max_mamba_cache_size", "enable_mamba_cache_stochastic_rounding"}:
        return "hybrid_mamba"
    if dest.startswith(("cuda_graph_", "disable_cuda_graph", "disable_prefill_cuda_graph", "disable_decode_cuda_graph")):
        return "cuda_graph"
    if dest.startswith(("moe_", "ep_", "deepep", "enable_eplb", "eplb_")) or dest == "ep_size":
        return "moe"
    if dest in {
        "tp_size", "pp_size", "dp_size", "dcp_size", "attn_cp_size", "moe_dp_size",
        "enable_dp_attention", "enable_dp_lm_head", "load_balance_method",
    }:
        return "parallelism"
    if "attention_backend" in dest or dest.endswith("gemm_backend") or dest in {
        "sampling_backend", "moe_runner_backend", "fp8_gemm_runner_backend",
        "fp4_gemm_runner_backend", "bf16_gemm_backend",
    }:
        return "kernel_backend"
    if dest in {
        "mem_fraction_static", "max_total_tokens", "kv_cache_dtype", "page_size",
        "disable_radix_cache", "radix_eviction_policy", "disable_chunked_prefix_cache",
    } or "cache" in dest:
        return "memory_cache"
    if dest in {
        "max_running_requests", "max_queued_requests", "chunked_prefill_size",
        "max_prefill_tokens", "prefill_max_requests", "schedule_policy",
        "schedule_conservativeness", "num_continuous_decode_steps", "enable_mixed_chunk",
        "disable_overlap_schedule", "enable_dynamic_chunking",
    }:
        return "scheduler"
    if any(token in dest for token in ("all_reduce", "msccl", "nccl", "symm_mem")):
        return "communication"
    if dest.startswith(("enable_metrics", "export_metrics", "profile", "log_")):
        return "observability"
    return "other"


def export_catalog(repository: Path) -> dict[str, Any]:
    repo_python = repository / "python"
    if not repo_python.is_dir():
        raise ValueError(f"SGLang python source not found: {repo_python}")
    sys.path.insert(0, str(repo_python))
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(parser)
    parameters = []
    for action in parser._actions:
        flags = [flag for flag in action.option_strings if flag.startswith("--")]
        if not flags:
            continue
        help_text = action.help if isinstance(action.help, str) else ""
        parameters.append({
            "dest": action.dest,
            "flags": flags,
            "primary_flag": flags[0],
            "default": json_value(action.default),
            "required": bool(action.required),
            "nargs": json_value(action.nargs),
            "choices": json_value(list(action.choices)) if action.choices is not None else None,
            "value_type": getattr(action.type, "__name__", None),
            "action": action_kind(action),
            "help": help_text,
            "deprecated": "deprecated" in help_text.lower(),
            "family": parameter_family(action.dest),
        })
    parameters.sort(key=lambda item: (item["family"], item["dest"]))
    family_counts: dict[str, int] = {}
    for item in parameters:
        family_counts[item["family"]] = family_counts.get(item["family"], 0) + 1
    return {
        "schema_version": 1,
        "repository": str(repository),
        "parameter_count": len(parameters),
        "family_counts": family_counts,
        "parameters": parameters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        catalog = export_catalog(Path(args.repository).expanduser().resolve())
        payload = json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        else:
            sys.stdout.write(payload)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
