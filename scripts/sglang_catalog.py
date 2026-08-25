#!/usr/bin/env python3
"""Export the installed SGLang server argparse surface as structured JSON."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
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
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(item) for item in value), key=str)
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
    server_args_source = inspect.getsourcefile(ServerArgs)
    source_path = Path(server_args_source).resolve() if server_args_source else None

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
        "extraction_mode": "runtime_argparse",
        "repository": str(repository),
        "server_args_source": str(source_path) if source_path else None,
        "server_args_source_sha256": (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            if source_path and source_path.is_file() else None
        ),
        "parameter_count": len(parameters),
        "family_counts": family_counts,
        "parameters": parameters,
    }


def export_catalog_static(repository: Path) -> dict[str, Any]:
    """Best-effort AST contract for dependency-light compatibility CI.

    Runtime InferOpt always uses ``export_catalog``. This fallback exists only
    so the scheduled main-branch audit can detect added/removed literal
    ``add_argument`` calls before a matching GPU image is available.
    """
    source = repository / "python" / "sglang" / "srt" / "server_args.py"
    if not source.is_file():
        raise ValueError(f"SGLang ServerArgs source not found: {source}")
    tree = ast.parse(source.read_text(encoding="utf-8", errors="replace"))
    constants: dict[str, Any] = {}
    for top_level in tree.body:
        if isinstance(top_level, (ast.Assign, ast.AnnAssign)):
            targets = (
                top_level.targets if isinstance(top_level, ast.Assign)
                else [top_level.target]
            )
            try:
                constant_value = ast.literal_eval(top_level.value)
            except (ValueError, TypeError):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = constant_value

    def literal(node: ast.AST | None) -> Any:
        if node is None:
            return None
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            if isinstance(node, ast.Name):
                return constants.get(node.id, node.id)
            if isinstance(node, ast.Attribute):
                parts = []
                current: ast.AST = node
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                return ".".join(reversed(parts))
            return None

    def type_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript):
            outer = type_name(node.value)
            elements = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            if outer in {"A", "Annotated", "Optional", "Union"} and elements:
                return type_name(elements[0])
            return outer
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return type_name(node.left) or type_name(node.right)
        return None

    def dataclass_metadata(annotation: ast.AST) -> tuple[str | None, dict[str, Any]]:
        if not isinstance(annotation, ast.Subscript) or type_name(annotation.value) not in {"A", "Annotated"}:
            return type_name(annotation), {}
        elements = list(annotation.slice.elts) if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
        value_type = type_name(elements[0]) if elements else None
        metadata: dict[str, Any] = {}
        for element in elements[1:]:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                metadata.setdefault("help", element.value)
                continue
            if not isinstance(element, ast.Call):
                continue
            callee = type_name(element.func)
            if callee != "Arg":
                continue
            for keyword in element.keywords:
                if keyword.arg:
                    metadata[keyword.arg] = literal(keyword.value)
        return value_type, metadata

    by_dest: dict[str, dict[str, Any]] = {}
    server_class = next(
        (
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
        ),
        None,
    )
    if isinstance(server_class, ast.ClassDef):
        for node in server_class.body:
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            dest = node.target.id
            value_type, metadata = dataclass_metadata(node.annotation)
            if metadata.get("no_cli") is True or value_type in {"ClassVar", None}:
                continue
            cli_name = metadata.get("cli_name")
            primary = (
                str(cli_name) if isinstance(cli_name, str) and cli_name.startswith("--")
                else "--" + dest.replace("_", "-")
            )
            aliases = metadata.get("aliases") if isinstance(metadata.get("aliases"), list) else []
            flags = list(dict.fromkeys([primary, *(
                alias for alias in aliases
                if isinstance(alias, str) and alias.startswith("--")
            )]))
            default = literal(node.value)
            required = node.value is None or metadata.get("required") is True
            action = metadata.get("action")
            if action is None and value_type == "bool":
                action = "store_false" if default is True else "store_true"
            action_name = {
                "store_true": "store_true", "store_false": "store_false",
                "append": "append",
            }.get(str(action), "_StoreAction")
            help_text = metadata.get("help") if isinstance(metadata.get("help"), str) else ""
            by_dest[dest] = {
                "dest": dest, "flags": flags, "primary_flag": flags[0],
                "default": default, "required": bool(required),
                "nargs": metadata.get("nargs"), "choices": metadata.get("choices"),
                "value_type": value_type, "action": action_name, "help": help_text,
                "deprecated": "deprecated" in help_text.lower(),
                "family": parameter_family(dest),
                "source_line": int(getattr(node, "lineno", 0) or 0),
                "declaration": "ServerArgs dataclass",
            }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        flags = [
            value for arg in node.args
            if isinstance((value := literal(arg)), str) and value.startswith("--")
        ]
        if not flags:
            continue
        keywords = {keyword.arg: literal(keyword.value) for keyword in node.keywords if keyword.arg}
        dest = keywords.get("dest") or flags[0][2:].replace("-", "_")
        action = keywords.get("action")
        action_name = {
            "store_true": "store_true", "store_false": "store_false",
            "append": "append",
        }.get(str(action), "_StoreAction")
        value_type = keywords.get("type")
        if isinstance(value_type, str):
            value_type = value_type.rsplit(".", 1)[-1]
        help_text = keywords.get("help") if isinstance(keywords.get("help"), str) else ""
        by_dest[str(dest)] = {
            "dest": str(dest), "flags": flags, "primary_flag": flags[0],
            "default": keywords.get("default"),
            "required": bool(keywords.get("required", False)),
            "nargs": keywords.get("nargs"), "choices": keywords.get("choices"),
            "value_type": value_type, "action": action_name, "help": help_text,
            "deprecated": "deprecated" in help_text.lower(),
            "family": parameter_family(str(dest)),
            "source_line": int(getattr(node, "lineno", 0) or 0),
        }
    parameters = sorted(by_dest.values(), key=lambda item: (item["family"], item["dest"]))
    family_counts: dict[str, int] = {}
    for item in parameters:
        family_counts[item["family"]] = family_counts.get(item["family"], 0) + 1
    return {
        "schema_version": 1,
        "extraction_mode": "static_ast_fallback",
        "repository": str(repository),
        "server_args_source": str(source),
        "server_args_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
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
