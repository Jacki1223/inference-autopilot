#!/usr/bin/env python3
"""Generate an auditable SGLang ServerArgs compatibility report for CI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optimization_rules import known_rule_parameters
from parameter_evolution import build_parameter_contract, diff_parameter_contract
from sglang_catalog import export_catalog, export_catalog_static


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_report(difference: dict, current: dict, unknown: list[str]) -> str:
    lines = [
        "# SGLang parameter compatibility update",
        "",
        f"- Contract: `{current.get('contract_hash')}`",
        f"- Diff status: `{difference.get('status')}`",
        f"- Parameters: `{len(current.get('parameters', {}))}`",
        "",
        "## Added",
        "",
        *(f"- `{name}`" for name in difference.get("added", [])),
        "",
        "## Removed",
        "",
        *(f"- `{name}`" for name in difference.get("removed", [])),
        "",
        "## Changed",
        "",
        *(f"- `{name}`: `{sorted(fields)}`" for name, fields in difference.get("changed", {}).items()),
        "",
        "## Current parameters without a versioned optimization rule",
        "",
        *(f"- `{name}`" for name in unknown),
        "",
        "Unknown parameters are not automatically trusted. Run InferOpt semantic/safety analysis "
        "and real benchmarks before promoting them to validated rules.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--contract-output", required=True)
    parser.add_argument("--diff-output", required=True)
    parser.add_argument("--report-output", required=True)
    args = parser.parse_args()
    repository = Path(args.repository).expanduser().resolve()
    try:
        catalog = export_catalog(repository)
    except Exception as exc:
        catalog = export_catalog_static(repository)
        catalog["runtime_extraction_error"] = f"{type(exc).__name__}: {exc}"
    current = build_parameter_contract(catalog)
    previous = None
    if args.baseline and Path(args.baseline).is_file():
        previous = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    difference = diff_parameter_contract(previous, current)
    known = known_rule_parameters()
    unknown = sorted(
        name for name, item in current["parameters"].items()
        if name not in known and item.get("cli_visible", True) and not item.get("deprecated")
    )
    write_json(Path(args.contract_output), current)
    write_json(Path(args.diff_output), {**difference, "unknown_rule_parameters": unknown})
    Path(args.report_output).write_text(
        markdown_report(difference, current, unknown), encoding="utf-8"
    )
    actionable = difference.get("status") == "changed" and bool(
        difference.get("added") or difference.get("removed") or difference.get("changed")
    )
    return 3 if actionable else 0


if __name__ == "__main__":
    raise SystemExit(main())
