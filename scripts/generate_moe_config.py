#!/usr/bin/env python3
"""Generate SGLang fused-MoE Triton configuration artifacts.

The default ``paired`` mode invokes SGLang's separate-kernel tuner and writes
both the regular and ``_down`` configuration files.  It intentionally refuses
to guess routing data: the official separate tuner requires top-k captures.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import autopilot


DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def _version(python: str, env: dict[str, str]) -> str:
    result = subprocess.run(
        [python, "-c", "import triton; print(triton.__version__)"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("cannot determine the installed Triton version")
    return result.stdout.strip()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    progress = autopilot.ProgressReporter(getattr(args, "progress", "plain"))
    progress.emit(
        "generate-moe-config", "validating paths, topology and tuner mode",
        completed=0, total=4,
    )
    repository = Path(args.repository).expanduser().resolve()
    python = str(Path(args.python).expanduser().resolve())
    model = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not repository.is_dir():
        raise ValueError(f"repository does not exist: {repository}")
    if not Path(python).is_file():
        raise ValueError(f"python executable does not exist: {python}")
    if not model.is_dir():
        raise ValueError(f"model directory does not exist: {model}")
    if args.tp_size <= 0 or args.ep_size <= 0 or args.tp_size % args.ep_size:
        raise ValueError("tp-size must be positive and divisible by ep-size")

    if args.mode == "paired":
        if not args.topk_ids_dir:
            raise ValueError(
                "paired mode requires --topk-ids-dir containing official SGLang top-k captures; "
                "refusing to fabricate an _down config"
            )
        topk_dir = Path(args.topk_ids_dir).expanduser().resolve()
        if not topk_dir.is_dir():
            raise ValueError(f"topk-ids-dir does not exist: {topk_dir}")
        tuner_rel = "benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py"
    else:
        topk_dir = None
        tuner_rel = "benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py"
    tuner = repository / tuner_rel
    if not tuner.is_file():
        raise ValueError(f"SGLang tuner not found: {tuner}")

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "tuner-work"
    work_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    repo_python = str(repository / "python")
    env["PYTHONPATH"] = repo_python + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    version = _version(python, env)
    progress.emit(
        "generate-moe-config", f"Triton {version} detected; preparing tuner command",
        completed=1, total=4,
    )
    command = [python, str(tuner), "--model", str(model), "--tp-size", str(args.tp_size), "--ep-size", str(args.ep_size)]
    if args.dtype != "auto":
        command += ["--dtype", args.dtype]
    if args.per_channel_quant:
        command.append("--per-channel-quant")
    if args.disable_shared_experts_fusion:
        command.append("--disable-shared-experts-fusion")
    if args.mode == "paired":
        command += ["--topk-ids-dir", str(topk_dir), "--tune"]
    else:
        command += ["--batch-sizes", *(str(value) for value in args.batch_sizes), "--tune"]
        if args.search_space_file:
            command += ["--search-space-file", str(Path(args.search_space_file).expanduser().resolve())]

    if not args.yes:
        raise ValueError("config generation is GPU-intensive; pass --yes after reviewing the command")
    print("[inferopt] running:", " ".join(command), file=sys.stderr, flush=True)
    stdout_log = output_dir / "tuner.stdout.log"
    stderr_log = output_dir / "tuner.stderr.log"
    progress.emit(
        "generate-moe-config", f"GPU tuner running; logs={stdout_log},{stderr_log}",
        completed=1, total=4,
    )
    with stdout_log.open("w", encoding="utf-8") as tuner_stdout, stderr_log.open(
        "w", encoding="utf-8"
    ) as tuner_stderr:
        process = subprocess.Popen(
            command, cwd=work_dir, env=env, text=True,
            stdout=tuner_stdout, stderr=tuner_stderr,
        )
        started = time.monotonic()
        last_heartbeat = started
        timeout = args.timeout_minutes * 60
        while process.poll() is None:
            now = time.monotonic()
            if now - started >= timeout:
                process.kill()
                process.wait(timeout=10)
                raise RuntimeError(
                    f"SGLang tuner exceeded {args.timeout_minutes:g} minutes; inspect {stdout_log} and {stderr_log}"
                )
            if now - last_heartbeat >= 30:
                progress.emit(
                    "generate-moe-config",
                    f"GPU tuner still running; elapsed={now - started:.0f}s, logs={stdout_log},{stderr_log}",
                    completed=1, total=4,
                )
                last_heartbeat = now
            time.sleep(1)
    if process.returncode != 0:
        raise RuntimeError(
            f"SGLang tuner exited with code {process.returncode}; inspect {stdout_log} and {stderr_log}"
        )
    progress.emit(
        "generate-moe-config", "tuner completed; validating generated config pairs",
        completed=2, total=4,
    )

    generated = sorted(work_dir.glob("E=*.json"))
    if not generated:
        raise RuntimeError("SGLang tuner completed without producing E=*.json files")
    names = {path.name for path in generated}
    if args.mode == "paired":
        up = {name for name in names if not name.endswith("_down.json")}
        down = {name.removesuffix("_down.json") for name in names if name.endswith("_down.json")}
        if not up or not down or not (up & down):
            raise RuntimeError("separate tuner did not produce matching normal and _down config files")

    config_root = output_dir / "config-root"
    target_dir = config_root / "configs" / f"triton_{version.replace('.', '_')}"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in generated:
        valid, reason = autopilot.validate_moe_config_artifact(source)
        if not valid:
            raise RuntimeError(reason or f"invalid config: {source.name}")
        target = target_dir / source.name
        shutil.copy2(source, target)
        copied.append(str(target))
    progress.emit(
        "generate-moe-config", f"validated and copied {len(copied)} config files",
        completed=4, total=4,
    )
    summary = {
        "status": "completed",
        "mode": args.mode,
        "triton_version": version,
        "config_root": str(config_root),
        "generated_files": copied,
        "command": command,
        "topk_ids_dir": str(topk_dir) if topk_dir else None,
        "deployment": f"export SGLANG_MOE_CONFIG_DIR={config_root}",
    }
    output = Path(args.output).expanduser().resolve() if args.output else output_dir / "result.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, help="SGLang source checkout")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used by SGLang")
    parser.add_argument("--model-path", required=True, help="local model checkpoint")
    parser.add_argument("--output-dir", required=True, help="private output directory")
    parser.add_argument("--output", help="summary JSON path")
    parser.add_argument("--mode", choices=["paired", "standard"], default="paired")
    parser.add_argument("--topk-ids-dir", help="required in paired mode")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["auto", "fp8_w8a8", "int8_w8a8", "int8_w8a16", "int4_w4a16"], default="auto")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--search-space-file")
    parser.add_argument("--per-channel-quant", action="store_true")
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    parser.add_argument("--timeout-minutes", type=float, default=120)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--progress", choices=["plain", "json", "none"], default="plain"
    )
    return parser


def main() -> int:
    try:
        _run(build_parser().parse_args())
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
