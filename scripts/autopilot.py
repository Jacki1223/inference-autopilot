#!/usr/bin/env python3
"""Run one-shot, hardware-aware SGLang deployment optimization."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.resources
import json
import math
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from autotune import (
    ALLOWED_ENV, candidate_matrix, capability_family, command_manifest,
    configuration_accelerator_count,
    decision_report, execute, execution_errors, write_json,
)
from inferopt import (
    METRIC_DIRECTIONS, RESOURCE_NORMALIZABLE_METRICS, RESOURCE_SCOPES,
    SLO_MAPPING, dump_json, load_json, resource_adjusted_metric_value,
    summary_accelerator_count,
)
from profile_sglang import diagnose_existing, run_profile
from trial_store import (
    compatibility_fingerprint as history_compatibility_fingerprint,
    ingest_final as ingest_trial_history,
    ingest_parameter_evidence,
    latest_parameter_contract,
    parameter_evidence_priors,
    save_parameter_contract,
    warm_start_candidates,
)
from optimization_rules import (
    MAGNITUDE_ORDER,
    RULESET_VERSION,
    classify_bottleneck,
    configuration_history_prior,
    dynamic_parameter_values,
    history_priors,
    known_rule_parameters,
    match_parameter_rules,
    parameter_submechanism,
    tiered_trial_budget,
)
from parameter_evolution import (
    analyze_parameter_evolution,
    build_parameter_contract,
    compact_evolution_summary,
    select_provisional_candidates,
    select_semantic_candidates,
)
from candidate_registry import (
    CANDIDATE_REGISTRY_SCHEMA_VERSION,
    CandidateRegistry,
    canonical_bottleneck_report,
    mark_registry_scheduled,
    normalize_mechanism,
    posterior_bottleneck_report,
    registry_from_search_plan,
    update_registry_from_aggregates,
)
from mechanism_search import (
    MECHANISM_SEARCH_SCHEMA_VERSION,
    adaptive_followup_schedule,
    contextual_semantic_mechanisms,
    initial_mechanism_schedule,
)


SEARCH_POLICY_VERSION = "2026.08.29.1"


def optimizer_contract() -> dict[str, Any]:
    """Version evidence-selection semantics independently from raw profiles."""
    return {
        "search_policy_version": SEARCH_POLICY_VERSION,
        "candidate_registry_schema": CANDIDATE_REGISTRY_SCHEMA_VERSION,
        "mechanism_search_schema": MECHANISM_SEARCH_SCHEMA_VERSION,
        "optimization_ruleset": RULESET_VERSION,
    }


REQUIRED_TOP_LEVEL = {
    "name",
    "repository",
    "python",
    "model_path",
    "workload",
    "slo",
    "objective",
    "budget",
    "output_dir",
}
OPTIONAL_TOP_LEVEL = {
    "offline",
    "allow_download",
    "env",
    "confirmation_repetitions",
    "port",
    "model",
    "profiling",
    "profile_dir",
    "measurement",
    "deployment_mode",
    "experiment_mode",
    "calibration",
    "search_depth",
    "knowledge",
    "capability_overrides",
    "deployment",
    "quality",
    "model_variants",
    "kernel_tuning",
    "parallel_trials",
    "max_gpus",
    "resume_run_dir",
    "history",
    "economics",
    "parameter_evolution",
}

# Cookbook content now lives with the SGLang source tree.  Keeping the
# documentation source aligned with the server checkout prevents recipes from
# proposing flags that belong to a different SGLang revision.
DEFAULT_COOKBOOK_REPOSITORY = "https://github.com/sgl-project/sglang.git"
COOKBOOK_DOCUMENT_EXTENSIONS = {".md", ".mdx"}
COOKBOOK_TUNABLE_FLAGS = {
    "attention_backend", "prefill_attention_backend", "decode_attention_backend",
    "chunked_prefill_size", "cuda_graph_max_bs_decode", "cuda_graph_max_bs_prefill",
    "enable_torch_compile", "torch_compile_max_bs", "disable_overlap_schedule",
    "enable_mixed_chunk", "enable_flashinfer_allreduce_fusion",
    "moe_runner_backend", "moe_a2a_backend", "moe_dp_size",
    "mamba_radix_cache_strategy", "max_mamba_cache_size", "mamba_ssm_dtype",
    "mamba_full_memory_ratio", "max_running_requests", "max_total_tokens",
    "mem_fraction_static", "num_continuous_decode_steps", "page_size",
    "schedule_conservativeness", "schedule_policy", "tp_size", "pp_size",
    "dp_size", "ep_size", "speculative_algorithm", "speculative_num_steps",
    "speculative_eagle_topk", "speculative_num_draft_tokens",
    "linear_attn_prefill_backend", "linear_attn_decode_backend",
    "ple_offload_embedding",
}
COOKBOOK_FLAG_ALIASES = {
    "tp": "tp_size", "tensor_parallel_size": "tp_size",
    "pp": "pp_size", "pipeline_parallel_size": "pp_size",
    "dp": "dp_size", "data_parallel_size": "dp_size",
    "ep": "ep_size", "expert_parallel_size": "ep_size",
    "speculative_algo": "speculative_algorithm",
}
COOKBOOK_BOOLEAN_FLAGS = {
    "enable_mixed_chunk", "enable_flashinfer_allreduce_fusion",
    "enable_torch_compile", "disable_overlap_schedule", "ple_offload_embedding",
}
COOKBOOK_NEGATED_BOOLEAN_FLAGS = {
    "no_ple_offload_embedding": "ple_offload_embedding",
}
COOKBOOK_REQUIRED_FUNCTIONAL_FLAGS = {
    "reasoning_parser", "tool_call_parser", "chat_template",
    "completion_template", "default_chat_template_kwargs",
}
MODE_CANDIDATE_LIMITS = {"fast": 8, "balanced": 14, "max": 28}


def normalized_experiment_mode(task: dict[str, Any]) -> str:
    """Map the retired rigorous spelling to the current max mode."""
    mode = task.get("experiment_mode", "balanced")
    return "max" if mode == "rigorous" else str(mode)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _linux_memory_value(path: Path, key: str) -> int | None:
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.startswith(key + ":"):
                continue
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                multiplier = 1024 if len(fields) >= 3 and fields[2].lower() == "kb" else 1
                return int(fields[1]) * multiplier
    except OSError:
        return None
    return None


def _integer_file(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def host_memory_inventory() -> dict[str, Any]:
    """Collect effective Host RAM and NUMA capacity without mutating the host."""
    meminfo = Path("/proc/meminfo")
    total = _linux_memory_value(meminfo, "MemTotal")
    available = _linux_memory_value(meminfo, "MemAvailable")
    cgroup_limit = _integer_file(Path("/sys/fs/cgroup/memory.max"))
    cgroup_current = _integer_file(Path("/sys/fs/cgroup/memory.current"))
    if cgroup_limit is None:
        cgroup_limit = _integer_file(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        cgroup_current = _integer_file(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    if isinstance(cgroup_limit, int) and cgroup_limit >= (1 << 60):
        cgroup_limit = None
    cgroup_available = (
        max(0, cgroup_limit - int(cgroup_current or 0))
        if isinstance(cgroup_limit, int) else None
    )
    available_values = [
        value for value in (available, cgroup_available)
        if isinstance(value, int) and value >= 0
    ]
    effective_available = min(available_values) if available_values else None
    numa_nodes = []
    for node in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        node_total = _linux_memory_value(node / "meminfo", "MemTotal")
        node_free = _linux_memory_value(node / "meminfo", "MemFree")
        numa_nodes.append({
            "node": node.name,
            "total_bytes": node_total,
            "free_bytes": node_free,
        })
    return {
        "total_bytes": total,
        "available_bytes": available,
        "cgroup_limit_bytes": cgroup_limit,
        "cgroup_current_bytes": cgroup_current,
        "effective_available_bytes": effective_available,
        "numa_nodes": numa_nodes,
        "numa_node_count": len(numa_nodes),
        "source": "linux_procfs_cgroup_sysfs",
        "available": isinstance(effective_available, int),
    }


def model_kv_bytes_per_token(model: dict[str, Any]) -> int | None:
    layers = model.get("num_hidden_layers")
    hidden = model.get("hidden_size")
    heads = model.get("num_attention_heads")
    kv_heads = model.get("num_key_value_heads", heads)
    if not all(isinstance(value, int) and value > 0 for value in (layers, hidden, heads, kv_heads)):
        return None
    head_dim = int(hidden) // int(heads)
    dtype = str(model.get("kv_cache_dtype") or model.get("checkpoint_dtype") or "bfloat16").lower()
    bytes_per_element = 1 if "fp8" in dtype or "int8" in dtype else 2
    return 2 * int(layers) * int(kv_heads) * head_dim * bytes_per_element


def prefix_workload_analysis(
    task: dict[str, Any], model: dict[str, Any], hardware: dict[str, Any],
    minimum_tp_size: int,
) -> dict[str, Any]:
    """Estimate reusable-prefix working set and device KV capacity conservatively."""
    workload = task.get("workload", {})
    declared_reuse = float(workload.get("prefix_reuse_ratio", 0.0) or 0.0)
    shared = workload.get("shared_prefix")
    if isinstance(shared, dict):
        prefix_tokens = int(shared.get("system_prompt_tokens", 0) or 0)
        groups = int(shared.get("groups", 1) or 1)
        working_set_tokens = prefix_tokens * groups
        reuse_ratio = prefix_tokens / max(1, int(workload.get("input_tokens", 1)))
        source = "synthetic_shared_prefix_exact"
        confidence = "exact"
    else:
        concurrency = int(workload.get("max_concurrency", 1) or 1)
        input_tokens = int(workload.get("input_tokens", 1) or 1)
        working_set_tokens = int(round(input_tokens * concurrency * declared_reuse))
        reuse_ratio = declared_reuse
        source = "declared_prefix_reuse_ratio"
        confidence = "declared"
    kv_bytes = model_kv_bytes_per_token(model)
    gpus = selected_gpus(task, hardware)
    memory_bytes = min(
        (int(gpu.get("memory_mib", 0)) * 1024**2 for gpu in gpus if int(gpu.get("memory_mib", 0)) > 0),
        default=0,
    )
    weight_per_rank = int(model.get("weight_bytes", 0) or 0) / max(1, minimum_tp_size)
    estimated_kv_pool_per_rank = max(0, memory_bytes * 0.80 - weight_per_rank - 2 * 1024**3)
    device_capacity_tokens = (
        int(estimated_kv_pool_per_rank * max(1, minimum_tp_size) / kv_bytes)
        if isinstance(kv_bytes, int) and kv_bytes > 0 else None
    )
    return {
        "prefix_reuse_ratio": round(reuse_ratio, 4),
        "working_set_tokens": working_set_tokens,
        "source": source,
        "confidence": confidence,
        "kv_bytes_per_token_estimate": kv_bytes,
        "device_kv_pool_bytes_per_rank_estimate": int(estimated_kv_pool_per_rank),
        "device_capacity_tokens_estimate": device_capacity_tokens,
        "working_set_exceeds_device_capacity": (
            working_set_tokens > device_capacity_tokens
            if isinstance(device_capacity_tokens, int) else None
        ),
        "policy": "capacity estimate is conservative and is replaced by resolved runtime KV evidence when available",
    }


class ProgressReporter:
    """Render concise one-shot execution progress without changing artifacts."""

    def __init__(self, mode: str = "plain") -> None:
        if mode not in {"plain", "json", "none"}:
            raise ValueError("progress mode must be plain, json, or none")
        self.mode = mode
        self.started = time.monotonic()
        # Progress is observational only.  A detached SSH/CI stdout pipe can
        # disappear while GPU trials remain healthy; never turn that into a
        # failed optimization run.
        self._output_available = True
        self._stage_state: dict[str, dict[str, Any]] = {}

    @staticmethod
    def bar(completed: int, total: int, width: int = 20) -> str:
        if total <= 0:
            return "[" + "-" * width + "] 0/0 0%"
        bounded = min(max(0, completed), total)
        filled = min(width, int(width * bounded / total))
        percent = int(100 * bounded / total)
        return f"[{'#' * filled}{'-' * (width - filled)}] {bounded}/{total} {percent}%"

    def emit(
        self, stage: str, message: str, *, completed: int | None = None, total: int | None = None,
    ) -> None:
        if not self._output_available:
            return
        elapsed = int(time.monotonic() - self.started)
        progress = (
            " " + self.bar(completed, total)
            if completed is not None and total is not None
            else ""
        )
        try:
            if self.mode == "none":
                return
            if self.mode == "json":
                print(json.dumps({
                    "event": "progress", "elapsed_sec": elapsed,
                    "stage": stage, "message": message,
                    "completed": completed, "total": total,
                }, sort_keys=True), file=sys.stderr, flush=True)
            else:
                print(
                    f"[inferopt +{elapsed // 60:02d}:{elapsed % 60:02d}] {stage}{progress}: {message}",
                    file=sys.stderr,
                    flush=True,
                )
        except BrokenPipeError:
            self._output_available = False
        except OSError as exc:
            if exc.errno == 32:  # EPIPE
                self._output_available = False
            else:
                raise

    def reset_stage(self, stage: str, total: int = 0) -> None:
        self._stage_state[stage] = {
            "total": max(0, int(total)), "active": set(), "done": set(),
        }

    def _counts(
        self, stage: str, event: dict[str, Any]
    ) -> tuple[dict[str, Any], int, int, int]:
        state = self._stage_state.setdefault(
            stage, {"total": 0, "active": set(), "done": set()}
        )
        event_total = int(event.get("trial_count", state["total"]) or 0)
        state["total"] = max(int(state["total"]), event_total)
        completed = len(state["done"])
        active = len(state["active"])
        queued = max(0, int(state["total"]) - completed - active)
        return state, completed, active, queued

    def trial(self, stage: str, event: dict[str, Any]) -> None:
        index = event["trial_index"]
        total = event["trial_count"]
        state, completed, active, queued = self._counts(stage, event)
        key = str(event.get("trial_name", index))
        if event["event"] == "trial_plan_updated":
            self.emit(
                stage,
                f"search plan updated: {event.get('message', 'trial set changed')}; "
                f"active={active}, queued={queued}, completed={completed}",
                completed=completed, total=state["total"],
            )
            return
        if event["event"] == "trial_skipped":
            state["active"].discard(key)
            state["done"].add(key)
            completed = len(state["done"])
            queued = max(0, int(state["total"]) - completed - len(state["active"]))
            self.emit(
                stage,
                f"trial {index}/{total} {event['trial_name']}: skipped "
                f"({event['capability']} unavailable: {event['reason']}); "
                f"active={len(state['active'])}, queued={queued}",
                completed=completed,
                total=state["total"],
            )
            return
        if event["event"] == "trial_started":
            state["active"].add(key)
            completed = len(state["done"])
            active = len(state["active"])
            queued = max(0, int(state["total"]) - completed - active)
            parallel = event.get("parallel_workers")
            worker_note = (
                f" in parallel batch of {parallel} exclusive-GPU workers"
                if isinstance(parallel, int) and parallel > 1 else ""
            )
            self.emit(
                stage,
                f"trial {index}/{total} {event['trial_name']}: queued for server startup{worker_note}; "
                f"active={active}, queued={queued}",
                completed=completed,
                total=state["total"],
            )
            return
        if event["event"] == "trial_phase":
            phase = str(event.get("phase", "working"))
            message = str(event.get("message", phase.replace("_", " ")))
            self.emit(
                stage,
                f"trial {index}/{total} {event['trial_name']} [{phase}]: {message}; "
                f"active={len(state['active'])}, queued={queued}",
                completed=len(state["done"]), total=state["total"],
            )
            return
        if event["event"] == "trial_worker_finished":
            state["active"].discard(key)
            state["done"].add(key)
            completed = len(state["done"])
            active = len(state["active"])
            queued = max(0, int(state["total"]) - completed - active)
            self.emit(
                stage,
                f"trial {index}/{total} {event['trial_name']}: GPU worker finished; "
                f"result accounting pending; active={active}, queued={queued}",
                completed=completed, total=state["total"],
            )
            return
        if event["event"] == "bayesian_update":
            posterior = event.get("posterior", {})
            self.emit(
                stage,
                "Bayesian update after "
                f"{posterior.get('blocks', 'unknown')} paired blocks: "
                f"{posterior.get('action', 'inconclusive')}",
                completed=len(state["done"]),
                total=state["total"],
            )
            return
        state["active"].discard(key)
        state["done"].add(key)
        completed = len(state["done"])
        active = len(state["active"])
        queued = max(0, int(state["total"]) - completed - active)
        if not event.get("ok"):
            self.emit(
                stage,
                f"trial {index}/{total} failed: {event.get('detail') or 'unknown error'}; "
                f"active={active}, queued={queued}",
                completed=completed,
                total=state["total"],
            )
            return
        metrics = event.get("metrics", {})
        rps = metrics.get("request_throughput_rps")
        p99 = metrics.get("p99_e2e_latency_ms")
        summary = []
        if isinstance(rps, (int, float)):
            summary.append(f"{rps:.3f} RPS")
        if isinstance(p99, (int, float)):
            summary.append(f"p99 E2E {p99:.1f} ms")
        summary.append("SLO pass" if event.get("slo_passed") else "SLO not passed")
        self.emit(
            stage,
            f"trial {index}/{total} completed ({', '.join(summary)}); "
            f"active={active}, queued={queued}",
            completed=completed,
            total=state["total"],
        )


def execute_with_progress(
    spec: dict[str, Any], reporter: ProgressReporter, stage: str
) -> dict[str, Any]:
    reporter.reset_stage(stage, int(spec.get("budget", {}).get("max_trials", 0)))
    reporter.emit(stage, "preparing experiment")
    report = execute(spec, progress=lambda event: reporter.trial(stage, event))
    reporter.emit(
        stage,
        f"finished: {report['completed_trials']}/{report['planned_trials']} trials, "
        f"{report['stop_reason']}",
        completed=(
            report["planned_trials"]
            if report["stop_reason"] == "completed_search"
            else report["completed_trials"] + len(report.get("skipped_capability_trials", []))
        ),
        total=report["planned_trials"],
    )
    return report


def run_readonly(
    command: list[str], timeout: int = 15, cwd: str | None = None, env: dict[str, str] | None = None
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
            env=env,
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError:
        return {"available": False}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "error": type(exc).__name__}


def workload_dataset(workload: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized benchmark data source for a task."""
    configured = workload.get("dataset")
    if configured is None:
        return {"name": "synthetic"}
    if not isinstance(configured, dict):
        return {"name": "invalid"}
    return configured


def structurally_valid_dataset_rows(path: Path, dataset_name: str, stop_after: int) -> int:
    """Count rows SGLang can inspect before tokenizer-dependent filtering."""
    def valid_conversation(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        conversations = value.get("conversations", value.get("conversation", []))
        if not isinstance(conversations, list) or len(conversations) < 2:
            return False
        for turn in conversations[:2]:
            if not isinstance(turn, dict):
                return False
            content = turn.get("content", turn.get("value"))
            if not isinstance(content, str) or not content.strip():
                return False
        return True

    count = 0
    if dataset_name == "custom":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if valid_conversation(value):
                    count += 1
                    if count >= stop_after:
                        break
        return count
    value = load_json(path)
    rows = value if isinstance(value, list) else []
    for row in rows:
        if valid_conversation(row):
            count += 1
            if count >= stop_after:
                break
    return count


def confirmation_request_count(task: dict[str, Any]) -> int:
    """Use half of the task measurement window for the expensive final pair."""
    measurement = task.get("measurement") or {}
    configured = measurement.get("confirmation_requests")
    if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
        requested = configured
    else:
        workload = task.get("workload") if isinstance(task.get("workload"), dict) else {}
        fallback = workload.get("num_prompts", 1)
        base_value = measurement.get("min_measurement_requests", fallback)
        base = int(base_value) if isinstance(base_value, int) and not isinstance(base_value, bool) else 1
        requested = max(1, math.ceil(base / 2))
    workload = task.get("workload") if isinstance(task.get("workload"), dict) else {}
    if task.get("slo"):
        # Tail-SLO confirmation needs several full concurrency waves even when
        # the normal half-size confirmation shortcut is smaller.
        requested = max(requested, int(workload.get("max_concurrency", 1)) * 5)
        if has_p99_latency_slo(task):
            request_waves = int(measurement.get("p99_request_waves", 10))
            requested = max(
                requested, int(workload.get("max_concurrency", 1)) * request_waves
            )
    return requested


OFFLINE_SCREENING_SATURATION_WAVES = 10
OFFLINE_REFINEMENT_SATURATION_WAVES = 10
OFFLINE_CONFIRMATION_SATURATION_WAVES = 10
OFFLINE_PROFILE_SATURATION_WAVES = 3


def offline_saturation_waves(
    task: dict[str, Any], *, confirmation: bool = False,
    phase: str = "screening",
) -> int:
    """Return one accurate offline no-SLO window for every candidate tier."""
    if confirmation or phase == "confirmation":
        return OFFLINE_CONFIRMATION_SATURATION_WAVES
    if phase in {"refinement", "composition"}:
        return OFFLINE_REFINEMENT_SATURATION_WAVES
    return OFFLINE_SCREENING_SATURATION_WAVES


def offline_saturation_request_count(
    task: dict[str, Any], *, confirmation: bool = False
) -> int | None:
    """Return the post-startup request floor for unconstrained offline tests.

    Omitting ``--max-concurrency`` only removes the client throttle; it does
    not create a backlog.  After SGLang reports its admission capacity, the
    benchmark must submit several capacity waves to measure saturation.
    """
    if not reference_baseline_mode(task):
        return None
    workload = task.get("workload")
    if not isinstance(workload, dict):
        return None
    capacity = workload.get(
        "observed_practical_capacity", workload.get("observed_admission_capacity")
    )
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        return None
    waves = offline_saturation_waves(task, confirmation=confirmation)
    return capacity * waves


def max_running_request_candidates(workload: dict[str, Any]) -> list[int]:
    """Probe the throughput plateau above SGLang's observed admission limit.

    The automatic value is the safe baseline, not proof that it is the
    throughput optimum.  Start with modest, eight-aligned increases so a
    failed memory admission is isolated to one candidate rather than forcing
    an arbitrary value such as 256 on every model.
    """
    capacity = workload.get("observed_admission_capacity")
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        return []

    def round_up_to_eight(value: float) -> int:
        return max(1, int(math.ceil(value / 8.0)) * 8)

    return list(dict.fromkeys([
        round_up_to_eight(capacity * 1.25),
        round_up_to_eight(capacity * 1.60),
    ]))


def has_latency_slo(task: dict[str, Any]) -> bool:
    """Return whether a task constrains average or tail latency."""
    slo = task.get("slo", {})
    return isinstance(slo, dict) and any(
        isinstance(key, str) and key.startswith(("mean_", "p99_"))
        for key in slo
    )


def has_p99_latency_slo(task: dict[str, Any]) -> bool:
    """Return whether a task needs the larger tail-latency sample window."""
    slo = task.get("slo", {})
    return isinstance(slo, dict) and any(
        isinstance(key, str) and key.startswith("p99_") for key in slo
    )


def uses_runtime_capacity_slo_calibration(task: dict[str, Any]) -> bool:
    """Whether online SLO calibration must start from SGLang's live capacity."""
    calibration = task.get("calibration") or {}
    return (
        task.get("deployment_mode") == "online_latency"
        and bool(task.get("slo"))
        and calibration.get("enabled", True) is not False
        and calibration.get("strategy", "adaptive") == "adaptive"
        and not isinstance(calibration.get("concurrencies"), list)
    )


def validate_task(task: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in sorted(REQUIRED_TOP_LEVEL - set(task)):
        errors.append(f"missing required field: {key}")
    for key in sorted(set(task) - REQUIRED_TOP_LEVEL - OPTIONAL_TOP_LEVEL):
        errors.append(f"unsupported field: {key}")
    name = task.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
        errors.append("name must be a safe 1-64 character identifier")
    for key in ("repository", "python", "model_path", "output_dir"):
        value = task.get(key)
        if not isinstance(value, str) or not Path(value).expanduser().is_absolute():
            errors.append(f"{key} must be an absolute path")
    if isinstance(task.get("repository"), str) and not Path(task["repository"]).is_dir():
        errors.append("repository must exist")
    if isinstance(task.get("python"), str) and not Path(task["python"]).is_file():
        errors.append("python must exist")
    if isinstance(task.get("model_path"), str) and not Path(task["model_path"]).is_dir():
        errors.append("model_path must be an existing local directory")
    if isinstance(task.get("output_dir"), str) and Path(task["output_dir"]).expanduser() == Path("/"):
        errors.append("output_dir must not be the filesystem root")
    port = task.get("port", 31000)
    if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
        errors.append("port must be an integer between 1024 and 65535")
    for section in ("workload", "slo", "objective", "budget"):
        if not isinstance(task.get(section), dict):
            errors.append(f"{section} must be an object")
    workload = task.get("workload", {})
    offline_task = task.get("deployment_mode") == "offline_throughput"
    required_workload_fields = ["input_tokens", "output_tokens", "num_prompts"]
    if not offline_task and not uses_runtime_capacity_slo_calibration(task):
        required_workload_fields.append("max_concurrency")
    for key in required_workload_fields:
        value = workload.get(key) if isinstance(workload, dict) else None
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"workload.{key} must be a positive integer")
    if offline_task and "max_concurrency" not in workload:
        bootstrap = workload.get("initial_backlog_requests")
        if not isinstance(bootstrap, int) or isinstance(bootstrap, bool) or bootstrap <= 0:
            errors.append("offline workload.initial_backlog_requests must be a positive integer")
    request_rate = workload.get("request_rate", "inf") if isinstance(workload, dict) else None
    if request_rate != "inf" and (
        not isinstance(request_rate, (int, float)) or isinstance(request_rate, bool) or request_rate <= 0
    ):
        errors.append("workload.request_rate must be positive or 'inf'")
    if isinstance(workload, dict) and workload.get("shared_prefix") is not None:
        try:
            shared_prefix_benchmark(workload)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    if isinstance(workload, dict):
        dataset = workload_dataset(workload)
        allowed_dataset_fields = {"name", "path", "apply_chat_template"}
        if any(key not in allowed_dataset_fields for key in dataset):
            errors.append("workload.dataset supports only name, path, and apply_chat_template")
        dataset_name = dataset.get("name", "synthetic")
        if dataset_name not in {"synthetic", "custom", "sharegpt"}:
            errors.append("workload.dataset.name must be synthetic, custom, or sharegpt")
        if dataset_name in {"custom", "sharegpt"}:
            dataset_path = dataset.get("path")
            path = Path(dataset_path).expanduser() if isinstance(dataset_path, str) else None
            if path is None or not path.is_absolute() or not path.is_file():
                errors.append("workload.dataset.path must be an existing absolute file")
            else:
                required_rows = confirmation_request_count(task)
                try:
                    available_rows = structurally_valid_dataset_rows(path, dataset_name, required_rows)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"workload dataset cannot be read: {exc}")
                else:
                    if available_rows < required_rows:
                        errors.append(
                            f"workload dataset has {available_rows} structurally valid requests; "
                            f"at least {required_rows} are required for final confirmation"
                        )
            if workload.get("shared_prefix") is not None:
                errors.append(
                    "workload.shared_prefix is only for synthetic generated-shared-prefix data; "
                    "real datasets must encode prefix reuse in their conversations"
                )
        apply_chat_template = dataset.get("apply_chat_template", False)
        if not isinstance(apply_chat_template, bool):
            errors.append("workload.dataset.apply_chat_template must be boolean")
    deployment_mode = task.get("deployment_mode", "online_latency")
    if deployment_mode not in {"online_latency", "offline_throughput"}:
        errors.append("deployment_mode must be online_latency or offline_throughput")
    if task.get("search_depth", "thorough") not in {"evidence_guided", "thorough"}:
        errors.append("search_depth must be evidence_guided or thorough")
    if task.get("experiment_mode", "balanced") not in {"fast", "balanced", "max", "rigorous"}:
        errors.append("experiment_mode must be fast, balanced, or max")
    evolution = task.get("parameter_evolution") or {}
    if not isinstance(evolution, dict):
        errors.append("parameter_evolution must be an object")
    else:
        supported_evolution = {
            "mode", "exploration_budget_pct", "minimum_confidence",
            "max_provisional_trials",
        }
        if any(key not in supported_evolution for key in evolution):
            errors.append(
                "parameter_evolution supports only mode, exploration_budget_pct, "
                "minimum_confidence, and max_provisional_trials"
            )
        if evolution.get("mode", "conservative") not in {"conservative", "experimental"}:
            errors.append("parameter_evolution.mode must be conservative or experimental")
        for key, maximum in (("exploration_budget_pct", 25), ("minimum_confidence", 1)):
            value = evolution.get(key)
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not 0 <= value <= maximum
            ):
                errors.append(f"parameter_evolution.{key} must be between 0 and {maximum}")
        provisional_trials = evolution.get("max_provisional_trials")
        if provisional_trials is not None and (
            not isinstance(provisional_trials, int) or isinstance(provisional_trials, bool)
            or not 0 <= provisional_trials <= 16
        ):
            errors.append("parameter_evolution.max_provisional_trials must be an integer from 0 through 16")
    parallel_trials = task.get("parallel_trials", 1)
    if (
        not isinstance(parallel_trials, int)
        or isinstance(parallel_trials, bool)
        or not 1 <= parallel_trials <= 16
    ):
        errors.append("parallel_trials must be an integer from 1 through 16")
    max_gpus = task.get("max_gpus")
    if max_gpus is not None and (
        not isinstance(max_gpus, int)
        or isinstance(max_gpus, bool)
        or not 1 <= max_gpus <= 1024
    ):
        errors.append("max_gpus must be a positive integer")
    budget = task.get("budget", {})
    for key in ("max_trials", "max_gpu_hours", "max_wall_time_minutes"):
        value = budget.get(key) if isinstance(budget, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            errors.append(f"budget.{key} must be positive")
    if isinstance(budget, dict) and not isinstance(budget.get("max_trials"), int):
        errors.append("budget.max_trials must be an integer")
    repetitions = task.get("confirmation_repetitions", 2)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or not 2 <= repetitions <= 9:
        errors.append("confirmation_repetitions must be an integer from 2 through 9")
    elif isinstance(budget, dict) and isinstance(budget.get("max_trials"), int):
        configured_adaptive = (task.get("measurement") or {}).get(
            "adaptive_confirmation_max_repetitions", repetitions
        )
        adaptive_repetitions = (
            configured_adaptive
            if isinstance(configured_adaptive, int)
            and not isinstance(configured_adaptive, bool)
            else repetitions
        )
        if adaptive_repetitions < repetitions:
            errors.append(
                "measurement.adaptive_confirmation_max_repetitions must be at least "
                "confirmation_repetitions"
            )
        minimum_trials = max(repetitions, adaptive_repetitions) * 2 + 3
        if budget["max_trials"] < minimum_trials:
            errors.append(
                f"budget.max_trials must be at least {minimum_trials} for profiling, screening, and confirmation"
            )
    environment = task.get("env", {})
    if not isinstance(environment, dict):
        errors.append("env must be an object")
    else:
        for key, value in environment.items():
            if key not in ALLOWED_ENV:
                errors.append(f"env contains unsupported key: {key}")
            if not isinstance(value, (str, int, float, bool)):
                errors.append(f"env.{key} must be scalar")
    objective = task.get("objective", {})
    if isinstance(objective, dict):
        metric = objective.get("metric")
        direction = objective.get("direction")
        if metric not in METRIC_DIRECTIONS:
            errors.append(f"unsupported objective.metric: {metric}")
        elif direction != METRIC_DIRECTIONS[metric]:
            errors.append(f"objective.direction for {metric} must be {METRIC_DIRECTIONS[metric]}")
        for key in ("min_improvement_pct", "max_regression_pct"):
            value = objective.get(key, 0)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                errors.append(f"objective.{key} must be non-negative")
        resource_scope = objective.get(
            "resource_scope",
            "per_gpu" if task.get("deployment_mode") == "offline_throughput" else "per_service",
        )
        if resource_scope not in RESOURCE_SCOPES:
            errors.append("objective.resource_scope must be per_service or per_gpu")
        elif resource_scope == "per_gpu" and metric not in RESOURCE_NORMALIZABLE_METRICS:
            errors.append(
                "objective.resource_scope=per_gpu requires a throughput or goodput objective"
            )
    slo = task.get("slo", {})
    if isinstance(slo, dict):
        for key, value in slo.items():
            if key not in SLO_MAPPING:
                errors.append(f"unsupported slo: {key}")
            elif not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                errors.append(f"slo.{key} must be non-negative")
    knowledge = task.get("knowledge", {})
    if not isinstance(knowledge, dict):
        errors.append("knowledge must be an object")
    else:
        supported = {
            "model_cookbook_url", "hardware_reference_urls", "require_cookbook",
            "cookbook_repository", "cookbook_snapshot_dir",
        }
        if any(key not in supported for key in knowledge):
            errors.append("knowledge contains an unsupported key")
        if "model_cookbook_url" in knowledge and not isinstance(knowledge["model_cookbook_url"], str):
            errors.append("knowledge.model_cookbook_url must be a URL string")
        if "require_cookbook" in knowledge and not isinstance(knowledge["require_cookbook"], bool):
            errors.append("knowledge.require_cookbook must be boolean")
        if "cookbook_repository" in knowledge and (
            not isinstance(knowledge["cookbook_repository"], str)
            or not knowledge["cookbook_repository"].startswith("https://")
        ):
            errors.append("knowledge.cookbook_repository must be an HTTPS Git URL")
        if "cookbook_snapshot_dir" in knowledge and (
            not isinstance(knowledge["cookbook_snapshot_dir"], str)
            or not Path(knowledge["cookbook_snapshot_dir"]).expanduser().is_absolute()
        ):
            errors.append("knowledge.cookbook_snapshot_dir must be an absolute path")
        if "hardware_reference_urls" in knowledge and (
            not isinstance(knowledge["hardware_reference_urls"], list)
            or any(not isinstance(url, str) for url in knowledge["hardware_reference_urls"])
        ):
            errors.append("knowledge.hardware_reference_urls must be an array of URL strings")
    capability_overrides = task.get("capability_overrides", {})
    if not isinstance(capability_overrides, dict):
        errors.append("capability_overrides must be an object")
    elif any(key not in {"mtp"} for key in capability_overrides):
        errors.append("capability_overrides supports only mtp")
    elif capability_overrides.get("mtp", "auto") not in {"auto", "disabled"}:
        errors.append("capability_overrides.mtp must be auto or disabled")
    deployment = task.get("deployment", {})
    if not isinstance(deployment, dict):
        errors.append("deployment must be an object")
    elif any(key not in {"allow_model_variant_recommendations", "allow_auto_model_switch"} for key in deployment):
        errors.append("deployment supports only allow_model_variant_recommendations and allow_auto_model_switch")
    elif deployment.get("allow_auto_model_switch", False):
        errors.append("deployment.allow_auto_model_switch is not supported; model changes require an explicit new task")
    quality = task.get("quality", {})
    if not isinstance(quality, dict):
        errors.append("quality must be an object")
    elif any(key not in {
        "evaluation_dataset", "max_regression_pct", "allow_kv_cache_precision_tuning",
        "attestation_path",
    } for key in quality):
        errors.append(
            "quality supports only evaluation_dataset, max_regression_pct, "
            "allow_kv_cache_precision_tuning, and attestation_path"
        )
    elif "evaluation_dataset" in quality and (
        not isinstance(quality["evaluation_dataset"], str)
        or not Path(quality["evaluation_dataset"]).expanduser().is_absolute()
        or not Path(quality["evaluation_dataset"]).expanduser().is_file()
    ):
        errors.append("quality.evaluation_dataset must be an existing absolute file")
    elif "max_regression_pct" in quality and (
        not isinstance(quality["max_regression_pct"], (int, float))
        or isinstance(quality["max_regression_pct"], bool)
        or quality["max_regression_pct"] < 0
    ):
        errors.append("quality.max_regression_pct must be non-negative")
    elif "allow_kv_cache_precision_tuning" in quality and not isinstance(
        quality["allow_kv_cache_precision_tuning"], bool
    ):
        errors.append("quality.allow_kv_cache_precision_tuning must be boolean")
    elif "attestation_path" in quality and (
        not isinstance(quality["attestation_path"], str)
        or not Path(quality["attestation_path"]).expanduser().is_absolute()
        or not Path(quality["attestation_path"]).expanduser().is_file()
    ):
        errors.append("quality.attestation_path must be an existing absolute JSON file")
    variants = task.get("model_variants", [])
    if not isinstance(variants, list):
        errors.append("model_variants must be an array")
    else:
        for index, variant in enumerate(variants):
            if not isinstance(variant, dict):
                errors.append(f"model_variants[{index}] must be an object")
                continue
            if set(variant) - {"name", "model_path", "quantization"}:
                errors.append(f"model_variants[{index}] supports only name, model_path, and quantization")
            if not isinstance(variant.get("name"), str) or not variant["name"]:
                errors.append(f"model_variants[{index}].name must be a non-empty string")
            path = variant.get("model_path")
            if not isinstance(path, str) or not Path(path).expanduser().is_absolute() or not Path(path).is_dir():
                errors.append(f"model_variants[{index}].model_path must be an existing absolute directory")
    if task.get("offline", True) is False and not task.get("allow_download", False):
        errors.append("offline=false requires allow_download=true")
    profiling = task.get("profiling", {})
    if not isinstance(profiling, dict):
        errors.append("profiling must be an object")
    elif profiling.get("enabled", True) is not True:
        errors.append("profiling.enabled must be true for automatic optimization")
    history = task.get("history", {})
    if not isinstance(history, dict):
        errors.append("history must be an object")
    elif any(key not in {"enabled", "database", "warm_start_limit"} for key in history):
        errors.append("history supports only enabled, database, and warm_start_limit")
    else:
        if "enabled" in history and not isinstance(history["enabled"], bool):
            errors.append("history.enabled must be boolean")
        database = history.get("database")
        if database is not None and (
            not isinstance(database, str)
            or not Path(database).expanduser().is_absolute()
            or Path(database).expanduser() == Path("/")
        ):
            errors.append("history.database must be an absolute non-root path")
        limit = history.get("warm_start_limit", 5)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 0 <= limit <= 20:
            errors.append("history.warm_start_limit must be an integer from 0 through 20")
    economics = task.get("economics", {})
    if not isinstance(economics, dict):
        errors.append("economics must be an object")
    elif any(key not in {
        "cost_per_gpu_hour", "currency", "power_cost_per_kwh"
    } for key in economics):
        errors.append(
            "economics supports only cost_per_gpu_hour, currency, and power_cost_per_kwh"
        )
    else:
        for key in ("cost_per_gpu_hour", "power_cost_per_kwh"):
            if key in economics and (
                not isinstance(economics[key], (int, float))
                or isinstance(economics[key], bool)
                or economics[key] < 0
            ):
                errors.append(f"economics.{key} must be non-negative")
        if "currency" in economics and (
            not isinstance(economics["currency"], str) or not economics["currency"]
        ):
            errors.append("economics.currency must be a non-empty string")
    kernel_tuning = task.get("kernel_tuning", {})
    if not isinstance(kernel_tuning, dict):
        errors.append("kernel_tuning must be an object")
    elif any(key not in {"mode", "timeout_minutes", "max_batch_sizes", "topk_ids_dir"} for key in kernel_tuning):
        errors.append("kernel_tuning supports only mode, timeout_minutes, max_batch_sizes, and topk_ids_dir")
    else:
        if kernel_tuning.get("mode", "detect_only") not in {"auto", "detect_only", "execute", "disabled"}:
            errors.append("kernel_tuning.mode must be detect_only, execute, or disabled (auto is a legacy alias for detect_only)")
        for key in ("timeout_minutes", "max_batch_sizes"):
            if key in kernel_tuning and (
                not isinstance(kernel_tuning[key], (int, float))
                or isinstance(kernel_tuning[key], bool)
                or kernel_tuning[key] <= 0
            ):
                errors.append(f"kernel_tuning.{key} must be positive")
        if "topk_ids_dir" in kernel_tuning and (
            not isinstance(kernel_tuning["topk_ids_dir"], str)
            or not Path(kernel_tuning["topk_ids_dir"]).expanduser().is_absolute()
            or not Path(kernel_tuning["topk_ids_dir"]).expanduser().is_dir()
        ):
            errors.append("kernel_tuning.topk_ids_dir must be an existing absolute directory")
    profile_dir = task.get("profile_dir")
    if profile_dir is not None and (
        not isinstance(profile_dir, str) or not Path(profile_dir).expanduser().is_absolute()
    ):
        errors.append("profile_dir must be an absolute path when provided")
    resume_run_dir = task.get("resume_run_dir")
    if resume_run_dir is not None:
        resume_path = (
            Path(resume_run_dir).expanduser()
            if isinstance(resume_run_dir, str) else None
        )
        if resume_path is None or not resume_path.is_absolute():
            errors.append("resume_run_dir must be an absolute path when provided")
        elif not resume_path.is_dir():
            errors.append("resume_run_dir must be an existing run directory")
        elif not (resume_path / "task.json").is_file():
            errors.append("resume_run_dir must contain task.json")
    measurement = task.get("measurement") or {}
    if not isinstance(measurement, dict):
        errors.append("measurement must be an object")
    elif any(key not in {
        "warmup_requests", "min_measurement_requests", "min_measurement_seconds",
        "confirmation_requests", "min_tail_samples", "near_slo_tail_samples",
        "near_slo_margin_pct", "p99_request_waves", "adaptive_confirmation_cv_pct",
        "adaptive_confirmation_max_repetitions",
        "adaptive_confirmation_min_measurement_seconds",
        "bayesian_sequential", "bayesian_min_blocks", "bayesian_max_blocks",
        "bayesian_accept_probability", "bayesian_reject_probability",
        "bayesian_prior_mean_pct", "bayesian_prior_strength",
    } for key in measurement):
        errors.append(
            "measurement supports only warmup_requests, min_measurement_requests, "
            "min_measurement_seconds, confirmation_requests, min_tail_samples, "
            "near_slo_tail_samples, near_slo_margin_pct, p99_request_waves, and "
            "adaptive confirmation controls"
        )
    else:
        for key in (
            "warmup_requests", "min_measurement_requests", "confirmation_requests",
            "min_tail_samples", "near_slo_tail_samples", "p99_request_waves",
            "adaptive_confirmation_max_repetitions",
            "bayesian_min_blocks", "bayesian_max_blocks",
        ):
            if key in measurement and (
                not isinstance(measurement[key], int)
                or isinstance(measurement[key], bool)
                or measurement[key] <= 0
            ):
                errors.append(f"measurement.{key} must be a positive integer")
        if "min_measurement_seconds" in measurement and (
            not isinstance(measurement["min_measurement_seconds"], (int, float))
            or isinstance(measurement["min_measurement_seconds"], bool)
            or measurement["min_measurement_seconds"] <= 0
        ):
            errors.append("measurement.min_measurement_seconds must be positive")
        if "near_slo_margin_pct" in measurement and (
            not isinstance(measurement["near_slo_margin_pct"], (int, float))
            or isinstance(measurement["near_slo_margin_pct"], bool)
            or not 0 <= measurement["near_slo_margin_pct"] <= 100
        ):
            errors.append("measurement.near_slo_margin_pct must be between 0 and 100")
        if "adaptive_confirmation_cv_pct" in measurement and (
            not isinstance(measurement["adaptive_confirmation_cv_pct"], (int, float))
            or isinstance(measurement["adaptive_confirmation_cv_pct"], bool)
            or not 0 <= measurement["adaptive_confirmation_cv_pct"] <= 100
        ):
            errors.append("measurement.adaptive_confirmation_cv_pct must be between 0 and 100")
        if "bayesian_sequential" in measurement and not isinstance(
            measurement["bayesian_sequential"], bool
        ):
            errors.append("measurement.bayesian_sequential must be boolean")
        for key in ("bayesian_accept_probability", "bayesian_reject_probability"):
            if key in measurement and (
                not isinstance(measurement[key], (int, float))
                or isinstance(measurement[key], bool)
                or not 0 < measurement[key] < 1
            ):
                errors.append(f"measurement.{key} must be between 0 and 1")
        if "adaptive_confirmation_min_measurement_seconds" in measurement and (
            not isinstance(
                measurement["adaptive_confirmation_min_measurement_seconds"], (int, float)
            )
            or isinstance(measurement["adaptive_confirmation_min_measurement_seconds"], bool)
            or measurement["adaptive_confirmation_min_measurement_seconds"] <= 0
        ):
            errors.append(
                "measurement.adaptive_confirmation_min_measurement_seconds must be positive"
            )
        if (
            isinstance(measurement.get("min_measurement_seconds"), (int, float))
            and isinstance(
                measurement.get("adaptive_confirmation_min_measurement_seconds"),
                (int, float),
            )
            and measurement["adaptive_confirmation_min_measurement_seconds"]
            < measurement["min_measurement_seconds"]
        ):
            errors.append(
                "measurement.adaptive_confirmation_min_measurement_seconds must be at least "
                "min_measurement_seconds"
            )
        if (
            isinstance(measurement.get("min_tail_samples"), int)
            and isinstance(measurement.get("near_slo_tail_samples"), int)
            and measurement["near_slo_tail_samples"] < measurement["min_tail_samples"]
        ):
            errors.append("measurement.near_slo_tail_samples must be at least min_tail_samples")
    calibration = task.get("calibration") or {}
    if not isinstance(calibration, dict):
        errors.append("calibration must be an object")
    else:
        supported = {
            "enabled", "min_concurrency", "max_concurrency", "fallback_max_concurrency", "max_steps",
            "stop_on_slo_failure", "concurrencies", "strategy",
        }
        if any(key not in supported for key in calibration):
            errors.append("calibration supports only strategy, enabled, concurrency range, fallback capacity, explicit concurrencies, max_steps, and stop_on_slo_failure")
        if "enabled" in calibration and not isinstance(calibration["enabled"], bool):
            errors.append("calibration.enabled must be boolean")
        for key in ("min_concurrency", "max_concurrency", "fallback_max_concurrency", "max_steps"):
            if key in calibration and (
                not isinstance(calibration[key], int) or isinstance(calibration[key], bool) or calibration[key] <= 0
            ):
                errors.append(f"calibration.{key} must be a positive integer")
        if "stop_on_slo_failure" in calibration and not isinstance(calibration["stop_on_slo_failure"], bool):
            errors.append("calibration.stop_on_slo_failure must be boolean")
        if calibration.get("strategy", "adaptive") not in {"adaptive", "full_curve"}:
            errors.append("calibration.strategy must be adaptive or full_curve")
        points = calibration.get("concurrencies")
        if points is not None:
            if (
                not isinstance(points, list) or not points or len(points) > 16
                or any(not isinstance(point, int) or isinstance(point, bool) or point <= 0 for point in points)
            ):
                errors.append("calibration.concurrencies must contain 1 through 16 positive integers")
            elif points != sorted(set(points)):
                errors.append("calibration.concurrencies must be unique and ascending")
            elif isinstance(workload, dict) and points[-1] != workload.get("max_concurrency"):
                errors.append("calibration.concurrencies must end at workload.max_concurrency")
        if (
            isinstance(calibration.get("min_concurrency"), int)
            and isinstance(calibration.get("max_concurrency"), int)
            and calibration["min_concurrency"] > calibration["max_concurrency"]
        ):
            errors.append("calibration.min_concurrency must not exceed calibration.max_concurrency")
    return errors


def materialize_runtime_task(task: dict[str, Any]) -> dict[str, Any]:
    """Provide an internal placeholder until runtime capacity is available."""
    runtime = deepcopy(task)
    objective = runtime.get("objective")
    if isinstance(objective, dict):
        objective.setdefault(
            "resource_scope",
            "per_gpu"
            if runtime.get("deployment_mode") == "offline_throughput"
            else "per_service",
        )
    workload = runtime.get("workload", {})
    if isinstance(workload, dict) and "max_concurrency" not in workload and (
        runtime.get("deployment_mode") == "offline_throughput"
        or uses_runtime_capacity_slo_calibration(runtime)
    ):
        workload["max_concurrency"] = 1
        workload["runtime_capacity_pending"] = True
    return runtime


def observed_admission_capacity(profile: dict[str, Any]) -> int | None:
    """Extract the loaded server's admission capacity without a task prior."""
    def find(value: Any) -> int | None:
        if isinstance(value, dict):
            candidate = value.get("max_running_requests")
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                return candidate
            for child in value.values():
                found = find(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find(child)
                if found is not None:
                    return found
        return None

    for source in (profile.get("startup_capacity"), profile.get("effective_server_config"), profile.get("server_info")):
        capacity = find(source)
        if capacity is not None:
            return capacity
    observations = profile.get("runtime_observations", {})
    for phase in ("prefill", "decode"):
        maximum = observations.get(phase, {}).get("running_requests", {}).get("max") if isinstance(observations, dict) else None
        if isinstance(maximum, int) and maximum > 0:
            return maximum
    return None


def observed_practical_capacity(
    profile: dict[str, Any], task: dict[str, Any]
) -> int | None:
    """Return the request pressure implied by KV tokens and workload shape.

    ``max_running_requests`` is commonly a static admission ceiling such as
    2048. It is not evidence that 2048 long-context requests fit. Prefer the
    profiler's shape-aware estimate and only fall back to the admission limit
    when the runtime exposes no token capacity.
    """
    startup = profile.get("startup_capacity")
    if isinstance(startup, dict):
        practical = startup.get("practical_request_capacity")
        if isinstance(practical, int) and not isinstance(practical, bool) and practical > 0:
            return practical
        token_capacity = startup.get("max_total_tokens")
        workload = task.get("workload", {})
        tokens_per_request = max(
            1,
            int(workload.get("input_tokens", 1))
            + int(workload.get("output_tokens", 1)),
        )
        if isinstance(token_capacity, int) and token_capacity > 0:
            return max(1, token_capacity // tokens_per_request)
    return observed_admission_capacity(profile)


def deployment_policy(task: dict[str, Any]) -> dict[str, Any]:
    mode = task.get("deployment_mode", "online_latency")
    if mode == "offline_throughput":
        return {
            "mode": mode,
            "objective": "maximize sustained aggregate throughput at the highest stable batch pressure",
            "latency_slo_required": False,
            "calibration_multiplier": 64,
            "calibration_floor": 32,
            "candidate_focus": ["batching", "KV_capacity", "scheduler", "CUDA_graph", "kernel_backends"],
            "final_acceptance": "declared SLOs plus error/correctness gates; latency is observational unless declared",
        }
    return {
        "mode": "online_latency",
        "objective": "maximize goodput or request throughput while every declared tail-latency SLO passes",
        "latency_slo_required": True,
        "calibration_multiplier": 16,
        "calibration_floor": 8,
        "candidate_focus": ["tail_latency", "scheduler", "CUDA_graph", "cache", "kernel_backends"],
        "final_acceptance": "every target-workload E2E/TTFT/TPOT/ITL and error SLO must pass on every confirmation repetition",
    }


def calibration_concurrencies(task: dict[str, Any]) -> list[int]:
    """Produce the requested calibration loads without changing workload semantics.

    An automatic capacity curve used to restart the server at every geometric
    point before the actual search.  Those points are useful when a user asks
    for a curve, but are redundant for a deployment decision at one declared
    target concurrency.  The default therefore measures that target first;
    an SLO failure triggers a bounded lower-load fallback in ``run_calibration``.
    Explicit points always retain their full-curve meaning.
    """
    calibration = task.get("calibration") or {}
    if calibration.get("enabled", True) is False:
        return []
    explicit_points = calibration.get("concurrencies")
    if isinstance(explicit_points, list):
        return list(explicit_points)
    workload = task["workload"]
    if workload.get("runtime_capacity_pending"):
        # The executor replaces this internal placeholder with the loaded
        # server's max_running_requests before it issues the first request.
        return [1]
    policy = deployment_policy(task)
    target = workload["max_concurrency"]
    if calibration.get("strategy", "adaptive") == "adaptive":
        return [target]
    range_requested = "min_concurrency" in calibration or "max_concurrency" in calibration
    start = int(calibration.get("min_concurrency", target))
    cap = int(calibration.get(
        "max_concurrency",
        target if range_requested else max(target * policy["calibration_multiplier"], policy["calibration_floor"]),
    ))
    requested_steps = int(calibration.get("max_steps", 5))
    repetitions = int(task.get("confirmation_repetitions", 2))
    # Reserve one profile, a baseline/candidate screen, and a full confirmation.
    affordable_steps = max(0, int(task["budget"]["max_trials"]) - (1 + 2 + 2 * repetitions))
    steps = min(requested_steps, affordable_steps)
    values: list[int] = []
    value = start
    while len(values) < steps and value <= cap:
        values.append(value)
        value *= 2
    if values and values[-1] != cap and len(values) < steps:
        values.append(cap)
    return values


def parse_nvidia_inventory() -> dict[str, Any] | None:
    query = run_readonly([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,driver_version,pci.bus_id,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if query.get("returncode") != 0 or not query.get("stdout"):
        return None
    gpus = []
    for line in query["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 9:
            continue
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "uuid": parts[2],
            "memory_mib": int(float(parts[3])),
            "memory_used_mib": int(float(parts[4])),
            "utilization_gpu_pct": float(parts[5]),
            "driver_version": parts[6],
            "pci_bus_id": parts[7],
            "compute_capability": parts[8],
        })
    if not gpus:
        return None
    topology = run_readonly(["nvidia-smi", "topo", "-m"])
    return {
        "vendor": "nvidia",
        "gpus": gpus,
        "topology": topology.get("stdout", ""),
        "topology_returncode": topology.get("returncode"),
    }


def parse_amd_inventory() -> dict[str, Any] | None:
    query = run_readonly(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if query.get("returncode") != 0 or not query.get("stdout"):
        return None
    try:
        payload = json.loads(query["stdout"])
    except json.JSONDecodeError:
        payload = {"raw": query["stdout"]}
    gpus = []
    if isinstance(payload, dict):
        for index, (card, values) in enumerate(sorted(payload.items())):
            if not isinstance(values, dict):
                continue
            lowered = {str(key).lower(): value for key, value in values.items()}
            name = next(
                (
                    str(value)
                    for key, value in lowered.items()
                    if any(token in key for token in ("card series", "product name", "device name"))
                ),
                str(card),
            )
            memory_bytes = next(
                (
                    int(re.sub(r"[^0-9]", "", str(value)))
                    for key, value in lowered.items()
                    if "vram total memory" in key and re.search(r"[0-9]", str(value))
                ),
                0,
            )
            gpus.append({"index": index, "name": name, "memory_mib": memory_bytes // 1024**2})
    topology = run_readonly(["rocm-smi", "--showtopo", "--json"])
    return {
        "vendor": "amd",
        "gpus": gpus,
        "rocm_smi": payload,
        "topology": topology.get("stdout", ""),
        "topology_returncode": topology.get("returncode"),
    }


def load_hardware_catalog() -> dict[str, Any]:
    try:
        resource = importlib.resources.files("inference_autopilot_data").joinpath("hardware-profiles.json")
        with importlib.resources.as_file(resource) as path:
            return load_json(path)
    except (ModuleNotFoundError, FileNotFoundError):
        # Direct script execution from a source checkout does not install the
        # package-data module, so retain the checked-out reference as fallback.
        path = Path(__file__).resolve().parent.parent / "references" / "hardware-profiles.json"
        return load_json(path)


def match_hardware_profile(inventory: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any] | None:
    names = " ".join(str(gpu.get("name", "")) for gpu in inventory.get("gpus", []))
    if not names.strip():
        names = json.dumps(inventory, sort_keys=True)
    for profile in catalog.get("profiles", []):
        if profile.get("vendor") != inventory.get("vendor"):
            continue
        if any(pattern.lower() in names.lower() for pattern in profile.get("name_patterns", [])):
            matched = deepcopy(profile)
            matched["matched_variant"] = next(
                (
                    variant
                    for variant in profile.get("known_variants", [])
                    if any(
                        pattern.lower() in names.lower()
                        for pattern in variant.get("name_patterns", [variant.get("name", "")])
                        if pattern
                    )
                ),
                None,
            )
            return matched
    return None


def topology_class(inventory: dict[str, Any]) -> str:
    if len(inventory.get("gpus", [])) <= 1:
        return "single-gpu"
    topology = str(inventory.get("topology", ""))
    if inventory.get("vendor") == "nvidia":
        if re.search(r"\bNV\d+\b", topology):
            return "nvlink-or-nvswitch"
        if any(token in topology for token in ("PIX", "PXB", "PHB", "SYS")):
            return "pcie"
    if inventory.get("vendor") == "amd" and topology:
        return "infinity-fabric-or-runtime-reported"
    return "unknown"


def framework_evidence(task: dict[str, Any]) -> dict[str, Any]:
    repository = Path(task["repository"])
    server_args = repository / "python" / "sglang" / "srt" / "server_args.py"
    source = server_args.read_text(encoding="utf-8") if server_args.is_file() else ""
    commit = run_readonly(["git", "rev-parse", "HEAD"], cwd=str(repository))
    runtime_env = os.environ.copy()
    repo_python = str(repository / "python")
    runtime_env["PYTHONPATH"] = repo_python + (
        os.pathsep + runtime_env["PYTHONPATH"] if runtime_env.get("PYTHONPATH") else ""
    )
    cli_help = run_readonly(
        [task["python"], "-m", "sglang.launch_server", "--help"],
        timeout=60,
        cwd=str(repository),
        env=runtime_env,
    )
    benchmark_probe = run_readonly(
        [
            task["python"], "-c",
            (
                "import importlib.util; "
                "mods=('sglang.benchmark.serving','sglang.bench_serving'); "
                "print(next((m for m in mods if importlib.util.find_spec(m) is not None), ''))"
            ),
        ],
        timeout=60,
        cwd=str(repository),
        env=runtime_env,
    )
    benchmark_module = (
        benchmark_probe.get("stdout", "").strip()
        if benchmark_probe.get("returncode") == 0
        else ""
    )
    if benchmark_module not in {"sglang.benchmark.serving", "sglang.bench_serving"}:
        benchmark_module = "sglang.bench_serving"
    benchmark_help = run_readonly(
        [task["python"], "-m", benchmark_module, "--help"],
        timeout=60,
        cwd=str(repository),
        env=runtime_env,
    )
    cli_text = cli_help.get("stdout", "") + "\n" + cli_help.get("stderr", "")
    cli_flags = sorted(set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", cli_text)))
    benchmark_help_text = (
        benchmark_help.get("stdout", "") + "\n" + benchmark_help.get("stderr", "")
    )
    runtime_probe = run_readonly(
        [
            task["python"], "-c",
            (
                "import importlib.metadata as m,json,sys; "
                "names=('sglang','torch','triton','flashinfer-python'); "
                "print(json.dumps({'python':sys.version.split()[0],"
                "'packages':{n:(m.version(n) if n in {d.metadata.get('Name','') for d in m.distributions()} else None) for n in names}}))"
            ),
        ],
        timeout=60, cwd=str(repository), env=runtime_env,
    )
    try:
        runtime_packages = json.loads(runtime_probe.get("stdout", ""))
    except json.JSONDecodeError:
        runtime_packages = {"error": runtime_probe.get("stderr") or "unavailable"}
    benchmark_cli_flags = sorted(set(re.findall(
        r"(?<![\w-])(--[a-z][a-z0-9-]*)", benchmark_help_text
    )))
    reserve_match = re.search(
        r"max\(self\.chunked_prefill_size,\s*\d+\)\s*\*\s*([0-9]+(?:\.[0-9]+)?)",
        source,
    )
    return {
        "repository": str(repository),
        "git_commit": commit.get("stdout") if commit.get("returncode") == 0 else None,
        "server_args_path": str(server_args) if server_args.is_file() else None,
        "server_args_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest() if source else None,
        "launch_server_help_sha256": hashlib.sha256(cli_text.encode("utf-8")).hexdigest() if cli_text else None,
        "launch_server_help_available": cli_help.get("returncode") == 0 and bool(cli_flags),
        "launch_server_help_error": cli_help.get("stderr") if cli_help.get("returncode") != 0 else None,
        "launch_server_cli_flags": cli_flags,
        "benchmark_module": benchmark_module,
        "benchmark_help_sha256": (
            hashlib.sha256(benchmark_help_text.encode("utf-8")).hexdigest()
            if benchmark_help_text else None
        ),
        "benchmark_help_available": (
            benchmark_help.get("returncode") == 0 and bool(benchmark_cli_flags)
        ),
        "benchmark_help_error": (
            benchmark_help.get("stderr") if benchmark_help.get("returncode") != 0 else None
        ),
        "runtime_packages": runtime_packages,
        "benchmark_cli_flags": benchmark_cli_flags,
        "chunk_activation_reserve_mib_per_token": (
            float(reserve_match.group(1)) if reserve_match else None
        ),
        "default_policy": "preserve defaults computed by this installed SGLang version; screen only explicit deltas",
    }


def _version_tuple(value: Any) -> tuple[int, int, int]:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", str(value or ""))
    return tuple(int(match.group(index)) for index in range(1, 4)) if match else (0, 0, 0)


def startup_acceleration_plan(
    task: dict[str, Any], discovery: dict[str, Any],
    search_plan: dict[str, Any] | None = None,
    profiling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess persistent SGLang Weight Cache as an execution accelerator.

    Weight Cache changes service lifecycle and startup cost, not steady-state
    throughput.  Keep it outside Candidate Registry and fail closed unless the
    current runtime exposes the 0.5.18+ contract and every trial can share one
    fixed non-speculative topology.
    """
    catalog = catalog_index(discovery)
    packages = discovery.get("framework", {}).get("runtime_packages", {}).get(
        "packages", {}
    )
    version = packages.get("sglang") if isinstance(packages, dict) else None
    live_contract = all(
        name in catalog for name in (
            "weight_cache_mode", "weight_cache_socket", "weight_cache_timeout",
        )
    )
    introduced = _version_tuple(version) >= (0, 5, 18)
    bundles = []
    if isinstance(search_plan, dict):
        bundles = [
            item for key in ("cookbook_candidate_bundles", "ranked_configuration_bundles")
            for item in search_plan.get(key, [])
            if isinstance(item, dict) and isinstance(item.get("config"), dict)
        ]
    else:
        bundles = cookbook_initial_search_plan(task, discovery).get(
            "cookbook_candidate_bundles", []
        )
    speculative_present = any(
        "speculative_algorithm" in item.get("config", {}) for item in bundles
    )
    minimum_tp = int(discovery.get("derived", {}).get("minimum_tp_size", 1) or 1)
    topology_varies = any(
        size != minimum_tp for size in supported_tp_sizes(discovery)
    )
    parallel_replicas = (
        int(task.get("parallel_trials", 1)) > 1
        and minimum_tp == 1
        and int(discovery.get("derived", {}).get("visible_gpu_count", 1)) > 1
    )
    reasons = []
    if not introduced:
        reasons.append("requires SGLang >=0.5.18")
    if not live_contract:
        reasons.append("installed ServerArgs does not expose the complete Weight Cache contract")
    if speculative_present:
        reasons.append("locally compatible speculative candidates are incompatible with Weight Cache")
    if topology_varies:
        reasons.append("candidate set changes TP topology; one daemon shard layout cannot serve every trial")
    if parallel_replicas:
        reasons.append(
            "parallel TP1 replicas cannot share the fixed rank0 daemon socket/shard layout"
        )
    startup = (
        (profiling or {}).get("runtime_observations", {}).get("startup", {})
        if isinstance(profiling, dict) else {}
    )
    capacity = (profiling or {}).get("startup_capacity", {}) if isinstance(profiling, dict) else {}
    eligible = introduced and live_contract and not reasons
    return {
        "schema_version": 1,
        "kind": "execution_acceleration",
        "feature": "persistent_weight_cache_daemon",
        "sglang_version": version,
        "introduced_in": "0.5.18",
        "live_contract_available": live_contract,
        "eligible": eligible,
        "automatic_execution_enabled": False,
        "decision": (
            "eligible_but_requires_owned_daemon_lifecycle"
            if eligible else "disabled"
        ),
        "reasons": reasons,
        "profiled_startup": deepcopy(startup),
        "required_capacity_pins": {
            key: capacity.get(key) for key in (
                "max_total_tokens", "max_mamba_cache_size",
            ) if isinstance(capacity.get(key), int) and capacity.get(key) > 0
        },
        "observed_safety_rule": (
            "IPC clients must pin profile-resolved KV/Mamba capacity; automatic "
            "mem_fraction sizing can over-allocate because IPC-mapped weights report zero client bytes"
        ),
        "lifecycle_policy": (
            "use a standalone daemon plus --weight-cache-mode client; engine daemon mode "
            "is co-terminal and does not accelerate the next restart"
        ),
        "recommendation": (
            "do not enable automatically until the run owns daemon startup, socket validation, "
            "capacity pins, and cleanup"
        ),
    }


def parameter_catalog(
    task: dict[str, Any], progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    exporter = Path(__file__).resolve().with_name("sglang_catalog.py")
    environment = os.environ.copy()
    environment.update({
        key: str(value) for key, value in task.get("env", {}).items()
        if key in ALLOWED_ENV
    })
    try:
        with tempfile.TemporaryDirectory(prefix="inferopt-catalog-") as temporary:
            output = Path(temporary) / "catalog.json"
            log_path = Path(temporary) / "catalog.log"
            command = [
                task["python"], str(exporter),
                "--repository", str(Path(task["repository"]).resolve()),
                "--output", str(output),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command, cwd=task["repository"], env=environment,
                    stdout=log, stderr=subprocess.STDOUT, text=True,
                )
                started = time.monotonic()
                last_heartbeat = started
                while process.poll() is None:
                    now = time.monotonic()
                    if now - started >= 180:
                        process.kill()
                        process.wait(timeout=10)
                        raise subprocess.TimeoutExpired(command, 180)
                    if progress and now - last_heartbeat >= 30:
                        progress(
                            f"ServerArgs exporter still running under {task['python']}; "
                            f"elapsed={now - started:.0f}s"
                        )
                        last_heartbeat = now
                    time.sleep(0.2)
            if process.returncode != 0:
                detail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
                raise RuntimeError(
                    "current-SGLang ServerArgs export failed under the configured Python: "
                    + (detail.strip() or f"exit {process.returncode}")
                )
            return load_json(output)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"could not execute current-SGLang ServerArgs exporter: {type(exc).__name__}"
        ) from exc


def model_inventory(model_path: str) -> dict[str, Any]:
    root = Path(model_path)
    config_path = root / "config.json"
    config = load_json(config_path) if config_path.is_file() else {}
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
    weight_files = [path for path in root.rglob("*") if path.is_file() and path.suffix in weight_suffixes]
    weight_bytes = sum(path.stat().st_size for path in weight_files)
    weight_manifest = [
        {"path": str(path.relative_to(root)), "size": path.stat().st_size}
        for path in sorted(weight_files)
    ]
    weight_manifest_sha256 = hashlib.sha256(
        json.dumps(weight_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    architectures = config.get("architectures", [])
    architecture_text = " ".join(str(value) for value in architectures).lower()
    model_type = str(config.get("model_type", "")).lower()
    text_config = config.get("text_config", config)
    language_config = text_config if isinstance(text_config, dict) else config
    # Multimodal and hybrid checkpoints often place the language-model
    # topology below text_config. Route every model-structure decision from
    # the same merged source so a Qwen3.5 MoE cannot be mistaken for dense.
    is_moe = bool(
        config.get("num_experts")
        or config.get("num_local_experts")
        or language_config.get("num_experts")
        or language_config.get("num_local_experts")
        or "moe" in architecture_text
        or "moe" in model_type
        or "moe" in str(language_config.get("model_type", "")).lower()
    )
    config_text = json.dumps(config).lower()
    # Some hybrid checkpoints describe their DeltaNet layers solely through a
    # model-type identifier, without spelling "mamba", "ssm", or
    # "linear_attention" in config.json.  Qwen3-Next is such a documented
    # hybrid architecture; treating it as dense discards its Cookbook cache
    # strategy before it can be screened.
    is_hybrid = (
        any(token in config_text for token in ("mamba", "ssm", "linear_attention", "deltanet"))
        or model_type in {"qwen3_next", "qwen3next"}
    )
    index_path = root / "model.safetensors.index.json"
    index = load_json(index_path) if index_path.is_file() else {}
    weight_map = index.get("weight_map", {}) if isinstance(index, dict) else {}
    mtp_weight_keys = [
        key for key in weight_map
        if "mtp" in key.lower() or "nextn" in key.lower() or "multi_token" in key.lower()
    ]
    mtp_files = sorted(path.name for path in root.glob("*mtp*.safetensors"))
    mtp_layers = (
        text_config.get(
            "num_nextn_predict_layers",
            text_config.get("num_mtp_layers", text_config.get("mtp_num_hidden_layers", 0)),
        )
        if isinstance(text_config, dict) else 0
    )
    has_mtp_weights = bool(
        mtp_files or mtp_weight_keys
        or isinstance(mtp_layers, int) and mtp_layers > 0
    )
    quantization_config = config.get("quantization_config", {})
    if not isinstance(quantization_config, dict):
        quantization_config = {}
    # Multimodal checkpoints commonly keep the language-model dimensions in
    # text_config. Prefer an explicit top-level value but otherwise use that
    # nested language configuration for capacity and parallelism checks.
    weight_block_size = quantization_config.get("weight_block_size")
    if not (
        isinstance(weight_block_size, list)
        and len(weight_block_size) == 2
        and all(isinstance(value, int) and value > 0 for value in weight_block_size)
    ):
        weight_block_size = None
    return {
        "checkpoint_name": root.name,
        "config_path": str(config_path) if config_path.is_file() else None,
        "config_sha256": (
            hashlib.sha256(config_path.read_bytes()).hexdigest()
            if config_path.is_file() else None
        ),
        "weight_manifest_sha256": weight_manifest_sha256,
        "weight_index_sha256": (
            hashlib.sha256(index_path.read_bytes()).hexdigest()
            if index_path.is_file() else None
        ),
        "architectures": architectures,
        "model_type": model_type,
        # A quantized checkpoint can still declare BF16 here for activations
        # and unquantized layers. Keep both facts instead of treating
        # torch_dtype as an instruction to override SGLang's model loader.
        "dtype": config.get("torch_dtype", config.get("dtype")),
        "checkpoint_dtype": config.get("torch_dtype", config.get("dtype")),
        "quantization": config.get("quantization", quantization_config.get("quant_method")),
        "weight_quantization": config.get("quantization", quantization_config.get("quant_method")),
        "quantization_algorithm": quantization_config.get("quant_algo"),
        "context_length": config.get("max_position_embeddings", language_config.get("max_position_embeddings")),
        "hidden_size": config.get("hidden_size", language_config.get("hidden_size")),
        "num_hidden_layers": config.get("num_hidden_layers", language_config.get("num_hidden_layers")),
        "num_attention_heads": config.get("num_attention_heads", language_config.get("num_attention_heads")),
        "num_key_value_heads": config.get("num_key_value_heads", language_config.get("num_key_value_heads")),
        "num_experts": config.get("num_experts", config.get("num_local_experts", language_config.get("num_experts", language_config.get("num_local_experts")))),
        "num_experts_per_tok": config.get("num_experts_per_tok", language_config.get("num_experts_per_tok")),
        "moe_intermediate_size": config.get("moe_intermediate_size", language_config.get("moe_intermediate_size")),
        "weight_block_size": weight_block_size,
        "is_moe": is_moe,
        "is_hybrid": is_hybrid,
        "num_mtp_layers": mtp_layers,
        "mtp_weight_files": mtp_files,
        "mtp_weight_key_count": len(mtp_weight_keys),
        "has_mtp_weights": has_mtp_weights,
        "weight_files": len(weight_files),
        "weight_bytes": weight_bytes,
        "weight_gib": weight_bytes / 1024**3,
    }


def cookbook_model_terms(model: dict[str, Any]) -> set[str]:
    """Return stable model identifiers for filename and document matching."""
    values = [
        *(str(item) for item in model.get("architectures", [])),
        str(model.get("model_type", "")),
        str(model.get("checkpoint_name", "")),
    ]
    terms: set[str] = set()
    for value in values:
        normalized = value.lower().replace("_", ".").replace("-", ".")
        terms.update(re.findall(r"[a-z]+\d+(?:\.\d+)?", normalized))
        if normalized:
            terms.add(normalized)
    return {term for term in terms if term and term not in {"for", "model"}}


def local_cookbook_roots(task: dict[str, Any]) -> list[Path]:
    """Find Cookbook trees that belong to the checked-out SGLang version."""
    knowledge = task.get("knowledge", {}) if isinstance(task.get("knowledge"), dict) else {}
    roots: list[Path] = []
    for raw in (task.get("repository"), knowledge.get("cookbook_snapshot_dir")):
        if not isinstance(raw, str) or not raw:
            continue
        base = Path(raw).expanduser()
        for relative in ("docs/cookbook", "docs_new/cookbook", "cookbook"):
            candidate = base / relative
            if candidate.is_dir() and candidate not in roots:
                roots.append(candidate)
    return roots


def cookbook_document_matches(root: Path, model: dict[str, Any]) -> list[Path]:
    """Rank local MD/MDX pages by model identity without executing documentation."""
    terms = cookbook_model_terms(model)
    exact_page_terms: set[str] = set()
    for raw in (model.get("model_type", ""), model.get("checkpoint_name", "")):
        normalized_model = str(raw).lower().replace("_", ".").replace("-", ".")
        match = re.search(r"[a-z]+\d+\.(?:[a-z]+|\d+)", normalized_model)
        if match:
            exact_page_terms.add(match.group(0))
    # Only numeric versions (for example Qwen3.5) require an exact page
    # match.  Architecture names such as ``qwen3_next`` also normalize to a
    # dotted form, but must match the Qwen3-Next page rather than be rejected
    # as a fictional decimal release.
    decimal_terms = {
        term for term in terms
        if re.fullmatch(r"[a-z]+\d+\.\d+", term)
    }
    specific_terms = decimal_terms or {term for term in terms if re.search(r"\d", term) or len(term) >= 8}
    ranked: list[tuple[int, Path]] = []
    exact_ranked: list[tuple[int, Path]] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in COOKBOOK_DOCUMENT_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        relative = str(path.relative_to(root)).lower().replace("_", ".").replace("-", ".")
        identity = (relative + "\n" + text[:2_000]).lower().replace("_", ".").replace("-", ".")
        normalized = (relative + "\n" + text[:32_000]).lower().replace("_", ".").replace("-", ".")
        if decimal_terms and not any(term in relative for term in decimal_terms):
            continue
        if not decimal_terms and specific_terms and not any(term in relative for term in specific_terms):
            continue
        score = sum(1 for term in terms if term in normalized)
        if score:
            if exact_page_terms and any(term in relative for term in exact_page_terms):
                exact_ranked.append((score, path))
            ranked.append((score, path))
    selected = exact_ranked or ranked
    return [path for _, path in sorted(selected, key=lambda item: (-item[0], str(item[1])))[:16]]


def cookbook_scalar(value: str) -> Any:
    """Parse a CLI scalar conservatively; unknown values remain strings."""
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", value):
        return float(value)
    return value


def cookbook_command_config(
    command: str, *, include_unrecognized: bool = False
) -> dict[str, Any]:
    """Extract launch options; unknown options remain evidence-only.

    The executable recipe path still uses the versioned allowlist.  Parameter
    evolution requests the full static option map so a newly documented flag
    can be classified without being executed as a trusted Cookbook bundle.
    """
    normalized = command.replace("\\\n", " ").replace("\n", " ").strip()
    try:
        tokens = shlex.split(normalized)
    except ValueError:
        return {}
    launch_at = None
    for index, token in enumerate(tokens):
        if token == "sglang" and index + 1 < len(tokens) and tokens[index + 1] == "serve":
            launch_at = index + 2
            break
        if token == "-m" and index + 1 < len(tokens) and tokens[index + 1] == "sglang.launch_server":
            launch_at = index + 2
            break
    if launch_at is None:
        return {}
    config: dict[str, Any] = {}
    index = launch_at
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        flag, separator, inline_value = token[2:].partition("=")
        normalized_flag = flag.replace("-", "_")
        negated_parameter = COOKBOOK_NEGATED_BOOLEAN_FLAGS.get(normalized_flag)
        parameter = negated_parameter or COOKBOOK_FLAG_ALIASES.get(
            normalized_flag, normalized_flag
        )
        if parameter not in COOKBOOK_TUNABLE_FLAGS and not include_unrecognized:
            index += 1
            continue
        if parameter in COOKBOOK_BOOLEAN_FLAGS:
            config[parameter] = negated_parameter is None
            index += 1
            continue
        if separator:
            config[parameter] = cookbook_scalar(inline_value)
            index += 1
            continue
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            config[parameter] = cookbook_scalar(tokens[index + 1])
            index += 2
            continue
        if include_unrecognized:
            config[parameter] = True
        index += 1
    return config


def cookbook_command_model_reference(command: str) -> str | None:
    """Return the documented checkpoint selector without resolving or loading it."""
    normalized = command.replace("\\\n", " ").replace("\n", " ").strip()
    try:
        tokens = shlex.split(normalized)
    except ValueError:
        return None
    for index, token in enumerate(tokens[:-1]):
        if token in {"--model-path", "--model"}:
            return Path(tokens[index + 1]).name.lower()
    return None


def _js_balanced_fragment(text: str, start: int) -> str | None:
    """Return one balanced JS object/array without evaluating JavaScript."""
    if start < 0 or start >= len(text) or text[start] not in "{[":
        return None
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    index = start
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and following == "/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char == "/" and following == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and following == "*":
            block_comment = True
            index += 2
            continue
        if char in "'\"`":
            quote = char
            index += 1
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
        index += 1
    return None


def _js_property_container(text: str, name: str, opener: str) -> str | None:
    match = re.search(rf"\b{re.escape(name)}\s*:\s*\{opener}", text)
    if match is None:
        return None
    start = text.find(opener, match.start())
    return _js_balanced_fragment(text, start)


def _js_top_level_objects(array_fragment: str) -> list[str]:
    """Split a literal JS array into its top-level object entries."""
    if not array_fragment.startswith("["):
        return []
    objects: list[str] = []
    index = 1
    line_comment = False
    block_comment = False
    while index < len(array_fragment) - 1:
        char = array_fragment[index]
        following = array_fragment[index + 1] if index + 1 < len(array_fragment) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and following == "/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if char == "/" and following == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and following == "*":
            block_comment = True
            index += 2
            continue
        if char == "{":
            fragment = _js_balanced_fragment(array_fragment, index)
            if fragment is None:
                break
            objects.append(fragment)
            index += len(fragment)
            continue
        if char in "'\"`":
            quote = char
            index += 1
            escaped = False
            while index < len(array_fragment):
                char = array_fragment[index]
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    index += 1
                    break
                index += 1
            continue
        index += 1
    return objects


def _js_string_literals(fragment: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"(['\"])(?:\\.|(?!\1).)*\1", fragment, re.DOTALL):
        try:
            value = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, str):
            values.append(value)
    return values


def _js_literal_object(fragment: str | None) -> dict[str, Any]:
    if not isinstance(fragment, str):
        return {}
    result: dict[str, Any] = {}
    scalar = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
        r"((?:'(?:\\.|[^'])*')|(?:\"(?:\\.|[^\"])*\")|true|false|null|-?\d+(?:\.\d+)?)"
    )
    for match in scalar.finditer(fragment):
        key, encoded = match.groups()
        if encoded == "true":
            value: Any = True
        elif encoded == "false":
            value = False
        elif encoded == "null":
            value = None
        else:
            try:
                value = ast.literal_eval(encoded)
            except (SyntaxError, ValueError):
                value = cookbook_scalar(encoded)
        result[key] = value
    return result


def _js_quoted_map(fragment: str | None) -> dict[str, str]:
    if not isinstance(fragment, str):
        return {}
    result: dict[str, str] = {}
    pattern = re.compile(
        r"(['\"])((?:\\.|(?!\1).)*)\1\s*:\s*"
        r"(['\"])((?:\\.|(?!\3).)*)\3",
        re.DOTALL,
    )
    for match in pattern.finditer(fragment):
        _, key, _, value = match.groups()
        result[key] = value
    return result


def _cookbook_automatic_behaviors(snippet_text: str, snippet_path: str) -> list[dict[str, Any]]:
    """Extract explicit auto/default behavior that intentionally emits no flag."""
    behaviors: list[dict[str, Any]] = []
    if (
        re.search(r"id\s*:\s*['\"]pleOffload['\"]", snippet_text)
        and "PLE Offload: auto-enabled for BF16 on CUDA, off otherwise" in snippet_text
    ):
        behaviors.append({
            "parameter": "ple_offload_embedding",
            "mode": "auto",
            "enabled_when": {"accelerator_runtime": "cuda", "checkpoint_precision": "bf16"},
            "otherwise": "off",
            "evidence": "PLE Offload: auto-enabled for BF16 on CUDA, off otherwise",
            "source": snippet_path,
            "policy": "record and verify the resolved runtime behavior; emit no flag for auto",
        })
    return behaviors


def _cookbook_deployment_cells(
    snippet_text: str, *, document_path: str, document_sha256: str,
    snippet_path: str,
) -> list[dict[str, Any]]:
    """Parse verified config.cells recipes from the current Cookbook format."""
    cells = _js_property_container(snippet_text, "cells", "[")
    if cells is None:
        return []
    model_names = _js_quoted_map(
        _js_property_container(snippet_text, "modelNames", "{")
    )
    automatic_behaviors = _cookbook_automatic_behaviors(snippet_text, snippet_path)
    recipes: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(_js_top_level_objects(cells), start=1):
        match = _js_literal_object(_js_property_container(cell, "match", "{"))
        if not match or re.search(r"\bverified\s*:\s*true\b", cell) is None:
            continue
        flags_fragment = _js_property_container(cell, "flags", "[")
        flags = _js_string_literals(flags_fragment or "")
        if not flags:
            continue
        command = "sglang serve " + " ".join(flags)
        config = cookbook_command_config(command)
        all_options = cookbook_command_config(command, include_unrecognized=True)
        required_functional_config = {
            key: value for key, value in all_options.items()
            if key in COOKBOOK_REQUIRED_FUNCTIONAL_FLAGS
        }
        unrecognized_config = {
            key: value for key, value in all_options.items()
            if key not in COOKBOOK_TUNABLE_FLAGS
            and key not in COOKBOOK_REQUIRED_FUNCTIONAL_FLAGS
            and key not in {"model_path", "host", "port"}
        }
        requirements: list[str] = []
        if "speculative_algorithm" in config:
            requirements.append("checkpoint.has_mtp_weights")
        hardware = str(match.get("hw", "")).lower()
        if hardware.startswith("mi"):
            requirements.append("amd_gpu")
        elif hardware:
            requirements.append("nvidia_gpu")
        variant = str(match.get("variant", "default"))
        quantization = str(match.get("quant", "")).lower()
        documented_model = model_names.get(f"{variant}|{quantization}")
        recipes.append({
            "name": (
                f"cookbook-{Path(document_path).stem.lower()}-"
                f"{hardware or 'any'}-{quantization or 'default'}-"
                f"{str(match.get('strategy', 'balanced')).lower()}"
            )[:120],
            "config": config,
            "required_functional_config": required_functional_config,
            "unrecognized_config": unrecognized_config,
            "source": {
                "path": document_path,
                "sha256": document_sha256,
                "snippet": snippet_path,
                "snippet_sha256": hashlib.sha256(
                    snippet_text.encode("utf-8")
                ).hexdigest(),
                "cell_index": cell_index,
                "kind": "verified_deployment_cell",
            },
            "requirements": sorted(set(requirements)),
            "hardware_affinity": [hardware] if hardware else [],
            "documented_model": documented_model,
            "cookbook_cell": match,
            "verified": True,
            "preserve_topology": True,
            "automatic_behaviors": deepcopy(automatic_behaviors),
            "documented_flags": flags,
        })
    return recipes


def cookbook_snippet_recipes_from_document(path: Path, root: Path, body: bytes) -> list[dict[str, Any]]:
    """Extract static option rules from a Cookbook command-generator snippet.

    Several current Cookbook pages keep their launch matrix in a local JSX
    component rather than in a shell fence. We statically read verified
    ``config.cells`` and literal ``commandRule`` strings from imports that
    remain under the same docs checkout; JavaScript is never evaluated.
    """
    text = body.decode("utf-8", errors="ignore")
    docs_root = root.parent
    docs_root_resolved = docs_root.resolve()
    imports = re.findall(
        r"^\s*import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]\s*;?",
        text,
        flags=re.MULTILINE,
    )
    fragments: list[tuple[str, dict[str, Any], list[str], str]] = []
    deployment_recipes: list[dict[str, Any]] = []
    relative = str(path.relative_to(root))
    digest = hashlib.sha256(body).hexdigest()
    seen_paths: set[Path] = set()
    for imported in imports:
        if not imported.startswith("/"):
            continue
        snippet = (docs_root / imported.lstrip("/")).resolve()
        try:
            snippet.relative_to(docs_root_resolved)
        except ValueError:
            continue
        if snippet in seen_paths or snippet.suffix.lower() not in {".js", ".jsx", ".ts", ".tsx"}:
            continue
        seen_paths.add(snippet)
        try:
            snippet_text = snippet.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        snippet_path = str(snippet.relative_to(docs_root_resolved))
        deployment_recipes.extend(_cookbook_deployment_cells(
            snippet_text,
            document_path=relative,
            document_sha256=digest,
            snippet_path=snippet_path,
        ))
        # Match only literal, value-gated command strings.  This covers the
        # Cookbook generator format while keeping arbitrary JS out of scope.
        rules = re.finditer(
            r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*\{.*?"
            r"commandRule\s*:\s*\(value\)\s*=>\s*value\s*===\s*['\"]([^'\"]+)['\"]\s*\?\s*"
            r"(['\"])((?:\\.|(?!\3).)*)\3\s*:\s*null",
            snippet_text,
            flags=re.DOTALL,
        )
        for rule in rules:
            option, enabled_value, _, encoded = rule.groups()
            try:
                command = bytes(encoded, "utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                command = encoded.replace("\\n", "\n").replace("\\\\", "\\")
            config = cookbook_command_config("sglang serve " + command)
            if not config:
                continue
            requirements: list[str] = []
            if "speculative_algorithm" in config:
                requirements.append("checkpoint.has_mtp_weights")
            if config.get("mamba_radix_cache_strategy") == "extra_buffer":
                config["page_size"] = 64
                requirements.extend(["nvidia_gpu", "checkpoint.is_hybrid", "page_size=64"])
            fragments.append((option, config, requirements, snippet_path))

    if not fragments:
        return deployment_recipes
    recipes: list[dict[str, Any]] = list(deployment_recipes)
    for index, (option, config, requirements, snippet_path) in enumerate(fragments, start=1):
        recipes.append({
            "name": f"cookbook-{path.stem.lower()}-generator-{option}",
            "config": config,
            "source": {
                "path": relative,
                "sha256": digest,
                "snippet": snippet_path,
                "rule_index": index,
            },
            "requirements": sorted(set(requirements)),
            "hardware_affinity": [],
            "documented_model": None,
            "generator_option": {"name": option, "enabled_value": enabled_value},
        })
    combined: dict[str, Any] = {}
    combined_requirements: set[str] = set()
    names: list[str] = []
    for option, config, requirements, _ in fragments:
        if any(key in combined and combined[key] != value for key, value in config.items()):
            continue
        combined.update(config)
        combined_requirements.update(requirements)
        names.append(option)
    if len(names) > 1:
        recipes.append({
            "name": f"cookbook-{path.stem.lower()}-generator-combined-{'-'.join(names)}",
            "config": combined,
            "source": {
                "path": relative,
                "sha256": digest,
                "kind": "compatible_generator_interaction",
            },
            "requirements": sorted(combined_requirements),
            "hardware_affinity": [],
            "documented_model": None,
            "generator_options": names,
        })
    return recipes


def cookbook_recipe_compatibility(recipe: dict[str, Any], model: dict[str, Any]) -> str | None:
    """Reject a documented variant when its explicit checkpoint identity conflicts."""
    documented = str(recipe.get("documented_model", "")).lower().replace("_", ".")
    if not documented:
        return None
    target = " ".join([
        str(model.get("checkpoint_name", "")), str(model.get("model_type", "")),
        *(str(value) for value in model.get("architectures", [])),
    ]).lower().replace("_", ".")
    documented_sizes = set(re.findall(r"\d+(?:\.\d+)?[bm]", documented))
    target_sizes = set(re.findall(r"\d+(?:\.\d+)?[bm]", target))
    if documented_sizes and target_sizes and not documented_sizes.intersection(target_sizes):
        return (
            "documented checkpoint variant does not match the local checkpoint size "
            f"(documented={sorted(documented_sizes)}, local={sorted(target_sizes)})"
        )
    documented_active = set(re.findall(r"a\d+(?:\.\d+)?b", documented))
    target_active = set(re.findall(r"a\d+(?:\.\d+)?b", target))
    if documented_active and target_active and not documented_active.intersection(target_active):
        return (
            "documented MoE active-parameter variant does not match the local checkpoint "
            f"(documented={sorted(documented_active)}, local={sorted(target_active)})"
        )
    if ("moe" in documented or documented_active) and not model.get("is_moe"):
        return "documented recipe targets a MoE checkpoint, but the local checkpoint is dense"
    documented_precision = set(re.findall(r"\b(?:fp8|bf16|bfloat16|int8|int4)\b", documented))
    target_precision = " ".join([
        str(model.get("weight_quantization", "")), str(model.get("checkpoint_dtype", "")),
        str(model.get("checkpoint_name", "")),
    ]).lower()
    if documented_precision and not any(token in target_precision for token in documented_precision):
        return (
            "documented checkpoint precision does not match local checkpoint metadata "
            f"(documented={sorted(documented_precision)})"
        )
    return None


def cookbook_recipes_from_document(path: Path, root: Path) -> list[dict[str, Any]]:
    """Turn documented SGLang launch blocks into auditable candidate recipes."""
    try:
        body = path.read_bytes()
    except OSError:
        return []
    text = body.decode("utf-8", errors="ignore")
    recipes: list[dict[str, Any]] = []
    blocks = re.finditer(
        r"```(?:bash|sh|shell|console)?[^\n]*\n(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block_index, match in enumerate(blocks):
        block = match.group(1)
        context = text[max(0, match.start() - 600):match.start()].lower()
        command_lines: list[str] = []
        active = False
        for raw_line in block.splitlines():
            line = raw_line.strip().lstrip("$").strip()
            if not active and ("sglang serve" in line or "sglang.launch_server" in line):
                active = True
            if active:
                command_lines.append(line)
                if not line.endswith("\\"):
                    command = "\n".join(command_lines)
                    config = cookbook_command_config(command)
                    all_options = cookbook_command_config(
                        command, include_unrecognized=True
                    )
                    required_functional_config = {
                        key: value for key, value in all_options.items()
                        if key in COOKBOOK_REQUIRED_FUNCTIONAL_FLAGS
                    }
                    unrecognized_config = {
                        key: value for key, value in all_options.items()
                        if key not in COOKBOOK_TUNABLE_FLAGS
                        and key not in COOKBOOK_REQUIRED_FUNCTIONAL_FLAGS
                        and key != "model_path"
                    }
                    if config or unrecognized_config:
                        relative = str(path.relative_to(root))
                        requirements: list[str] = []
                        if "speculative_algorithm" in config:
                            requirements.append("checkpoint.has_mtp_weights")
                        if config.get("mamba_radix_cache_strategy") == "extra_buffer":
                            requirements.append("nvidia_gpu")
                            requirements.append("checkpoint.is_hybrid")
                            if config.get("page_size") != 64:
                                # The published Qwen hybrid recipe requires this pair.
                                config["page_size"] = 64
                                requirements.append("page_size=64")
                        if config.get("enable_flashinfer_allreduce_fusion") and int(config.get("tp_size", 1)) < 2:
                            requirements.append("tp_size>=2")
                        if re.search(r"\b(?:amd|mi300|mi325|mi355|rocm)\b", context):
                            requirements.append("amd_gpu")
                        elif re.search(r"\b(?:nvidia|h100|h200|h800|b200|b300)\b", context):
                            requirements.append("nvidia_gpu")
                        hardware_affinity = sorted(set(re.findall(
                            r"\b(?:a100|h100|h200|h800|b200|b300|gb300|mi300x|mi325x|mi350x|mi355x)\b",
                            context,
                        )))
                        recipes.append({
                            "name": f"cookbook-{path.stem.lower()}-{block_index + 1}",
                            "config": config,
                            "required_functional_config": required_functional_config,
                            "unrecognized_config": unrecognized_config,
                            "source": {
                                "path": relative,
                                "sha256": hashlib.sha256(body).hexdigest(),
                                "block_index": block_index,
                            },
                            "requirements": requirements,
                            "hardware_affinity": hardware_affinity,
                            "documented_model": cookbook_command_model_reference(
                                command
                            ),
                        })
                    active = False
                    command_lines = []
    return recipes


def local_cookbook_evidence(task: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Read matching recipes from the local SGLang checkout before network fallbacks."""
    roots = local_cookbook_roots(task)
    documents: list[dict[str, Any]] = []
    recipes: list[dict[str, Any]] = []
    tuning_tips: list[dict[str, Any]] = []
    automatic_behaviors: list[dict[str, Any]] = []
    excluded_recipes: list[dict[str, Any]] = []
    seen_document_hashes: set[str] = set()
    for root in roots:
        checkout = root.parents[1] if root.parent.name in {"docs", "docs_new"} else root
        commit = run_readonly(["git", "rev-parse", "HEAD"], cwd=str(checkout))
        for path in cookbook_document_matches(root, model):
            try:
                body = path.read_bytes()
            except OSError:
                continue
            digest = hashlib.sha256(body).hexdigest()
            # The current SGLang checkout is searched before the downloaded
            # snapshot.  Identical pages must not create duplicate recipe
            # trials merely because both sources retain the same revision.
            if digest in seen_document_hashes:
                continue
            seen_document_hashes.add(digest)
            documents.append({
                "root": str(root), "path": str(path.relative_to(root)),
                "sha256": digest,
                "commit": commit.get("stdout") if commit.get("returncode") == 0 else None,
            })
            # Keep prose advice only when it names a real launch flag. This
            # preserves the human rationale from Cookbook without inventing
            # candidates from unconstrained natural-language parsing.
            text = body.decode("utf-8", errors="ignore")
            for line in text.splitlines():
                if "--" not in line:
                    continue
                flags = re.findall(r"--([a-z0-9-]+)", line.lower())
                if not flags:
                    continue
                normalized = [
                    COOKBOOK_FLAG_ALIASES.get(flag.replace("-", "_"), flag.replace("-", "_"))
                    for flag in flags
                ]
                supported = sorted(set(flag for flag in normalized if flag in COOKBOOK_TUNABLE_FLAGS))
                unrecognized = sorted(set(normalized) - set(supported))
                if supported:
                    tuning_tips.append({
                        "parameters": supported,
                        "unrecognized_parameters": unrecognized,
                        "text": line.strip()[:500],
                        "path": str(path.relative_to(root)),
                    })
                elif unrecognized:
                    tuning_tips.append({
                        "parameters": [],
                        "unrecognized_parameters": unrecognized,
                        "text": line.strip()[:500],
                        "path": str(path.relative_to(root)),
                        "policy": "evidence for parameter evolution; never executable as a trusted recipe",
                    })
            for recipe in cookbook_recipes_from_document(path, root):
                incompatibility = cookbook_recipe_compatibility(recipe, model)
                if incompatibility:
                    excluded_recipes.append({
                        "name": recipe["name"],
                        "documented_model": recipe.get("documented_model"),
                        "reason": incompatibility,
                    })
                    continue
                recipes.append(recipe)
            for recipe in cookbook_snippet_recipes_from_document(path, root, body):
                incompatibility = cookbook_recipe_compatibility(recipe, model)
                if incompatibility:
                    excluded_recipes.append({
                        "name": recipe["name"],
                        "documented_model": recipe.get("documented_model"),
                        "reason": incompatibility,
                    })
                    continue
                automatic_behaviors.extend(
                    deepcopy(recipe.get("automatic_behaviors", []))
                )
                recipes.append(recipe)
    return {
        "status": "available" if documents else "not_found",
        "source": "local_sglang_checkout" if documents else None,
        "documents": documents,
        "recipes": recipes,
        "tuning_tips": list({
            json.dumps(item, sort_keys=True): item for item in tuning_tips
        }.values()),
        "automatic_behaviors": list({
            json.dumps(item, sort_keys=True): item for item in automatic_behaviors
        }.values()),
        "excluded_recipes": excluded_recipes,
        "policy": "parsed launch commands become executable candidates; documented flag tips are retained as auditable routing evidence",
    }


def inferred_cookbook_url(model: dict[str, Any]) -> str | None:
    identity = " ".join([
        *(str(item) for item in model.get("architectures", [])),
        str(model.get("model_type", "")),
        str(model.get("checkpoint_name", "")),
    ]).lower().replace("_", ".")
    normalized_identity = identity.replace("-", ".")
    if "qwen3.8.flash.next" in normalized_identity:
        return "https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next"
    if "qwen3.8" in normalized_identity:
        return "https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8"
    if "qwen3.next" in identity or "qwen3next" in identity:
        return "https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-Next"
    if "qwen3.5" in identity:
        return "https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5"
    if "qwen3.6" in identity:
        return "https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6"
    if "qwen3" in identity:
        return "https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3"
    return None


def fetch_reference(url: str, task: dict[str, Any]) -> dict[str, Any]:
    """Fetch an explicitly supplied official reference and retain only evidence metadata."""
    if not url.startswith("https://"):
        return {"url": url, "status": "rejected", "reason": "only HTTPS references are allowed"}
    environment = task.get("env", {}) if isinstance(task.get("env"), dict) else {}
    proxies = {
        scheme: environment[key]
        for scheme, key in (("http", "http_proxy"), ("https", "https_proxy"))
        if isinstance(environment.get(key), str) and environment[key]
    }
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
        request = urllib.request.Request(url, headers={"User-Agent": "inference-autopilot/1.0"})
        with opener.open(request, timeout=20) as response:
            body = response.read(2_000_000)
        text = body.decode("utf-8", errors="replace")
        return {
            "url": url,
            "status": "fetched",
            "retrieved_at": utc_now(),
            "sha256": hashlib.sha256(body).hexdigest(),
            "text": text,
        }
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return {"url": url, "status": "unavailable", "reason": type(exc).__name__}


def cookbook_snapshot_evidence(snapshot_dir: Path, model: dict[str, Any]) -> dict[str, Any]:
    """Describe a local cookbook clone without relying on branch names.

    The clone is a reproducibility artifact, not an executable configuration
    source. Every candidate still has to pass the locally discovered SGLang
    CLI contract before a server is launched.
    """
    if not snapshot_dir.is_dir() or not (snapshot_dir / ".git").exists():
        return {"status": "unavailable", "path": str(snapshot_dir), "reason": "not_a_git_checkout"}
    commit = run_readonly(["git", "rev-parse", "HEAD"], cwd=str(snapshot_dir))
    terms = set()
    for architecture in model.get("architectures", []):
        value = str(architecture).lower().replace("_", ".")
        for token in re.findall(r"[a-z]+\d+(?:\.\d+)?", value):
            terms.add(token)
    model_type = str(model.get("model_type", "")).lower().replace("_", ".")
    if model_type:
        terms.add(model_type)
    matches: list[dict[str, str]] = []
    for path in snapshot_dir.rglob("*"):
        if path.suffix.lower() not in COOKBOOK_DOCUMENT_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
            body = path.read_bytes()
        except OSError:
            continue
        normalized = body.decode("utf-8", errors="ignore").lower().replace("_", ".")
        if terms and not any(term in normalized for term in terms):
            continue
        matches.append({
            "path": str(path.relative_to(snapshot_dir)),
            "sha256": hashlib.sha256(body).hexdigest(),
        })
        if len(matches) == 32:
            break
    return {
        "status": "available",
        "path": str(snapshot_dir),
        "commit": commit.get("stdout") if commit.get("returncode") == 0 else None,
        "matched_markdown": matches,
        "policy": "snapshot is retained for offline review; only local-CLI-compatible candidates may execute",
    }


def provision_cookbook_snapshot(
    task: dict[str, Any], run_root: Path,
    progress: Callable[[str, str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create or reuse a private shallow cookbook snapshot for an execution run.

    This is intentionally called by `run`, not `doctor` or `plan`, so the
    latter remain non-mutating inspection commands. A failed download is
    evidence, not an optimization failure, unless the task requires cookbook
    material.
    """
    prepared = deepcopy(task)
    knowledge = prepared.setdefault("knowledge", {})
    local = local_cookbook_evidence(prepared, model_inventory(prepared["model_path"]))
    if local["status"] == "available":
        if progress:
            progress("local", "using Cookbook pages from the selected SGLang checkout")
        return prepared, local
    configured = knowledge.get("cookbook_snapshot_dir")
    snapshot_dir = Path(configured).expanduser() if configured else run_root / "knowledge" / "sglang-docs"
    repository = knowledge.get("cookbook_repository", DEFAULT_COOKBOOK_REPOSITORY)
    if snapshot_dir.is_dir() and (snapshot_dir / ".git").exists():
        if progress:
            progress("reuse", f"reusing existing snapshot at {snapshot_dir}")
        knowledge["cookbook_snapshot_dir"] = str(snapshot_dir)
        return prepared, local_cookbook_evidence(prepared, model_inventory(prepared["model_path"]))
    if not prepared.get("allow_download", False):
        if progress:
            progress("skip", "download disabled and no local Cookbook page matched")
        return prepared, {
            "status": "not_requested", "path": str(snapshot_dir),
            "reason": "allow_download=false; no network cookbook snapshot was created",
        }
    if snapshot_dir.exists():
        return prepared, {
            "status": "unavailable", "path": str(snapshot_dir),
            "reason": "snapshot path exists but is not a Git checkout",
        }
    try:
        snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(snapshot_dir.parent, 0o700)
        environment = os.environ.copy()
        for key in ("http_proxy", "https_proxy"):
            value = prepared.get("env", {}).get(key)
            if isinstance(value, str) and value:
                environment[key] = value
        if progress:
            progress("clone", f"cloning the documentation snapshot from {repository}")
        clone_command = [
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            str(repository), str(snapshot_dir),
        ]
        with (run_root / "cookbook-clone.log").open("w", encoding="utf-8") as clone_log:
            clone = subprocess.Popen(
                clone_command, stdout=clone_log, stderr=subprocess.STDOUT,
                text=True, env=environment,
            )
            clone_started = time.monotonic()
            last_heartbeat = clone_started
            while clone.poll() is None:
                now = time.monotonic()
                if now - clone_started >= 120:
                    clone.kill()
                    clone.wait(timeout=10)
                    raise subprocess.TimeoutExpired(clone_command, 120)
                if progress and now - last_heartbeat >= 30:
                    progress("clone", f"Cookbook clone still running; elapsed={now - clone_started:.0f}s")
                    last_heartbeat = now
                time.sleep(1)
        result = subprocess.CompletedProcess(clone_command, clone.returncode, "", "")
        if result.returncode != 0:
            return prepared, {
                "status": "unavailable", "path": str(snapshot_dir),
                "repository": repository, "reason": "git_clone_failed",
            }
        if progress:
            progress(
                "sparse-checkout",
                "selecting Cookbook pages and their static deployment config snippets",
            )
        sparse = subprocess.run(
            [
                "git", "sparse-checkout", "set",
                "docs/cookbook", "docs_new/cookbook", "docs/src/snippets",
            ],
            capture_output=True, text=True, timeout=60, check=False, cwd=str(snapshot_dir), env=environment,
        )
        if sparse.returncode != 0:
            return prepared, {
                "status": "unavailable", "path": str(snapshot_dir),
                "repository": repository, "reason": "git_sparse_checkout_failed",
            }
    except (OSError, subprocess.TimeoutExpired):
        return prepared, {
            "status": "unavailable", "path": str(snapshot_dir),
            "repository": repository, "reason": "git_clone_unavailable",
        }
    knowledge["cookbook_snapshot_dir"] = str(snapshot_dir)
    if progress:
        progress("complete", "Cookbook snapshot ready; parsing model-matched pages")
    return prepared, local_cookbook_evidence(prepared, model_inventory(prepared["model_path"]))


def cookbook_evidence(task: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    knowledge = task.get("knowledge", {}) if isinstance(task.get("knowledge"), dict) else {}
    url = knowledge.get("model_cookbook_url") or inferred_cookbook_url(model)
    required = bool(knowledge.get("require_cookbook", False))
    local = local_cookbook_evidence(task, model)
    snapshot_dir = knowledge.get("cookbook_snapshot_dir")
    snapshot = (
        cookbook_snapshot_evidence(Path(snapshot_dir).expanduser(), model)
        if isinstance(snapshot_dir, str) else None
    )
    if url is None and local["status"] != "available":
        return {
            "status": "not_matched", "required": required, "model_profile": None,
            "repository_snapshot": snapshot,
        }
    if url is None:
        url = "local://sglang-checkout/cookbook"
    fetched = fetch_reference(url, task) if (
        url.startswith("https://") and task.get("allow_download", False) and local["status"] != "available"
    ) else {
        "url": url, "status": "not_fetched", "reason": "allow_download=false"
    }
    text = str(fetched.pop("text", ""))
    normalized = re.sub(r"\s+", " ", text).lower()
    # Cookbook pages share a navigation tree that mentions adjacent model
    # families. The fetched body is evidence for flags and constraints, not
    # model identity: choose the exact canonical page path before inspecting
    # text, otherwise a Qwen3 page can be mistaken for Qwen3.5 from its nav.
    canonical_page = url.rstrip("/").lower()
    if canonical_page.endswith("/qwen3.5"):
        qwen35, qwen36, qwen3 = True, False, False
    elif canonical_page.endswith("/qwen3.6"):
        qwen35, qwen36, qwen3 = False, True, False
    elif canonical_page.endswith("/qwen3"):
        qwen35, qwen36, qwen3 = False, False, True
    else:
        qwen35 = "qwen3.5" in normalized
        qwen36 = "qwen3.6" in normalized and not qwen35
        qwen3 = "qwen3 moe" in normalized and not qwen35 and not qwen36
    claims = {
        "mtp_eagle": "speculative-algorithm eagle" in normalized,
        "mtp_nextn": "speculative-algo nextn" in normalized or "speculative-algorithm nextn" in normalized,
        "mamba_extra_buffer": "extra_buffer" in normalized and "mamba" in normalized,
        "page_size_64": "page-size 64" in normalized or "page size 64" in normalized,
        "spec_v2": "sglang_enable_spec_v2" in normalized,
        "expert_parallelism": "expert parallel" in normalized,
    }
    profile = None
    # A checked-out Cookbook is authoritative for its own model variants.
    # Built-in recipes are only an offline fallback when no local page exists.
    use_builtin_profile = local["status"] != "available"
    if qwen35 and use_builtin_profile:
        # Qwen3.5's checkpoint-integrated MTP uses NEXTN.  It is not the
        # EAGLE draft-model path used by the Qwen3.6 cookbook.  Bundles are
        # subsequently checked against the locally discovered ServerArgs, so
        # an older/newer SGLang checkout cannot receive unknown flags.
        profile = {
            "name": "qwen3.5-hybrid-mtp",
            "requires_mtp_weights": True,
            "initial_bundles": [
                {
                    "name": "qwen35-prefix-cache-lpm",
                    "config": {
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "page_size": 64,
                        "schedule_policy": "lpm",
                        "mem_fraction_static": 0.8,
                    },
                },
                {
                    "name": "qwen35-mtp-nextn-3-1-4",
                    "config": {
                        "speculative_algorithm": "NEXTN",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mem_fraction_static": 0.8,
                    },
                },
                {
                    "name": "qwen35-mtp-prefix-cache-lpm",
                    "config": {
                        "speculative_algorithm": "NEXTN",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "page_size": 64,
                        "schedule_policy": "lpm",
                        "mem_fraction_static": 0.8,
                    },
                },
            ],
            "claims": claims,
        }
    elif qwen36 and use_builtin_profile:
        profile = {
            "name": "qwen3.6-hybrid-mtp",
            "requires_mtp_weights": True,
            "initial_bundles": [
                {
                    "name": "qwen36-prefix-cache-lpm",
                    "config": {
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "schedule_policy": "lpm",
                        "mem_fraction_static": 0.8,
                    },
                },
                {
                    "name": "qwen36-mtp-eagle-3-1-4",
                    "config": {
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mem_fraction_static": 0.8,
                    },
                },
                {
                    "name": "qwen36-mtp-prefix-cache-lpm",
                    "config": {
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "page_size": 64,
                        "mem_fraction_static": 0.8,
                    },
                },
                {
                    "name": "qwen36-mtp-prefix-memory-075",
                    "config": {
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "page_size": 64,
                        "schedule_policy": "lpm",
                        "mem_fraction_static": 0.75,
                    },
                },
                {
                    "name": "qwen36-mtp-prefix-admission-graph",
                    "config": {
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "page_size": 64,
                        "schedule_policy": "lpm",
                        "mem_fraction_static": 0.8,
                        "max_running_requests": 16,
                        "cuda_graph_max_bs_decode": 8,
                    },
                },
                {
                    "name": "qwen36-mtp-prefix-continuous-decode-2",
                    "config": {
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "page_size": 64,
                        "schedule_policy": "lpm",
                        "mem_fraction_static": 0.8,
                        "num_continuous_decode_steps": 2,
                    },
                },
                {
                    "name": "qwen36-mtp-prefix-moe-triton",
                    "config": {
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "page_size": 64,
                        "schedule_policy": "lpm",
                        "mem_fraction_static": 0.8,
                        "moe_runner_backend": "triton",
                    },
                },
            ],
            "claims": claims,
        }
        if task.get("capability_overrides", {}).get("mtp") == "disabled":
            profile["initial_bundles"] = [
                bundle for bundle in profile["initial_bundles"]
                if "speculative_algorithm" not in bundle["config"]
            ]
            profile["mtp_status"] = "disabled_by_task"
    elif qwen3 and use_builtin_profile:
        # Qwen3's published EAGLE example relies on a separately supplied
        # draft checkpoint. Do not manufacture a speculative candidate from
        # the target checkpoint alone. Expert parallelism is instead routed
        # from the model's exact FP8 shape and the local ServerArgs contract.
        profile = {
            "name": "qwen3-moe",
            "requires_mtp_weights": False,
            "initial_bundles": [],
            "claims": claims,
            "speculative_policy": "requires_explicit_compatible_draft_model",
            "parallelism_policy": "consider EP only when TP/EP and FP8 block dimensions are mathematically compatible",
        }
    extracted_recipes = []
    for recipe in local.get("recipes", []):
        config = deepcopy(recipe.get("config", {}))
        # Documentation commands often encode one measured topology (for
        # example TP=8 on H200).  Topology is selected separately from the
        # local GPU pool; retaining it would make a generic recipe invalid on
        # otherwise compatible hardware.
        preserve_topology = bool(recipe.get("preserve_topology"))
        documented_topology = {
            parameter: config[parameter]
            for parameter in ("tp_size", "pp_size", "dp_size", "ep_size")
            if parameter in config
        }
        if not preserve_topology:
            for parameter in documented_topology:
                config.pop(parameter, None)
        if config or recipe.get("required_functional_config"):
            extracted_recipes.append({
                **recipe,
                "config": config,
                "topology_adaptation": {
                    "documented": documented_topology,
                    "policy": (
                        "verified deployment cells preserve their atomic topology when it "
                        "fits this host; legacy prose recipes remain topology evidence and "
                        "are adapted to the local GPU pool"
                    ),
                    "preserved": preserve_topology,
                },
            })
    if extracted_recipes:
        if profile is None:
            profile = {
                "name": "local-cookbook-recipes",
                "requires_mtp_weights": False,
                "initial_bundles": [],
                "claims": claims,
            }
        seen = {
            json.dumps(bundle.get("config", {}), sort_keys=True)
            for bundle in profile.get("initial_bundles", [])
        }
        for recipe in extracted_recipes:
            if not recipe.get("config"):
                continue
            signature = json.dumps(recipe["config"], sort_keys=True)
            if signature not in seen:
                profile["initial_bundles"].append(recipe)
                seen.add(signature)
        functional_values: dict[str, dict[str, Any]] = {}
        for recipe in extracted_recipes:
            for parameter, value in recipe.get("required_functional_config", {}).items():
                functional_values.setdefault(parameter, {})[
                    json.dumps(value, sort_keys=True)
                ] = value
        profile["required_functional_config"] = {
            parameter: deepcopy(next(iter(values.values())))
            for parameter, values in functional_values.items()
            if len(values) == 1
        }
        profile["functional_config_conflicts"] = {
            parameter: sorted(values)
            for parameter, values in functional_values.items()
            if len(values) > 1
        }
        profile["automatic_behaviors"] = deepcopy(
            local.get("automatic_behaviors", [])
        )
        profile["task_deployment_mode"] = deployment_policy(task)["mode"]
    return {
        **fetched,
        "status": "available" if local["status"] == "available" else fetched.get("status"),
        "required": required,
        "repository_snapshot": snapshot,
        "local_checkout": local,
        "model_profile": profile,
        "claims": claims,
        "checkpoint_verification": {
            "has_mtp_weights": bool(model.get("has_mtp_weights")),
            "mtp_weight_key_count": model.get("mtp_weight_key_count", 0),
        },
    }


def selected_gpus(task: dict[str, Any], inventory: dict[str, Any]) -> list[dict[str, Any]]:
    gpus = inventory.get("gpus", [])
    if not gpus:
        return []
    task_env = task.get("env", {})
    visible = task_env.get(
        "CUDA_VISIBLE_DEVICES",
        task_env.get(
            "HIP_VISIBLE_DEVICES",
            os.environ.get("CUDA_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES")),
        ),
    )
    max_gpus = task.get("max_gpus")
    if visible is None:
        return gpus[:max_gpus] if isinstance(max_gpus, int) else gpus
    identifiers = [item.strip() for item in str(visible).split(",") if item.strip() and item.strip() != "-1"]
    if not identifiers:
        return []
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("GPU visibility contains duplicate indexes or UUIDs")
    by_index = {int(gpu["index"]): gpu for gpu in gpus if str(gpu.get("index", "")).isdigit()}
    by_uuid = {str(gpu.get("uuid")): gpu for gpu in gpus if gpu.get("uuid")}
    selected = []
    for identifier in identifiers:
        gpu = by_index.get(int(identifier)) if identifier.isdigit() else by_uuid.get(identifier)
        if gpu is not None:
            selected.append(gpu)
    if len(selected) != len(identifiers):
        discovered = [gpu.get("index") for gpu in gpus]
        raise ValueError(
            f"GPU visibility {identifiers} does not match discovered indexes/UUIDs; discovered indexes: {discovered}"
        )
    return selected[:max_gpus] if isinstance(max_gpus, int) else selected


def selected_gpu_identifiers(task: dict[str, Any], inventory: dict[str, Any]) -> list[str]:
    """Return stable physical identifiers for the task's resource allocation."""
    selected = selected_gpus(task, inventory)
    task_env = task.get("env", {})
    visible = task_env.get("CUDA_VISIBLE_DEVICES", task_env.get("HIP_VISIBLE_DEVICES"))
    if visible is not None:
        identifiers = [item.strip() for item in str(visible).split(",") if item.strip()]
        return identifiers[:len(selected)]
    return [str(gpu["index"]) for gpu in selected]


def task_on_gpus(
    task: dict[str, Any], identifiers: list[str], *, port: int, parallel_trials: int,
) -> dict[str, Any]:
    """Create a stage-local task constrained to an explicit GPU pool."""
    constrained = deepcopy(task)
    env = deepcopy(constrained.get("env", {}))
    env.pop("HIP_VISIBLE_DEVICES", None)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(identifiers)
    constrained["env"] = env
    constrained["max_gpus"] = len(identifiers)
    constrained["parallel_trials"] = max(1, min(parallel_trials, len(identifiers)))
    constrained["port"] = port
    return constrained


def visible_gpu_count(task: dict[str, Any], inventory: dict[str, Any]) -> int:
    return len(selected_gpus(task, inventory))


def kv_heads_support_tp(kv_heads: int | None, tp_size: int) -> bool:
    """Match SGLang's sharded-or-replicated GQA head rule."""
    if not isinstance(kv_heads, int) or kv_heads <= 0:
        return True
    return (
        kv_heads % tp_size == 0
        if kv_heads >= tp_size
        else tp_size % kv_heads == 0
    )


def minimum_tp(task: dict[str, Any], inventory: dict[str, Any], model: dict[str, Any]) -> int:
    gpus = selected_gpus(task, inventory)
    count = len(gpus)
    if not count:
        raise ValueError("no visible GPUs remain after applying the visibility environment")
    if all(gpu.get("memory_mib", 0) > 0 for gpu in gpus):
        memory_bytes = min(gpu["memory_mib"] for gpu in gpus) * 1024**2
    else:
        raise ValueError("GPU memory could not be discovered; refusing to guess tensor parallelism")
    # Keep allocator/runtime reserve explicit instead of applying both a 12%
    # weight multiplier and a second per-GPU reserve.
    weight_required = model.get("weight_bytes", 0) * 1.08
    workload = task.get("workload", {})
    kv_estimate = (
        estimate_kv_cache_bytes(model, workload)
        if isinstance(workload, dict)
        and all(
            isinstance(workload.get(key), int) and workload.get(key) > 0
            for key in ("input_tokens", "output_tokens", "max_concurrency")
        )
        else {"available": False, "estimated_bytes": 0}
    )
    kv_required = float(kv_estimate.get("estimated_bytes", 0) or 0)
    per_gpu_runtime_reserve = max(4 * 1024**3, memory_bytes * 0.05)
    heads = model.get("num_attention_heads")
    kv_heads = model.get("num_key_value_heads")
    for tp in range(1, count + 1):
        if isinstance(heads, int) and heads > 0 and heads % tp:
            continue
        if not kv_heads_support_tp(kv_heads, tp):
            continue
        required_per_gpu = (weight_required + kv_required) / tp + per_gpu_runtime_reserve
        if required_per_gpu <= memory_bytes * 0.88:
            return tp
    raise ValueError(
        "model weights, workload KV estimate, and runtime reserve do not fit a legal "
        "tensor-parallel subset of the visible GPUs"
    )


def supported_tp_sizes(discovery: dict[str, Any]) -> list[int]:
    """Return topology- and model-safe tensor-parallel sizes.

    This is intentionally conservative about model dimensions, but a service
    may legally use a subset of the visible GPUs.  Divisibility of the *host*
    GPU count is a replica-placement concern, not a TP legality constraint.
    """
    count = int(discovery["derived"]["visible_gpu_count"])
    minimum = int(discovery["derived"]["minimum_tp_size"])
    model = discovery.get("model", {})
    heads = model.get("num_attention_heads")
    kv_heads = model.get("num_key_value_heads")
    sizes: list[int] = []
    for tp in range(minimum, count + 1):
        if isinstance(heads, int) and heads > 0 and heads % tp:
            continue
        if not kv_heads_support_tp(kv_heads, tp):
            continue
        sizes.append(tp)
    if not sizes:
        raise ValueError(
            "no tensor-parallel size is compatible with the visible GPU count and model head dimensions"
        )
    return sizes


def supported_ep_sizes(
    discovery: dict[str, Any], tp_size: int | None = None,
) -> list[int]:
    """Return EP degrees that preserve the model's FP8 MoE tile contract.

    SGLang's Qwen3 FP8 guidance constrains the expert's tensor-parallel
    dimension: ``(moe_intermediate_size / (TP / EP))`` must be divisible by
    the FP8 weight block-N size.  This check intentionally refuses to infer
    candidates when the checkpoint does not publish these dimensions; a
    startup failure after a long model load is not a useful search result.
    """
    model = discovery.get("model", {})
    if not model.get("is_moe"):
        return []
    tp = int(tp_size or discovery.get("derived", {}).get("minimum_tp_size") or 1)
    intermediate = model.get("moe_intermediate_size")
    block_shape = model.get("weight_block_size")
    if not (
        isinstance(intermediate, int) and intermediate > 0
        and isinstance(block_shape, list) and len(block_shape) == 2
        and isinstance(block_shape[1], int) and block_shape[1] > 0
    ):
        return []
    block_n = block_shape[1]
    valid: list[int] = []
    for ep in range(2, tp + 1):
        if tp % ep:
            continue
        moe_tp = tp // ep
        if intermediate % moe_tp:
            continue
        if (intermediate // moe_tp) % block_n:
            continue
        valid.append(ep)
    return valid


def deployment_feasibility(
    task: dict[str, Any], inventory: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Assess whether the checkpoint can launch on the selected local GPUs."""
    gpus = selected_gpus(task, inventory)
    if not gpus:
        return {
            "status": "insufficient_inventory",
            "reason": "no GPUs remain after applying the visibility selection",
        }
    if any(not isinstance(gpu.get("memory_mib"), int) or gpu["memory_mib"] <= 0 for gpu in gpus):
        return {
            "status": "insufficient_inventory",
            "reason": "every selected GPU must have discovered memory capacity",
        }
    try:
        minimum = minimum_tp(task, inventory, model)
        supported = supported_tp_sizes({
            "derived": {"visible_gpu_count": len(gpus), "minimum_tp_size": minimum},
            "model": model,
        })
    except ValueError as exc:
        return {
            "status": "requires_parallel_or_variant",
            "selected_gpu_count": len(gpus),
            "reason": str(exc),
        }
    return {
        "status": "deployable_as_is",
        "selected_gpu_count": len(gpus),
        "minimum_tp_size": minimum,
        "supported_tp_sizes": supported,
        "reason": (
            "checkpoint weights fit a legal tensor-parallel layout on the selected local GPUs; "
            "the server launch remains the final allocator/KV-cache feasibility check"
        ),
    }


def estimate_kv_cache_bytes(model: dict[str, Any], workload: dict[str, Any]) -> dict[str, Any]:
    """Return a transparent BF16 KV upper estimate for feasibility routing."""
    layers = model.get("num_hidden_layers")
    hidden = model.get("hidden_size")
    heads = model.get("num_attention_heads")
    kv_heads = model.get("num_key_value_heads") or heads
    if not all(isinstance(value, int) and value > 0 for value in (layers, hidden, heads, kv_heads)):
        return {"available": False, "reason": "model config lacks attention dimensions"}
    head_dim = hidden // heads
    if head_dim <= 0:
        return {"available": False, "reason": "invalid attention dimensions"}
    # K and V, BF16/FP16 storage. Hybrid models may use less; this is an
    # admission estimate, not a claimed allocator allocation.
    bytes_per_token = layers * kv_heads * head_dim * 2 * 2
    tokens = (int(workload["input_tokens"]) + int(workload["output_tokens"])) * int(workload["max_concurrency"])
    return {
        "available": True,
        "assumption": "BF16/FP16 K/V upper estimate before prefix reuse and allocator effects",
        "bytes_per_token": bytes_per_token,
        "target_tokens": tokens,
        "estimated_bytes": bytes_per_token * tokens,
    }


def single_gpu_feasibility(task: dict[str, Any], inventory: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Assess one visible GPU without launching SGLang or changing a model."""
    gpus = selected_gpus(task, inventory)
    if len(gpus) != 1 or not isinstance(gpus[0].get("memory_mib"), int) or gpus[0]["memory_mib"] <= 0:
        return {
            "status": "insufficient_inventory",
            "reason": "single-GPU feasibility requires exactly one visible GPU with discovered memory",
        }
    gpu_bytes = gpus[0]["memory_mib"] * 1024**2
    usable_bytes = int(gpu_bytes * 0.88)
    weights = int(model.get("weight_bytes", 0))
    kv = estimate_kv_cache_bytes(model, task["workload"])
    kv_bytes = int(kv.get("estimated_bytes", 0))
    runtime_reserve = max(4 * 1024**3, int(gpu_bytes * 0.05))
    estimated_peak = int(weights * 1.12) + kv_bytes + runtime_reserve
    headroom = usable_bytes - estimated_peak
    status = "deployable_as_is" if headroom >= 0 else "requires_parallel_or_variant"
    required_gpus = max(1, math.ceil(estimated_peak / max(1, usable_bytes)))
    profile = match_hardware_profile(inventory, load_hardware_catalog()) or {}
    precisions = profile.get("precision", [])
    suggestions = []
    if status != "deployable_as_is" and task.get("deployment", {}).get("allow_model_variant_recommendations", True):
        if "fp8" in precisions:
            suggestions.append({"target": "fp8 checkpoint", "reason": "hardware catalog reports FP8 capability; verify the exact checkpoint/backend and quality before use"})
        if any(item in precisions for item in ("mxfp4", "mxfp6", "mxfp8")):
            suggestions.append({"target": "FP4/MXFP4 checkpoint", "reason": "hardware catalog reports low-precision capability; requires an explicit compatible checkpoint and quality validation"})
    return {
        "status": status,
        "scope": "single_gpu_estimate",
        "gpu": {"name": gpus[0].get("name"), "memory_gib": round(gpu_bytes / 1024**3, 2), "usable_gib": round(usable_bytes / 1024**3, 2)},
        "model_weight_gib": round(weights / 1024**3, 2),
        "weight_reserve_gib": round(weights * 1.12 / 1024**3, 2),
        "runtime_reserve_gib": round(runtime_reserve / 1024**3, 2),
        "kv_cache_estimate": {**kv, "estimated_gib": round(kv_bytes / 1024**3, 2)},
        "estimated_peak_gib": round(estimated_peak / 1024**3, 2),
        "estimated_headroom_gib": round(headroom / 1024**3, 2),
        "minimum_gpu_count_estimate": required_gpus,
        "recommendations": suggestions,
        "quality_gate": {
            "required_for_model_variant": bool(suggestions),
            "evaluation_dataset": task.get("quality", {}).get("evaluation_dataset"),
            "max_regression_pct": task.get("quality", {}).get("max_regression_pct"),
            "policy": "never switch, download, or claim a quantized model is acceptable without an explicit quality evaluation",
        },
    }


def power_of_two_floor(value: int) -> int:
    return 1 << max(0, int(math.log2(max(1, value))))


def power_of_two_ceil(value: int) -> int:
    floor = power_of_two_floor(value)
    return floor if floor == value else floor * 2


def expected_prefill_tokens(workload: dict[str, Any]) -> int:
    """Estimate uncached prefill work per request for candidate routing.

    This is a prior only. The benchmark and cache metrics remain the source of
    truth because a declared prefix-reuse ratio need not equal the observed hit
    rate.
    """
    input_tokens = int(workload["input_tokens"])
    reuse_ratio = float(workload.get("prefix_reuse_ratio", 0.0))
    return max(1, int(round(input_tokens * max(0.0, 1.0 - reuse_ratio))))


def chunk_candidates(
    task: dict[str, Any], framework_default: int | None = None
) -> list[int]:
    """Return a workload-centred chunk sweep, including the resolved default.

    `chunked_prefill_size` is a prefill-batch budget, not a per-request value.
    The relevant first-order boundary is therefore concurrent uncached prefill
    tokens. SGLang's current-version resolved default is deliberately retained
    as an anchor instead of embedding a GPU-specific default in this tool.
    """
    workload = task["workload"]
    boundary = expected_prefill_tokens(workload) * int(workload["max_concurrency"])
    if boundary < 256:
        return []
    lower = max(256, power_of_two_floor(boundary) // 2)
    centre = max(256, power_of_two_floor(boundary))
    upper = max(256, power_of_two_ceil(boundary))
    resolved_default = framework_default if isinstance(framework_default, int) and framework_default > 0 else None
    # Do not grow activation allocations without a bounded local reason. The
    # default is still included as an anchor even if it falls outside this cap.
    cap = max(16384, (resolved_default or upper) * 4)
    sweep_end = min(cap, max(upper * 2, (resolved_default or upper) // 2))
    values: set[int] = set()
    value = lower
    while value <= sweep_end:
        values.add(value)
        value *= 2
    values.update({centre, upper})
    # A chunk is a *batch* budget, but a long individual request is still a
    # useful whole-prefill reference point.  In particular, this distinguishes
    # interleaved prefill from allowing one full prompt to proceed before
    # decode.  Keep it bounded by the same local allocation cap and let the
    # memory feasibility model make the final admission decision.
    request_anchor = power_of_two_ceil(int(workload["input_tokens"]))
    if int(workload["input_tokens"]) >= 1024 and request_anchor <= cap:
        values.add(request_anchor)
    if resolved_default is not None:
        values.add(resolved_default)
    return sorted(value for value in values if 256 <= value <= cap)


def chunk_memory_feasibility(
    task: dict[str, Any],
    discovery: dict[str, Any],
    effective: dict[str, Any],
    candidates: list[int],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Filter chunk sizes using the installed SGLang default-memory model.

    SGLang derives mem_fraction_static after resolving the chunk size. The
    activation-reserve coefficient is parsed from the current checkout rather
    than embedded here. The actual server launch remains the source of truth.
    """
    base_chunk = effective.get("chunked_prefill_size")
    base_fraction = effective.get("mem_fraction_static")
    tp_size = int(effective.get("tp_size", discovery.get("derived", {}).get("minimum_tp_size", 1)) or 1)
    pp_size = int(effective.get("pp_size", 1) or 1)
    gpus = selected_gpus(task, discovery.get("hardware", {}))
    gpu_memory_mib = min(
        (float(gpu.get("memory_mib", 0)) for gpu in gpus if gpu.get("memory_mib")),
        default=0.0,
    )
    weight_bytes = float(discovery.get("model", {}).get("weight_bytes") or 0)
    reserve_mib_per_token = discovery.get("framework", {}).get(
        "chunk_activation_reserve_mib_per_token"
    )
    if not (
        isinstance(base_chunk, int)
        and base_chunk > 0
        and isinstance(base_fraction, (int, float))
        and gpu_memory_mib > 0
        and weight_bytes > 0
        and isinstance(reserve_mib_per_token, (int, float))
        and reserve_mib_per_token > 0
    ):
        return sorted(set(candidates)), []

    weight_mib_per_gpu = weight_bytes / (1024**2) / max(1, tp_size * pp_size)
    fixed_headroom_mib = max(2048.0, weight_mib_per_gpu * 0.03)
    minimum_static_fraction = (weight_mib_per_gpu + fixed_headroom_mib) / gpu_memory_mib
    safety_margin = 0.01
    feasible: list[int] = []
    excluded: list[dict[str, Any]] = []
    for value in sorted(set(candidates)):
        predicted_fraction = (
            float(base_fraction)
            - (value - base_chunk) * float(reserve_mib_per_token) / gpu_memory_mib
        )
        evidence = {
            "chunked_prefill_size": value,
            "reserve_mib_per_token": reserve_mib_per_token,
            "predicted_mem_fraction_static": round(predicted_fraction, 3),
            "minimum_estimated_static_fraction": round(minimum_static_fraction, 3),
        }
        if predicted_fraction + 1e-9 < minimum_static_fraction + safety_margin:
            evidence["reason"] = (
                "predicted SGLang activation reserve leaves insufficient static memory for model weights"
            )
            excluded.append(evidence)
        else:
            feasible.append(value)
    return feasible, excluded


def rank_chunk_candidates(
    task: dict[str, Any],
    default_chunk: int | None,
    candidates: list[int],
    runtime_prefill: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    """Order chunk candidates by workload objective instead of numeric value."""
    values = sorted(set(candidates))
    if not isinstance(default_chunk, int) or default_chunk <= 0:
        return values, {
            "strategy": "workload_boundary_without_resolved_default",
            "reason": "the runtime did not expose a resolved chunked-prefill default",
        }
    lower = sorted((value for value in values if value < default_chunk), reverse=True)
    upper = sorted(value for value in values if value > default_chunk)
    mode = deployment_policy(task)["mode"]
    latency_slos = has_latency_slo(task)
    queue_pct = runtime_prefill.get("queue_nonempty_batch_pct")
    configured_concurrency = task["workload"].get("max_concurrency")
    observed_running = runtime_prefill.get("running_requests", {})
    observed_concurrency = (
        observed_running.get("p95") if isinstance(observed_running, dict) else None
    )
    concurrency_evidence = (
        int(configured_concurrency)
        if isinstance(configured_concurrency, int) and configured_concurrency > 0
        else max(1, int(observed_concurrency))
        if isinstance(observed_concurrency, (int, float)) and observed_concurrency > 0
        else 1
    )
    latency_pressure = latency_slos or (
        isinstance(queue_pct, (int, float)) and queue_pct >= 10.0
    )
    if mode == "online_latency" and latency_pressure:
        ordered = [*lower, *upper]
        strategy = "latency_interleaving_first"
        reason = (
            "online tail-latency or prefill-queue evidence prioritizes the nearest smaller chunks, "
            "then tests larger throughput-oriented chunks"
        )
    else:
        ordered = [*upper, *lower]
        strategy = "throughput_amortization_first"
        reason = (
            "offline or objective-only execution prioritizes the nearest larger chunks to reduce "
            "prefill fragmentation, then measures smaller sensitivity anchors"
        )
    return ordered, {
        "strategy": strategy,
        "reason": reason,
        "resolved_default": default_chunk,
        "expected_uncached_tokens_per_request": expected_prefill_tokens(task["workload"]),
        "concurrent_uncached_tokens": (
            expected_prefill_tokens(task["workload"])
            * concurrency_evidence
        ),
        "concurrency_evidence": (
            "configured_max_concurrency"
            if isinstance(configured_concurrency, int) and configured_concurrency > 0
            else "observed_prefill_running_requests_p95"
            if isinstance(observed_concurrency, (int, float)) and observed_concurrency > 0
            else "single_request_fallback_for_unbounded_preprofile_planning"
        ),
        "prefill_queue_nonempty_batch_pct": queue_pct,
        "declared_latency_slo": latency_slos,
        "ordered_candidates": ordered,
    }


def long_context_capacity_bundles(
    task: dict[str, Any],
    discovery: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    effective: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build coupled prefill candidates for long-context offline throughput.

    ``max_prefill_tokens`` is only useful when the scheduler can actually
    issue a correspondingly large prefill chunk.  Screening either flag alone
    therefore produces a false negative for the common 8K-default / 16K+
    prompt shape.  These bounded bundles establish the coarse joint response
    before scalar refinement spends restarts around a local point.
    """
    workload = task["workload"]
    if (
        deployment_policy(task)["mode"] != "offline_throughput"
        or int(workload["input_tokens"]) < 8192
        or "chunked_prefill_size" not in catalog
        or "max_prefill_tokens" not in catalog
    ):
        return []
    default_chunk = effective.get("chunked_prefill_size")
    default_prefill = effective.get("max_prefill_tokens")
    if not isinstance(default_chunk, int) or default_chunk <= 0:
        return []
    if not isinstance(default_prefill, int) or default_prefill <= 0:
        return []
    context_length = discovery["model"].get("context_length")
    ceiling = context_length if isinstance(context_length, int) and context_length > 0 else None
    chunks, _ = chunk_memory_feasibility(
        task, discovery, effective,
        [power_of_two_ceil(int(workload["input_tokens"])), power_of_two_ceil(int(workload["input_tokens"]) * 2)],
    )
    bundles: list[dict[str, Any]] = []
    for chunk in sorted(set(chunks)):
        if chunk <= default_chunk:
            continue
        prefill = max(default_prefill * 2, chunk * 2)
        if ceiling is not None:
            prefill = min(prefill, ceiling)
        if prefill <= default_prefill:
            continue
        bundles.append({
            "name": f"long-context-prefill-{chunk}-budget-{prefill}",
            "config": {
                "chunked_prefill_size": chunk,
                "max_prefill_tokens": prefill,
            },
            "priority": "medium",
            "reason": "jointly increase the prefill issue size and its admission budget for long-context offline traffic",
            "evidence": [
                f"input_tokens={workload['input_tokens']}",
                f"resolved_chunked_prefill_size={default_chunk}",
                f"resolved_max_prefill_tokens={default_prefill}",
            ],
        })
    return bundles


def shared_prefix_benchmark(workload: dict[str, Any]) -> dict[str, Any] | None:
    """Translate task-level shared-prefix intent to SGLang's native dataset."""
    config = workload.get("shared_prefix")
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("workload.shared_prefix must be an object")
    groups = int(config.get("groups", 1))
    prompts_per_group = int(config.get("prompts_per_group", math.ceil(workload["num_prompts"] / groups)))
    system_tokens = int(config.get("system_prompt_tokens", max(1, workload["input_tokens"] * 3 // 4)))
    question_tokens = int(config.get("question_tokens", workload["input_tokens"] - system_tokens))
    if groups <= 0 or prompts_per_group <= 0 or system_tokens <= 0 or question_tokens <= 0:
        raise ValueError("workload.shared_prefix groups, prompts_per_group, and token lengths must be positive")
    if system_tokens + question_tokens != workload["input_tokens"]:
        raise ValueError("workload.shared_prefix system_prompt_tokens + question_tokens must equal workload.input_tokens")
    return {
        "dataset_name": "generated-shared-prefix",
        "num_prompts": groups * prompts_per_group,
        "gsp_num_groups": groups,
        "gsp_prompts_per_group": prompts_per_group,
        "gsp_system_prompt_len": system_tokens,
        "gsp_question_len": question_tokens,
        "gsp_output_len": workload["output_tokens"],
        "gsp_range_ratio": float(config.get("range_ratio", 1.0)),
        # Preserve locality by default: a workload declared as shared-prefix
        # should exercise the cache rather than deliberately defeating it.
        "gsp_ordered": bool(config.get("ordered", True)),
    }


def offline_trial_request_window(
    workload: dict[str, Any], requested_requests: int, requested_warmups: int,
    *, unbounded_client_concurrency: bool,
) -> tuple[int, int]:
    """Choose a short initial offline window from load and token shape.

    A fixed request count under-samples low concurrency while a fixed multiple
    grows without bound at very high concurrency.  Five full pressure waves
    are a useful screening minimum: at concurrency 8 that means 40 requests,
    enough to observe queueing and KV turnover.  Cap the initial screen at 512
    requests because a single high-concurrency wave already contributes many
    observations.  ``run_trial`` expands the window when its duration gate
    proves the sample too short.
    """
    concurrency = max(1, int(workload["max_concurrency"]))
    del requested_requests
    initial_requests = min(512, max(40, concurrency * 5))
    if not unbounded_client_concurrency:
        # A bounded client must at least fill its configured concurrency once.
        initial_requests = max(concurrency, initial_requests)
    warmups = min(requested_warmups, 32, max(8, math.ceil(initial_requests / 10)))
    return initial_requests, warmups


def build_execution_spec(
    task: dict[str, Any],
    discovery: dict[str, Any],
    *,
    stage_name: str,
    baseline: dict[str, Any],
    space: dict[str, list[Any]],
    max_trials: int,
    repetitions: int,
    remaining_gpu_hours: float,
    remaining_wall_minutes: float,
) -> dict[str, Any]:
    compatibility = runtime_compatibility_constraints(discovery)
    baseline = {**baseline, **compatibility["required_config"]}
    workload = task["workload"]
    measurement = task.get("measurement") or {}
    min_requests = int(measurement.get("min_measurement_requests", max(256, workload["max_concurrency"] * 64)))
    if stage_name == "confirm":
        min_requests = confirmation_request_count(task)
    warmup_requests = int(measurement.get("warmup_requests", max(32, workload["max_concurrency"] * 8)))
    minimum_duration = float(measurement.get("min_measurement_seconds", 15))
    p99_request_waves = int(measurement.get("p99_request_waves", 10))
    # Candidate ranking needs enough steady-state evidence to eliminate large
    # regressions, not the full confirmation window. run_trial will expand the
    # request count if this bounded window is too short for the target model.
    shared_prefix = shared_prefix_benchmark(workload)
    offline_unbounded = (
        deployment_policy(task)["mode"] == "offline_throughput"
        and not task.get("slo")
    )
    if stage_name in {"screen", "interact"}:
        p99_constrained = has_p99_latency_slo(task)
        if deployment_policy(task)["mode"] == "offline_throughput":
            # Offline screening only nominates candidates. Keep enough
            # requests to observe sustained batching, but reserve the full
            # workload contract for confirmation.
            min_requests = min(min_requests, max(64, workload["max_concurrency"] * 4))
            warmup_requests = min(warmup_requests, max(16, workload["max_concurrency"]))
            minimum_duration = min(minimum_duration, 15.0 if p99_constrained else 10.0)
        else:
            min_requests = min(min_requests, max(40, workload["max_concurrency"] * 5))
            warmup_requests = min(warmup_requests, max(16, workload["max_concurrency"] * 2))
            minimum_duration = min(minimum_duration, 15.0 if p99_constrained else 10.0)
    if task.get("slo"):
        min_requests = max(min_requests, workload["max_concurrency"] * 5)
    if deployment_policy(task)["mode"] == "offline_throughput" and (
        stage_name.startswith("calibrate") or stage_name in {"screen", "interact"}
    ):
        min_requests, warmup_requests = offline_trial_request_window(
            workload,
            min_requests,
            warmup_requests,
            unbounded_client_concurrency=offline_unbounded,
        )
    # The bootstrap profile runs before SGLang has disclosed its capacity.
    # Every following no-SLO offline trial must leave a queue behind that
    # capacity, otherwise an unbounded client can still under-drive the server.
    saturation_requests = offline_saturation_request_count(
        task, confirmation=stage_name == "confirm"
    )
    if saturation_requests is not None:
        min_requests = max(min_requests, saturation_requests)
    if has_p99_latency_slo(task):
        # Apply this after offline-window shaping so the ten-wave p99 contract
        # cannot be reduced back to the generic five-wave screen.
        min_requests = max(
            min_requests, workload["max_concurrency"] * p99_request_waves
        )
    model = discovery["model"]
    inventory = discovery["hardware"]
    gpu_count = visible_gpu_count(task, inventory)
    gpu_model = "unknown"
    selected = selected_gpus(task, inventory)
    if selected:
        gpu_model = selected[0]["name"]
    requested_parallel_trials = int(task.get("parallel_trials", 1))
    gpu_signatures = {
        (str(gpu.get("name", "unknown")), int(gpu.get("memory_mib", 0)))
        for gpu in selected
    }
    # Cross-card comparison is only meaningful when the worker devices are
    # the same SKU and capacity. Mixed selections still work serially.
    parallel_workers = (
        min(requested_parallel_trials, len(selected))
        if stage_name in {"screen", "interact"} and len(gpu_signatures) == 1
        else 1
    )
    # With no latency SLO, offline throughput should not be artificially
    # limited by the client. SGLang still applies its resolved KV/admission
    # limits, while the benchmark supplies a backlog for it to schedule.
    unbounded_client_concurrency = offline_unbounded
    benchmark = {
        "dataset_name": "random-ids",
        # workload.num_prompts describes the workload shape, whereas the
        # measurement contract controls the required sample count. Do not let
        # an init-time default of 1024 force every candidate into a long run.
        "num_prompts": (
            min_requests if unbounded_client_concurrency
            else max(workload["max_concurrency"], min_requests)
        ),
        "random_input_len": workload["input_tokens"],
        "random_output_len": workload["output_tokens"],
        "random_range_ratio": 1.0,
        "request_rate": workload.get("request_rate", "inf"),
        **({} if unbounded_client_concurrency else {"max_concurrency": workload["max_concurrency"]}),
        "unbounded_concurrency": unbounded_client_concurrency,
        # Calibration can set this after the server has loaded.  It causes
        # autotune to query SGLang's resolved KV/admission limit and use that
        # value for the first SLO capacity probe instead of this task hint.
        "auto_max_concurrency": False,
        "warmup_requests": warmup_requests,
        "min_measurement_seconds": minimum_duration,
        "p99_request_waves": p99_request_waves if has_p99_latency_slo(task) else 0,
        "seed": 1,
        "output_details": True,
    }
    dataset = workload_dataset(workload)
    if dataset.get("name") in {"custom", "sharegpt"}:
        benchmark.update({
            "dataset_name": dataset["name"],
            "dataset_path": dataset["path"],
            "apply_chat_template": bool(dataset.get("apply_chat_template", False)),
        })
        context_length = task.get("model", {}).get("context_length", model.get("context_length"))
        if isinstance(context_length, int) and context_length > 0:
            benchmark["sharegpt_context_len"] = context_length
    if shared_prefix is not None:
        # shared_prefix_benchmark includes the task-level prompt count as a
        # convenient standalone default. Execution stages own their sample
        # count, however: screening must begin from its bounded request floor
        # and only expand when the duration gate proves that floor too short.
        # Updating the whole mapping first used to silently restore the init
        # default (typically 1024) for every shared-prefix trial.
        shared_prefix = dict(shared_prefix)
        shared_prefix.pop("num_prompts", None)
        benchmark.update(shared_prefix)
        groups = benchmark["gsp_num_groups"]
        benchmark["gsp_prompts_per_group"] = math.ceil(benchmark["num_prompts"] / groups)
        benchmark["num_prompts"] = groups * benchmark["gsp_prompts_per_group"]
    if float(workload.get("prefix_reuse_ratio", 0.0)) > 0 or shared_prefix is not None:
        # Each retry/repetition starts from an equivalent cache state while
        # still exercising prefix reuse within the benchmark window.
        benchmark["flush_cache"] = True
    spec = {
        "name": f"{task['name']}-{stage_name}"[:64],
        "mode": "execute",
        "framework": "sglang",
        "repository": task["repository"],
        "model": {
            "path": task["model_path"],
            "architecture": "moe" if model.get("is_moe") else "dense",
            "context_length": task.get("model", {}).get("context_length", model.get("context_length")),
            "detected_checkpoint_dtype": model.get("checkpoint_dtype", model.get("dtype")),
            "detected_weight_quantization": model.get("weight_quantization", model.get("quantization")),
            "detected_quantization_algorithm": model.get("quantization_algorithm"),
            "runtime_compatibility": deepcopy(compatibility),
        },
        "hardware": {
            "hosts": 1,
            "gpus_per_host": gpu_count,
            "gpu_model": gpu_model,
            "interconnect": "runtime-discovered",
        },
        "workload": {
            "arrival": "closed_loop" if workload.get("request_rate", "inf") == "inf" else "poisson",
            "request_rate": workload.get("request_rate", "inf"),
            "max_concurrency": workload["max_concurrency"],
            "num_prompts": workload["num_prompts"],
            "input_tokens": {"p50": workload["input_tokens"], "p95": workload["input_tokens"]},
            "output_tokens": {"p50": workload["output_tokens"], "p95": workload["output_tokens"]},
            "prefix_reuse_ratio": workload.get("prefix_reuse_ratio", 0.0),
            "dataset": deepcopy(dataset),
        },
        "slo": task["slo"],
        "objective": task["objective"],
        "deployment_mode": task.get("deployment_mode", "online_latency"),
        "experiment_fingerprint": experiment_fingerprint(task, discovery),
        "optimizer_contract": optimizer_contract(),
        "budget": {
            "max_trials": max_trials,
            "max_gpu_hours": max(0.01, remaining_gpu_hours),
            "max_wall_time_minutes": max(1, remaining_wall_minutes),
            "max_consecutive_failures": 3,
        },
        "scope": {
            "allow_launch": True,
            "allow_download": task.get("allow_download", False),
            "allow_profiling": False,
            "allow_parameter_changes": True,
            "allow_code_changes": False,
            "allow_kernel_changes": False,
            "production": False,
            "output_dir": str(Path(task["output_dir"]) / "stages"),
        },
        "execution": {
            "python": task["python"],
            "benchmark_module": discovery.get("framework", {}).get(
                "benchmark_module", "sglang.bench_serving"
            ),
            "host": "127.0.0.1",
            "port": task.get("port", 31000),
            "offline": task.get("offline", True),
            "require_accelerator": True,
            "startup_timeout_sec": 1200,
            "benchmark_timeout_sec": 1800,
            "shutdown_timeout_sec": 60,
            "env": task.get("env", {}),
            # One-pass screening trials are resource-packed onto disjoint GPU
            # sets. Capacity calibration and repeated confirmation stay serial;
            # the controller may separately pipeline Nsys with spare-GPU priors.
            "parallel_trials": parallel_workers,
            "gpu_allocation": "exclusive",
            "parallel_policy": {
                "requested_workers": requested_parallel_trials,
                "effective_workers": parallel_workers,
                "reason": (
                    "eligible same-SKU screening workers receive disjoint GPU sets and exclusive ports"
                    if parallel_workers > 1 else
                    "serial stage, insufficient selected GPUs, or mixed GPU SKU/capacity selection"
                ),
            },
            "parameter_bindings": execution_parameter_bindings(discovery),
        },
        "benchmark": benchmark,
        "search": {
            "strategy": "one_factor",
            "repetitions": repetitions,
            "order": "interleaved",
            "max_cv_pct": 10,
            "min_confirm_repetitions": task.get("confirmation_repetitions", 2),
            "require_all_slo_pass": True,
            "baseline": baseline,
            "compatibility_baseline": deepcopy(compatibility["required_config"]),
            "compatibility_evidence": deepcopy(compatibility["evidence"]),
            "space": space,
            "parameter_order": list(space),
        },
    }
    # Only user-declared precision overrides belong on the launch command.
    # Quantized HF checkpoints often declare torch_dtype=bfloat16 for
    # activations while their actual weights are FP8/INT4. SGLang's `auto`
    # path reads both fields correctly from the local checkpoint.
    declared_model = task.get("model", {})
    if isinstance(declared_model, dict) and declared_model.get("dtype"):
        spec["model"]["dtype"] = declared_model["dtype"]
    if isinstance(declared_model, dict) and declared_model.get("quantization"):
        spec["model"]["quantization"] = declared_model["quantization"]
    return spec


def calibration_spec(
    task: dict[str, Any], discovery: dict[str, Any], concurrency: int,
    remaining_gpu_hours: float, remaining_wall_minutes: float,
    auto_capacity_probe: bool = False,
    initial_unbounded_probe: bool = False,
) -> dict[str, Any]:
    calibrated_task = deepcopy(task)
    calibrated_task["workload"]["max_concurrency"] = concurrency
    # Capacity points below the target do not need the target's warmup and
    # request count. Keep a meaningful floor while scaling with actual load.
    target_concurrency = max(1, task["workload"]["max_concurrency"])
    scale = min(1.0, concurrency / target_concurrency)
    measurement = calibrated_task.get("measurement") or {}
    target_requests = int(measurement.get("min_measurement_requests", max(64, concurrency * 16)))
    calibrated_task["measurement"] = {
        **measurement,
        "warmup_requests": max(16, math.ceil(int(measurement.get("warmup_requests", 32)) * scale)),
        "min_measurement_requests": max(32, math.ceil(target_requests * scale)),
    }
    if deployment_policy(task)["mode"] == "offline_throughput":
        calibrated_task["measurement"]["warmup_requests"] = min(
            calibrated_task["measurement"]["warmup_requests"], max(16, concurrency)
        )
        calibrated_task["measurement"]["min_measurement_requests"] = min(
            calibrated_task["measurement"]["min_measurement_requests"], max(64, concurrency * 4)
        )
    calibrated_task["workload"]["num_prompts"] = max(
        calibrated_task["measurement"]["min_measurement_requests"], concurrency * 8
    )
    spec = build_execution_spec(
        calibrated_task,
        discovery,
        stage_name=f"calibrate-c{concurrency}",
        baseline={"tp_size": discovery["derived"]["minimum_tp_size"]},
        space={},
        max_trials=1,
        repetitions=1,
        remaining_gpu_hours=remaining_gpu_hours,
        remaining_wall_minutes=remaining_wall_minutes,
    )
    # Only the first adaptive SLO probe is automatic.  Once its resolved
    # capacity is observed, lower probes must use explicit client limits so
    # the halving/binary-search bracket is meaningful.
    spec["benchmark"]["auto_max_concurrency"] = auto_capacity_probe
    if auto_capacity_probe:
        spec["benchmark"]["unbounded_concurrency"] = initial_unbounded_probe
    return spec


def run_calibration(
    task: dict[str, Any], discovery: dict[str, Any], root: Path,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Measure the capacity curve in one resident baseline-server session."""
    policy = deployment_policy(task)
    calibration_config = task.get("calibration") or {}
    runtime_capacity_pending = bool(task["workload"].get("runtime_capacity_pending"))
    explicit_curve = isinstance(calibration_config.get("concurrencies"), list)
    adaptive = calibration_config.get("strategy", "adaptive") == "adaptive" and not explicit_curve
    adaptive_slo_search = adaptive and bool(task.get("slo"))
    initial_unbounded_probe = (
        policy["mode"] == "offline_throughput" and bool(task.get("slo"))
    )
    concurrencies = calibration_concurrencies(task)
    if not concurrencies:
        return {
            "policy": policy, "target_concurrency": (
                "runtime_resolved" if runtime_capacity_pending else task["workload"]["max_concurrency"]
            ),
            "points": [], "selected_analysis_concurrency": task["workload"]["max_concurrency"],
            "stopped_before_requested_cap": False, "strategy": "disabled",
            "approx_gpu_hours": 0.0, "completed_trials": 0, "server_sessions": 0,
        }
    fixed_reserve = 1 + 2 + confirmation_trial_reserve(task)
    affordable_steps = max(1, int(task["budget"]["max_trials"]) - fixed_reserve)
    configured_steps = calibration_config.get("max_steps")
    max_steps = min(
        affordable_steps,
        int(configured_steps) if isinstance(configured_steps, int) else (
            max(3, math.ceil(math.log2(max(1, task["workload"]["max_concurrency"]))) + 2)
            if adaptive_slo_search else len(concurrencies)
        ),
    )
    first_concurrency = concurrencies[0]
    spec = calibration_spec(
        task, discovery, first_concurrency,
        float(task["budget"]["max_gpu_hours"]),
        float(task["budget"]["max_wall_time_minutes"]),
        auto_capacity_probe=adaptive_slo_search,
        initial_unbounded_probe=initial_unbounded_probe,
    )
    spec["budget"]["max_trials"] = max_steps
    spec["benchmark"]["flush_cache"] = True
    spec["benchmark"]["calibration_session"] = {
        "strategy": "adaptive_slo" if adaptive_slo_search else "fixed_curve",
        "concurrencies": concurrencies,
        "min_concurrency": int(calibration_config.get("min_concurrency", 1)),
        "max_steps": max_steps,
        "request_waves": (
            int((task.get("measurement") or {}).get("p99_request_waves", 10))
            if has_p99_latency_slo(task)
            else 5
        ),
        "requested_concurrency": (
            "runtime_resolved" if runtime_capacity_pending
            else task["workload"]["max_concurrency"]
        ),
        "initial_unbounded_probe": initial_unbounded_probe,
        "stop_on_slo_failure": bool(
            policy["mode"] == "online_latency"
            and calibration_config.get("stop_on_slo_failure", True)
        ),
    }
    fallback = calibration_config.get("fallback_max_concurrency")
    if isinstance(fallback, int) and not isinstance(fallback, bool) and fallback > 0:
        spec["benchmark"]["calibration_session"]["fallback_max_concurrency"] = fallback
    stage = "capacity (one resident baseline server)"
    report = execute_with_progress(spec, progress, stage) if progress else execute(spec)
    points: list[dict[str, Any]] = []
    for row in report.get("results", []):
        if not row.get("ok"):
            continue
        concurrency = row.get("calibration_concurrency")
        if not isinstance(concurrency, int) or concurrency <= 0:
            continue
        status = row.get("status", {})
        slo_passed = bool(row.get("slo", {}).get("passed"))
        valid = slo_passed or (policy["mode"] == "offline_throughput" and not task["slo"])
        points.append({
            "concurrency": concurrency,
            "requested_concurrency": (
                "runtime_resolved" if runtime_capacity_pending and not points
                else task["workload"]["max_concurrency"] if not points else concurrency
            ),
            "resolved_server_max_running_requests": status.get("resolved_server_max_running_requests"),
            "capacity_source": status.get("resolved_capacity_source"),
            "effective_num_prompts": row.get("effective_num_prompts"),
            "run_dir": report.get("run_dir"),
            "stop_reason": report.get("stop_reason"),
            "metrics": row.get("metrics", {}),
            "slo_passed": slo_passed,
            "valid_for_analysis": valid,
        })
    valid_points = [point for point in points if point["valid_for_analysis"]]
    target = task["workload"]["max_concurrency"]
    selected = max((point["concurrency"] for point in valid_points), default=target)
    return {
        "policy": policy,
        "target_concurrency": "runtime_resolved" if runtime_capacity_pending else target,
        "points": points,
        "selected_analysis_concurrency": selected,
        "stopped_before_requested_cap": (
            bool(points)
            and not adaptive
            and len(points) < len(concurrencies)
        ),
        "strategy": "adaptive_target_first" if adaptive else "full_curve",
        "approx_gpu_hours": float(report.get("approx_gpu_hours", 0)),
        "completed_trials": len(points),
        "server_sessions": int(report.get("completed_server_sessions", 0)),
        "run_dir": report.get("run_dir"),
    }


def required_benchmark_cli_flags(
    task: dict[str, Any], model: dict[str, Any],
) -> set[str]:
    """Return every benchmark flag this task can emit during execution."""
    required = {
        "--backend", "--base-url", "--model", "--dataset-name", "--num-prompts",
        "--ready-check-timeout-sec", "--warmup-requests", "--seed", "--output-file",
        "--disable-tqdm", "--output-details",
    }
    dataset = workload_dataset(task["workload"])
    if dataset.get("name") in {"custom", "sharegpt"}:
        required.add("--dataset-path")
        declared_context = task.get("model", {}).get("context_length")
        if isinstance(declared_context, int) or isinstance(model.get("context_length"), int):
            required.add("--sharegpt-context-len")
        if dataset.get("apply_chat_template", False):
            required.add("--apply-chat-template")
    elif task["workload"].get("shared_prefix") is None:
        required.update({
            "--random-input-len", "--random-output-len", "--random-range-ratio",
            "--tokenize-prompt",
        })
    if task["workload"].get("shared_prefix") is not None:
        required.update({
            "--gsp-num-groups", "--gsp-prompts-per-group",
            "--gsp-system-prompt-len", "--gsp-question-len", "--gsp-output-len",
            "--gsp-range-ratio", "--gsp-ordered",
        })
    if not (
        deployment_policy(task)["mode"] == "offline_throughput" and not task.get("slo")
    ):
        required.add("--max-concurrency")
    if task["workload"].get("request_rate", "inf") != "inf":
        required.add("--request-rate")
    if reference_baseline_mode(task):
        required.add("--flush-cache")
    return required


def discover(
    task: dict[str, Any], progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    if progress:
        progress.emit("discover", "reading GPU inventory", completed=0, total=6)
    hardware = parse_nvidia_inventory() or parse_amd_inventory()
    if hardware is None:
        raise RuntimeError("no supported NVIDIA or AMD accelerator inventory available")
    catalog = load_hardware_catalog()
    if progress:
        progress.emit("discover", "reading checkpoint metadata", completed=1, total=6)
    model = model_inventory(task["model_path"])
    if progress:
        progress.emit("discover", "matching local/current Cookbook evidence", completed=2, total=6)
    cookbook = cookbook_evidence(task, model)
    snapshot = cookbook.get("repository_snapshot", {})
    if not isinstance(snapshot, dict):
        snapshot = {}
    local_checkout = cookbook.get("local_checkout", {})
    cookbook_available = (
        cookbook.get("status") == "fetched"
        or snapshot.get("status") == "available"
        or (isinstance(local_checkout, dict) and local_checkout.get("status") == "available")
    )
    if cookbook.get("required") and not cookbook_available:
        raise RuntimeError(
            "required model cookbook could not be fetched before optimization: "
            + str(cookbook.get("reason") or cookbook.get("status"))
        )
    profile = match_hardware_profile(hardware, catalog)
    tp_size = minimum_tp(task, hardware, model)
    host_memory = host_memory_inventory()
    prefix_analysis = prefix_workload_analysis(
        task, model, hardware, tp_size
    )
    if progress:
        progress.emit("discover", "checking SGLang CLI and benchmark surfaces", completed=3, total=6)
    framework = framework_evidence(task)
    if not framework["launch_server_help_available"]:
        raise RuntimeError(
            "could not obtain the current sglang.launch_server CLI surface: "
            + str(framework.get("launch_server_help_error") or "unknown error")
        )
    if not framework["benchmark_help_available"]:
        raise RuntimeError(
            "could not obtain the current SGLang serving benchmark CLI surface: "
            + str(framework.get("benchmark_help_error") or "unknown error")
        )
    benchmark_flags = set(framework["benchmark_cli_flags"])
    required_benchmark_flags = required_benchmark_cli_flags(task, model)
    missing_benchmark_flags = sorted(required_benchmark_flags - benchmark_flags)
    if missing_benchmark_flags:
        raise RuntimeError(
            "installed SGLang benchmark is missing required flags for this workload: "
            + ", ".join(missing_benchmark_flags)
        )
    if progress:
        progress.emit("discover", "exporting the live ServerArgs contract", completed=4, total=6)
    catalog_progress = (
        lambda message: progress.emit(
            "discover catalog", message, completed=4, total=6
        )
        if progress is not None else None
    )
    parameters = parameter_catalog(task, progress=catalog_progress)
    cli_flags = set(framework["launch_server_cli_flags"])
    compatible_parameters = []
    for item in parameters["parameters"]:
        item = deepcopy(item)
        item["cli_visible"] = bool(set(item["flags"]) & cli_flags)
        compatible_parameters.append(item)
    parameters["parameters"] = compatible_parameters
    parameters["cli_visible_parameter_count"] = sum(
        1 for item in compatible_parameters if item["cli_visible"]
    )
    parameters["parameter_contract"] = {
        "source": "ServerArgs.add_cli_args + current sglang.launch_server --help",
        "server_args_sha256": framework["server_args_sha256"],
        "launch_server_help_sha256": framework["launch_server_help_sha256"],
    }
    discovery = {
        "collected_at": utc_now(),
        "hardware": hardware,
        "hardware_profile": profile,
        "topology_class": topology_class(hardware),
        "hardware_catalog_retrieved_at": catalog.get("retrieved_at"),
        "framework": framework,
        "parameter_catalog": parameters,
        "model": model,
        "host_memory": host_memory,
        "cookbook": cookbook,
        "derived": {
            "visible_gpu_count": visible_gpu_count(task, hardware),
            "visible_gpus": selected_gpus(task, hardware),
            "minimum_tp_size": tp_size,
            "typical_prefill_batch_tokens": expected_prefill_tokens(task["workload"])
            * task["workload"]["max_concurrency"],
            "expected_uncached_prefill_tokens_per_request": expected_prefill_tokens(task["workload"]),
            "chunked_prefill_candidates": chunk_candidates(task),
            "prefix_workload_analysis": prefix_analysis,
        },
    }
    history = task.get("history", {}) if isinstance(task.get("history"), dict) else {}
    history_enabled = history.get("enabled", True)
    database = history_database_path(task)
    current_contract = build_parameter_contract(parameters, framework)
    previous_contract = (
        latest_parameter_contract(
            database, exclude_contract_hash=current_contract.get("contract_hash")
        )
        if history_enabled else None
    )
    known_parameters = known_rule_parameters() | set(COOKBOOK_TUNABLE_FLAGS)
    if progress:
        progress.emit(
            "discover", "diffing parameter contracts and classifying new flags",
            completed=5, total=6,
        )
    evolution = analyze_parameter_evolution(
        task, discovery, previous_contract, known_parameters=known_parameters
    )
    evolution["persistence"] = {
        "enabled": bool(history_enabled),
        "database": str(database),
        "previous_contract_available": previous_contract is not None,
        "policy": (
            "successful runs save the complete contract and exact-fingerprint parameter evidence"
            if history_enabled else
            "disabled; every unseen environment remains a conservative first observation"
        ),
    }
    if history_enabled and evolution.get("provisional_candidates"):
        _, components = history_compatibility_fingerprint(task, discovery)
        priors = parameter_evidence_priors(
            database, components["framework_fingerprint"],
            {
                item["parameter"]
                for item in evolution.get("provisional_candidates", [])
            },
            compatibility_components=components,
        )
        minimum_gain = float(task.get("objective", {}).get("min_improvement_pct", 1.0))
        for item in evolution.get("parameters", []):
            if item.get("parameter") in priors:
                item["history_prior"] = deepcopy(priors[item["parameter"]])
        for item in evolution.get("provisional_candidates", []):
            if item.get("parameter") in priors:
                item["history_prior"] = deepcopy(priors[item["parameter"]])
                by_value = {
                    json.dumps(sample.get("value"), sort_keys=True): sample
                    for sample in priors[item["parameter"]]
                }
                item["candidate_values"] = sorted(
                    item.get("candidate_values", []),
                    key=lambda value: (
                        by_value.get(json.dumps(value, sort_keys=True), {}).get(
                            "median_improvement_pct"
                        ) is not None,
                        by_value.get(json.dumps(value, sort_keys=True), {}).get(
                            "median_improvement_pct"
                        ) or float("-inf"),
                    ),
                    reverse=True,
                )
                supported = [
                    sample for sample in priors[item["parameter"]]
                    if sample.get("samples", 0) >= 2
                    and sample.get("successful_samples") == sample.get("samples")
                    and sample.get("slo_pass_samples") == sample.get("samples")
                    and isinstance(sample.get("median_improvement_pct"), (int, float))
                    and sample["median_improvement_pct"] >= minimum_gain
                ]
                if supported:
                    item["state"] = "experimentally_supported"
                    item["local_promotion"] = {
                        "state": "experimentally_supported",
                        "evidence": deepcopy(supported),
                        "policy": "local exact-fingerprint evidence only; global validated rules still require review",
                    }
        promoted_count = sum(
            1 for item in evolution.get("provisional_candidates", [])
            if item.get("state") == "experimentally_supported"
        )
        if promoted_count:
            evolution.setdefault("state_counts", {})["experimentally_supported"] = promoted_count
            evolution["state_counts"]["provisional"] = max(
                0, int(evolution["state_counts"].get("provisional", 0)) - promoted_count
            )
    evolution["current_contract"] = {
        key: deepcopy(value)
        for key, value in evolution.get("current_contract", {}).items()
        if key != "parameters"
    }
    evolution["parameters"] = [
        {
            key: deepcopy(item.get(key))
            for key in (
                "parameter", "state", "reason", "family", "submechanism",
                "confidence", "value_strategy", "risk", "history_prior",
            )
            if key in item
        }
        for item in evolution.get("parameters", [])
    ]
    discovery["parameter_evolution"] = evolution
    if progress:
        progress.emit(
            "discover", "hardware/model/parameter discovery completed",
            completed=6, total=6,
        )
    return discovery


def parameter_evolution_preflight(
    task: dict[str, Any], hardware: dict[str, Any], model: dict[str, Any],
    framework: dict[str, Any],
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Return a doctor-safe contract diff without downloading Cookbook data."""
    parameters = parameter_catalog(
        task,
        progress=(
            lambda message: progress.emit(
                "doctor catalog", message, completed=3, total=6
            )
            if progress is not None else None
        ),
    )
    cli_flags = set(framework.get("launch_server_cli_flags", []))
    parameters["parameters"] = [
        {
            **deepcopy(item),
            "cli_visible": bool(set(item.get("flags", [])) & cli_flags),
        }
        for item in parameters.get("parameters", [])
    ]
    mini_discovery = {
        "hardware": hardware,
        "hardware_profile": match_hardware_profile(
            hardware, load_hardware_catalog()
        ),
        "topology_class": topology_class(hardware),
        "model": model,
        "framework": framework,
        "parameter_catalog": parameters,
        "cookbook": {},
        "derived": {
            "visible_gpu_count": visible_gpu_count(task, hardware),
        },
    }
    current = build_parameter_contract(parameters, framework)
    history = task.get("history", {}) if isinstance(task.get("history"), dict) else {}
    previous = (
        latest_parameter_contract(
            history_database_path(task),
            exclude_contract_hash=current.get("contract_hash"),
        )
        if history.get("enabled", True) else None
    )
    evolution = analyze_parameter_evolution(
        task, mini_discovery, previous,
        known_parameters=known_rule_parameters() | set(COOKBOOK_TUNABLE_FLAGS),
    )
    evolution["persistence"] = {
        "enabled": history.get("enabled", True),
        "database": str(history_database_path(task)),
        "previous_contract_available": previous is not None,
    }
    return compact_evolution_summary(evolution)


def catalog_index(discovery: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["dest"]: item
        for item in discovery["parameter_catalog"]["parameters"]
        if not item.get("deprecated") and item.get("cli_visible", True)
    }


def runtime_compatibility_constraints(discovery: dict[str, Any]) -> dict[str, Any]:
    """Return non-negotiable launch settings implied by model/runtime support.

    These settings establish a runnable baseline and are never reported as a
    tuning gain. The rules are deliberately narrow and are validated against
    the installed ServerArgs catalog before use.
    """
    model = discovery.get("model", {})
    catalog = catalog_index(discovery)
    required_config: dict[str, Any] = {}
    excluded_values: dict[str, dict[str, str]] = {}
    evidence: list[str] = []
    quant_algo = str(model.get("quantization_algorithm") or "").upper()
    if model.get("is_moe") and quant_algo == "NVFP4":
        metadata = catalog.get("moe_runner_backend", {})
        choices = metadata.get("choices")
        cutlass_supported = not isinstance(choices, list) or "flashinfer_cutlass" in choices
        if cutlass_supported and metadata:
            required_config["moe_runner_backend"] = "flashinfer_cutlass"
            evidence.append(
                "NVFP4 MoE requires SGLang flashinfer_cutlass; other runners raise "
                "NotImplementedError during CUDA Graph warmup"
            )
            if isinstance(choices, list):
                excluded_values["moe_runner_backend"] = {
                    str(value): "unsupported for NVFP4 MoE by the installed SGLang runtime"
                    for value in choices if value != "flashinfer_cutlass"
                }
    cookbook_profile = (
        discovery.get("cookbook", {}).get("model_profile", {})
        if isinstance(discovery.get("cookbook"), dict) else {}
    )
    functional_config = {}
    if isinstance(cookbook_profile, dict):
        # An empty selected recipe means that no performance recipe survived
        # qualification; it must not erase model-wide functional settings
        # such as reasoning/tool parsers.
        functional_config = (
            cookbook_profile.get("selected_required_functional_config")
            or cookbook_profile.get("required_functional_config", {})
        )
    for parameter, value in functional_config.items():
        metadata = catalog.get(parameter)
        if not isinstance(metadata, dict):
            evidence.append(
                f"excluded Cookbook functional requirement {parameter}={value!r}: absent from current ServerArgs"
            )
            continue
        choices = metadata.get("choices")
        if isinstance(choices, list) and value not in choices:
            evidence.append(
                f"excluded Cookbook functional requirement {parameter}={value!r}: incompatible current choices"
            )
            continue
        required_config[parameter] = value
        evidence.append(
            f"required Cookbook functional setting {parameter}={value!r}"
        )
    return {
        "required_config": required_config,
        "excluded_values": excluded_values,
        "evidence": evidence,
        "policy": "compatibility requirements define the runnable baseline and are not optimization wins",
    }


def cookbook_automatic_behavior_report(
    discovery: dict[str, Any], profile: dict[str, Any]
) -> list[dict[str, Any]]:
    """Compare documented auto behavior with the resolved runtime when observable."""
    cookbook_profile = (
        discovery.get("cookbook", {}).get("model_profile", {})
        if isinstance(discovery.get("cookbook"), dict) else {}
    )
    behaviors = (
        cookbook_profile.get("automatic_behaviors", [])
        if isinstance(cookbook_profile, dict) else []
    )
    effective = (
        profile.get("effective_server_config", {})
        if isinstance(profile, dict) else {}
    )
    vendor = str(discovery.get("hardware", {}).get("vendor", "")).lower()
    precision = _cookbook_checkpoint_precision(discovery.get("model", {}))
    reports: list[dict[str, Any]] = []
    for behavior in behaviors:
        if not isinstance(behavior, dict):
            continue
        parameter = behavior.get("parameter")
        condition = behavior.get("enabled_when", {})
        expected = bool(
            condition.get("accelerator_runtime") == "cuda"
            and vendor == "nvidia"
            and condition.get("checkpoint_precision") == precision
        )
        observed = effective.get(parameter) if isinstance(parameter, str) else None
        status = (
            "verified" if isinstance(observed, bool) and observed == expected else
            "mismatch" if isinstance(observed, bool) else
            "unobservable"
        )
        reports.append({
            **deepcopy(behavior),
            "expected_enabled": expected,
            "observed_resolved_value": observed,
            "status": status,
            "detected_vendor": vendor or None,
            "detected_checkpoint_precision": precision,
        })
    return reports


def parameter_value_runtime_compatible(
    discovery: dict[str, Any], parameter: str, value: Any,
) -> tuple[bool, str | None]:
    exclusions = runtime_compatibility_constraints(discovery)["excluded_values"].get(
        parameter, {}
    )
    reason = exclusions.get(str(value))
    return reason is None, reason


def execution_parameter_bindings(discovery: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Freeze the discovered ServerArgs bindings into reproducible trial specs."""
    return {
        item["dest"]: {
            "primary_flag": item["primary_flag"],
            "action": item.get("action"),
            "value_type": item.get("value_type"),
            "choices": item.get("choices"),
        }
        for item in discovery["parameter_catalog"]["parameters"]
        if item.get("cli_visible", True) and not item.get("deprecated")
    }


def add_ranked_candidate(
    ranked: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    parameter: str,
    values: list[Any],
    reason: str,
    evidence: list[str],
    tier: str = "evidence",
) -> None:
    metadata = catalog.get(parameter)
    if metadata is None:
        return
    choices = metadata.get("choices")
    filtered = []
    for value in values:
        if choices is not None and value not in choices:
            continue
        # Do not discard the catalog default here. SGLang may resolve a
        # model-specific effective value that differs from the static parser
        # default; returning to the static default is then a real candidate.
        # The effective-baseline filter runs after the server has exposed its
        # resolved configuration.
        if value in filtered:
            continue
        filtered.append(value)
    if filtered:
        existing = next((item for item in ranked if item["parameter"] == parameter), None)
        if existing is not None:
            existing["values"] = list(dict.fromkeys([*existing["values"], *filtered]))
            existing["evidence"] = list(dict.fromkeys([*existing["evidence"], *evidence]))
            existing["tiers"] = list(dict.fromkeys([
                *existing.get("tiers", ["evidence"]), tier,
            ]))
            return
        ranked.append({
            "parameter": parameter,
            "values": filtered,
            "family": metadata["family"],
            "flag": metadata["primary_flag"],
            "installed_default": metadata.get("default"),
            "reason": reason,
            "evidence": evidence,
            "semantics": metadata.get("help"),
            "tiers": [tier],
        })


def add_cookbook_hardware_prior_candidates(
    ranked: list[dict[str, Any]], catalog: dict[str, dict[str, Any]],
    discovery: dict[str, Any],
) -> list[dict[str, Any]]:
    """Promote evidence-only Cookbook scalars to bounded one-factor trials."""
    profile = (
        discovery.get("cookbook", {}).get("model_profile", {})
        if isinstance(discovery.get("cookbook"), dict) else {}
    )
    evidence_items = (
        profile.get("hardware_adaptation_evidence", [])
        if isinstance(profile, dict) else []
    )
    admitted: list[dict[str, Any]] = []
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        for parameter, value in item.get("parameter_values", {}).items():
            before = next(
                (len(group.get("values", [])) for group in ranked if group.get("parameter") == parameter),
                0,
            )
            add_ranked_candidate(
                ranked, catalog, parameter, [value],
                "test a portable value documented on non-matching Cookbook hardware",
                [
                    f"cookbook.source={item.get('name')}",
                    f"cookbook.qualification={item.get('qualification')}",
                    f"cookbook.documented_hardware={item.get('documented_hardware')}",
                    f"cookbook.detected_hardware={item.get('detected_hardware')}",
                    "cookbook.policy=evidence_only_one_factor_measurement_required",
                ],
                tier="cookbook_hardware_prior",
            )
            group = next(
                (group for group in ranked if group.get("parameter") == parameter), None
            )
            after = len(group.get("values", [])) if isinstance(group, dict) else 0
            if after > before:
                admitted.append({
                    "parameter": parameter, "value": deepcopy(value),
                    "source": item.get("name"),
                    "qualification": item.get("qualification"),
                })
    return admitted


def hardware_backends(discovery: dict[str, Any]) -> dict[str, list[str]]:
    profile = discovery.get("hardware_profile") or {}
    architecture = profile.get("architecture")
    vendor = discovery["hardware"].get("vendor")
    if vendor == "amd":
        return {"attention": ["aiter", "triton"], "moe": ["aiter", "triton"]}
    if architecture == "blackwell":
        return {
            "attention": ["trtllm_mha", "fa4", "flashinfer", "triton"],
            # Keep these names aligned with the installed ServerArgs choices.
            # Older aliases silently fell through catalog filtering and made
            # flashinfer_cutlass impossible to test on Blackwell.
            "moe": ["flashinfer_cutlass", "flashinfer_trtllm", "deep_gemm", "triton"],
        }
    if architecture == "hopper":
        return {
            "attention": ["fa3", "flashinfer", "triton"],
            "moe": ["deep_gemm", "flashinfer_trtllm", "triton"],
        }
    return {"attention": ["flashinfer", "triton"], "moe": ["triton"]}


def prometheus_value(lines: list[str], metric: str) -> float | None:
    pattern = re.compile(rf"^{re.escape(metric)}(?:\{{.*\}})?\s+([-+0-9.eE]+)$")
    values = []
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            try:
                values.append(float(match.group(1)))
            except ValueError:
                pass
    return values[-1] if values else None


def experiment_fingerprint(task: dict[str, Any], discovery: dict[str, Any]) -> str:
    """Identify failure evidence that is safe to reuse across local stages."""
    payload = {
        "repository": str(Path(task["repository"]).resolve()),
        "framework_commit": discovery.get("framework", {}).get("git_commit"),
        "model_path": str(Path(task["model_path"]).resolve()),
        "model_weight_bytes": discovery.get("model", {}).get("weight_bytes"),
        "hardware": [
            {
                "name": gpu.get("name"),
                "memory_mib": gpu.get("memory_mib"),
                "compute_capability": gpu.get("compute_capability"),
            }
            for gpu in selected_gpus(task, discovery.get("hardware", {}))
        ],
        "workload": task.get("workload", {}),
        "visibility": {
            key: task.get("env", {}).get(key, os.environ.get(key))
            for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
            if key in task.get("env", {}) or key in os.environ
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def history_database_path(task: dict[str, Any]) -> Path:
    history = task.get("history", {}) if isinstance(task.get("history"), dict) else {}
    configured = history.get("database")
    return (
        Path(configured).expanduser()
        if isinstance(configured, str)
        else Path(task["output_dir"]).expanduser() / "inferopt-history.sqlite3"
    )


def history_search_evidence(
    task: dict[str, Any], discovery: dict[str, Any]
) -> dict[str, Any]:
    history = task.get("history", {}) if isinstance(task.get("history"), dict) else {}
    enabled = history.get("enabled", True)
    database = history_database_path(task)
    fingerprint, components = history_compatibility_fingerprint(task, discovery)
    candidates = (
        warm_start_candidates(
            database, fingerprint, limit=int(history.get("warm_start_limit", 5))
        )
        if enabled else []
    )
    quality = task.get("quality", {}) if isinstance(task.get("quality"), dict) else {}
    if not quality.get("allow_kv_cache_precision_tuning", False):
        candidates = [
            item for item in candidates
            if str(item.get("config", {}).get("kv_cache_dtype", "auto")).lower()
            not in {"fp8_e4m3", "fp8_e5m2", "mxfp8", "nvfp4", "fp4_mx_block16"}
        ]
    return {
        "enabled": enabled,
        "database": str(database),
        "compatibility_fingerprint": fingerprint,
        "components": components,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "policy": "form weak priors only from exact-compatible model/framework/hardware/workload fingerprints; never create history candidate trials",
    }


def apply_history_priors_to_search_plan(
    search_plan: dict[str, Any], history_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Attach compatible history as priors and remove legacy history trials."""
    prior_evidence = history_priors(history_evidence.get("candidates", []))
    search_plan["history"] = {
        **history_evidence,
        "priors": prior_evidence,
        "warm_start_bundle_names": [],
        "prior_only": True,
    }
    search_plan["ranked_configuration_bundles"] = [
        bundle for bundle in search_plan.get("ranked_configuration_bundles", [])
        if not str(bundle.get("name", "")).startswith("history-")
    ]
    return search_plan


def known_failed_candidates(
    task: dict[str, Any], discovery: dict[str, Any]
) -> set[tuple[str, str]]:
    """Reuse only definitive failures from the exact same local experiment context."""
    failed: set[tuple[str, str]] = set()
    output_dir = task.get("output_dir")
    if not isinstance(output_dir, str):
        return failed
    root = Path(output_dir) / "stages"
    if not root.is_dir():
        return failed
    fingerprint = experiment_fingerprint(task, discovery)
    definitive_failures = {
        "backend_incompatible", "configuration", "dependency_missing",
        "memory_infeasible", "oom",
    }
    for result_path in root.glob("*/results.json"):
        spec_path = result_path.parent / "spec.json"
        if not spec_path.is_file():
            continue
        try:
            prior_spec = load_json(spec_path)
            if prior_spec.get("experiment_fingerprint") != fingerprint:
                continue
            baseline = prior_spec.get("search", {}).get("baseline", {})
            rows = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(baseline, dict) or not isinstance(rows, list):
            continue
        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("ok")
                or row.get("kind") != "candidate"
                or row.get("status", {}).get("failure_class") not in definitive_failures
            ):
                continue
            config = row.get("config", {})
            if not isinstance(config, dict):
                continue
            changed = [(key, value) for key, value in config.items() if baseline.get(key) != value]
            if len(changed) == 1:
                key, value = changed[0]
                failed.add((key, json.dumps(value, sort_keys=True)))
    return failed


def known_failed_configuration_deltas(
    task: dict[str, Any], discovery: dict[str, Any]
) -> set[str]:
    """Return definitive failed multi-parameter deltas for this exact context."""
    failed: set[str] = set()
    output_dir = task.get("output_dir")
    if not isinstance(output_dir, str):
        return failed
    root = Path(output_dir) / "stages"
    if not root.is_dir():
        return failed
    fingerprint = experiment_fingerprint(task, discovery)
    definitive = {
        "backend_incompatible", "configuration", "dependency_missing",
        "memory_infeasible", "oom",
    }
    for result_path in root.glob("*/results.json"):
        spec_path = result_path.parent / "spec.json"
        if not spec_path.is_file():
            continue
        try:
            prior_spec = load_json(spec_path)
            if prior_spec.get("experiment_fingerprint") != fingerprint:
                continue
            baseline = prior_spec.get("search", {}).get("baseline", {})
            rows = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(baseline, dict) or not isinstance(rows, list):
            continue
        for row in rows:
            if (
                not isinstance(row, dict) or row.get("ok")
                or row.get("kind") != "candidate"
                or row.get("status", {}).get("failure_class") not in definitive
            ):
                continue
            config = row.get("config", {})
            if not isinstance(config, dict):
                continue
            delta = {
                key: value for key, value in config.items()
                if baseline.get(key) != value
            }
            if len(delta) > 1:
                failed.add(json.dumps(delta, sort_keys=True, separators=(",", ":")))
    return failed


def parameter_audit(
    catalog: dict[str, dict[str, Any]],
    ranked: list[dict[str, Any]],
    discovery: dict[str, Any],
    task: dict[str, Any],
    configuration_bundles: list[dict[str, Any]] | None = None,
    semantic_decisions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Account for every current-version CLI parameter in a search plan.

    A serving binary exposes many control-plane, topology, and diagnostic
    options. Treating all of them as independent performance dials would make
    an unbounded and invalid experiment. This audit makes every exclusion
    explicit instead of silently hiding it behind a small candidate list.
    """
    selected = {item["parameter"] for item in ranked}
    for bundle in configuration_bundles or []:
        if isinstance(bundle, dict) and isinstance(bundle.get("config"), dict):
            selected.update(bundle["config"])
    cookbook = discovery.get("cookbook", {})
    profile = cookbook.get("model_profile", {}) if isinstance(cookbook, dict) else {}
    for bundle in profile.get("initial_bundles", []) if isinstance(profile, dict) else []:
        if isinstance(bundle, dict) and isinstance(bundle.get("config"), dict):
            selected.update(bundle["config"])
    gpu_count = discovery["derived"]["visible_gpu_count"]
    is_moe = bool(discovery["model"].get("is_moe"))
    has_draft_model = (
        bool(task.get("speculative"))
        or bool(task.get("draft_model_path"))
        or (
            bool(discovery["model"].get("has_mtp_weights"))
            and task.get("capability_overrides", {}).get("mtp") != "disabled"
        )
    )
    static_identity = {
        "model_path", "served_model_name", "host", "port", "api_key",
        "tokenizer_path", "chat_template", "revision", "trust_remote_code",
    }
    evolution_by_parameter = {
        item.get("parameter"): item
        for item in discovery.get("parameter_evolution", {}).get("parameters", [])
        if isinstance(item, dict) and item.get("parameter")
    }
    semantic_by_parameter = {
        item.get("parameter"): item
        for item in semantic_decisions or []
        if isinstance(item, dict) and item.get("parameter")
    }
    entries: list[dict[str, Any]] = []
    summary: dict[str, int] = {}
    for parameter, metadata in sorted(catalog.items()):
        state = "excluded"
        reason = "not selected by the measured bottleneck and workload routing"
        if metadata["family"] == "speculative" and task.get("capability_overrides", {}).get("mtp") == "disabled":
            state = "inapplicable"
            reason = "MTP/speculative decoding is explicitly disabled for this SGLang-model compatibility combination"
        elif parameter in selected:
            state = "selected"
            reason = "scheduled for an isolated, SLO-gated measurement"
        elif metadata["family"] == "observability":
            reason = "diagnostic or logging control; changing it would perturb the measurement"
        elif parameter in static_identity:
            reason = "model, endpoint, or compatibility identity; not a serving-path tuning dial"
        elif metadata["family"] == "speculative" and not has_draft_model:
            state = "inapplicable"
            reason = "speculative decoding requires an explicitly supplied compatible draft model and acceptance workload"
        elif metadata["family"] == "parallelism" and gpu_count == 1:
            state = "inapplicable"
            reason = "multi-GPU parallelism setting is not meaningful on the discovered single-GPU topology"
        elif metadata["family"] == "communication" and gpu_count == 1:
            state = "inapplicable"
            reason = "collective-communication setting is not meaningful on the discovered single-GPU topology"
        elif metadata["family"] == "moe" and not is_moe:
            state = "inapplicable"
            reason = "MoE-specific setting is not applicable to the detected dense model"
        elif metadata.get("required") or metadata.get("action") == "append":
            reason = "structured deployment/control-plane input; it is not safely enumerable as a scalar startup candidate"
        evolution = evolution_by_parameter.get(parameter, {})
        semantic_decision = semantic_by_parameter.get(parameter, {})
        if parameter not in selected and semantic_decision:
            semantic_reasons = semantic_decision.get("reasons", [])
            if semantic_reasons:
                reason = "semantic context gate: " + "; ".join(
                    str(value) for value in semantic_reasons
                )
        if parameter not in selected and evolution.get("state") in {
            "unclassified", "control_plane", "quality_sensitive", "unsafe"
        }:
            reason = "parameter evolution: " + str(evolution.get("reason"))
        entries.append({
            "parameter": parameter,
            "flag": metadata["primary_flag"],
            "family": metadata["family"],
            "state": state,
            "reason": reason,
            "evolution_state": evolution.get("state"),
            "evolution_confidence": evolution.get("confidence"),
            "submechanism": evolution.get("submechanism"),
        })
        summary[state] = summary.get(state, 0) + 1
    return {
        "policy": "all CLI-visible, non-deprecated SGLang ServerArgs are accounted for; only compatible serving-path dials enter isolated SLO-gated experiments",
        "summary": summary,
        "parameters": entries,
    }


def _cookbook_checkpoint_precision(model: dict[str, Any]) -> str | None:
    identity = " ".join([
        str(model.get("checkpoint_name", "")),
        str(model.get("weight_quantization", "")),
        str(model.get("quantization", "")),
        str(model.get("checkpoint_dtype", "")),
        str(model.get("dtype", "")),
    ]).lower()
    if "nvfp4" in identity or "modelopt_fp4" in identity:
        return "nvfp4"
    if "fp8" in identity:
        return "fp8"
    if "bfloat16" in identity or "bf16" in identity:
        return "bf16"
    return None


def _cookbook_strategy_rank(mode: str, strategy: str | None) -> int:
    normalized = str(strategy or "balanced").lower()
    order = (
        {"high-throughput": 0, "balanced": 1, "low-latency": 2}
        if mode == "offline_throughput" else
        {"low-latency": 0, "balanced": 1, "high-throughput": 2}
    )
    return order.get(normalized, 3)


COOKBOOK_HARDWARE_ARCHITECTURE = {
    "a100": ("nvidia", "ampere"),
    "h100": ("nvidia", "hopper"),
    "h200": ("nvidia", "hopper"),
    "h800": ("nvidia", "hopper"),
    "b200": ("nvidia", "blackwell"),
    "b300": ("nvidia", "blackwell"),
    "gb300": ("nvidia", "blackwell"),
    "mi300x": ("amd", "cdna3"),
    "mi325x": ("amd", "cdna3"),
    "mi350x": ("amd", "cdna4"),
    "mi355x": ("amd", "cdna4"),
}

COOKBOOK_PORTABLE_PARAMETERS = {
    "mamba_ssm_dtype", "mamba_radix_cache_strategy",
    "speculative_algorithm", "speculative_num_steps",
    "speculative_eagle_topk", "speculative_num_draft_tokens",
    "chunked_prefill_size", "max_prefill_tokens",
}
COOKBOOK_SAME_VENDOR_PRIOR_PARAMETERS = COOKBOOK_PORTABLE_PARAMETERS | {
    "mem_fraction_static", "page_size", "max_total_tokens",
    "schedule_policy", "schedule_conservativeness",
    "num_continuous_decode_steps", "enable_mixed_chunk",
}
COOKBOOK_HARDWARE_BACKEND_PARAMETERS = {
    "attention_backend", "prefill_attention_backend", "decode_attention_backend",
    "linear_attn_prefill_backend", "linear_attn_decode_backend",
    "moe_runner_backend", "moe_a2a_backend", "bf16_gemm_backend",
    "fp8_gemm_runner_backend", "fp4_gemm_runner_backend",
}


def _detected_hardware_identity(discovery: dict[str, Any]) -> dict[str, Any]:
    names = " ".join(
        str(gpu.get("name", "")).lower()
        for gpu in discovery.get("hardware", {}).get("gpus", [])
    )
    tokens = set(re.findall(
        r"\b(?:a100|h100|h200|h800|b200|b300|gb300|mi300x|mi325x|mi350x|mi355x)\b",
        names,
    ))
    vendor = str(discovery.get("hardware", {}).get("vendor", "")).lower()
    architectures = {
        COOKBOOK_HARDWARE_ARCHITECTURE[token][1]
        for token in tokens if token in COOKBOOK_HARDWARE_ARCHITECTURE
    }
    profile_architecture = str(
        (discovery.get("hardware_profile") or {}).get("architecture", "")
    ).lower()
    if profile_architecture:
        architectures.add(profile_architecture)
    return {
        "tokens": tokens,
        "vendor": vendor,
        "architectures": architectures,
        "names": names,
    }


def _cookbook_hardware_qualification(
    affinity: set[str], discovery: dict[str, Any]
) -> dict[str, Any]:
    detected = _detected_hardware_identity(discovery)
    if not affinity:
        return {**detected, "level": "generic", "documented_architectures": set()}
    if affinity.intersection(detected["tokens"]):
        return {**detected, "level": "exact_verified", "documented_architectures": {
            COOKBOOK_HARDWARE_ARCHITECTURE[value][1]
            for value in affinity if value in COOKBOOK_HARDWARE_ARCHITECTURE
        }}
    documented_pairs = {
        COOKBOOK_HARDWARE_ARCHITECTURE[value]
        for value in affinity if value in COOKBOOK_HARDWARE_ARCHITECTURE
    }
    documented_vendors = {value[0] for value in documented_pairs}
    documented_architectures = {value[1] for value in documented_pairs}
    if documented_architectures.intersection(detected["architectures"]):
        level = "hardware_adapted"
    elif detected["vendor"] and detected["vendor"] in documented_vendors:
        level = "evidence_only_same_vendor"
    else:
        level = "evidence_only_cross_vendor"
    return {
        **detected, "level": level,
        "documented_architectures": documented_architectures,
        "documented_vendors": documented_vendors,
    }


def _nearest_legal_degree(documented: int, legal: list[int]) -> int | None:
    if not legal:
        return None
    return min(legal, key=lambda value: (abs(value - documented), -value))


def _adapt_cookbook_config(
    config: dict[str, Any], discovery: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Adapt topology/backend details while preserving auditable source values."""
    adapted = deepcopy(config)
    changes: list[dict[str, Any]] = []
    documented_tp = int(adapted.get("tp_size", 1) or 1)
    legal_tp = supported_tp_sizes(discovery)
    if documented_tp not in legal_tp:
        replacement = _nearest_legal_degree(documented_tp, legal_tp)
        if replacement is None:
            return None, [{"parameter": "tp_size", "reason": "no legal local TP degree"}]
        adapted["tp_size"] = replacement
        changes.append({
            "parameter": "tp_size", "documented": documented_tp,
            "adapted": replacement, "reason": "nearest legal local TP degree",
        })
    local_tp = int(adapted.get("tp_size", 1) or 1)
    if "ep_size" in adapted:
        documented_ep = int(adapted.get("ep_size", 1) or 1)
        legal_ep = supported_ep_sizes(discovery, local_tp)
        if not legal_ep:
            experts = discovery.get("model", {}).get("num_experts")
            legal_ep = [
                value for value in range(1, local_tp + 1)
                if local_tp % value == 0
                and (not isinstance(experts, int) or experts <= 0 or experts % value == 0)
            ]
        if documented_ep not in legal_ep:
            replacement = _nearest_legal_degree(documented_ep, legal_ep or [1])
            adapted["ep_size"] = int(replacement or 1)
            changes.append({
                "parameter": "ep_size", "documented": documented_ep,
                "adapted": int(replacement or 1),
                "reason": "nearest mathematically legal local EP degree",
            })
    visible = int(discovery.get("derived", {}).get("visible_gpu_count", 1) or 1)
    for parameter in ("pp_size", "dp_size"):
        if parameter not in adapted:
            continue
        documented = int(adapted.get(parameter, 1) or 1)
        other = "dp_size" if parameter == "pp_size" else "pp_size"
        product = local_tp * documented * int(adapted.get(other, 1) or 1)
        if product > visible:
            adapted[parameter] = 1
            changes.append({
                "parameter": parameter, "documented": documented, "adapted": 1,
                "reason": "documented topology exceeds the selected GPU pool",
            })
    for parameter in sorted(COOKBOOK_HARDWARE_BACKEND_PARAMETERS & set(adapted)):
        metadata = catalog.get(parameter)
        value = adapted[parameter]
        choices = metadata.get("choices") if isinstance(metadata, dict) else None
        if not isinstance(metadata, dict) or (
            isinstance(choices, list) and value not in choices
        ):
            adapted.pop(parameter, None)
            changes.append({
                "parameter": parameter, "documented": value, "adapted": None,
                "reason": "backend is absent from or incompatible with the local ServerArgs contract",
            })
    return adapted, changes


def _cookbook_evidence_only_values(
    config: dict[str, Any], *, same_vendor: bool,
    catalog: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    allowed = (
        COOKBOOK_SAME_VENDOR_PRIOR_PARAMETERS
        if same_vendor else COOKBOOK_PORTABLE_PARAMETERS
    )
    retained: dict[str, Any] = {}
    removed: list[str] = []
    for parameter, value in config.items():
        metadata = catalog.get(parameter)
        choices = metadata.get("choices") if isinstance(metadata, dict) else None
        if (
            parameter in allowed
            and isinstance(metadata, dict)
            and (not isinstance(choices, list) or value in choices)
        ):
            retained[parameter] = deepcopy(value)
        else:
            removed.append(parameter)
    return retained, sorted(removed)


def cookbook_candidate_bundles(
    discovery: dict[str, Any], catalog: dict[str, dict[str, Any]],
    task: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate cookbook configuration bundles against the local CLI and checkpoint."""
    cookbook = discovery.get("cookbook", {})
    profile = cookbook.get("model_profile") if isinstance(cookbook, dict) else None
    if not isinstance(profile, dict):
        return [], []
    excluded: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    hardware_names = " ".join(
        str(gpu.get("name", "")).lower()
        for gpu in discovery.get("hardware", {}).get("gpus", [])
    )
    detected_affinity = set(re.findall(
        r"\b(?:a100|h100|h200|h800|b200|b300|gb300|mi300x|mi325x|mi350x|mi355x)\b",
        hardware_names,
    ))
    mode = (
        deployment_policy(task)["mode"] if isinstance(task, dict)
        else str(profile.get("task_deployment_mode", "online_latency"))
    )
    local_precision = _cookbook_checkpoint_precision(discovery.get("model", {}))
    visible_gpus = int(discovery.get("derived", {}).get("visible_gpu_count", 1) or 1)
    strategy_floor: dict[tuple[tuple[str, ...], str], int] = {}
    for source_bundle in profile.get("initial_bundles", []):
        source_cell = source_bundle.get("cookbook_cell", {})
        if not isinstance(source_cell, dict) or not source_cell:
            continue
        source_precision = str(source_cell.get("quant", "")).lower()
        if source_precision and local_precision and source_precision != local_precision:
            continue
        source_affinity = tuple(sorted(
            str(value).lower()
            for value in source_bundle.get("hardware_affinity", []) if value
        ))
        key = (source_affinity, source_precision)
        rank = _cookbook_strategy_rank(mode, source_cell.get("strategy"))
        strategy_floor[key] = min(rank, strategy_floor.get(key, rank))
    adaptation_evidence: list[dict[str, Any]] = []
    for source_bundle in profile.get("initial_bundles", []):
        bundle = deepcopy(source_bundle)
        config = bundle.get("config", {})
        cell = bundle.get("cookbook_cell", {})
        requirements = set(bundle.get("requirements", []))
        affinity = {
            str(value).lower() for value in bundle.get("hardware_affinity", [])
            if value
        }
        documented_precision = str(cell.get("quant", "")).lower()
        if documented_precision and local_precision and documented_precision != local_precision:
            excluded.append({
                "name": bundle.get("name"),
                "reason": "Cookbook deployment cell precision does not match the local checkpoint",
                "documented_precision": documented_precision,
                "detected_precision": local_precision,
            })
            continue
        qualification = _cookbook_hardware_qualification(affinity, discovery)
        qualification_level = str(qualification["level"])
        adaptation_changes: list[dict[str, Any]] = []
        if qualification_level == "hardware_adapted":
            adapted_config, adaptation_changes = _adapt_cookbook_config(
                config, discovery, catalog
            )
            if adapted_config is None:
                excluded.append({
                    "name": bundle.get("name"),
                    "reason": "Cookbook recipe could not be adapted to a legal local topology",
                    "required_hardware": sorted(affinity),
                    "detected_hardware": sorted(detected_affinity),
                    "adaptation_changes": adaptation_changes,
                })
                continue
            config = adapted_config
            bundle["config"] = config
            bundle["name"] = f"{bundle.get('name', 'cookbook')}-adapted"[:120]
        elif qualification_level.startswith("evidence_only"):
            source_strategy_rank = _cookbook_strategy_rank(
                mode, cell.get("strategy") if isinstance(cell, dict) else None
            )
            floor = strategy_floor.get(
                (tuple(sorted(affinity)), documented_precision),
                source_strategy_rank,
            )
            if source_strategy_rank > floor:
                excluded.append({
                    "name": bundle.get("name"),
                    "reason": "Cookbook evidence-only strategy is not preferred for the requested mode",
                    "qualification": qualification_level,
                    "deployment_mode": mode,
                    "documented_strategy": cell.get("strategy"),
                })
                continue
            portable_values, removed_parameters = _cookbook_evidence_only_values(
                config,
                same_vendor=qualification_level == "evidence_only_same_vendor",
                catalog=catalog,
            )
            evidence_record = {
                "name": bundle.get("name"),
                "qualification": qualification_level,
                "documented_hardware": sorted(affinity),
                "required_hardware": sorted(affinity),
                "detected_hardware": sorted(detected_affinity),
                "documented_architectures": sorted(
                    qualification.get("documented_architectures", [])
                ),
                "detected_architectures": sorted(
                    qualification.get("architectures", [])
                ),
                "parameter_values": portable_values,
                "removed_parameters": removed_parameters,
                "required_functional_config": deepcopy(
                    bundle.get("required_functional_config", {})
                ),
                "source": deepcopy(bundle.get("source", {})),
                "policy": (
                    "evidence-only values may seed one-factor local trials; the documented "
                    "bundle and hardware-specific backends/topology are never executed"
                ),
            }
            adaptation_evidence.append(evidence_record)
            excluded.append({
                "name": bundle.get("name"),
                "reason": "Cookbook recipe retained as evidence-only for non-matching hardware",
                **evidence_record,
            })
            continue
        needs_mtp = (
            "checkpoint.has_mtp_weights" in requirements
            or (profile.get("requires_mtp_weights") and "speculative_algorithm" in config)
        )
        if needs_mtp and not discovery["model"].get("has_mtp_weights"):
            excluded.append({
                "name": bundle.get("name"),
                "reason": "cookbook recipe requires checkpoint MTP weights, but none were found locally",
            })
            continue
        if "checkpoint.is_hybrid" in requirements and not discovery["model"].get("is_hybrid"):
            excluded.append({
                "name": bundle.get("name"),
                "reason": "cookbook recipe requires a hybrid/Mamba checkpoint, but local model metadata is not hybrid",
            })
            continue
        if "nvidia_gpu" in requirements and discovery.get("hardware", {}).get("vendor") != "nvidia":
            excluded.append({
                "name": bundle.get("name"),
                "reason": "cookbook recipe requires NVIDIA GPU support",
            })
            continue
        if "amd_gpu" in requirements and discovery.get("hardware", {}).get("vendor") != "amd":
            excluded.append({
                "name": bundle.get("name"),
                "reason": "cookbook recipe requires AMD GPU support",
            })
            continue
        if "tp_size>=2" in requirements and int(config.get("tp_size", 1)) < 2:
            excluded.append({
                "name": bundle.get("name"),
                "reason": "cookbook recipe enables all-reduce fusion but does not supply a legal TP topology",
            })
            continue
        topology_gpus = (
            int(config.get("tp_size", 1) or 1)
            * int(config.get("pp_size", 1) or 1)
            * int(config.get("dp_size", 1) or 1)
        )
        ep_size = int(config.get("ep_size", 1) or 1)
        if topology_gpus > visible_gpus or ep_size > max(
            int(config.get("tp_size", 1) or 1), topology_gpus
        ):
            excluded.append({
                "name": bundle.get("name"),
                "reason": "Cookbook deployment cell topology does not fit the selected GPU pool",
                "required_gpus": topology_gpus,
                "visible_gpus": visible_gpus,
                "tp_size": int(config.get("tp_size", 1) or 1),
                "ep_size": ep_size,
            })
            continue
        missing = [name for name in config if name not in catalog]
        invalid = [
            name for name, value in config.items()
            if name in catalog
            and catalog[name].get("choices") is not None
            and value not in catalog[name]["choices"]
        ]
        if missing or invalid:
            excluded.append({
                "name": bundle.get("name"),
                "reason": "bundle is absent from or incompatible with the installed SGLang CLI",
                "missing_parameters": missing,
                "invalid_parameters": invalid,
            })
            continue
        candidate = deepcopy(bundle)
        strategy = str(cell.get("strategy", "balanced")) if isinstance(cell, dict) else "balanced"
        base_priority = (
            "high" if _cookbook_strategy_rank(mode, strategy) == 0 else
            "medium" if _cookbook_strategy_rank(mode, strategy) == 1 else "low"
        )
        candidate["priority"] = (
            "medium" if qualification_level == "hardware_adapted" and base_priority == "high"
            else "low" if qualification_level == "hardware_adapted"
            else base_priority
        )
        candidate["selection_evidence"] = {
            "deployment_mode": mode,
            "documented_strategy": strategy,
            "strategy_rank": _cookbook_strategy_rank(mode, strategy),
            "documented_precision": documented_precision or None,
            "detected_precision": local_precision,
            "hardware_affinity": sorted(affinity),
            "verified_cell": bool(bundle.get("verified")),
            "hardware_qualification": qualification_level,
            "documented_hardware": sorted(affinity),
            "detected_hardware": sorted(detected_affinity),
            "documented_architectures": sorted(
                qualification.get("documented_architectures", [])
            ),
            "detected_architectures": sorted(
                qualification.get("architectures", [])
            ),
            "adaptation_changes": adaptation_changes,
        }
        candidate["risk"] = {
            "level": (
                "hardware_unverified"
                if qualification_level == "hardware_adapted" else "safe"
            ),
            "quality_sensitive": False,
            "requires_local_measurement": qualification_level == "hardware_adapted",
        }
        bundles.append(candidate)
    verified_cells = [
        item for item in bundles
        if item.get("selection_evidence", {}).get("verified_cell")
    ]
    if verified_cells:
        best_strategy_rank = min(
            int(item["selection_evidence"]["strategy_rank"])
            for item in verified_cells
        )
        retained: list[dict[str, Any]] = []
        for item in bundles:
            evidence = item.get("selection_evidence", {})
            if (
                evidence.get("verified_cell")
                and int(evidence.get("strategy_rank", 3)) > best_strategy_rank
            ):
                excluded.append({
                    "name": item.get("name"),
                    "reason": "Cookbook deployment strategy is not preferred for the requested mode",
                    "deployment_mode": mode,
                    "documented_strategy": evidence.get("documented_strategy"),
                })
                continue
            retained.append(item)
        bundles = retained
    bundles.sort(key=lambda item: (
        int(item.get("selection_evidence", {}).get("strategy_rank", 3)),
        str(item.get("name", "")),
    ))
    selected_functional: dict[str, dict[str, Any]] = {}
    for item in bundles:
        for parameter, value in item.get("required_functional_config", {}).items():
            selected_functional.setdefault(parameter, {})[
                json.dumps(value, sort_keys=True)
            ] = value
    for evidence_item in adaptation_evidence:
        for parameter, value in evidence_item.get(
            "required_functional_config", {}
        ).items():
            selected_functional.setdefault(parameter, {})[
                json.dumps(value, sort_keys=True)
            ] = value
    profile["selected_required_functional_config"] = {
        parameter: deepcopy(next(iter(values.values())))
        for parameter, values in selected_functional.items()
        if len(values) == 1
    }
    profile["hardware_adaptation_evidence"] = adaptation_evidence
    return bundles, excluded


def cookbook_initial_search_plan(task: dict[str, Any], discovery: dict[str, Any]) -> dict[str, Any]:
    catalog = catalog_index(discovery)
    bundles, exclusions = cookbook_candidate_bundles(discovery, catalog, task)
    failed_bundles = known_failed_configuration_deltas(task, discovery)
    retained: list[dict[str, Any]] = []
    for bundle in bundles:
        signature = json.dumps(
            bundle.get("config", {}), sort_keys=True, separators=(",", ":")
        )
        if signature in failed_bundles:
            exclusions.append({
                "name": bundle.get("name"),
                "reason": "exact Cookbook bundle has a definitive failure in this experiment fingerprint",
            })
        else:
            retained.append(bundle)
    bundles = retained
    ranked: list[dict[str, Any]] = []
    hardware_prior_candidates = add_cookbook_hardware_prior_candidates(
        ranked, catalog, discovery
    )
    tp_candidates = [
        size for size in supported_tp_sizes(discovery)
        if size > discovery["derived"]["minimum_tp_size"]
    ]
    if tp_candidates:
        add_ranked_candidate(
            ranked, catalog, "tp_size", tp_candidates,
            "compare legal tensor-parallel groups on the discovered multi-GPU topology before accepting a single-GPU baseline",
            [
                f"visible_gpu_count={discovery['derived']['visible_gpu_count']}",
                f"topology={discovery['topology_class']}",
                f"supported_tp_sizes={supported_tp_sizes(discovery)}",
            ],
            tier="topology",
        )
    plan = {
        "schema_version": 1,
        "phase": "cookbook_initialization",
        "ranked_parameter_groups": ranked,
        "cookbook_candidate_bundles": bundles,
        "cookbook_bundle_exclusions": exclusions,
        "cookbook_hardware_prior_candidates": hardware_prior_candidates,
        "parameter_audit": parameter_audit(catalog, [], discovery, task),
        "policy": "benchmark locally compatible model-cookbook configuration bundles across capability, prefix cache, scheduler, memory, CUDA Graph, and MoE families before profiling; retain only SLO-valid evidence",
    }
    plan["candidate_registry"] = registry_from_search_plan(plan)
    return plan


def preprofile_search_plan(task: dict[str, Any], discovery: dict[str, Any]) -> dict[str, Any]:
    """Build high-impact workload candidates that do not require trace evidence.

    These candidates are safe to screen on otherwise-idle same-SKU GPUs while
    GPU0 captures the baseline trace. Trace-dependent backends and topology
    changes remain deferred until profiling is complete.
    """
    plan = cookbook_initial_search_plan(task, discovery)
    catalog = catalog_index(discovery)
    ranked = plan["ranked_parameter_groups"]
    workload = task["workload"]
    evidence = [
        "stage=preprofile_workload_prior",
        f"deployment_mode={deployment_policy(task)['mode']}",
        f"input_tokens={workload['input_tokens']}",
        f"output_tokens={workload['output_tokens']}",
        f"prefix_reuse_ratio={workload.get('prefix_reuse_ratio', 0.0)}",
    ]
    if deployment_policy(task)["mode"] == "offline_throughput":
        add_ranked_candidate(
            ranked, catalog, "enable_mixed_chunk", [True],
            "test prefill/decode overlap under offline backlog pressure", evidence,
            tier="workload_prior",
        )
        add_ranked_candidate(
            ranked, catalog, "num_continuous_decode_steps", [2, 4],
            "amortize scheduler work for throughput-oriented continuous batching", evidence,
            tier="workload_prior",
        )
        add_ranked_candidate(
            ranked, catalog, "schedule_conservativeness", [0.6, 0.3],
            "test more aggressive admission under sustained offline backlog", evidence,
            tier="workload_prior",
        )
    if int(workload["input_tokens"]) >= 1024:
        default_chunk = catalog.get("chunked_prefill_size", {}).get("default")
        chunks = [value for value in chunk_candidates(task, default_chunk) if value != default_chunk]
        if chunks:
            chunks, strategy = rank_chunk_candidates(task, default_chunk, chunks, {})
            add_ranked_candidate(
                ranked, catalog, "chunked_prefill_size", chunks,
                strategy["reason"], evidence + [f"framework_default={default_chunk}"],
                tier="workload_prior",
            )
    if int(workload["input_tokens"]) >= 8192 or float(workload.get("prefix_reuse_ratio", 0.0)) >= 0.2:
        add_ranked_candidate(
            ranked, catalog, "page_size", [16, 32],
            "test KV page granularity for long-context or prefix-reusing traffic", evidence,
            tier="workload_prior",
        )
    if float(workload.get("prefix_reuse_ratio", 0.0)) >= 0.2:
        add_ranked_candidate(
            ranked, catalog, "schedule_policy", ["lpm"],
            "route shared-prefix requests with longest-prefix-match scheduling", evidence,
            tier="workload_prior",
        )
    plan["phase"] = "preprofile_parallel_screen"
    plan["policy"] = (
        "screen only locally compatible, single-GPU workload/Cookbook priors on spare GPUs "
        "while the baseline trace runs; defer trace-routed and multi-GPU mechanisms"
    )
    plan["candidate_registry"] = registry_from_search_plan(plan)
    return plan


def diagnosed_search_plan(
    task: dict[str, Any], discovery: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    diagnosis = profile["diagnosis"]
    bottleneck_classification = classify_bottleneck(task, discovery, profile)
    canonical_bottleneck = canonical_bottleneck_report(
        bottleneck_classification, diagnosis
    )
    legacy_adapter = {
        "prefill_attention_bound": "attention",
        "decode_attention_bound": "attention",
        "moe_compute_bound": "moe_compute",
        "gdn_state_compute_bound": "gdn_state_compute",
        "kv_memory_capacity_bound": "memory_transfer",
        "communication_bound": "communication",
        "host_scheduler_bound": "host_or_scheduler_stall",
        "mixed_or_unknown": "mixed_gpu_compute",
    }
    # The old imperative branches remain an adapter during migration. Their
    # input now comes from the canonical classifier, never directly from a
    # second Nsys taxonomy.
    primary = legacy_adapter[canonical_bottleneck["primary"]]
    secondary = {
        legacy_adapter[value]
        for value in canonical_bottleneck.get("secondary", [])
        if value in legacy_adapter
    }
    shares = diagnosis.get("shares_pct", {})
    timing_comparable = (
        diagnosis.get("profiling_run_performance_comparable") is True
        or "profiling_run_performance_comparable" not in diagnosis
    )
    # Nsys instrumentation can distort end-to-end timing without invalidating
    # the relative CUDA kernel execution shares. Keep GPU-active kernel shares
    # for backend routing; only host-gap/CUDA-API timing is disabled below.
    routing_shares = shares
    if not timing_comparable:
        # Instrumentation changed end-to-end throughput materially. Preserve
        # the trace for operator diagnosis, but do not let inferred timeline
        # bottlenecks or kernel shares reorder deployment candidates.
        if primary in {"host_or_scheduler_stall", "cpu_gpu_synchronization"}:
            primary = "profile_timing_distorted"
        secondary.clear()
    routing_diagnosis = deepcopy(diagnosis)
    routing_diagnosis["primary_bottleneck"] = canonical_bottleneck["primary"]
    routing_diagnosis["secondary_bottlenecks"] = canonical_bottleneck["secondary"]
    routing_diagnosis["trace_parameter_routing_enabled"] = timing_comparable
    catalog = catalog_index(discovery)
    ranked: list[dict[str, Any]] = []
    cookbook_bundles, cookbook_bundle_exclusions = cookbook_candidate_bundles(
        discovery, catalog, task
    )
    cookbook_hardware_prior_candidates = add_cookbook_hardware_prior_candidates(
        ranked, catalog, discovery
    )
    workload = task["workload"]
    concurrency = workload["max_concurrency"]
    boundary = discovery["derived"]["typical_prefill_batch_tokens"]
    backends = hardware_backends(discovery)
    evidence = [
        f"nsys.primary_bottleneck={primary}",
        f"nsys.timing_comparable={timing_comparable}",
    ]
    effective = profile.get("effective_server_config", {})
    prometheus_lines = profile.get("prometheus", {}).get("selected_samples", [])
    runtime = profile.get("runtime_observations", {})
    runtime_decode = runtime.get("decode", {}) if isinstance(runtime, dict) else {}
    runtime_prefill = runtime.get("prefill", {}) if isinstance(runtime, dict) else {}
    runtime_moe = runtime.get("moe", {}) if isinstance(runtime, dict) else {}
    missing_moe_config = bool(runtime_moe.get("missing_tuned_config"))
    decode_graph_coverage = runtime_decode.get("cuda_graph_coverage_pct")
    prefill_graph_coverage = runtime_prefill.get("cuda_graph_coverage_pct")
    cached_token_share = runtime_prefill.get("cached_token_share_pct")
    prefill_queue_pct = runtime_prefill.get("queue_nonempty_batch_pct")
    decode_graph_active = (
        any("mode=\"decode_cuda_graph\"" in line for line in prometheus_lines)
        or (isinstance(decode_graph_coverage, (int, float)) and decode_graph_coverage >= 95.0)
    )
    resolved_decode_graph_max = effective.get("cuda_graph_max_bs_decode")
    # A large configured graph ceiling is not a serving-throughput regression:
    # SGLang only replays the graphs that its scheduler actually reaches.
    # Lowering a well-covered default mostly measures recapture/startup cost
    # and produced misleading candidates such as 8/16 for a capacity-50 run.
    # Tune graph size only when logs show that the active decode batches are
    # not covered, or when runtime demand exceeds the configured ceiling.
    decode_running = runtime_decode.get("running_requests", {})
    observed_decode_batch = (
        decode_running.get("max") if isinstance(decode_running, dict) else None
    )
    decode_graph_capacity_shortfall = (
        isinstance(resolved_decode_graph_max, int)
        and isinstance(observed_decode_batch, int)
        and observed_decode_batch > resolved_decode_graph_max
    )
    decode_graph_needs_tuning = (
        not decode_graph_active
        or isinstance(decode_graph_coverage, (int, float)) and decode_graph_coverage < 95.0
        or decode_graph_capacity_shortfall
    )
    prefill_running = runtime_prefill.get("running_requests", {})
    observed_prefill_batch = (
        prefill_running.get("p95") if isinstance(prefill_running, dict) else None
    )
    known_failures = known_failed_candidates(task, discovery)
    known_failed_bundles = known_failed_configuration_deltas(task, discovery)
    retained_cookbook_bundles: list[dict[str, Any]] = []
    for bundle in cookbook_bundles:
        signature = json.dumps(
            bundle.get("config", {}), sort_keys=True, separators=(",", ":")
        )
        if signature in known_failed_bundles:
            cookbook_bundle_exclusions.append({
                "name": bundle.get("name"),
                "reason": "exact bundle has a definitive failure in this experiment fingerprint",
            })
        else:
            retained_cookbook_bundles.append(bundle)
    cookbook_bundles = retained_cookbook_bundles
    queue_reqs = prometheus_value(prometheus_lines, "sglang:num_queue_reqs")
    token_usage = prometheus_value(prometheus_lines, "sglang:token_usage")
    retractions = prometheus_value(prometheus_lines, "sglang:num_retracted_reqs")
    # A closed-loop client normally has no waiting queue while it keeps all
    # target requests in flight. A post-run queue sample cannot suppress
    # scheduler tuning for that workload shape.
    underdriven = (
        workload.get("request_rate", "inf") != "inf"
        and queue_reqs == 0
        and token_usage is not None
        and token_usage < 0.5
    )
    mode = deployment_policy(task)["mode"]
    profile_metrics = profile.get("benchmark", {}).get("metrics", {})
    mean_e2e = profile_metrics.get("mean_e2e_latency_ms")
    mean_ttft = profile_metrics.get("mean_ttft_ms")
    output_token_ratio = float(workload["output_tokens"]) / max(
        1.0, float(workload["input_tokens"] + workload["output_tokens"])
    )
    decode_latency_share = (
        max(0.0, float(mean_e2e) - float(mean_ttft)) / float(mean_e2e)
        if isinstance(mean_e2e, (int, float)) and mean_e2e > 0
        and isinstance(mean_ttft, (int, float))
        else None
    )
    mtp_relevant = (
        decode_latency_share >= 0.20
        if decode_latency_share is not None else output_token_ratio >= 0.20
    )
    if not mtp_relevant:
        retained_bundles = []
        for bundle in cookbook_bundles:
            if "speculative_algorithm" in bundle.get("config", {}):
                cookbook_bundle_exclusions.append({
                    "name": bundle.get("name"),
                    "reason": "decode share is below the 20% MTP relevance floor for this workload",
                })
            else:
                retained_bundles.append(bundle)
        cookbook_bundles = retained_bundles
    evidence.extend([
        f"prometheus.queue_reqs={queue_reqs}",
        f"prometheus.token_usage={token_usage}",
        f"prometheus.retractions={retractions}",
        f"scheduler_log.decode_cuda_graph_coverage_pct={decode_graph_coverage}",
        f"scheduler_log.prefill_cuda_graph_coverage_pct={prefill_graph_coverage}",
        f"scheduler_log.prefill_cached_token_share_pct={cached_token_share}",
        f"scheduler_log.prefill_queue_nonempty_batch_pct={prefill_queue_pct}",
        f"workload.output_token_ratio={output_token_ratio}",
        f"profile.decode_latency_share={decode_latency_share}",
        f"mtp_relevant={mtp_relevant}",
    ])
    quality_gated_candidates: list[dict[str, Any]] = []
    kv_metadata = catalog.get("kv_cache_dtype")
    if isinstance(kv_metadata, dict):
        choices = kv_metadata.get("choices")
        fp8_kv_values = [
            value for value in (choices or ["fp8_e4m3", "fp8_e5m2"])
            if isinstance(value, str) and value.lower() in {"fp8_e4m3", "fp8_e5m2"}
        ]
        fp8_kv_values.sort(key=lambda value: (value.lower() != "fp8_e4m3", value))
        kv_opt_in = bool(
            task.get("quality", {}).get("allow_kv_cache_precision_tuning", False)
        )
        quality_gated_candidates.append({
            "parameter": "kv_cache_dtype",
            "values": fp8_kv_values,
            "enabled": kv_opt_in and bool(fp8_kv_values),
            "quality_risk": "reduced KV-cache precision can change model outputs",
            "policy": "never benchmark or recommend FP8 KV cache without explicit task opt-in",
        })
        if kv_opt_in and fp8_kv_values:
            token_usage_summary = runtime_decode.get("token_usage_ratio", {})
            token_usage_p95 = (
                token_usage_summary.get("p95")
                if isinstance(token_usage_summary, dict) else None
            )
            add_ranked_candidate(
                ranked, catalog, "kv_cache_dtype", fp8_kv_values,
                "measure explicitly authorized FP8 KV-cache bandwidth/capacity trade-offs",
                evidence + [
                    "quality.allow_kv_cache_precision_tuning=true",
                    f"decode_token_usage_p95={token_usage_p95}",
                    f"attention_kernel_pct={shares.get('attention_kernels', 0)}",
                ],
                tier="quality_opt_in",
            )

    # Offline optimization deliberately runs at calibrated batch pressure. It
    # must explore capacity controls even when a short profiler range happens
    # not to contain a queue sample. Online mode keeps these controls behind
    # the tail-latency and queue evidence below.
    if mode == "offline_throughput":
        # Do not manufacture throughput candidates for max_running_requests.
        # In an unbounded no-SLO run it is an admission ceiling, not a speed
        # mechanism; SGLang's observed automatic capacity is used only to size
        # the benchmark backlog. Online/SLO calibration may still set a limit
        # to enforce the selected concurrency contract.
        base_mem_fraction = effective.get("mem_fraction_static", catalog.get("mem_fraction_static", {}).get("default"))
        if isinstance(base_mem_fraction, (int, float)) and not isinstance(base_mem_fraction, bool):
            # Allocation is nonlinear near the usable-memory boundary. A fixed
            # +/-2 point probe misses the useful region on large checkpoints,
            # while a blind high value merely turns tuning into OOMs.
            values = [
                round(min(0.95, float(base_mem_fraction) + step), 3)
                for step in (0.02, 0.05, 0.08)
            ]
            values.append(round(max(0.60, float(base_mem_fraction) - 0.03), 3))
            add_ranked_candidate(
                ranked, catalog, "mem_fraction_static",
                values,
                "progressively probe KV allocation above the resolved SGLang default and refine the measured allocator boundary",
                evidence + [f"resolved_mem_fraction_static={base_mem_fraction}"],
                tier="capacity",
            )
        base_prefill = effective.get(
            "max_prefill_tokens", catalog.get("max_prefill_tokens", {}).get("default")
        )
        if isinstance(base_prefill, int) and not isinstance(base_prefill, bool) and base_prefill > 0:
            context_length = discovery["model"].get("context_length")
            expanded_prefill = max(base_prefill * 2, int(workload["input_tokens"]) * 2)
            if isinstance(context_length, int) and context_length > 0:
                expanded_prefill = min(expanded_prefill, context_length)
            if expanded_prefill > base_prefill:
                add_ranked_candidate(
                    ranked, catalog, "max_prefill_tokens", [expanded_prefill],
                    "compare workload-shaped prefill admission budgets above or below the resolved runtime value",
                    evidence + [
                        f"resolved_max_prefill_tokens={base_prefill}",
                        f"input_tokens={workload['input_tokens']}",
                    ],
                    tier="capacity",
                )
        add_ranked_candidate(
            ranked, catalog, "schedule_conservativeness", [0.3, 0.6, 1.1],
            "sweep admission aggressiveness under calibrated offline queue pressure", evidence,
        )
        add_ranked_candidate(
            ranked, catalog, "num_continuous_decode_steps", [2, 4, 8],
            "amortize scheduler work for throughput-oriented continuous batching", evidence,
        )
        add_ranked_candidate(
            ranked, catalog, "enable_mixed_chunk", [True],
            "test mixed prefill/decode batching only for throughput-oriented execution", evidence,
        )

    if (
        discovery["model"].get("is_hybrid")
        or workload["input_tokens"] >= 8192
        or workload.get("prefix_reuse_ratio", 0) >= 0.2
    ):
        page_values = [16, 32, 64] if discovery["model"].get("is_hybrid") else [16, 32]
        add_ranked_candidate(
            ranked, catalog, "page_size", page_values,
            "test KV page granularity for hybrid state-cache layout, long contexts, or real prefix reuse",
            evidence + [
                f"input_tokens={workload['input_tokens']}",
                f"prefix_reuse_ratio={workload.get('prefix_reuse_ratio', 0)}",
            ],
            tier="workload_trace_coverage",
        )

    if (
        workload["input_tokens"] >= 1024
        and isinstance(prefill_graph_coverage, (int, float))
        and prefill_graph_coverage < 95.0
    ):
        prefill_target = max(1, int(observed_prefill_batch or 1))
        prefill_graph_bs = 1 << math.ceil(math.log2(prefill_target))
        add_ranked_candidate(
            ranked, catalog, "cuda_graph_max_bs_prefill",
            [prefill_graph_bs, prefill_graph_bs * 2],
            "extend prefill CUDA Graph coverage to the batch size observed in SGLang logs",
            evidence + [
                f"observed_prefill_batch_p95={observed_prefill_batch}",
                f"prefill_cuda_graph_coverage_pct={prefill_graph_coverage}",
            ],
            tier="workload_trace_coverage",
        )

    tp_candidates = [
        size for size in supported_tp_sizes(discovery)
        if size > discovery["derived"]["minimum_tp_size"]
    ]
    if tp_candidates:
        add_ranked_candidate(
            ranked, catalog, "tp_size", tp_candidates,
            "compare legal tensor-parallel groups against the profiled deployment; retain a larger group only with SLO-valid measured gain",
            evidence + [f"topology={discovery['topology_class']}", f"supported_tp_sizes={supported_tp_sizes(discovery)}"],
            tier="topology",
        )

    if (
        workload["input_tokens"] >= 1024
        and isinstance(prefill_queue_pct, (int, float))
        and prefill_queue_pct >= 10.0
    ):
        add_ranked_candidate(
            ranked, catalog, "enable_mixed_chunk", [True],
            "prefill and decode overlap while prefill batches queue; test mixed chunking before low-leverage cache sensitivity sweeps",
            evidence + [f"prefill_queue_nonempty_batch_pct={prefill_queue_pct}"],
            tier="workload_trace_coverage",
        )

    if task.get("search_depth", "thorough") == "thorough":
        # Sensitivity candidates provide coverage when the target workload is
        # low-load or a short trace cannot expose every capacity knob. They are
        # deliberately one-factor and remain subject to the same SLO gates.
        base_mem_fraction = effective.get("mem_fraction_static", catalog.get("mem_fraction_static", {}).get("default"))
        if isinstance(base_mem_fraction, (int, float)) and not isinstance(base_mem_fraction, bool):
            add_ranked_candidate(
                ranked, catalog, "mem_fraction_static",
                [round(max(0.60, float(base_mem_fraction) - 0.03), 3), round(min(0.97, float(base_mem_fraction) + 0.02), 3)],
                "sensitivity screen KV allocation around the resolved current-version default",
                evidence + [f"resolved_mem_fraction_static={base_mem_fraction}"], tier="sensitivity",
            )
        add_ranked_candidate(
            ranked, catalog, "num_continuous_decode_steps", [2, 4],
            "sensitivity screen scheduler amortization while retaining tail-latency gates", evidence, tier="sensitivity",
        )
        if decode_graph_needs_tuning:
            graph_target = max(concurrency, int(observed_decode_batch or 1))
            graph_bs = 1 << math.ceil(math.log2(max(1, graph_target)))
            add_ranked_candidate(
                ranked, catalog, "cuda_graph_max_bs_decode", [graph_bs, graph_bs * 2],
                "sensitivity screen incomplete CUDA Graph decode coverage", evidence, tier="sensitivity",
            )
        sensitivity_pages = [1, 16, 32, 64] if discovery["model"].get("is_hybrid") else [1, 16, 32]
        add_ranked_candidate(
            ranked, catalog, "page_size", sensitivity_pages,
            "sensitivity screen KV page granularity with full SLO regression checks", evidence, tier="sensitivity",
        )
        if workload["input_tokens"] >= 1024:
            add_ranked_candidate(
                ranked, catalog, "enable_mixed_chunk", [True],
                "sensitivity screen mixed prefill/decode batching for non-trivial prompts", evidence, tier="sensitivity",
            )

    dependent_bundles: list[dict[str, Any]] = long_context_capacity_bundles(
        task, discovery, catalog, effective
    )
    if discovery["model"].get("is_hybrid"):
        if "mamba_radix_cache_strategy" in catalog and "page_size" in catalog:
            dependent_bundles.append({
                "name": "hybrid-mamba-extra-buffer-page-64",
                "config": {"mamba_radix_cache_strategy": "extra_buffer", "page_size": 64},
                "reason": "the Cookbook Mamba-V2 path requires page_size=64 with extra_buffer; it is throughput-relevant even without shared-prefix traffic",
                "evidence": evidence + [
                    "cookbook.mamba_radix_cache_strategy=extra_buffer",
                    "cookbook.page_size=64",
                ],
            })
        base_mamba_ratio = effective.get(
            "mamba_full_memory_ratio",
            catalog.get("mamba_full_memory_ratio", {}).get("default"),
        )
        if (
            "mamba_full_memory_ratio" in catalog
            and isinstance(base_mamba_ratio, (int, float))
            and not isinstance(base_mamba_ratio, bool)
        ):
            for ratio in sorted({
                round(max(0.5, float(base_mamba_ratio) - 0.15), 2),
                round(min(1.0, float(base_mamba_ratio) + 0.05), 2),
            }):
                if ratio == float(base_mamba_ratio):
                    continue
                config = {"mamba_full_memory_ratio": ratio}
                if "mamba_radix_cache_strategy" in catalog and "page_size" in catalog:
                    config.update({
                        "mamba_radix_cache_strategy": "extra_buffer", "page_size": 64,
                    })
                dependent_bundles.append({
                    "name": f"hybrid-mamba-v2-memory-{ratio:g}",
                    "config": config,
                    "reason": "measure Cookbook-guided Mamba cache-memory allocation together with the required V2 cache path",
                    "evidence": evidence + [
                        "cookbook.mamba_full_memory_ratio",
                        f"resolved_mamba_full_memory_ratio={base_mamba_ratio}",
                    ],
                })

    # MTP must not be reduced to a single cookbook smoke command.  Once the
    # local checkpoint and current ServerArgs accept a documented speculative
    # recipe, probe one shallower and one deeper draft horizon.  They remain
    # bounded, model-native bundles rather than a blind Cartesian sweep.
    if discovery["model"].get("has_mtp_weights") and mtp_relevant:
        mtp_recipe = next(
            (
                bundle for bundle in cookbook_bundles
                if isinstance(bundle.get("config"), dict)
                and "speculative_algorithm" in bundle["config"]
            ),
            None,
        )
        if mtp_recipe is not None:
            base = dict(mtp_recipe["config"])
            for steps, drafts, label in ((2, 3, "shallow"), (4, 5, "deep")):
                variant = {
                    **base,
                    "speculative_num_steps": steps,
                    "speculative_num_draft_tokens": drafts,
                }
                dependent_bundles.append({
                    "name": f"mtp-{label}-{steps}-{drafts}",
                    "config": variant,
                    "reason": "refine the locally compatible Cookbook MTP horizon; acceptance telemetry determines whether a further sweep is justified",
                    "evidence": evidence + [
                        f"cookbook_base={mtp_recipe.get('name')}",
                        f"speculative_num_steps={steps}",
                        f"speculative_num_draft_tokens={drafts}",
                    ],
                })

    if normalized_experiment_mode(task) in {"balanced", "max"} and primary in {
        "gemm_compute", "moe_compute", "mixed_gpu_compute"
    } and "enable_torch_compile" in catalog:
        compile_config: dict[str, Any] = {"enable_torch_compile": True}
        if "torch_compile_max_bs" in catalog:
            compile_config["torch_compile_max_bs"] = min(
                256, 1 << math.ceil(math.log2(max(1, concurrency)))
            )
        dependent_bundles.append({
            "name": "compute-torch-compile",
            "config": compile_config,
            "reason": "measure one bounded torch.compile bundle when Nsys attributes the serving path to compute kernels",
            "evidence": evidence + [f"profile_primary={primary}"],
        })

    if (
        normalized_experiment_mode(task) == "max"
        and primary in {"host_or_scheduler_stall", "cpu_gpu_synchronization"}
    ):
        add_ranked_candidate(
            ranked, catalog, "disable_overlap_schedule", [True],
            "diagnostic max-mode comparison when trace evidence attributes material time to scheduler synchronization",
            evidence,
            tier="trace_diagnostic",
        )

    if primary in {"host_or_scheduler_stall", "cpu_gpu_synchronization"} and not underdriven:
        add_ranked_candidate(
            ranked, catalog, "num_continuous_decode_steps", [2, 4],
            "amortize scheduler and launch overhead; TTFT remains a hard gate",
            evidence + [f"gpu_active_pct={diagnosis.get('gpu_timeline_active_pct')}"]
        )
        add_ranked_candidate(
            ranked, catalog, "scheduler_recv_interval", [2, 4],
            "reduce scheduler polling overhead when GPU timeline contains host-side gaps", evidence
        )
        if concurrency >= 16:
            add_ranked_candidate(
                ranked, catalog, "tokenizer_worker_num", [2, 4],
                "parallelize tokenizer work at high request concurrency", evidence
            )

    if primary == "attention" or "attention" in secondary or routing_shares.get("attention_kernels", 0) >= 20:
        phase = "prefill_attention_backend" if workload["input_tokens"] >= workload["output_tokens"] else "decode_attention_backend"
        active_attention_backend = effective.get(phase) or effective.get("attention_backend")
        alternate_attention_backends = [
            backend for backend in backends["attention"] if backend != active_attention_backend
        ]
        add_ranked_candidate(
            ranked, catalog, phase,
            alternate_attention_backends,
            "compare installed, hardware-compatible attention implementations for the dominant phase",
            evidence + [
                f"attention_kernel_pct={routing_shares.get('attention_kernels', 0):.3f}",
                f"resolved_active_attention_backend={active_attention_backend}",
            ]
        )
        if phase != "attention_backend":
            add_ranked_candidate(
                ranked, catalog, "attention_backend", alternate_attention_backends,
                "compare the full attention backend; phase-only overrides do not establish end-to-end equivalence",
                evidence + [
                    f"attention_kernel_pct={routing_shares.get('attention_kernels', 0):.3f}",
                    f"resolved_active_attention_backend={effective.get('attention_backend')}",
                ],
            )

    if discovery["model"].get("is_moe"):
        profile_tp = effective.get("tp_size", discovery["derived"]["minimum_tp_size"])
        ep_candidates = supported_ep_sizes(discovery, profile_tp)
        # EP removes MoE tensor-parallel work but adds dispatch and collective
        # traffic. Limit automatic probing to request pressure where that
        # trade-off is plausible, and retain the mathematical proof in the
        # plan so a proposed topology is independently auditable.
        if (
            workload["max_concurrency"] >= 8
            and ep_candidates
            and "ep_size" in catalog
        ):
            intermediate = discovery["model"].get("moe_intermediate_size")
            block_shape = discovery["model"].get("weight_block_size")
            add_ranked_candidate(
                ranked, catalog, "ep_size", ep_candidates,
                "compare mathematically legal expert-parallel degrees under sustained MoE request pressure",
                evidence + [
                    f"profile_tp_size={profile_tp}",
                    f"moe_intermediate_size={intermediate}",
                    f"fp8_weight_block_size={block_shape}",
                    f"legal_ep_sizes={ep_candidates}",
                    "official_qwen3_rule=(moe_intermediate_size/(TP/EP)) % weight_block_N == 0",
                ],
                tier="topology",
            )
        moe_backend_relevant = bool(
            missing_moe_config
            or routing_shares.get("moe_kernels", 0) >= 10.0
            or canonical_bottleneck["primary"] == "moe_compute_bound"
            or "moe_compute_bound" in canonical_bottleneck.get("secondary", [])
        )
        if moe_backend_relevant:
            add_ranked_candidate(
                ranked, catalog, "moe_runner_backend", backends["moe"],
                "compare MoE runners only after trace or runtime logs establish material MoE backend evidence",
                evidence + [
                    f"moe_kernel_pct={routing_shares.get('moe_kernels', 0):.3f}",
                    f"sglang_log.missing_moe_config={missing_moe_config}",
                    f"sglang_log.missing_moe_config_count={runtime_moe.get('missing_config_count', 0)}",
                ]
            )
        if discovery["derived"]["visible_gpu_count"] > 1 and discovery["derived"]["minimum_tp_size"] > 1:
            add_ranked_candidate(
                ranked, catalog, "enable_dp_attention", [True],
                "test data-parallel attention for a tensor-parallel MoE deployment; retain it only when trace and SLO gates improve",
                evidence + [f"topology={discovery['topology_class']}"]
            )

    if primary == "gemm_compute" and not discovery["model"].get("is_moe"):
        add_ranked_candidate(
            ranked, catalog, "bf16_gemm_backend", ["cutedsl"],
            "compare the installed BF16 GEMM backend on supported Blackwell hardware", evidence
        )

    if (primary == "communication" or "communication" in secondary) and discovery["derived"]["visible_gpu_count"] > 1:
        topology = discovery["topology_class"]
        add_ranked_candidate(
            ranked, catalog, "enable_mscclpp", [True],
            "test small-message collective optimization after nsys attributes material GPU time to communication",
            evidence + [f"topology={topology}", f"communication_pct={routing_shares.get('communication_kernels', 0):.3f}"]
        )
        add_ranked_candidate(
            ranked, catalog, "disable_custom_all_reduce", [True],
            "measure NCCL fallback against SGLang custom all-reduce on the discovered topology", evidence
        )

    if primary == "memory_transfer" or "memory_transfer" in secondary:
        add_ranked_candidate(
            ranked, catalog, "page_size", [1, 16, 32, 64],
            "measure KV page granularity when memory operations are material", evidence
        )

    if (
        primary in {"host_or_scheduler_stall", "mixed_gpu_compute"}
        or "cuda_synchronization" in secondary
    ) and decode_graph_needs_tuning:
        graph_target = max(concurrency, int(observed_decode_batch or 1))
        graph_bs = max(1, 1 << math.ceil(math.log2(max(1, graph_target))))
        add_ranked_candidate(
            ranked, catalog, "cuda_graph_max_bs_decode", [graph_bs, graph_bs * 2],
            "extend decode CUDA Graph coverage to the observed active batch size",
            evidence + [
                f"resolved_cuda_graph_max_bs_decode={resolved_decode_graph_max}",
                f"observed_decode_batch={observed_decode_batch}",
                f"decode_graph_coverage_pct={decode_graph_coverage}",
            ]
        )

    if workload.get("prefix_reuse_ratio", 0.0) >= 0.2:
        add_ranked_candidate(
            ranked, catalog, "schedule_policy", ["lpm"],
            "exploit declared prefix reuse with longest-prefix-match scheduling",
            [f"prefix_reuse_ratio={workload.get('prefix_reuse_ratio')}"]
        )
    elif workload.get("prefix_reuse_ratio", 0.0) == 0:
        add_ranked_candidate(
            ranked, catalog, "disable_radix_cache", [True],
            "measure radix-cache bookkeeping cost because the workload declares no prefix reuse",
            ["prefix_reuse_ratio=0"]
        )

    default_chunk = effective.get("chunked_prefill_size")
    chunks = chunk_candidates(task, default_chunk)
    if isinstance(default_chunk, int) and default_chunk >= 512:
        # Give a smaller value a workload meaning. For shared-prefix traffic,
        # this is commonly the uncached request suffix, not an arbitrary half.
        uncached_anchor = max(256, power_of_two_ceil(expected_prefill_tokens(workload)))
        latency_pressure = (
            deployment_policy(task)["mode"] == "online_latency"
            and (
                bool(task.get("slo"))
                or isinstance(prefill_queue_pct, (int, float)) and prefill_queue_pct >= 10.0
            )
        )
        if latency_pressure and uncached_anchor < default_chunk:
            chunks.append(uncached_anchor)
        larger = sorted(value for value in chunks if value > default_chunk)
        if larger:
            midpoint = (default_chunk + larger[0]) // 2
            chunks.append(max(256, midpoint // 256 * 256))
    chunks, excluded_chunks = chunk_memory_feasibility(task, discovery, effective, chunks)
    nondefault_chunks = [value for value in chunks if value != default_chunk]
    nondefault_chunks, chunk_strategy = rank_chunk_candidates(
        task, default_chunk, nondefault_chunks, runtime_prefill
    )
    if nondefault_chunks:
        add_ranked_candidate(
            ranked, catalog, "chunked_prefill_size", nondefault_chunks,
            chunk_strategy["reason"],
            [
                f"expected_prefill_tokens_per_request={expected_prefill_tokens(workload)}",
                f"concurrent_prefill_boundary={boundary}",
                f"resolved_sglang_default={default_chunk}",
                f"memory_feasibility_excluded={len(excluded_chunks)}",
                f"candidate_order={nondefault_chunks}",
            ]
        )
    if queue_reqs is not None and queue_reqs > 0:
        if token_usage is not None and token_usage < 0.8:
            add_ranked_candidate(
                ranked, catalog, "schedule_conservativeness", [0.3, 0.6],
                "queue is nonempty while KV use is low, indicating conservative admission",
                evidence,
            )
        elif retractions is not None and retractions > 0:
            add_ranked_candidate(
                ranked, catalog, "schedule_conservativeness", [1.1, 1.3],
                "retractions indicate KV pressure; reduce unsafe admission", evidence,
            )

    trigger_plan = match_parameter_rules(
        task, discovery, profile, bottleneck_classification, set(catalog)
    )
    matched_parameters = set(trigger_plan["parameters"])
    existing_by_parameter = {item["parameter"]: item for item in ranked}
    for parameter, trigger in trigger_plan["parameters"].items():
        metadata = catalog.get(parameter)
        if not isinstance(metadata, dict):
            continue
        dynamic_values, value_evidence = dynamic_parameter_values(
            parameter, task, discovery, profile, metadata.get("default")
        )
        if parameter == "chunked_prefill_size" and dynamic_values:
            dynamic_values, dynamic_exclusions = chunk_memory_feasibility(
                task, discovery, effective, [
                    int(value) for value in dynamic_values
                    if isinstance(value, int) and not isinstance(value, bool)
                ],
            )
            excluded_chunks.extend(dynamic_exclusions)
            value_evidence["memory_feasibility_excluded"] = len(dynamic_exclusions)
        choices = metadata.get("choices")
        if isinstance(choices, list):
            dynamic_values = [value for value in dynamic_values if value in choices]
        item = existing_by_parameter.get(parameter)
        if item is None and dynamic_values:
            add_ranked_candidate(
                ranked, catalog, parameter, dynamic_values,
                f"triggered by {', '.join(trigger['rule_ids'])}",
                [json.dumps(value_evidence, sort_keys=True)],
                tier="trigger_rule",
            )
            item = next(
                (candidate for candidate in ranked if candidate["parameter"] == parameter),
                None,
            )
            if item is not None:
                existing_by_parameter[parameter] = item
        elif item is not None and dynamic_values:
            item["values"] = list(dict.fromkeys(dynamic_values))
        if item is not None:
            item["trigger"] = deepcopy(trigger)
            item["trigger_magnitude"] = trigger["magnitude"]
            item["value_strategy"] = value_evidence
            item["evidence"] = list(dict.fromkeys([
                *item.get("evidence", []),
                f"bottleneck.primary={bottleneck_classification['primary']}",
                f"trigger.rules={trigger['rule_ids']}",
                f"value.strategy={value_evidence.get('strategy')}",
            ]))

    quality_opt_in_parameters = {
        item["parameter"] for item in quality_gated_candidates if item.get("enabled")
    }
    ranked = [
        item for item in ranked
        if item["parameter"] in matched_parameters
        or item["parameter"] in quality_opt_in_parameters
        or "cookbook_hardware_prior" in item.get("tiers", [])
    ]
    evolution = deepcopy(discovery.get("parameter_evolution", {}))
    provisional_selected = select_provisional_candidates(
        evolution, bottleneck_classification
    )
    executable_provisional: list[dict[str, Any]] = []
    blocked_provisional_dependencies: list[dict[str, Any]] = []
    for provisional in provisional_selected:
        parameter = provisional["parameter"]
        metadata = catalog.get(parameter)
        if not isinstance(metadata, dict):
            continue
        companion_options = provisional.get("companion_configs", [])
        atomic_config: dict[str, Any] = {}
        if companion_options:
            compatible_companion = None
            incompatibilities = []
            for companion in companion_options:
                missing = [name for name in companion if name not in catalog]
                invalid = [
                    name for name, value in companion.items()
                    if name in catalog
                    and isinstance(catalog[name].get("choices"), list)
                    and value not in catalog[name]["choices"]
                ]
                if not missing and not invalid:
                    compatible_companion = companion
                    break
                incompatibilities.append({
                    "missing_parameters": missing,
                    "invalid_parameters": invalid,
                })
            if compatible_companion is None:
                blocked_provisional_dependencies.append({
                    "parameter": parameter,
                    "reason": "no Cookbook companion bundle is compatible with the current ServerArgs contract",
                    "incompatibilities": incompatibilities,
                })
                continue
            atomic_config = deepcopy(compatible_companion)
        provisional["atomic_config"] = atomic_config
        executable_provisional.append(provisional)
        add_ranked_candidate(
            ranked, catalog, parameter, provisional.get("candidate_values", []),
            "provisional SGLang parameter selected by contract diff and local semantic evidence",
            [
                f"parameter_evolution.state={provisional.get('state')}",
                f"parameter_evolution.confidence={provisional.get('confidence')}",
                f"parameter_evolution.submechanism={provisional.get('submechanism')}",
                *provisional.get("evidence", {}).get("mechanism_evidence", []),
            ],
            tier="parameter_evolution",
        )
        item = next(
            (candidate for candidate in ranked if candidate["parameter"] == parameter),
            None,
        )
        if item is not None:
            item.update({
                "provisional": True,
                "parameter_evolution": deepcopy(provisional),
                "trigger_magnitude": "medium",
                "submechanism": provisional.get("submechanism"),
                "value_strategy": provisional.get("value_strategy", {}),
            })

    # The live ServerArgs contract is larger than InferOpt's versioned rule
    # library and keeps evolving.  Route every other safe, bounded parameter
    # through the same semantic/context gate instead of requiring a bespoke
    # ``if parameter == ...`` branch.  Known rules and verified Cookbook
    # bundles keep precedence; semantic candidates only fill uncovered
    # mechanisms and remain ordinary measured experiments.
    covered_parameters = {
        item["parameter"] for item in ranked if isinstance(item.get("parameter"), str)
    }
    for bundle in [*cookbook_bundles, *dependent_bundles]:
        if isinstance(bundle, dict) and isinstance(bundle.get("config"), dict):
            covered_parameters.update(bundle["config"])
    semantic_selection = select_semantic_candidates(
        evolution, task, discovery, profile, bottleneck_classification,
        existing_parameters=covered_parameters,
    )
    semantic_runtime_exclusions: list[dict[str, Any]] = []
    compatible_semantic_configurations: list[dict[str, Any]] = []
    for configuration in semantic_selection.get("configurations", []):
        incompatibilities = []
        for parameter, value in configuration.get("config", {}).items():
            compatible, reason = parameter_value_runtime_compatible(
                discovery, parameter, value
            )
            if not compatible:
                incompatibilities.append({
                    "parameter": parameter, "value": value, "reason": reason,
                })
        if incompatibilities:
            semantic_runtime_exclusions.append({
                "name": configuration.get("name"),
                "config": deepcopy(configuration.get("config", {})),
                "incompatibilities": incompatibilities,
            })
        else:
            compatible_semantic_configurations.append(configuration)
    semantic_selection["configurations"] = compatible_semantic_configurations
    semantic_selection["runtime_compatibility_exclusions"] = semantic_runtime_exclusions
    for configuration in semantic_selection.get("configurations", []):
        relationships = configuration.get("relationships", {})
        dependent_bundles.append({
            **deepcopy(configuration),
            "source_type": "parameter_capability_registry",
            "dependencies": deepcopy(relationships.get("dependencies", [])),
            "conflicts": deepcopy(relationships.get("conflicts", [])),
            "priority": "medium",
        })
    runtime_compatibility = runtime_compatibility_constraints(discovery)
    runtime_compatibility_exclusions: list[dict[str, Any]] = []
    for item in ranked:
        original = list(item["values"])
        filtered_values = []
        for value in original:
            if (item["parameter"], json.dumps(value, sort_keys=True)) in known_failures:
                continue
            compatible, reason = parameter_value_runtime_compatible(
                discovery, item["parameter"], value
            )
            if not compatible:
                runtime_compatibility_exclusions.append({
                    "parameter": item["parameter"], "value": value, "reason": reason,
                })
                continue
            filtered_values.append(value)
        item["values"] = filtered_values
        if len(item["values"]) != len(original):
            item["evidence"].append(
                "excluded values that were prior local failures or incompatible with the detected model/runtime"
            )
    ranked = [item for item in ranked if item["values"]]
    # Trigger-derived values can replace the earlier heuristic chunk list.
    # Keep the public strategy evidence aligned with the exact, post-filter
    # order that screening will consume instead of reporting a stale list.
    chunk_group = next(
        (item for item in ranked if item["parameter"] == "chunked_prefill_size"),
        None,
    )
    if chunk_group is not None:
        final_chunk_order, objective_chunk_strategy = rank_chunk_candidates(
            task,
            effective.get("chunked_prefill_size"),
            [
                int(value) for value in chunk_group["values"]
                if isinstance(value, int) and not isinstance(value, bool)
            ],
            runtime_prefill,
        )
        chunk_group["values"] = final_chunk_order
        value_strategy = deepcopy(chunk_group.get("value_strategy", {}))
        chunk_reason = (
            "record the exact post-compatibility screening order for workload-derived "
            "uncached-prefill boundaries and sensitivity anchors"
        )
        chunk_strategy = {
            **chunk_strategy,
            **value_strategy,
            **objective_chunk_strategy,
            "value_set_strategy": value_strategy.get("strategy"),
            "strategy": objective_chunk_strategy.get(
                "strategy", chunk_strategy.get("strategy", "workload_derived_candidates")
            ),
            "reason": chunk_reason,
            "ordered_candidates": final_chunk_order,
            "candidate_order_source": "ranked_parameter_group_after_runtime_filters",
        }
        chunk_group["reason"] = chunk_reason
        chunk_group["evidence"] = [
            item for item in chunk_group.get("evidence", [])
            if not str(item).startswith("candidate_order=")
        ]
        chunk_group["evidence"].append(f"candidate_order={final_chunk_order}")
    retained_dependent_bundles: list[dict[str, Any]] = []
    for bundle in dependent_bundles:
        signature = json.dumps(
            bundle.get("config", {}), sort_keys=True, separators=(",", ":")
        )
        if signature in known_failed_bundles:
            cookbook_bundle_exclusions.append({
                "name": bundle.get("name"),
                "reason": "exact dependent bundle has a definitive failure in this experiment fingerprint",
            })
        else:
            retained_dependent_bundles.append(bundle)
    dependent_bundles = retained_dependent_bundles
    family_coverage: dict[str, dict[str, Any]] = {}
    submechanism_coverage: dict[str, dict[str, Any]] = {}
    for metadata in catalog.values():
        family = metadata["family"]
        family_coverage.setdefault(family, {"available_parameters": 0, "selected_parameters": []})
        family_coverage[family]["available_parameters"] += 1
    for item in ranked:
        item["submechanism"] = item.get("submechanism") or parameter_submechanism(
            item["parameter"], item["family"]
        )
        family_coverage[item["family"]]["selected_parameters"].append(item["parameter"])
        submechanism_coverage.setdefault(
            item["submechanism"], {"selected_parameters": []}
        )["selected_parameters"].append(item["parameter"])
    search_plan = {
        "schema_version": 6,
        "profiler_evidence": diagnosis,
        "routing_evidence": routing_diagnosis,
        "bottleneck_classification": bottleneck_classification,
        "canonical_bottleneck": canonical_bottleneck,
        "trigger_rule_plan": trigger_plan,
        "ranked_parameter_groups": ranked,
        "cookbook_candidate_bundles": cookbook_bundles,
        "cookbook_bundle_exclusions": cookbook_bundle_exclusions,
        "cookbook_hardware_prior_candidates": cookbook_hardware_prior_candidates,
        "policy": "match the bottleneck/workload/model/hardware tuple against declarative rules, then order only within the matched parameter set; coupled model-native bundles remain atomic experiments",
        "deployment_mode": mode,
        "parameter_family_coverage": family_coverage,
        "parameter_submechanism_coverage": submechanism_coverage,
        "runtime_compatibility": runtime_compatibility,
        "runtime_compatibility_exclusions": runtime_compatibility_exclusions,
        "parameter_evolution": {
            **compact_evolution_summary(evolution),
            "selected_for_exploration": [
                item["parameter"] for item in executable_provisional
            ],
            "blocked_dependencies": blocked_provisional_dependencies,
        },
        "parameter_capability_registry": {
            "schema_version": evolution.get("schema_version"),
            "audited_parameter_count": len(evolution.get("parameters", [])),
            "state_counts": deepcopy(evolution.get("state_counts", {})),
            "semantic_selection": deepcopy(semantic_selection),
            "policy": (
                "every visible ServerArgs parameter is classified before selection; "
                "versioned rules, Cookbook evidence, inferred semantics, dependencies, "
                "conflicts, applicability, risk, and measured bottlenecks use one gate"
            ),
        },
        "parameter_audit": parameter_audit(
            catalog, ranked, discovery, task,
            [*cookbook_bundles, *dependent_bundles],
            semantic_selection.get("decisions", []),
        ),
        "workload_assessment": {
            "underdriven": underdriven,
            "reason": (
                "queue is empty and KV use is below 50%; run a closed-loop load calibration before interpreting scheduler or memory tuning"
                if underdriven else "workload produced sufficient queue or KV pressure for parameter routing"
            ),
        },
        "profile_timing_comparable": timing_comparable,
        "runtime_moe_config_missing": missing_moe_config,
        "runtime_moe_config_evidence": runtime_moe,
        "mtp_relevance": {
            "relevant": mtp_relevant,
            "minimum_decode_share": 0.20,
            "decode_latency_share": decode_latency_share,
            "output_token_ratio_fallback": output_token_ratio,
            "policy": "use measured decode latency share when available; token ratio is only a pre-profile fallback",
        },
        "quality_gated_candidates": quality_gated_candidates,
        "excluded_prior_failures": [
            {"parameter": key, "value": json.loads(value)}
            for key, value in sorted(known_failures)
        ],
        "ranked_configuration_bundles": dependent_bundles,
        "resolved_baseline": {
            key: effective.get(key)
            for key in (
                "chunked_prefill_size", "mem_fraction_static", "max_running_requests",
                "max_total_tokens", "max_prefill_tokens", "kv_cache_dtype", "moe_runner_backend",
                "cuda_graph_max_bs_decode", "cuda_graph_max_bs_prefill",
                "attention_backend", "prefill_attention_backend", "decode_attention_backend",
            )
            if key in effective
        },
        "excluded_chunked_prefill_candidates": excluded_chunks,
        "chunked_prefill_strategy": chunk_strategy,
    }
    # Registry is the canonical candidate ledger. Existing ranked fields are
    # retained during migration as adapters and for archived artifact readers.
    search_plan["candidate_registry"] = registry_from_search_plan(search_plan)
    search_plan["candidate_registry"]["migration"] = {
        "mode": "active_registry_with_legacy_adapter",
        "legacy_ranked_parameter_groups_retained": True,
        "policy": (
            "all candidate sources are merged into the registry before scheduling; "
            "legacy ranked groups remain an audited adapter until replay parity is established"
        ),
    }
    return search_plan


def profile_spec(
    task: dict[str, Any], discovery: dict[str, Any], baseline: dict[str, Any] | None = None
) -> dict[str, Any]:
    spec = build_execution_spec(
        task, discovery, stage_name="profile", baseline={
            "tp_size": discovery["derived"]["minimum_tp_size"],
            "enable_metrics": True, **(baseline or {}),
        },
        space={}, max_trials=1, repetitions=1,
        remaining_gpu_hours=float(task["budget"]["max_gpu_hours"]),
        remaining_wall_minutes=float(task["budget"]["max_wall_time_minutes"]),
    )
    spec["scope"]["allow_profiling"] = True
    profile_concurrency = max(1, int(task["workload"]["max_concurrency"]))
    # Nsight is used for bottleneck classification, not latency ranking. Three
    # closed-loop waves provide repeated prefill/decode behavior without
    # turning a diagnostic trace into another full confirmation benchmark.
    # Keep very high-concurrency traces bounded, but always admit one full wave.
    profile_prompts = max(
        profile_concurrency,
        min(256, max(32, profile_concurrency * 3)),
    )
    benchmark = spec["benchmark"]
    benchmark["profile_capacity_waves"] = OFFLINE_PROFILE_SATURATION_WAVES
    benchmark["num_prompts"] = profile_prompts
    # generated-shared-prefix uses prompts-per-group as its effective sample
    # count. Updating only --num-prompts silently leaves the original large
    # group count in place, turning a bounded trace into a full workload run.
    if benchmark.get("dataset_name") == "generated-shared-prefix":
        groups = max(1, int(benchmark["gsp_num_groups"]))
        benchmark["gsp_prompts_per_group"] = max(1, math.ceil(profile_prompts / groups))
        benchmark["num_prompts"] = groups * benchmark["gsp_prompts_per_group"]
    # A profile needs warm kernels and a steady-state interval, not the full
    # confirmation warmup budget. Bound both before the preflight can expand
    # request count to satisfy its duration gate.
    benchmark["warmup_requests"] = min(
        int(benchmark["warmup_requests"]),
        max(16, int(task["workload"]["max_concurrency"]) * 2),
    )
    benchmark["min_measurement_seconds"] = min(
        float(benchmark["min_measurement_seconds"]), 5.0
    )
    return spec


def profile_matches_task(profile: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    path = Path(str(profile.get("run_dir", ""))) / "spec.json"
    if not path.is_file():
        return ["profile artifact is missing its immutable spec.json"]
    observed = load_json(path)
    expected_pairs = {
        "repository": (observed.get("repository"), expected.get("repository")),
        "model.path": (observed.get("model", {}).get("path"), expected.get("model", {}).get("path")),
        "model.dtype": (observed.get("model", {}).get("dtype"), expected.get("model", {}).get("dtype")),
        "model.quantization": (observed.get("model", {}).get("quantization"), expected.get("model", {}).get("quantization")),
        "model.context_length": (observed.get("model", {}).get("context_length"), expected.get("model", {}).get("context_length")),
        "input_tokens": (observed.get("benchmark", {}).get("random_input_len"), expected.get("benchmark", {}).get("random_input_len")),
        "output_tokens": (observed.get("benchmark", {}).get("random_output_len"), expected.get("benchmark", {}).get("random_output_len")),
        "max_concurrency": (observed.get("benchmark", {}).get("max_concurrency"), expected.get("benchmark", {}).get("max_concurrency")),
        "baseline_config": (observed.get("search", {}).get("baseline"), expected.get("search", {}).get("baseline")),
        "execution.env": (observed.get("execution", {}).get("env", {}), expected.get("execution", {}).get("env", {})),
        "experiment_fingerprint": (
            observed.get("experiment_fingerprint"), expected.get("experiment_fingerprint")
        ),
    }
    return [
        f"profile does not match current {name}: {actual!r} != {wanted!r}"
        for name, (actual, wanted) in expected_pairs.items()
        if actual != wanted
    ]


def annotate_profile_comparability(
    profiling: dict[str, Any], calibration: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
    max_regression_pct: float = 15.0,
) -> dict[str, Any]:
    """Compare profiled throughput with calibration or the later raw baseline."""
    diagnosis = profiling.setdefault("diagnosis", {})
    profile_rps = profiling.get("benchmark", {}).get("metrics", {}).get("request_throughput_rps")
    selected_concurrency = calibration.get("selected_analysis_concurrency")
    baseline_point = next(
        (
            point for point in calibration.get("points", [])
            if point.get("concurrency") == selected_concurrency and point.get("valid_for_analysis")
        ),
        None,
    )
    baseline_rps = (
        baseline_metrics.get("request_throughput_rps")
        if isinstance(baseline_metrics, dict)
        else baseline_point.get("metrics", {}).get("request_throughput_rps")
        if isinstance(baseline_point, dict) else None
    )
    comparison_source = (
        "screening_baseline" if isinstance(baseline_metrics, dict)
        else "calibration" if isinstance(baseline_point, dict)
        else "pending_unprofiled_baseline"
    )
    comparable: bool | None = None
    regression_pct = None
    if (
        isinstance(profile_rps, (int, float))
        and isinstance(baseline_rps, (int, float))
        and baseline_rps > 0
    ):
        regression_pct = (float(baseline_rps) - float(profile_rps)) / float(baseline_rps) * 100
        comparable = regression_pct <= max_regression_pct
    diagnosis.update({
        "profiling_run_performance_comparable": comparable,
        "profile_comparability_status": (
            "comparable" if comparable is True
            else "distorted" if comparable is False
            else "pending_baseline"
        ),
        "profile_comparison_source": comparison_source,
        "profile_request_throughput_rps": profile_rps,
        "unprofiled_request_throughput_rps": baseline_rps,
        "profile_throughput_regression_pct": round(regression_pct, 3) if regression_pct is not None else None,
        "profile_comparability_max_regression_pct": max_regression_pct,
        "timing_evidence_policy": (
            "host-gap and CUDA-API timing may route parameters"
            if comparable is True
            else "ignore host-gap and CUDA-API timing for parameter routing; retain kernel execution shares only"
            if comparable is False
            else "defer host-gap and CUDA-API routing until an unprofiled baseline is measured"
        ),
    })
    return profiling


def refresh_search_plan_after_baseline(
    task: dict[str, Any], discovery: dict[str, Any], profiling: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Refresh post-profile routing without discarding measured Registry state."""
    refreshed = diagnosed_search_plan(task, discovery, profiling)
    merged = deepcopy(current)

    def unique_values(values: list[Any]) -> list[Any]:
        output: list[Any] = []
        seen: set[str] = set()
        for value in values:
            signature = json.dumps(value, sort_keys=True, default=str)
            if signature in seen:
                continue
            seen.add(signature)
            output.append(deepcopy(value))
        return output

    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for source in (
        refreshed.get("ranked_parameter_groups", []),
        current.get("ranked_parameter_groups", []),
    ):
        for group in source:
            if not isinstance(group, dict) or not isinstance(group.get("parameter"), str):
                continue
            parameter = group["parameter"]
            if parameter not in groups:
                groups[parameter] = deepcopy(group)
                order.append(parameter)
                continue
            existing = groups[parameter]
            existing["values"] = unique_values([
                *existing.get("values", []), *group.get("values", []),
            ])
            existing["evidence"] = unique_values([
                *existing.get("evidence", []), *group.get("evidence", []),
            ])
            existing_rules = existing.setdefault("trigger", {}).setdefault(
                "rule_ids", []
            )
            existing["trigger"]["rule_ids"] = unique_values([
                *existing_rules,
                *group.get("trigger", {}).get("rule_ids", []),
            ])
    merged["ranked_parameter_groups"] = [groups[name] for name in order]

    existing_registry = CandidateRegistry.from_dict(
        current.get("candidate_registry", {})
    )
    refreshed_registry = CandidateRegistry.from_dict(
        refreshed.get("candidate_registry", {})
    )
    existing_registry.canonical_bottleneck = deepcopy(
        refreshed_registry.canonical_bottleneck
    )
    for candidate in refreshed_registry.candidates.values():
        sources = candidate.get("sources", []) or [{"type": "post_baseline_refresh"}]
        for source in sources:
            existing_registry.propose(
                name=str(candidate.get("name", "candidate")),
                config_delta=deepcopy(candidate.get("config_delta", {})),
                env_delta=deepcopy(candidate.get("env_delta", {})),
                mechanism=str(candidate.get("mechanism", "unknown")),
                source=deepcopy(source),
                expected_impact=str(candidate.get("expected_impact", "medium")),
                parameter=candidate.get("parameter"),
                value=deepcopy(candidate.get("value")),
                value_rank=int(candidate.get("value_rank", 0)),
                dependencies=deepcopy(candidate.get("dependencies", [])),
                conflicts=deepcopy(candidate.get("conflicts", [])),
                risk=deepcopy(candidate.get("risk", {})),
                value_strategy=deepcopy(candidate.get("value_strategy", {})),
                selection_score=float(candidate.get("selection_score", 0.0)),
                selection_evidence=deepcopy(
                    candidate.get("selection_evidence", {})
                ),
                state=str(candidate.get("state", "eligible")),
                decision_reason=candidate.get("decision_reason"),
            )
    merged["candidate_registry"] = existing_registry.to_dict()
    for key in (
        "bottleneck_classification", "canonical_bottleneck", "routing_evidence",
        "trigger_rule_plan", "parameter_priority_scores", "parameter_match_order",
        "parameter_capability_registry", "workload_assessment",
        "profile_timing_comparable",
    ):
        if key in refreshed:
            merged[key] = deepcopy(refreshed[key])
    merged["post_screen_profile_refresh"] = {
        "completed": True,
        "comparison_source": profiling.get("diagnosis", {}).get(
            "profile_comparison_source"
        ),
        "comparability_status": profiling.get("diagnosis", {}).get(
            "profile_comparability_status"
        ),
        "policy": (
            "screening baseline resolves offline Nsys timing comparability; refreshed "
            "bottleneck rules are eligible only for refinement and later stages"
        ),
    }
    return merged


def core_serving_parameter_order(
    task: dict[str, Any], discovery: dict[str, Any], search_plan: dict[str, Any],
) -> list[str]:
    """Order only trigger-matched knobs; history is a bounded tie-break prior."""
    ranked = [
        item for item in search_plan.get("ranked_parameter_groups", [])
        if isinstance(item, dict) and item.get("parameter") and item.get("values")
    ]
    original_index = {item["parameter"]: index for index, item in enumerate(ranked)}
    by_parameter = {item["parameter"]: item for item in ranked}
    strong = search_plan.get("trigger_rule_plan", {}).get("strong_candidates", [])
    strong_index = {parameter: index for index, parameter in enumerate(strong)}
    parameter_priors = search_plan.get("history", {}).get("priors", {}).get(
        "parameter_priors", {}
    )

    def prior_support(parameter: str) -> float:
        samples = parameter_priors.get(parameter, [])
        return max(
            (
                float(item.get("mean_improvement_pct", 0))
                * min(1.0, math.log2(1 + int(item.get("samples", 1))) / 3)
                for item in samples if isinstance(item, dict)
            ),
            default=0.0,
        )

    ordered = sorted(
        (item["parameter"] for item in ranked),
        key=lambda parameter: (
            -MAGNITUDE_ORDER.get(by_parameter[parameter].get("trigger_magnitude", "low"), 1),
            strong_index.get(parameter, len(strong) + original_index[parameter]),
            -prior_support(parameter),
            original_index[parameter],
        ),
    )
    search_plan["parameter_match_order"] = [
        {
            "parameter": parameter,
            "magnitude": by_parameter[parameter].get("trigger_magnitude", "low"),
            "rule_ids": by_parameter[parameter].get("trigger", {}).get("rule_ids", []),
            "history_prior_support": round(prior_support(parameter), 4),
            "candidate_values": deepcopy(by_parameter[parameter].get("values", [])),
        }
        for parameter in ordered
    ]
    # Retain this field for archived report readers, but its score is now only
    # the trigger magnitude rather than a global cross-scenario points table.
    search_plan["parameter_priority_scores"] = [
        {
            "parameter": parameter,
            "score": MAGNITUDE_ORDER.get(
                by_parameter[parameter].get("trigger_magnitude", "low"), 1
            ),
            "candidate_values": deepcopy(by_parameter[parameter].get("values", [])),
            "reason": by_parameter[parameter].get("reason"),
            "evidence": deepcopy(by_parameter[parameter].get("evidence", [])),
        }
        for parameter in ordered
    ]
    return list(dict.fromkeys(ordered))


def candidate_differs_from_effective_baseline(
    parameter: str,
    value: Any,
    anchor_config: dict[str, Any],
    effective_config: dict[str, Any],
) -> bool:
    """Avoid spending a restart on a value SGLang already resolved.

    The launch baseline intentionally contains only explicit user/controller
    choices. ServerArgs, however, derives several serving defaults at runtime.
    Comparing a proposed value only with the sparse launch dictionary makes a
    no-op look like a tuning candidate.
    """
    if parameter in anchor_config:
        return anchor_config[parameter] != value
    return parameter not in effective_config or effective_config[parameter] != value


def candidate_config_differs_from_effective_baseline(
    config: dict[str, Any],
    anchor_config: dict[str, Any],
    effective_config: dict[str, Any],
) -> bool:
    return any(
        candidate_differs_from_effective_baseline(parameter, value, anchor_config, effective_config)
        for parameter, value in config.items()
    )


def reference_baseline_mode(task: dict[str, Any]) -> bool:
    """Return whether one preserved baseline supports candidate-only confirmation."""
    return task.get("deployment_mode") == "offline_throughput" and not task.get("slo")


def configure_offline_reference_window(
    spec: dict[str, Any], task: dict[str, Any], *,
    phase: str = "screening",
) -> None:
    """Use one matched, cache-flushed window for screening and confirmation.

    The confirmation contract is intentionally applied to every candidate in
    this screen. A baseline that already meets it can then be preserved without
    running a second, nearly identical benchmark against the same service.
    """
    workload = task.get("workload") if isinstance(task.get("workload"), dict) else {}
    capacity = workload.get(
        "observed_practical_capacity", workload.get("observed_admission_capacity")
    )
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise ValueError(
            "offline no-SLO screening requires observed practical/admission capacity from the "
            "unbounded baseline profile before candidate benchmarking"
        )
    reference_prompts = confirmation_request_count(task)
    waves = offline_saturation_waves(
        task, confirmation=phase == "confirmation", phase=phase
    )
    saturation_requests = capacity * waves
    if saturation_requests is not None:
        reference_prompts = max(reference_prompts, saturation_requests)
    reference_duration = float(
        (task.get("measurement") or {}).get("min_measurement_seconds", 15)
    )
    spec["benchmark"].update({
        "num_prompts": reference_prompts,
        "min_measurement_seconds": max(
            float(spec["benchmark"].get("min_measurement_seconds", 0)),
            reference_duration,
        ),
        "flush_cache": True,
        "baseline_reference_num_prompts": reference_prompts,
        "baseline_reference_min_measurement_seconds": reference_duration,
        "saturation_capacity": capacity,
        "saturation_waves": waves,
    })
    spec.setdefault("search", {})["measurement_tier"] = phase
    if spec["benchmark"].get("dataset_name") == "generated-shared-prefix":
        groups = max(1, int(spec["benchmark"]["gsp_num_groups"]))
        spec["benchmark"]["gsp_prompts_per_group"] = max(
            1, math.ceil(reference_prompts / groups)
        )
        spec["benchmark"]["num_prompts"] = (
            groups * spec["benchmark"]["gsp_prompts_per_group"]
        )
        spec["benchmark"]["baseline_reference_num_prompts"] = spec["benchmark"]["num_prompts"]


def confirmation_trial_reserve(task: dict[str, Any]) -> int:
    """Reserve only the minimum complete paired decision blocks.

    Additional Bayesian blocks are elastic consumers of genuinely unused
    exploration budget. Reserving an extra block before a viable candidate
    existed systematically starved fast and balanced discovery.
    """
    repetitions = effective_confirmation_repetitions(task)
    measurement = task.get("measurement") or {}
    minimum_repetitions = repetitions
    if measurement.get("bayesian_sequential", False):
        minimum_repetitions = max(
            repetitions, int(measurement.get("bayesian_min_blocks", repetitions))
        )
    return minimum_repetitions * 2


def task_trial_budget(task: dict[str, Any]) -> dict[str, Any]:
    """Allocate trials while reserving only minimum paired confirmation."""
    allocation = tiered_trial_budget(int(task["budget"]["max_trials"]))
    planned = allocation["planned"]
    desired_confirmation = min(
        max(2, int(task["budget"]["max_trials"]) - 1),
        confirmation_trial_reserve(task),
    )
    original_confirmation = int(planned["confirmation"])
    extra = max(0, desired_confirmation - original_confirmation)
    from_refinement = min(extra, int(planned["refinement"]))
    planned["refinement"] -= from_refinement
    extra -= from_refinement
    if extra:
        transferable_discovery = max(0, int(planned["discovery"]) - 1)
        from_discovery = min(extra, transferable_discovery)
        planned["discovery"] -= from_discovery
        extra -= from_discovery
    planned["confirmation"] = desired_confirmation - extra
    released_confirmation = max(0, original_confirmation - planned["confirmation"])
    planned["discovery"] += released_confirmation
    total = max(1, int(allocation["total_trials"]))
    allocation["percentages"] = {
        key: round(int(value) / total * 100, 2)
        for key, value in planned.items()
    }
    allocation["confirmation_policy"] = (
        "reserve only the minimum paired Bayesian blocks; additional blocks "
        "may use only budget left after discovery/refinement"
    )
    return allocation


def required_mechanism_coverage(task: dict[str, Any]) -> int:
    """Return the minimum distinct serving mechanisms for a defensible screen."""
    return {
        "fast": 3,
        "balanced": 6,
        "max": 12,
    }.get(normalized_experiment_mode(task), 6)


def required_mechanism_classes(
    task: dict[str, Any], discovery: dict[str, Any], search_plan: dict[str, Any] | None = None
) -> list[str]:
    """Describe model/workload mechanisms that need evidence, not knob count."""
    required = {"scheduling", "capacity"}
    model = discovery.get("model", {})
    catalog = catalog_index(discovery)
    moe_parameters = {"moe_runner_backend", "ep_size", "moe_a2a_backend", "moe_dp_size"}
    planned_parameters = {
        item.get("parameter")
        for item in (search_plan or {}).get("ranked_parameter_groups", [])
        if isinstance(item, dict)
    }
    planned_bundle_parameters = {
        parameter
        for key in ("cookbook_candidate_bundles", "ranked_configuration_bundles")
        for bundle in (search_plan or {}).get(key, [])
        if isinstance(bundle, dict) and isinstance(bundle.get("config"), dict)
        for parameter in bundle["config"]
    }
    has_executable_moe_candidate = bool(
        moe_parameters & (planned_parameters | planned_bundle_parameters)
    ) if isinstance(search_plan, dict) else any(
        parameter in catalog for parameter in moe_parameters
    )
    if model.get("is_moe") and has_executable_moe_candidate:
        required.add("moe")
    if planned_parameters & {
        "attention_backend", "prefill_attention_backend", "decode_attention_backend"
    }:
        required.add("attention")
    if model.get("is_hybrid") and "mamba_radix_cache_strategy" in catalog:
        required.add("mamba")
    cookbook_bundles, _ = cookbook_candidate_bundles(discovery, catalog, task)
    has_compatible_mtp = any(
        "speculative_algorithm" in bundle.get("config", {}) for bundle in cookbook_bundles
    )
    if (
        model.get("has_mtp_weights")
        and has_compatible_mtp
        and task.get("capability_overrides", {}).get("mtp") != "disabled"
        and (
            not isinstance(search_plan, dict)
            or search_plan.get("mtp_relevance", {}).get("relevant", True)
        )
    ):
        required.add("mtp")
    minimum_tp = discovery.get("derived", {}).get("minimum_tp_size", 1)
    if any(size > minimum_tp for size in supported_tp_sizes(discovery)):
        required.add("topology")
    return sorted(required)


def configuration_mechanism_classes(config: dict[str, Any]) -> set[str]:
    """Map a measured configuration to serving mechanisms it actually exercised."""
    mechanisms: set[str] = set()
    keys = set(config)
    if keys & {
        "chunked_prefill_size", "enable_mixed_chunk", "schedule_policy",
        "schedule_conservativeness", "num_continuous_decode_steps",
    }:
        mechanisms.add("scheduling")
    if keys & {"mem_fraction_static", "max_prefill_tokens", "max_total_tokens", "page_size"}:
        mechanisms.add("capacity")
    if keys & {"moe_runner_backend", "ep_size", "moe_dp_size", "moe_a2a_backend"}:
        mechanisms.add("moe")
    if keys & {
        "attention_backend", "prefill_attention_backend", "decode_attention_backend"
    }:
        mechanisms.add("attention")
    if keys & {
        "mamba_radix_cache_strategy", "mamba_full_memory_ratio", "max_mamba_cache_size",
        "mamba_ssm_dtype",
    }:
        mechanisms.add("mamba")
    if "speculative_algorithm" in keys:
        mechanisms.add("mtp")
    if keys & {"tp_size", "pp_size", "dp_size", "ep_size", "enable_dp_attention"}:
        mechanisms.add("topology")
    return mechanisms


def recommendation_quality_gate(
    task: dict[str, Any], recommendation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed when a performance winner changes numerical precision."""
    config = (
        recommendation.get("config", recommendation)
        if isinstance(recommendation, dict) else {}
    )
    kv_dtype = str(config.get("kv_cache_dtype", "auto")).lower()
    precision_change = kv_dtype in {"fp8_e4m3", "fp8_e5m2", "mxfp8", "nvfp4", "fp4_mx_block16"}
    quality = task.get("quality", {}) if isinstance(task.get("quality"), dict) else {}
    if not precision_change:
        return {
            "required": False, "passed": True, "state": "not_required",
            "reason": "recommended configuration does not change cache/model precision",
        }
    attestation_path = quality.get("attestation_path")
    attestation: dict[str, Any] | None = None
    attestation_error: str | None = None
    attestation_sha256: str | None = None
    dataset_sha256: str | None = None
    if isinstance(attestation_path, str):
        path = Path(attestation_path).expanduser()
        try:
            raw = path.read_bytes()
            attestation_sha256 = hashlib.sha256(raw).hexdigest()
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("attestation root must be an object")
            attestation = value
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            attestation_error = str(exc)
    evaluation_dataset = quality.get("evaluation_dataset")
    if isinstance(evaluation_dataset, str):
        try:
            digest = hashlib.sha256()
            with Path(evaluation_dataset).expanduser().open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            dataset_sha256 = digest.hexdigest()
        except OSError as exc:
            attestation_error = str(exc)
    maximum_regression = float(quality.get("max_regression_pct", 5.0))
    required_attestation_fields = {
        "approved", "method", "dataset_sha256", "metric",
        "baseline_score", "candidate_score", "regression_pct",
    }
    attested = bool(
        isinstance(attestation, dict)
        and required_attestation_fields.issubset(attestation)
        and attestation.get("approved") is True
        and isinstance(attestation.get("method"), str)
        and bool(attestation.get("method"))
        and isinstance(attestation.get("dataset_sha256"), str)
        and len(attestation.get("dataset_sha256", "")) == 64
        and dataset_sha256 is not None
        and attestation.get("dataset_sha256") == dataset_sha256
        and isinstance(attestation.get("metric"), str)
        and isinstance(attestation.get("baseline_score"), (int, float))
        and isinstance(attestation.get("candidate_score"), (int, float))
        and isinstance(attestation.get("regression_pct"), (int, float))
        and not isinstance(attestation.get("regression_pct"), bool)
        and float(attestation["regression_pct"]) <= maximum_regression
        and str(attestation.get("kv_cache_dtype", kv_dtype)).lower() == kv_dtype
    )
    if attested:
        return {
            "required": True,
            "passed": True,
            "state": "externally_attested",
            "parameter": "kv_cache_dtype",
            "value": kv_dtype,
            "evaluation_dataset": quality.get("evaluation_dataset"),
            "evaluation_dataset_sha256": dataset_sha256,
            "max_regression_pct": maximum_regression,
            "attestation_path": str(Path(attestation_path).expanduser()),
            "attestation_sha256": attestation_sha256,
            "attestation": deepcopy(attestation),
            "reason": "external quality evidence explicitly approves this KV-cache precision",
        }
    return {
        "required": True,
        "passed": False,
        "state": "quality_unverified",
        "parameter": "kv_cache_dtype",
        "value": kv_dtype,
        "evaluation_dataset": quality.get("evaluation_dataset"),
        "evaluation_dataset_sha256": dataset_sha256,
        "max_regression_pct": quality.get("max_regression_pct"),
        "attestation_path": attestation_path,
        "attestation_sha256": attestation_sha256,
        "attestation_error": attestation_error,
        "reason": (
            "performance confirmation does not validate model-output quality; provide a valid "
            "external quality attestation before authorizing deployment"
        ),
    }


def cost_per_token_summary(
    task: dict[str, Any], decision: dict[str, Any], fallback_gpu_count: int,
    experiment_gpu_hours: float,
) -> dict[str, Any]:
    economics = task.get("economics", {}) if isinstance(task.get("economics"), dict) else {}
    rate = economics.get("cost_per_gpu_hour")
    currency = str(economics.get("currency", "USD"))
    if not isinstance(rate, (int, float)) or rate <= 0:
        return {
            "available": False,
            "currency": currency,
            "reason": "set economics.cost_per_gpu_hour to compute serving cost",
        }

    def costs(metrics: dict[str, Any], gpu_count: int) -> dict[str, Any]:
        hourly_cost = float(rate) * gpu_count
        output_tps = metrics.get("output_throughput_tps")
        total_tps = metrics.get("total_throughput_tps")
        error_rate = float(metrics.get("error_rate", 0) or 0)

        def per_million(value: Any) -> float | None:
            return (
                hourly_cost * 1_000_000 / (float(value) * 3600)
                if isinstance(value, (int, float)) and value > 0 else None
            )

        valid_output_tps = (
            float(output_tps) * max(0.0, 1.0 - error_rate)
            if isinstance(output_tps, (int, float)) else None
        )
        return {
            "hourly_gpu_cost": hourly_cost,
            "cost_per_million_output_tokens": per_million(output_tps),
            "cost_per_million_total_tokens": per_million(total_tps),
            "cost_per_million_slo_valid_output_tokens": per_million(valid_output_tps),
            "output_tokens_per_hour": (
                float(output_tps) * 3600 if isinstance(output_tps, (int, float)) else None
            ),
            "total_tokens_per_hour": (
                float(total_tps) * 3600 if isinstance(total_tps, (int, float)) else None
            ),
        }

    aggregates = decision.get("aggregates", []) if isinstance(decision, dict) else []
    baseline = next((item for item in aggregates if item.get("kind") == "baseline"), None)
    winner = decision.get("recommended_configuration") if isinstance(decision, dict) else None
    def reported_gpu_count(item: Any) -> int:
        if not isinstance(item, dict):
            return fallback_gpu_count
        resources = item.get("resources")
        config = item.get("config")
        if isinstance(resources, dict) and isinstance(
            resources.get("accelerator_count"), int
        ):
            return summary_accelerator_count(item)
        if isinstance(config, dict) and any(
            key in config for key in ("tp_size", "pp_size", "dp_size")
        ):
            return summary_accelerator_count(item)
        return fallback_gpu_count

    baseline_gpu_count = reported_gpu_count(baseline)
    winner_gpu_count = reported_gpu_count(winner)
    baseline_cost = (
        costs(baseline.get("metrics", {}), baseline_gpu_count)
        if isinstance(baseline, dict) else None
    )
    winner_cost = (
        costs(winner.get("metrics", {}), winner_gpu_count)
        if isinstance(winner, dict) else None
    )
    savings = None
    if baseline_cost and winner_cost:
        old = baseline_cost["cost_per_million_total_tokens"]
        new = winner_cost["cost_per_million_total_tokens"]
        if isinstance(old, (int, float)) and isinstance(new, (int, float)) and old > 0:
            savings = {
                "cost_per_million_total_tokens": old - new,
                "relative_pct": (old - new) / old * 100,
            }
    return {
        "available": True,
        "currency": currency,
        "cost_per_gpu_hour": float(rate),
        "baseline_gpu_count": baseline_gpu_count,
        "winner_gpu_count": winner_gpu_count,
        "experiment_cost": experiment_gpu_hours * float(rate),
        "baseline": baseline_cost,
        "winner": winner_cost,
        "savings": savings,
        "policy": "user-supplied GPU-hour price; no cloud price is inferred from GPU name",
    }


def effective_confirmation_repetitions(task: dict[str, Any]) -> int:
    """Return the explicit confirmation contract (new tasks default to two)."""
    return int(task.get("confirmation_repetitions", 2))


def task_at_calibrated_concurrency(
    task: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    """Return the workload contract used by every post-calibration stage."""
    calibrated = deepcopy(task)
    if task.get("slo"):
        selected = calibration.get("selected_analysis_concurrency")
        if isinstance(selected, int) and selected > 0:
            calibrated["workload"]["max_concurrency"] = selected
            if has_p99_latency_slo(task):
                measurement = calibrated.get("measurement") or {}
                waves = int(measurement.get("p99_request_waves", 10))
                calibrated["measurement"] = {
                    **measurement,
                    "confirmation_requests": selected * waves,
                }
    return calibrated


def screening_spec(
    task: dict[str, Any], discovery: dict[str, Any], search_plan: dict[str, Any],
    remaining_gpu_hours: float | None = None, remaining_wall_minutes: float | None = None,
    remaining_trials: int | None = None, baseline: dict[str, Any] | None = None,
    confirmation_reserve_trials: int | None = None, anchor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_trials = int(task["budget"]["max_trials"]) if remaining_trials is None else remaining_trials
    budget_allocation = search_plan.get("budget_allocation") or task_trial_budget(task)
    confirmation_reserve = (
        int(budget_allocation["planned"]["confirmation"])
        if confirmation_reserve_trials is None
        else confirmation_reserve_trials
    )
    # Reserve a compact interaction pass. Single-parameter effects below the
    # practical threshold can be real but only become deployable when a
    # compatible combination is measured and then confirmed.
    # A one-factor screen is only a routing stage. Reserve at least one
    # post-screen composition in ordinary balanced runs, otherwise several
    # individually small but compatible gains can never become a deployment
    # candidate. Thorough mode keeps room for a second composition.
    mode_name = normalized_experiment_mode(task)
    mode_candidate_limit = MODE_CANDIDATE_LIMITS.get(mode_name, 12)
    # Max mode spends every currently available screening slot on coverage.
    # Interaction trials are allocated only after the atomic/coarse screen has
    # produced compatible positive seeds; reserving them up front previously
    # truncated second chunk/memory values without ever running a combination.
    desired_interactions = (
        int(budget_allocation["planned"]["refinement"])
        if confirmation_reserve_trials is None else 0
    )
    # Coverage takes precedence over speculative combination work. A balanced
    # run must first measure its baseline plus six distinct high-impact
    # mechanisms; otherwise an empty interaction reserve can silently reduce
    # the useful search to only a couple of flags.
    minimum_screen_trials = 1 + min(
        required_mechanism_coverage(task), mode_candidate_limit
    )
    interaction_reserve = min(
        desired_interactions,
        max(0, total_trials - confirmation_reserve - minimum_screen_trials),
    )
    screening_trials = max(1, total_trials - confirmation_reserve - interaction_reserve)
    tp_size = discovery["derived"]["minimum_tp_size"]
    compatibility = runtime_compatibility_constraints(discovery)
    baseline_config = {
        "tp_size": tp_size, **(baseline or {}), **compatibility["required_config"]
    }
    anchor_config = {
        **baseline_config, **(anchor or {}), **compatibility["required_config"]
    }
    effective_config = {
        parameter: value
        for parameter, value in search_plan.get("resolved_baseline", {}).items()
        if value is not None
    }
    candidate_budget = max(0, screening_trials - 1)
    minimum_successes_before_early_stop = {
        "fast": 3,
        "balanced": 6,
        "max": 12,
    }.get(mode_name, 6)
    candidate_budget = min(candidate_budget, mode_candidate_limit)
    bundles = [
        *search_plan.get("cookbook_candidate_bundles", []),
        *search_plan.get("ranked_configuration_bundles", []),
    ]
    valid_bundles = [
        bundle for bundle in bundles
        if isinstance(bundle, dict) and isinstance(bundle.get("config"), dict)
        and not str(bundle.get("name", "")).startswith("history-")
    ]
    initial_cookbook_phase = search_plan.get("phase") == "cookbook_initialization"
    if initial_cookbook_phase:
        # The initial stage establishes hardware topology and model-native
        # capabilities. Keep a compatible MTP/speculative bundle ahead of
        # cache-only bundles; otherwise a long-prompt core screen can spend
        # every slot on generic scheduler controls without ever testing the
        # checkpoint capability that motivated the cookbook lookup.
        valid_bundles.sort(
            key=lambda bundle: 0 if "speculative_algorithm" in bundle["config"] else 1
        )
    history_quota = 0
    priority_bundles = [
        bundle for bundle in valid_bundles
        if bundle.get("priority") == "high"
    ]
    strong_parameters = search_plan.get("trigger_rule_plan", {}).get(
        "strong_candidates", []
    )
    strong_index = {parameter: index for index, parameter in enumerate(strong_parameters)}
    priority_bundles.sort(key=lambda bundle: min(
        (strong_index.get(parameter, len(strong_index) + 1) for parameter in bundle["config"]),
        default=len(strong_index) + 1,
    ))
    for mechanism in ("mtp", "mamba"):
        representative = next((
            bundle for bundle in valid_bundles
            if bundle not in priority_bundles
            and (
                mechanism == "mtp" and "speculative_algorithm" in bundle.get("config", {})
                or mechanism == "mamba"
                and "speculative_algorithm" not in bundle.get("config", {})
                and any(key.startswith("mamba_") for key in bundle.get("config", {}))
            )
        ), None)
        if representative is not None:
            priority_bundles.append(representative)
    priority_bundles.sort(key=lambda bundle: min(
        (strong_index.get(parameter, len(strong_index) + 1) for parameter in bundle["config"]),
        default=len(strong_index) + 1,
    ))
    regular_bundles = [bundle for bundle in valid_bundles if bundle not in priority_bundles]
    provisional_groups = [
        item for item in search_plan.get("ranked_parameter_groups", [])
        if isinstance(item, dict) and item.get("provisional") and item.get("values")
    ]
    requested_provisional_slots = int(
        search_plan.get("parameter_evolution", {}).get(
            "exploration_budget", {}
        ).get("slots", 0) or 0
    )
    reserved_provisional_slots = min(
        candidate_budget, requested_provisional_slots, len(provisional_groups)
    )
    reserved_bundle_slots = min(candidate_budget, len(priority_bundles))
    reserved_anchor_slots = 1 if anchor_config != baseline_config else 0
    selection_budget = max(
        0, candidate_budget - reserved_bundle_slots - reserved_anchor_slots
        - reserved_provisional_slots
    )
    # Cookbook and dependent bundles are useful experiments, but they are not
    # allowed to consume the first slots. The first slots establish workload
    # coverage across the core serving controls, then trace-ranked candidates
    # and bundles compete for whatever remains.
    selected: list[tuple[str, Any]] = []
    covered_submechanisms: set[str] = set()
    ranked = search_plan["ranked_parameter_groups"]
    by_parameter = {
        item["parameter"]: item for item in ranked
        if item.get("values")
    }

    def submechanism(parameter: str) -> str:
        item = by_parameter[parameter]
        return str(item.get("submechanism") or parameter_submechanism(
            parameter, item.get("family")
        ))

    def selected_parameters() -> set[str]:
        return {parameter for parameter, _ in selected}

    def select_parameter(parameter: str, *, require_new_submechanism: bool = False) -> bool:
        """Select one executable value without treating a family as exclusive."""
        item = by_parameter.get(parameter)
        if item is None or parameter in selected_parameters() or len(selected) >= selection_budget:
            return False
        mechanism = submechanism(parameter)
        if require_new_submechanism and mechanism in covered_submechanisms:
            return False
        value = next(
            (
                candidate for candidate in item["values"]
                if candidate_differs_from_effective_baseline(
                    parameter, candidate, anchor_config, effective_config
                )
            ),
            None,
        )
        if value is None:
            return False
        selected.append((parameter, value))
        covered_submechanisms.add(mechanism)
        return True

    if initial_cookbook_phase:
        topology = by_parameter.get("tp_size")
        if topology is not None:
            for value in topology["values"]:
                if len(selected) >= selection_budget:
                    break
                if candidate_differs_from_effective_baseline(
                    "tp_size", value, anchor_config, effective_config
                ):
                    selected.append(("tp_size", value))
                    covered_submechanisms.add(submechanism("tp_size"))
    # Capacity controls are mandatory for offline no-SLO optimization.  A
    # fixed candidate budget must not silently turn a throughput claim into a
    # test of scheduler defaults only.
    mandatory_capacity = (
        ("mem_fraction_static", "max_prefill_tokens")
        if reference_baseline_mode(task)
        else ()
    )
    mandatory_model_mechanisms: list[str] = []
    if discovery["model"].get("is_moe"):
        mandatory_model_mechanisms.append("moe_runner_backend")
    attention_share = float(
        search_plan.get("profiler_evidence", {}).get("shares_pct", {}).get("attention_kernels", 0) or 0
    )
    if attention_share >= 20:
        mandatory_model_mechanisms.append(
            "prefill_attention_backend"
            if int(task["workload"]["input_tokens"]) >= int(task["workload"]["output_tokens"])
            else "decode_attention_backend"
        )
    for parameter in (*mandatory_capacity, *mandatory_model_mechanisms):
        select_parameter(parameter)
    priority_order = [
        parameter for parameter in core_serving_parameter_order(
            task, discovery, search_plan
        )
        if not by_parameter.get(parameter, {}).get("provisional")
    ]
    # First establish breadth across causal sub-mechanisms rather than coarse
    # ServerArgs families.  Then reserve distinct-parameter evidence for every
    # high-magnitude trigger, so one memory/cache knob cannot stand in for all
    # other controls in that family. Remaining slots follow global priority.
    breadth_targets = {"fast": 3, "balanced": 10, "max": 20}
    breadth_budget = max(
        len(selected), min(selection_budget, breadth_targets.get(mode_name, minimum_successes_before_early_stop))
    )
    mechanism_coverage_target = min(
        breadth_budget, required_mechanism_coverage(task)
    )
    for parameter in priority_order:
        if len(covered_submechanisms) >= mechanism_coverage_target:
            break
        if select_parameter(parameter, require_new_submechanism=True) and parameter == "ep_size":
            # A legal EP degree is a topology choice, not a low-priority
            # scalar sensitivity point.  Test every mathematically legal
            # degree before spending the last slots on secondary scheduler
            # values; EP=2 does not establish the behavior of EP=4.
            item = by_parameter[parameter]
            for additional in item["values"]:
                if len(selected) >= breadth_budget:
                    break
                pair = (parameter, additional)
                if (
                    candidate_differs_from_effective_baseline(
                        parameter, additional, anchor_config, effective_config
                    )
                    and pair not in selected
                ):
                    selected.append(pair)

    high_rule_floor = {
        "fast": 1, "balanced": 2, "max": 1_000_000,
    }.get(mode_name, 2)
    high_rule_coverage: dict[str, list[str]] = {}
    trigger_matches = search_plan.get("trigger_rule_plan", {}).get("matches", [])
    for rule in trigger_matches:
        if rule.get("magnitude") != "high" or len(selected) >= breadth_budget:
            continue
        eligible = [
            parameter for parameter in rule.get("parameters", [])
            if parameter in by_parameter
        ]
        target = min(len(eligible), high_rule_floor)
        for parameter in eligible:
            already = [name for name in eligible if name in selected_parameters()]
            if len(already) >= target or len(selected) >= breadth_budget:
                break
            select_parameter(parameter)
        high_rule_coverage[str(rule.get("id"))] = [
            parameter for parameter in eligible if parameter in selected_parameters()
        ]

    for parameter in priority_order:
        if len(selected) >= breadth_budget:
            break
        select_parameter(parameter)
    for parameter in priority_order:
        item = by_parameter[parameter]
        for value in item["values"]:
            if len(selected) >= selection_budget:
                break
            pair = (item["parameter"], value)
            if (
                candidate_differs_from_effective_baseline(
                    item["parameter"], value, anchor_config, effective_config
                )
                and pair not in selected
            ):
                selected.append(pair)
        if len(selected) >= selection_budget:
            break
    configurations: list[dict[str, Any]] = []
    # The profile may use a fast, SLO-valid seed from the Cookbook stage, but
    # it must re-enter the target-workload screen as a candidate against the
    # original SGLang-default launch. It is never silently promoted to the
    # final baseline on one exploratory observation.
    if anchor_config != baseline_config:
        configurations.append({
            "name": "preprofile-seed",
            "config": anchor_config,
        })
    configurations.extend(
        {
            "name": bundle["name"],
            "config": {**anchor_config, **bundle["config"]},
            **({"env": bundle["env"]} if isinstance(bundle.get("env"), dict) and bundle["env"] else {}),
        }
        for bundle in priority_bundles[:reserved_bundle_slots]
        if (
            bundle.get("env")
            or candidate_config_differs_from_effective_baseline(
                bundle["config"], anchor_config, effective_config
            )
        )
    )
    provisional_configurations: list[dict[str, Any]] = []
    for item in provisional_groups:
        if len(provisional_configurations) >= reserved_provisional_slots:
            break
        value = next(
            (
                candidate for candidate in item["values"]
                if candidate_differs_from_effective_baseline(
                    item["parameter"], candidate, anchor_config, effective_config
                )
            ),
            None,
        )
        if value is None:
            continue
        provisional_configurations.append({
            "name": f"provisional-{item['parameter']}-{str(value).lower()}"[:96],
            "config": {
                **anchor_config,
                **item.get("parameter_evolution", {}).get("atomic_config", {}),
                item["parameter"]: value,
            },
            "provisional_parameter": item["parameter"],
            "provisional_state": "provisional",
            "provisional_atomic_config": deepcopy(
                item.get("parameter_evolution", {}).get("atomic_config", {})
            ),
        })
    known_before_provisional = (
        min(
            len(selected),
            max(1, required_mechanism_coverage(task) - 1),
        )
        if provisional_configurations else len(selected)
    )
    for parameter, value in selected[:known_before_provisional]:
        if len(configurations) >= candidate_budget:
            break
        configurations.append({
            "name": f"{parameter}-{str(value).lower()}"[:96],
            "config": {**anchor_config, parameter: value},
        })
    configurations.extend(provisional_configurations)
    for parameter, value in selected[known_before_provisional:]:
        if len(configurations) >= candidate_budget:
            break
        configurations.append({
            "name": f"{parameter}-{str(value).lower()}"[:96],
            "config": {**anchor_config, parameter: value},
        })
    # History is a warm start, not a substitute for current-model discovery.
    # Keep at least half the available candidate slots for novel local evidence.
    for bundle in regular_bundles:
        if len(configurations) >= candidate_budget:
            break
        if not (
            bundle.get("env")
            or candidate_config_differs_from_effective_baseline(
                bundle["config"], anchor_config, effective_config
            )
        ):
            continue
        configurations.append({
            "name": bundle["name"],
            "config": {**anchor_config, **bundle["config"]},
            **({"env": bundle["env"]} if isinstance(bundle.get("env"), dict) and bundle["env"] else {}),
        })
    # Configurations can overlap when a Cookbook bundle and a workload delta
    # resolve to the same launch. Deduplicate here rather than paying for two
    # complete server restarts.
    unique_configurations: list[dict[str, Any]] = []
    seen_configurations: set[str] = set()
    baseline_signature = json.dumps({"config": baseline_config, "env": {}}, sort_keys=True)
    for configuration in configurations:
        signature = json.dumps(
            {"config": configuration["config"], "env": configuration.get("env", {})},
            sort_keys=True,
        )
        if signature == baseline_signature or signature in seen_configurations:
            continue
        seen_configurations.add(signature)
        unique_configurations.append(configuration)
    configurations = unique_configurations[:candidate_budget]
    # Cookbook/topology screening and the post-profile screen use the same
    # target workload.  A configuration that completed in the former is
    # evidence, not a new parameter candidate.  Keep the current baseline
    # measurement for comparability, but avoid restarting a server merely to
    # re-measure an identical non-baseline configuration such as TP=2 after
    # TP=2/TP=4 were already compared during cookbook initialization.
    prior_configurations = {
        json.dumps(config, sort_keys=True)
        for config in search_plan.get("previously_evaluated_configurations", [])
        if isinstance(config, dict)
    }
    if prior_configurations:
        configurations = [
            item for item in configurations
            if item.get("env") or json.dumps(item["config"], sort_keys=True) not in prior_configurations
        ]
    registry_schedule: dict[str, Any] | None = None
    registry_value = search_plan.get("candidate_registry")
    if isinstance(registry_value, dict) and registry_value.get("candidates"):
        mandatory_registry_mechanisms: list[str] = []
        compatible_cookbook_bundles = [
            bundle for bundle in search_plan.get("cookbook_candidate_bundles", [])
            if isinstance(bundle, dict) and isinstance(bundle.get("config"), dict)
        ]
        if (
            search_plan.get("mtp_relevance", {}).get("relevant", True)
            and any(
                "speculative_algorithm" in bundle["config"]
                for bundle in compatible_cookbook_bundles
            )
        ):
            mandatory_registry_mechanisms.append("speculative_decoding")
        if any(
            "mamba_radix_cache_strategy" in bundle["config"]
            and "speculative_algorithm" not in bundle["config"]
            for bundle in compatible_cookbook_bundles
        ):
            mandatory_registry_mechanisms.append("state_space_cache")
        # High-confidence semantic matches come from the live ServerArgs
        # contract and current (hardware, model, workload, SLO, bottleneck)
        # context.  Reserve a bounded number of mechanism slots so newly
        # understood controls such as a cache tier are actually measured,
        # instead of merely appearing as eligible in the final report.
        mandatory_registry_mechanisms.extend(
            contextual_semantic_mechanisms(
                registry_value,
                limit={"fast": 1, "balanced": 2, "max": 4}.get(mode_name, 2),
            )
        )
        # A legal multi-GPU topology is a materially different deployment,
        # not a scalar sensitivity point.  Measure one representative whenever
        # multiple selected GPUs are available; resource-adjusted comparison
        # still prevents a wider topology from winning on raw throughput alone.
        available_registry_mechanisms = {
            normalize_mechanism(item.get("mechanism"))
            for item in registry_value.get("candidates", [])
            if item.get("state") == "eligible"
        }
        if (
            len(discovery.get("hardware", {}).get("gpus", [])) > 1
            and "parallel_topology" in available_registry_mechanisms
        ):
            mandatory_registry_mechanisms.append("parallel_topology")
        mandatory_registry_mechanisms = list(dict.fromkeys(
            normalize_mechanism(value) for value in mandatory_registry_mechanisms
        ))
        registry_schedule = initial_mechanism_schedule(
            registry_value,
            budget=candidate_budget,
            mandatory_parameters=[
                *mandatory_capacity, *mandatory_model_mechanisms,
            ],
            mandatory_mechanisms=mandatory_registry_mechanisms,
            provisional_slots=reserved_provisional_slots,
            breadth_target=required_mechanism_coverage(task),
            max_values_per_parameter={
                "fast": 1, "balanced": 2, "max": 3,
            }.get(mode_name, 2),
        )
        registry = CandidateRegistry.from_dict(registry_value)
        registry_configurations: list[dict[str, Any]] = []
        if anchor_config != baseline_config:
            registry_configurations.append({
                "name": "preprofile-seed", "config": anchor_config,
                "registry_candidate_id": None,
            })
        for candidate in registry_schedule["selected"]:
            full_config = {**anchor_config, **candidate.get("config_delta", {})}
            candidate_env = deepcopy(candidate.get("env_delta", {}))
            if not candidate_env and not candidate_config_differs_from_effective_baseline(
                candidate.get("config_delta", {}), anchor_config, effective_config
            ):
                registry.transition(
                    candidate["candidate_id"], "already_effective",
                    "candidate equals the current effective SGLang baseline",
                )
                continue
            name = str(candidate.get("name", "candidate"))
            if candidate.get("risk", {}).get("provisional"):
                name = "provisional-" + name
            registry_configurations.append({
                "name": name[:96],
                "config": full_config,
                **({"env": candidate_env} if candidate_env else {}),
                "registry_candidate_id": candidate["candidate_id"],
                "registry_mechanism": candidate.get("mechanism"),
                **(
                    {"provisional_parameter": candidate.get("parameter")}
                    if candidate.get("risk", {}).get("provisional") else {}
                ),
                **(
                    {"provisional_state": "provisional"}
                    if candidate.get("risk", {}).get("provisional") else {}
                ),
            })
        deduped_registry_configurations: list[dict[str, Any]] = []
        seen_registry_configurations: set[str] = set()
        for configuration in registry_configurations:
            signature = json.dumps(
                {
                    "config": configuration["config"],
                    "env": configuration.get("env", {}),
                }, sort_keys=True,
            )
            if signature == baseline_signature or signature in seen_registry_configurations:
                continue
            seen_registry_configurations.add(signature)
            deduped_registry_configurations.append(configuration)
        configurations = deduped_registry_configurations[:candidate_budget]
        scheduled_ids = [
            item["registry_candidate_id"] for item in configurations
            if isinstance(item.get("registry_candidate_id"), str)
        ]
        for candidate_id in scheduled_ids:
            registry.mark_scheduled(candidate_id, "screening")
        scheduled_mechanisms = sorted({
            normalize_mechanism(item.get("registry_mechanism"))
            for item in configurations if item.get("registry_candidate_id")
        })
        registry_schedule["scheduled_candidate_ids"] = scheduled_ids
        registry_schedule["scheduled_mechanisms"] = scheduled_mechanisms
        registry_schedule["covered_mechanisms"] = scheduled_mechanisms
        search_plan["candidate_registry"] = registry.to_dict()
        search_plan["mechanism_schedule"] = registry_schedule
    provisional_coverage_floor = max(
        (
            index + 1 for index, item in enumerate(configurations)
            if str(item.get("name", "")).startswith("provisional-")
        ),
        default=0,
    )
    if configurations:
        ranked_by_parameter = {
            item["parameter"]: item for item in ranked
            if isinstance(item, dict) and item.get("parameter")
        }
        priority_by_parameter = {
            item["parameter"]: item
            for item in search_plan.get("parameter_priority_scores", [])
            if isinstance(item, dict) and item.get("parameter")
        }
        selection_evidence: list[dict[str, Any]] = []
        for configuration in configurations:
            changed = {
                key: value for key, value in configuration["config"].items()
                if anchor_config.get(key) != value and key != "tp_size"
            }
            if len(changed) == 1:
                parameter, value = next(iter(changed.items()))
                ranked_item = ranked_by_parameter.get(parameter, {})
                priority = priority_by_parameter.get(parameter, {})
                selection_evidence.append({
                    "name": configuration["name"],
                    "parameter": parameter,
                    "value": value,
                    "family": ranked_item.get("family", "unknown"),
                    "submechanism": ranked_item.get("submechanism") or parameter_submechanism(
                        parameter, ranked_item.get("family")
                    ),
                    "tiers": ranked_item.get("tiers", []),
                    "priority_score": priority.get("score"),
                    "trigger_magnitude": ranked_item.get("trigger_magnitude"),
                    "trigger_rule_ids": ranked_item.get("trigger", {}).get("rule_ids", []),
                    "provisional": bool(ranked_item.get("provisional")),
                    "reason": ranked_item.get("reason"),
                    "evidence": ranked_item.get("evidence", []),
                })
            else:
                selection_evidence.append({
                    "name": configuration["name"],
                    "parameters": sorted(changed),
                    "family": "bundle",
                    "provisional": str(configuration.get("name", "")).startswith(
                        "provisional-"
                    ),
                    "provisional_parameter": configuration.get(
                        "provisional_parameter"
                    ),
                    "reason": (
                        "provisional parameter with current-Cookbook compatible dependencies"
                        if str(configuration.get("name", "")).startswith("provisional-")
                        else "compatible model/Cookbook or dependent configuration bundle"
                    ),
                })
        spec = explicit_configuration_spec(
            task, discovery,
            stage_name="screen",
            baseline=baseline_config,
            configurations=configurations,
            max_trials=1 + len(configurations),
            repetitions=1,
            remaining_gpu_hours=float(task["budget"]["max_gpu_hours"]) if remaining_gpu_hours is None else remaining_gpu_hours,
            remaining_wall_minutes=float(task["budget"]["max_wall_time_minutes"]) if remaining_wall_minutes is None else remaining_wall_minutes,
        )
        spec["search"].update({
            "candidate_limit": mode_candidate_limit,
            "selection_policy": (
                "cover causal sub-mechanisms first; retain multiple distinct parameters from "
                "high-magnitude trigger rules; then refine values without family-level exclusion"
            ),
            "budget_allocation": deepcopy(budget_allocation),
            "selected_parameter_candidates": [item["name"] for item in configurations],
            "history_candidate_quota": history_quota,
            "history_candidates_selected": [
                item["name"] for item in configurations if item["name"].startswith("history-")
            ],
            "provisional_parameter_candidates": [
                item["name"] for item in configurations
                if item["name"].startswith("provisional-")
            ],
            "provisional_parameter_names": [
                item["provisional_parameter"] for item in provisional_configurations
            ],
            "provisional_exploration_budget": {
                "requested_slots": requested_provisional_slots,
                "reserved_slots": reserved_provisional_slots,
                "selected_slots": len(provisional_configurations),
                "policy": "reserved from discovery candidates; confirmation budget is untouched",
            },
            "mandatory_mechanism_parameters": [
                parameter for parameter in (*mandatory_capacity, *mandatory_model_mechanisms)
                if parameter in by_parameter
            ],
            "mandatory_registry_mechanisms": deepcopy(
                registry_schedule.get("mandatory_mechanisms", [])
                if isinstance(registry_schedule, dict) else []
            ),
            "mechanism_coverage_target": mechanism_coverage_target,
            "covered_submechanisms": sorted(covered_submechanisms),
            "high_magnitude_rule_parameter_floor": (
                "all" if mode_name == "max" else high_rule_floor
            ),
            "high_magnitude_rule_coverage": high_rule_coverage,
            "deferred_triggered_parameters": [
                {
                    "parameter": parameter,
                    "submechanism": submechanism(parameter),
                    "rule_ids": by_parameter[parameter].get("trigger", {}).get("rule_ids", []),
                    "reason": "candidate budget exhausted after sub-mechanism and high-trigger coverage",
                }
                for parameter in priority_order if parameter not in selected_parameters()
            ],
            "selection_evidence": selection_evidence,
            "mechanism_schedule": deepcopy(registry_schedule),
            "required_mechanism_coverage": (
                registry_schedule.get("covered_mechanisms", [])
                if isinstance(registry_schedule, dict) else []
            ),
        })
        if confirmation_reserve_trials is None and reference_baseline_mode(task):
            configure_offline_reference_window(spec, task)
        if confirmation_reserve_trials is None and mode_name != "max":
            required_early_stop_parameters = {
                parameter for parameter in (
                    *mandatory_capacity,
                    *mandatory_model_mechanisms,
                    *(
                        name for names in high_rule_coverage.values()
                        for name in names
                    ),
                )
            }
            coverage_positions = [
                index + 1
                for index, configuration in enumerate(configurations)
                if required_early_stop_parameters.intersection(
                    key for key, value in configuration.get("config", {}).items()
                    if anchor_config.get(key) != value
                )
            ]
            early_stop_coverage_floor = max(
                [provisional_coverage_floor, *coverage_positions], default=0
            )
            spec["search"].update({
                "min_successful_candidates_before_early_stop": min(
                    max(
                        minimum_successes_before_early_stop,
                        early_stop_coverage_floor,
                    ),
                    len(configurations),
                ),
                "early_stop_coverage_floor": early_stop_coverage_floor,
                "early_stop_improvement_pct": 3.0,
            })
        return spec
    space: dict[str, list[Any]] = {}
    for parameter, value in selected:
        space.setdefault(parameter, []).append(value)
    candidate_count = sum(len(values) for values in space.values())
    max_trials = min(screening_trials, 1 + candidate_count)
    spec = build_execution_spec(
        task,
        discovery,
        stage_name="screen",
        baseline=baseline_config,
        space=space,
        max_trials=max_trials,
        repetitions=1,
        remaining_gpu_hours=float(task["budget"]["max_gpu_hours"]) if remaining_gpu_hours is None else remaining_gpu_hours,
        remaining_wall_minutes=float(task["budget"]["max_wall_time_minutes"]) if remaining_wall_minutes is None else remaining_wall_minutes,
    )
    spec["search"].update({
        "candidate_limit": mode_candidate_limit,
        "selection_policy": (
            "cover the highest expected-impact workload/trace mechanism first; test one "
            "representative value per parameter before spending restarts on local refinement"
        ),
        "selected_parameter_candidates": [
            f"{parameter}-{str(value).lower()}" for parameter, value in selected
        ],
    })
    if confirmation_reserve_trials is None and reference_baseline_mode(task):
        configure_offline_reference_window(spec, task)
    if confirmation_reserve_trials is None and candidate_count > 0 and mode_name != "max":
        spec["search"].update({
            "min_successful_candidates_before_early_stop": min(
                minimum_successes_before_early_stop, candidate_count
            ),
            "early_stop_improvement_pct": 3.0,
        })
    return spec


def initial_cookbook_trial_budget(task: dict[str, Any], completed_calibration_trials: int) -> int:
    """Limit exploratory Cookbook trials so profiling and real parameter tuning remain possible."""
    confirmation_reserve = confirmation_trial_reserve(task)
    profile_reserve = 1
    desired_candidates = {
        "fast": 4,
        "balanced": 6,
        "max": 10,
    }.get(normalized_experiment_mode(task), 6)
    available_before_optional_interaction = (
        int(task["budget"]["max_trials"])
        - completed_calibration_trials
        - profile_reserve
        - confirmation_reserve
    )
    desired_interactions = (
        0 if normalized_experiment_mode(task) == "max"
        else 3 if task.get("search_depth", "thorough") == "thorough" else 2
    )
    interaction_reserve = min(
        desired_interactions,
        max(0, available_before_optional_interaction - (1 + desired_candidates) - 2),
    )
    available_after_fixed = available_before_optional_interaction - interaction_reserve
    # Reserve a baseline plus enough profiler-directed parameters to make the
    # expensive trace useful, while retaining two slots for a baseline and at
    # least one model-native Cookbook candidate on tighter custom budgets.
    parameter_screen_reserve = min(
        1 + desired_candidates,
        max(4, available_after_fixed - 2),
    )
    return max(
        0,
        available_after_fixed
        - parameter_screen_reserve
    )


def explicit_configuration_spec(
    task: dict[str, Any],
    discovery: dict[str, Any],
    *,
    stage_name: str,
    baseline: dict[str, Any],
    configurations: list[dict[str, Any]],
    max_trials: int,
    repetitions: int,
    remaining_gpu_hours: float,
    remaining_wall_minutes: float,
    include_baseline: bool = True,
    reference_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required_config = runtime_compatibility_constraints(discovery)["required_config"]
    configurations = [
        {**item, "config": {**item["config"], **required_config}}
        for item in configurations
    ]
    spec = build_execution_spec(
        task,
        discovery,
        stage_name=stage_name,
        baseline=baseline,
        space={},
        max_trials=max_trials,
        repetitions=repetitions,
        remaining_gpu_hours=remaining_gpu_hours,
        remaining_wall_minutes=remaining_wall_minutes,
    )
    spec["search"].update({
        "strategy": "explicit_configurations",
        "explicit_configurations": configurations,
        "parameter_order": [],
        "include_baseline": include_baseline,
        **({"reference_baseline": deepcopy(reference_baseline)} if reference_baseline is not None else {}),
    })
    return spec


def single_gpu_preprofile_spec(spec: dict[str, Any], task: dict[str, Any]) -> dict[str, Any] | None:
    """Keep only independent candidates suitable for the spare-GPU pipeline."""
    matrix = candidate_matrix(spec)
    baseline = next((item for item in matrix if item["kind"] == "baseline"), None)
    candidates = [
        {
            "name": item["name"],
            "config": deepcopy(item["config"]),
            **({"env": deepcopy(item["env"])} if item.get("env") else {}),
        }
        for item in matrix
        if item["kind"] == "candidate"
        and configuration_accelerator_count(spec, item["config"]) == 1
    ]
    if baseline is None or not candidates:
        return None
    filtered = deepcopy(spec)
    filtered["search"].update({
        "strategy": "explicit_configurations",
        "space": {},
        "parameter_order": [],
        "explicit_configurations": candidates,
        "selected_parameter_candidates": [item["name"] for item in candidates],
    })
    filtered["budget"]["max_trials"] = 1 + len(candidates)
    if reference_baseline_mode(task):
        configure_offline_reference_window(filtered, task)
    return filtered


def measured_reference_baseline(result: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the preserved baseline used by later candidate-only screens."""
    row = next(
        (item for item in result.get("results", []) if item.get("kind") == "baseline" and item.get("ok")),
        None,
    )
    if row is None:
        return None
    reference = row.get("confirmation_reference")
    metrics = reference.get("metrics") if isinstance(reference, dict) else row.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None
    return {
        "config": deepcopy(row.get("config", {})),
        "env": deepcopy(row.get("env", {})),
        "resources": deepcopy(row.get("resources", {})),
        "metrics": deepcopy(metrics),
        "source_run_dir": result.get("run_dir"),
        "source": "parallel_preprofile_baseline",
    }


def apply_reference_baseline(spec: dict[str, Any], reference: dict[str, Any]) -> None:
    """Convert a screen to candidate-only execution against measured evidence."""
    spec["search"].update({
        "include_baseline": False,
        "reference_baseline": deepcopy(reference),
        "min_confirm_repetitions": 1,
    })
    spec["budget"]["max_trials"] = max(1, int(spec["budget"]["max_trials"]) - 1)
    spec["benchmark"].pop("baseline_reference_num_prompts", None)
    spec["benchmark"].pop("baseline_reference_min_measurement_seconds", None)


def merge_screening_evidence(
    spec: dict[str, Any], current: dict[str, Any], prior: dict[str, Any],
) -> dict[str, Any]:
    """Re-rank unique candidate evidence from concurrent and trace-routed screens."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in (prior, current):
        for row in source.get("results", []):
            if row.get("kind") != "candidate":
                continue
            signature = json.dumps(
                {"config": row.get("config", {}), "env": row.get("env", {})}, sort_keys=True
            )
            if signature in seen:
                continue
            seen.add(signature)
            rows.append(deepcopy(row))
    decision = decision_report(spec, rows)
    merged = deepcopy(current)
    merged.update(decision)
    merged["results"] = rows
    merged["evidence_sources"] = [prior.get("run_dir"), current.get("run_dir")]
    merged["evidence_candidate_count"] = len(rows)
    return merged


def fastest_slo_valid_configuration(result: dict[str, Any], objective: dict[str, Any]) -> dict[str, Any] | None:
    """Select a representative profile baseline without applying final decision gates.

    The pre-profile run is deliberately an exploratory coarse screen. It must
    profile the best observed SLO-valid configuration even when a one-sample
    comparison cannot yet clear the final minimum-improvement threshold.
    """
    metric = objective["metric"]
    maximize = objective["direction"] == "maximize"
    resource_scope = objective.get("resource_scope", "per_service")
    eligible = [
        item for item in result.get("aggregates", [])
        if item.get("all_repetitions_slo_passed")
        and resource_adjusted_metric_value(item, metric, resource_scope) is not None
    ]
    if not eligible:
        return None
    return deepcopy(sorted(
        eligible,
        key=lambda item: float(
            resource_adjusted_metric_value(item, metric, resource_scope)
        ),
        reverse=maximize,
    )[0]["config"])


def interaction_spec(
    task: dict[str, Any],
    discovery: dict[str, Any],
    search_plan: dict[str, Any],
    screen: dict[str, Any],
    remaining_trials: int,
    remaining_gpu_hours: float,
    remaining_wall_minutes: float,
    phase: str = "combined",
    candidate_slot_limit: int | None = None,
) -> dict[str, Any] | None:
    """Measure refinement first, then compositions from refined evidence."""
    if phase not in {"combined", "refinement", "composition"}:
        raise ValueError("interaction phase must be combined, refinement, or composition")
    baseline = deepcopy(screen["aggregates"][0]["config"])
    threshold_seeds = [
        item for item in screen["aggregates"][1:]
        if item.get("screening_accepted") and item.get("comparison", {}).get("improvement_pct") is not None
    ]
    # Keep stable positive deltas as interaction seeds even when each one is
    # below the user-facing practical-improvement threshold. The interaction
    # itself is still benchmarked and must pass the normal confirmation gates.
    optional_seeds = [
        item for item in screen["aggregates"][1:]
        if item not in threshold_seeds
        and item.get("stable")
        and item.get("all_repetitions_slo_passed")
        and item.get("comparison", {}).get("secondary_regressions_passed")
        and (item.get("comparison", {}).get("improvement_pct") or 0) > 0
    ]
    threshold_seeds.sort(key=lambda item: item["comparison"]["improvement_pct"], reverse=True)
    optional_seeds.sort(key=lambda item: item["comparison"]["improvement_pct"], reverse=True)
    seeds = [*threshold_seeds, *optional_seeds]

    baseline_metrics = screen["aggregates"][0].get("metrics", {})
    offline_racing_recheck = (
        reference_baseline_mode(task)
        and phase == "refinement"
        and offline_saturation_waves(task, phase="screening")
        < offline_saturation_waves(task, phase="refinement")
    )
    # The coarse offline screen uses three capacity waves. Re-run one matched
    # five-wave baseline for exact refinement promotion. Composition then
    # reuses that same-tier reference instead of loading another baseline.
    reuse_reference = bool(baseline_metrics) and not offline_racing_recheck
    budget_allocation = search_plan.get("budget_allocation") or task_trial_budget(task)
    confirmation_reserve = int(budget_allocation["planned"]["confirmation"])
    baseline_trials = 0 if reuse_reference else 1
    reclaimed_discovery = max(
        0, int(budget_allocation["planned"]["discovery"])
        - int(screen.get("completed_trials", 0))
    )
    refinement_cap = int(budget_allocation["planned"]["refinement"]) + reclaimed_discovery
    if phase == "composition":
        refinement_cap = max(0, remaining_trials - confirmation_reserve - baseline_trials)
    candidate_slots = min(
        refinement_cap,
        max(0, remaining_trials - confirmation_reserve - baseline_trials),
    )
    if isinstance(candidate_slot_limit, int) and candidate_slot_limit >= 0:
        candidate_slots = min(candidate_slots, candidate_slot_limit)
    if candidate_slots == 0:
        return None

    registry_candidates_by_id = {
        str(item.get("candidate_id")): item
        for item in search_plan.get("candidate_registry", {}).get("candidates", [])
        if isinstance(item, dict) and item.get("candidate_id")
    }

    def measured_mechanism(item: dict[str, Any]) -> str:
        candidate_id = item.get("registry_candidate_id")
        if isinstance(candidate_id, str) and candidate_id in registry_candidates_by_id:
            return normalize_mechanism(
                registry_candidates_by_id[candidate_id].get("mechanism")
            )
        changed = [
            key for key, value in item.get("config", {}).items()
            if key != "tp_size" and baseline.get(key) != value
        ]
        if len(changed) == 1:
            group = next(
                (
                    value for value in search_plan.get("ranked_parameter_groups", [])
                    if value.get("parameter") == changed[0]
                ),
                {},
            )
            return normalize_mechanism(
                group.get("submechanism")
                or parameter_submechanism(changed[0], group.get("family"))
            )
        return "configuration_bundle"

    # A short offline screen is a routing observation. Promote the exact
    # positive configuration into the longer saturation tier before exploring
    # a neighboring value. Otherwise 32768 -> 65536 was incorrectly described
    # as a refinement of the measured 32768 signal, and exact flags such as LPM
    # had no path to higher-confidence evidence at all.
    racing_rechecks: list[dict[str, Any]] = []
    if phase == "refinement" and offline_racing_recheck:
        positive = [*threshold_seeds, *optional_seeds]
        positive.sort(
            key=lambda item: float(item["comparison"]["improvement_pct"]),
            reverse=True,
        )
        by_mechanism: dict[str, list[dict[str, Any]]] = {}
        for item in positive:
            by_mechanism.setdefault(measured_mechanism(item), []).append(item)
        ordered_mechanisms = sorted(
            by_mechanism,
            key=lambda name: -float(
                by_mechanism[name][0]["comparison"]["improvement_pct"]
            ),
        )
        promoted: list[dict[str, Any]] = []
        for mechanism in ordered_mechanisms:
            promoted.append(by_mechanism[mechanism][0])
        promoted_ids = {id(item) for item in promoted}
        promoted.extend(item for item in positive if id(item) not in promoted_ids)
        seen_rechecks: set[str] = set()
        for item in promoted:
            signature = json.dumps(
                {"config": item.get("config", {}), "env": item.get("env", {})},
                sort_keys=True,
            )
            if signature in seen_rechecks:
                continue
            seen_rechecks.add(signature)
            racing_rechecks.append({
                "name": f"recheck-{item.get('configuration_name', 'candidate')}"[:96],
                "config": deepcopy(item.get("config", {})),
                **({"env": deepcopy(item["env"])} if item.get("env") else {}),
                **(
                    {"registry_candidate_id": item["registry_candidate_id"]}
                    if isinstance(item.get("registry_candidate_id"), str) else {}
                ),
                "parent": item.get("configuration_name"),
                "promotion_mechanism": measured_mechanism(item),
                "reason": (
                    "exact coarse positive promoted to the matched saturation "
                    "window before any neighboring value is explored"
                ),
            })
            if len(racing_rechecks) >= candidate_slots:
                break

    # Successive refinement: use measured coarse results to choose which
    # parameter neighborhoods deserve more values. This replaces the old
    # behavior where one representative value was often the only observation
    # for a continuous/nonlinear control.
    ranked_groups = {
        item.get("parameter"): item
        for item in search_plan.get("ranked_parameter_groups", [])
        if isinstance(item, dict) and isinstance(item.get("values"), list)
    }
    evaluated_signatures = {
        json.dumps({"config": item.get("config", {}), "env": item.get("env", {})}, sort_keys=True)
        for item in screen.get("aggregates", [])
    }
    refinement_parents: list[tuple[float, dict[str, Any], str]] = []
    for item in screen.get("aggregates", [])[1:]:
        if not item.get("stable") or not item.get("all_repetitions_slo_passed"):
            continue
        changed = {
            key: value for key, value in item.get("config", {}).items()
            if baseline.get(key) != value
        }
        if len(changed) != 1:
            continue
        parameter = next(iter(changed))
        if parameter not in ranked_groups:
            continue
        improvement = item.get("comparison", {}).get("improvement_pct")
        if not isinstance(improvement, (int, float)):
            continue
        refinement_parents.append((float(improvement), item, parameter))
    refinement_parent_limit = {
        "fast": 1, "balanced": 2, "max": 4,
    }.get(normalized_experiment_mode(task), 2)
    refinements: list[dict[str, Any]] = []
    refined_parameters: set[str] = set()
    for _, parent, parameter in sorted(refinement_parents, reverse=True, key=lambda row: row[0]):
        if parameter in refined_parameters or len(refined_parameters) >= refinement_parent_limit:
            continue
        refined_parameters.add(parameter)
        for value in ranked_groups[parameter]["values"]:
            config = {**baseline, parameter: value}
            signature = json.dumps({"config": config, "env": {}}, sort_keys=True)
            if signature in evaluated_signatures:
                continue
            evaluated_signatures.add(signature)
            refinements.append({
                "name": f"refine-{parameter}-{str(value).lower()}"[:96],
                "config": config,
                "parent": parent.get("configuration_name"),
                "reason": "successive refinement around the best measured coarse value",
            })

    # A positive member of a mechanism is evidence to inspect its unmeasured
    # siblings, not evidence that the whole coarse catalog family is solved.
    # Promote siblings that share a matched rule first, then those that share
    # a causal sub-mechanism. This gives high-value parameters omitted by the
    # coarse budget a bounded path back into the search.
    evaluated_parameters: set[str] = set()
    positive_parent_parameters: list[str] = []
    for item in screen.get("aggregates", [])[1:]:
        changed = {
            key: value for key, value in item.get("config", {}).items()
            if baseline.get(key) != value
        }
        if len(changed) != 1:
            continue
        parameter = next(iter(changed))
        evaluated_parameters.add(parameter)
        improvement = item.get("comparison", {}).get("improvement_pct")
        if (
            item.get("stable")
            and item.get("all_repetitions_slo_passed")
            and item.get("comparison", {}).get("secondary_regressions_passed")
            and isinstance(improvement, (int, float))
            and improvement > 0
        ):
            positive_parent_parameters.append(parameter)

    sibling_ranked: list[tuple[int, int, str, str, dict[str, Any]]] = []
    for parent_parameter in positive_parent_parameters:
        parent_group = ranked_groups.get(parent_parameter, {})
        parent_rules = set(parent_group.get("trigger", {}).get("rule_ids", []))
        parent_submechanism = parent_group.get("submechanism") or parameter_submechanism(
            parent_parameter, parent_group.get("family")
        )
        for parameter, group in ranked_groups.items():
            if parameter in evaluated_parameters or parameter == parent_parameter:
                continue
            rules = set(group.get("trigger", {}).get("rule_ids", []))
            current_submechanism = group.get("submechanism") or parameter_submechanism(
                parameter, group.get("family")
            )
            shared_rules = parent_rules & rules
            if shared_rules:
                relationship = "shared_trigger_rule"
                relationship_rank = 0
            elif current_submechanism == parent_submechanism:
                relationship = "shared_submechanism"
                relationship_rank = 1
            else:
                continue
            sibling_ranked.append((
                relationship_rank,
                -MAGNITUDE_ORDER.get(group.get("trigger_magnitude", "low"), 1),
                parameter,
                parent_parameter,
                {"relationship": relationship, "shared_rules": sorted(shared_rules)},
            ))

    sibling_cap = max(1, candidate_slots // 3)
    sibling_refinements: list[dict[str, Any]] = []
    promoted_siblings: set[str] = set()
    for _, _, parameter, parent_parameter, relationship in sorted(
        sibling_ranked, key=lambda row: row[:4]
    ):
        if len(sibling_refinements) >= sibling_cap or parameter in promoted_siblings:
            continue
        value = next(
            (
                candidate for candidate in ranked_groups[parameter]["values"]
                if baseline.get(parameter) != candidate
            ),
            None,
        )
        if value is None:
            continue
        config = {**baseline, parameter: value}
        signature = json.dumps({"config": config, "env": {}}, sort_keys=True)
        if signature in evaluated_signatures:
            continue
        evaluated_signatures.add(signature)
        promoted_siblings.add(parameter)
        sibling_refinements.append({
            "name": f"refine-sibling-{parameter}-{str(value).lower()}"[:96],
            "config": config,
            "parent": parent_parameter,
            "reason": (
                "unmeasured sibling promoted after a positive related parameter; "
                f"relationship={relationship['relationship']}"
            ),
            "sibling_relationship": relationship,
        })

    # Model-native bundles have conditional parameters and cannot be refined
    # as one scalar. Once a representative MTP or Mamba configuration runs
    # successfully, promote untested compatible variants into this second
    # stage instead of consuming the coarse mechanism-coverage budget.
    successful_model_mechanisms: set[str] = set()
    for item in screen.get("aggregates", [])[1:]:
        if not item.get("stable") or not item.get("all_repetitions_slo_passed"):
            continue
        changed = {
            key: value for key, value in item.get("config", {}).items()
            if baseline.get(key) != value
        }
        if "speculative_algorithm" in changed:
            successful_model_mechanisms.add("mtp")
        if any(key.startswith("mamba_") for key in changed):
            successful_model_mechanisms.add("mamba")
    model_bundle_limit = {
        "fast": 1, "balanced": 2, "max": 4,
    }.get(normalized_experiment_mode(task), 2)
    model_bundle_refinements: list[dict[str, Any]] = []
    model_bundle_counts: dict[str, int] = {}
    for bundle in [
        *search_plan.get("cookbook_candidate_bundles", []),
        *search_plan.get("ranked_configuration_bundles", []),
    ]:
        config = bundle.get("config", {}) if isinstance(bundle, dict) else {}
        mechanism = (
            "mtp" if "speculative_algorithm" in config
            else "mamba" if any(key.startswith("mamba_") for key in config)
            else None
        )
        if mechanism not in successful_model_mechanisms:
            continue
        if model_bundle_counts.get(mechanism, 0) >= model_bundle_limit:
            continue
        candidate = {**baseline, **config}
        signature = json.dumps({"config": candidate, "env": bundle.get("env", {})}, sort_keys=True)
        if signature in evaluated_signatures:
            continue
        evaluated_signatures.add(signature)
        model_bundle_counts[mechanism] = model_bundle_counts.get(mechanism, 0) + 1
        model_bundle_refinements.append({
            "name": f"refine-{bundle.get('name', mechanism)}"[:96],
            "config": candidate,
            **({"env": deepcopy(bundle["env"])} if isinstance(bundle.get("env"), dict) else {}),
            "parent": mechanism,
            "reason": "conditional model-native refinement after a compatible representative completed",
        })
    # Round-robin the three refinement sources. A model-native variant, a
    # local value refinement, or a sibling promotion cannot consume every
    # slot merely because its list was assembled first.
    refinement_buckets = [model_bundle_refinements, refinements, sibling_refinements]
    registry_refinements: list[dict[str, Any]] = []
    registry_followup: dict[str, Any] | None = None
    if phase == "refinement" and isinstance(search_plan.get("candidate_registry"), dict):
        # Legacy refinement generation above records proposed values in
        # ``evaluated_signatures`` before the canonical Registry schedule is
        # built.  Those proposals were never executed, so they must not hide
        # the same Registry candidates and collapse refinement to an empty
        # stage.  Rebuild this set strictly from measured screen evidence.
        registry_evaluated_signatures = {
            json.dumps(
                {"config": item.get("config", {}), "env": item.get("env", {})},
                sort_keys=True,
            )
            for item in screen.get("aggregates", [])
            if isinstance(item, dict)
        }
        remaining_refinement_slots = max(
            0, candidate_slots - len(racing_rechecks)
        )
        disabled_capability_families = {
            str(item.get("family"))
            for item in screen.get("disabled_capabilities", [])
            if isinstance(item, dict) and item.get("family")
        }
        registry_followup = adaptive_followup_schedule(
            search_plan["candidate_registry"],
            budget=remaining_refinement_slots,
            minimum_improvement_pct=float(task["objective"].get("min_improvement_pct", 0)),
            anchor_config=baseline,
            evaluated_configurations=[
                item.get("config", {}) for item in screen.get("aggregates", [])
            ],
            max_values_per_mechanism=(
                1 if normalized_experiment_mode(task) == "fast"
                else 3 if normalized_experiment_mode(task) == "balanced"
                else 6
            ),
        )
        for candidate in registry_followup["selected"]:
            config = candidate.get("full_config", {})
            signature = json.dumps({"config": config, "env": candidate.get("env_delta", {})}, sort_keys=True)
            if signature in registry_evaluated_signatures:
                continue
            registry_evaluated_signatures.add(signature)
            registry_refinements.append({
                "name": f"refine-{candidate.get('name', candidate['candidate_id'])}"[:96],
                "config": config,
                **({"env": deepcopy(candidate["env_delta"])} if candidate.get("env_delta") else {}),
                "registry_candidate_id": candidate["candidate_id"],
                "parent": "mechanism_schedule",
                "reason": "mechanism-level adaptive follow-up after positive or uncertain evidence",
            })
    refinement_buckets = (
        [racing_rechecks, registry_refinements]
        if registry_followup is not None
        else [racing_rechecks, *refinement_buckets]
    )
    fair_refinements: list[dict[str, Any]] = []
    offset = 0
    while any(offset < len(bucket) for bucket in refinement_buckets):
        for bucket in refinement_buckets:
            if offset < len(bucket):
                fair_refinements.append(bucket[offset])
        offset += 1
    disabled_capability_families = {
        str(item.get("family"))
        for item in screen.get("disabled_capabilities", [])
        if isinstance(item, dict) and item.get("family")
    }
    capability_pruned_refinements: list[dict[str, Any]] = []
    refinements = []
    for item in fair_refinements:
        family = capability_family({
            "name": item.get("name"),
            "configuration_name": item.get("name"),
            "config": item.get("config", {}),
        })
        if family in disabled_capability_families:
            capability_pruned_refinements.append({
                "name": item.get("name"), "capability": family,
                "reason": "capability was disabled by an earlier definitive failure",
            })
            continue
        refinements.append(item)

    # Champion augmentation: every compatible positive mechanism gets a direct
    # measured edge against the strongest current configuration. This replaces
    # the fixed top-2 composition menu that left mixed-chunk/max-prefill
    # untested despite positive standalone evidence.
    threshold_ids = {id(item) for item in threshold_seeds}
    primary = seeds[0] if seeds else None
    if phase == "composition" and primary is not None:
        possible = [
            (primary, item) for item in seeds[1:]
            if not str(item.get("configuration_name", "")).startswith("combine-")
        ]
    else:
        possible = list(combinations(seeds, 2))
        possible.extend(
            combination
            for size in range(3, len(seeds) + 1)
            for combination in combinations(seeds, size)
        )
        possible.sort(key=lambda group: (
            0 if len(group) == 2 and primary is not None and primary in group and all(id(item) in threshold_ids for item in group) else
            1 if len(group) == 2 and all(id(item) in threshold_ids for item in group) else
            2 if all(id(item) in threshold_ids for item in group) else
            3 if len(group) == 2 and primary is not None and primary in group else
            4,
            len(group),
            -sum(float(item["comparison"]["improvement_pct"]) for item in group),
        ))

    compatible_configurations: list[dict[str, Any]] = []
    # Do not regenerate a composition that the atomic screen already measured.
    # Cross-stage duplicates cost a full model reload without adding evidence.
    signatures: set[str] = {
        json.dumps(
            {"config": item.get("config", {}), "env": item.get("env", {})},
            sort_keys=True,
        )
        for item in screen.get("aggregates", [])[1:]
        if isinstance(item, dict) and isinstance(item.get("config"), dict)
    }
    for group in possible:
        combined_changes: dict[str, Any] = {}
        combined_env: dict[str, Any] = {}
        conflict = False
        for item in group:
            changes = {
                key: value for key, value in item["config"].items()
                if baseline.get(key) != value
            }
            item_env = item.get("env", {})
            if any(key in combined_changes and combined_changes[key] != value for key, value in changes.items()):
                conflict = True
                break
            if any(key in combined_env and combined_env[key] != value for key, value in item_env.items()):
                conflict = True
                break
            combined_changes.update(changes)
            combined_env.update(item_env)
        if conflict or (not combined_changes and not combined_env):
            continue
        combined = deepcopy(baseline)
        combined.update(combined_changes)
        signature = json.dumps({"config": combined, "env": combined_env}, sort_keys=True)
        member_signatures = {
            json.dumps({"config": item["config"], "env": item.get("env", {})}, sort_keys=True)
            for item in group
        }
        if signature in signatures or signature in member_signatures:
            continue
        signatures.add(signature)
        names = [item["configuration_name"] for item in group]
        configuration = {
            "name": ("combine-" + "-and-".join(names))[:96],
            "config": combined,
            **({"env": deepcopy(combined_env)} if combined_env else {}),
        }
        if phase == "composition" and isinstance(search_plan.get("candidate_registry"), dict):
            registry = CandidateRegistry.from_dict(search_plan["candidate_registry"])
            parent_ids = []
            for item in group:
                candidate_id = item.get("registry_candidate_id")
                if isinstance(candidate_id, str):
                    parent_ids.append(candidate_id)
                    continue
                item_delta = {
                    key: value for key, value in item.get("config", {}).items()
                    if baseline.get(key) != value
                }
                signature = json.dumps(item_delta, sort_keys=True, separators=(",", ":"))
                matched = next(
                    (
                        value.get("candidate_id") for value in registry.candidates.values()
                        if json.dumps(value.get("config_delta", {}), sort_keys=True, separators=(",", ":"))
                        == signature
                    ),
                    None,
                )
                if isinstance(matched, str):
                    parent_ids.append(matched)
            # Materialize only combinations that survive the trial-budget
            # slice. Keeping every Cartesian possibility in the Registry made
            # reports grow quadratically while almost all entries stayed idle.
            configuration["_registry_parent_ids"] = sorted(set(parent_ids))
            configuration["_registry_parent_names"] = names
            configuration["_registry_config_delta"] = combined_changes
            configuration["_registry_env_delta"] = combined_env
        compatible_configurations.append(configuration)
    if phase == "refinement":
        configurations = refinements[:candidate_slots]
    elif phase == "composition":
        configurations = compatible_configurations[:candidate_slots]
    else:
        # Backward-compatible combined planning for direct callers. The full
        # autopilot workflow uses two phases so refined winners can compose.
        refinement_quota = min(len(refinements), max(1, candidate_slots // 2))
        configurations = refinements[:refinement_quota]
        configurations.extend(
            compatible_configurations[: max(0, candidate_slots - len(configurations))]
        )
        if len(configurations) < candidate_slots:
            configurations.extend(
                refinements[refinement_quota: candidate_slots - len(configurations) + refinement_quota]
            )
    configurations = configurations[:candidate_slots]
    if not configurations:
        return None
    if isinstance(search_plan.get("candidate_registry"), dict):
        registry = CandidateRegistry.from_dict(search_plan["candidate_registry"])
        for configuration in configurations:
            candidate_id = configuration.get("registry_candidate_id")
            if phase == "composition":
                candidate_id = registry.propose(
                    name=configuration["name"],
                    config_delta=deepcopy(configuration.pop("_registry_config_delta", {})),
                    env_delta=deepcopy(configuration.pop("_registry_env_delta", {})),
                    mechanism="interaction",
                    source={
                        "type": "adaptive_interaction",
                        "parents": configuration.pop("_registry_parent_names", []),
                    },
                    expected_impact="medium",
                    decision_reason="compatible positive mechanism combination",
                )
                registry.candidates[candidate_id]["parent_ids"] = configuration.pop(
                    "_registry_parent_ids", []
                )
                configuration["registry_candidate_id"] = candidate_id
                configuration["registry_mechanism"] = "interaction"
            if isinstance(candidate_id, str) and candidate_id in registry.candidates:
                registry.mark_scheduled(candidate_id, phase)
        search_plan["candidate_registry"] = registry.to_dict()
    spec = explicit_configuration_spec(
        task, discovery,
        stage_name="interact",
        baseline=baseline,
        configurations=configurations,
        max_trials=baseline_trials + len(configurations),
        repetitions=1,
        remaining_gpu_hours=remaining_gpu_hours,
        remaining_wall_minutes=remaining_wall_minutes,
        include_baseline=not reuse_reference,
        reference_baseline=(
            {
                "config": deepcopy(screen["aggregates"][0]["config"]),
                "env": deepcopy(screen["aggregates"][0].get("env", {})),
                "resources": deepcopy(
                    screen["aggregates"][0].get("resources", {})
                ),
                "metrics": deepcopy(baseline_metrics),
            }
            if reuse_reference else None
        ),
    )
    spec["search"].update({
        "interaction_policy": (
            "successive refinement and compatible composition are separate measured phases; "
            "composition is regenerated from refined winners"
        ),
        "interaction_phase": phase,
        "adaptive_refinement_parents": [item["parent"] for item in refinements],
        "adaptive_refinement_candidates": [item["name"] for item in refinements],
        "sibling_refinement_candidates": [item["name"] for item in sibling_refinements],
        "mechanism_refinement_candidates": [item["name"] for item in registry_refinements],
        "racing_recheck_candidates": [item["name"] for item in racing_rechecks],
        "racing_recheck_policy": (
            "promote exact positive coarse configurations with mechanism diversity; "
            "neighboring values consume only slots left after exact promotion"
        ),
        "capability_pruned_refinements": deepcopy(
            capability_pruned_refinements
        ),
        "mechanism_followup": deepcopy(registry_followup),
        "sibling_refinement_policy": (
            "positive parameters promote unmeasured shared-rule or shared-submechanism siblings; "
            "sources are round-robin scheduled within the bounded refinement budget"
        ),
        "threshold_seed_names": [item["configuration_name"] for item in threshold_seeds],
        "optional_positive_seed_names": [item["configuration_name"] for item in optional_seeds],
        "candidate_slots": candidate_slots,
        "budget_allocation": deepcopy(budget_allocation),
        "reclaimed_discovery_trials": reclaimed_discovery,
        "generated_combinations": len(configurations),
        "compatible_combinations": len(compatible_configurations),
        "budget_omitted_combinations": max(0, len(compatible_configurations) - len(configurations)),
        "augmentation_champion": (
            primary.get("configuration_name") if isinstance(primary, dict) else None
        ),
        "champion_augmentation_candidates": [
            item["name"] for item in compatible_configurations
        ],
        "champion_augmentation_policy": (
            "test the current champion plus every compatible positive peer in "
            "descending standalone evidence order until the residual budget is exhausted"
        ),
        "measurement_tier": (
            f"{phase}_race" if offline_racing_recheck
            else phase if phase in {"refinement", "composition"}
            else "matched_reference"
        ),
    })
    long_reference = screen["aggregates"][0].get("confirmation_reference")
    if reference_baseline_mode(task):
        configure_offline_reference_window(spec, task, phase=phase)
    elif isinstance(long_reference, dict):
        spec["benchmark"].update({
            "num_prompts": int(long_reference["num_prompts"]),
            "min_measurement_seconds": float(
                long_reference.get("measurement_validity", {}).get(
                    "minimum_duration_sec",
                    (task.get("measurement") or {}).get("min_measurement_seconds", 15),
                )
            ),
            "flush_cache": True,
        })
        if spec["benchmark"].get("dataset_name") == "generated-shared-prefix":
            groups = max(1, int(spec["benchmark"]["gsp_num_groups"]))
            spec["benchmark"]["gsp_prompts_per_group"] = max(
                1, math.ceil(int(long_reference["num_prompts"]) / groups)
            )
            spec["benchmark"]["num_prompts"] = groups * spec["benchmark"]["gsp_prompts_per_group"]
    return spec


def merge_stage_evidence(
    screen: dict[str, Any], *reports: dict[str, Any] | None,
    include_screen_usage: bool = True,
) -> dict[str, Any]:
    """Merge measured stage candidates against one preserved baseline."""
    merged = deepcopy(screen)
    baseline = deepcopy(screen.get("aggregates", [None])[0])
    candidates = [
        deepcopy(item) for item in screen.get("aggregates", [])[1:]
        if isinstance(item, dict)
    ] if include_screen_usage else []
    results = list(deepcopy(screen.get("results", []))) if include_screen_usage else []
    completed_trials = int(screen.get("completed_trials", 0)) if include_screen_usage else 0
    approx_gpu_hours = float(screen.get("approx_gpu_hours", 0)) if include_screen_usage else 0.0
    planned_trials = int(screen.get("planned_trials", 0)) if include_screen_usage else 0
    for report in reports:
        if not isinstance(report, dict):
            continue
        candidates.extend(
            deepcopy(item) for item in report.get("aggregates", [])[1:]
            if isinstance(item, dict)
        )
        results.extend(deepcopy(report.get("results", [])))
        completed_trials += int(report.get("completed_trials", 0))
        approx_gpu_hours += float(report.get("approx_gpu_hours", 0))
        planned_trials += int(report.get("planned_trials", 0))
    merged.update({
        "aggregates": ([baseline] if isinstance(baseline, dict) else []) + candidates,
        "results": results,
        "completed_trials": completed_trials,
        "approx_gpu_hours": approx_gpu_hours,
        "planned_trials": planned_trials,
        "stop_reason": "completed_search",
        "stage_merge_policy": "preserved baseline plus atomic, refined, and composed candidates",
    })
    return merged


def confirmation_spec(
    task: dict[str, Any],
    discovery: dict[str, Any],
    screen: dict[str, Any],
    remaining_trials: int,
    remaining_gpu_hours: float,
    remaining_wall_minutes: float,
) -> dict[str, Any]:
    baseline_aggregate = screen["aggregates"][0]
    long_reference = baseline_aggregate.get("confirmation_reference")
    has_reference = reference_baseline_mode(task) and isinstance(long_reference, dict)
    if reference_baseline_mode(task) and not has_reference:
        raise ValueError(
            "offline no-SLO confirmation requires the matched long-window baseline captured "
            "during parameter screening"
        )
    baseline = deepcopy(baseline_aggregate["config"])
    winner = screen.get("screening_winner")
    history_prior = winner.get("history_prior") if isinstance(winner, dict) else None
    configurations: list[dict[str, Any]] = []
    if winner is not None:
        candidate_config = winner["config"]
        candidate_env = winner.get("env", {})
        if candidate_config != baseline or candidate_env:
            configurations.append({
                "name": "selected-candidate",
                "config": candidate_config,
                **({"env": candidate_env} if candidate_env else {}),
            })
    # A candidate decision always uses paired repeated baseline and candidate
    # windows. One preserved baseline sample cannot establish a two-sample
    # confidence interval. Reference-only mode is retained only when there is
    # no positive candidate to investigate.
    reference_only = has_reference and not configurations
    measurement = task.get("measurement") or {}
    bayesian_enabled = bool(measurement.get("bayesian_sequential", False))
    repetitions = (
        1 if reference_only
        else int(measurement.get("bayesian_min_blocks", effective_confirmation_repetitions(task)))
        if bayesian_enabled
        else effective_confirmation_repetitions(task)
    )
    required = repetitions * (1 if reference_only and configurations else 2 if configurations else 1)
    if remaining_trials < required:
        raise ValueError(f"insufficient remaining trial budget for confirmation: need {required}, have {remaining_trials}")
    if remaining_gpu_hours <= 0:
        raise ValueError("GPU-hour budget exhausted before confirmation")
    if remaining_wall_minutes <= 0:
        raise ValueError("wall-time budget exhausted before confirmation")
    configured_adaptive_repetitions = int(
        measurement.get(
            "bayesian_max_blocks" if bayesian_enabled else "adaptive_confirmation_max_repetitions",
            repetitions,
        )
    )
    requested_adaptive_extra = max(0, configured_adaptive_repetitions - repetitions) * 2
    # Admit as many complete paired blocks as the residual budget can fund.
    # Requiring room for *all* blocks up to bayesian_max_blocks discarded five
    # usable trials in a real 20-trial run even though two additional AB pairs
    # would have resolved a 2.4% candidate.
    available_adaptive_trials = max(0, remaining_trials - required)
    adaptive_extra_trials = (
        min(
            requested_adaptive_extra,
            (available_adaptive_trials // 2) * 2,
        )
        if not reference_only and requested_adaptive_extra > 0 else 0
    )
    budgeted_max_repetitions = repetitions + adaptive_extra_trials // 2
    spec = explicit_configuration_spec(
        task, discovery,
        stage_name="confirm",
        baseline=baseline,
        configurations=configurations or [{"name": "baseline-repeat", "config": baseline}],
        max_trials=required + adaptive_extra_trials,
        repetitions=repetitions,
        remaining_gpu_hours=remaining_gpu_hours,
        remaining_wall_minutes=remaining_wall_minutes,
        include_baseline=not reference_only,
        reference_baseline=(
            {
                "config": deepcopy(baseline_aggregate["config"]),
                "env": deepcopy(baseline_aggregate.get("env", {})),
                "resources": deepcopy(
                    baseline_aggregate.get("resources", {})
                ),
                "metrics": deepcopy(long_reference["metrics"]),
                "measurement": deepcopy(long_reference.get("measurement_validity", {})),
                "num_prompts": long_reference["num_prompts"],
                "dataset_name": long_reference.get("dataset_name"),
            }
            if reference_only else None
        ),
    )
    if reference_baseline_mode(task) and configurations:
        configure_offline_reference_window(
            spec, task, phase="confirmation"
        )
    if not reference_only and repetitions > 1:
        # Repetitions are separate benchmark windows, not separate model loads.
        # A cache flush restores a comparable starting state while the server
        # remains resident for all windows of the same configuration.
        spec["search"]["reuse_server_across_repetitions"] = True
        spec["search"]["min_confirm_repetitions"] = repetitions
        spec["search"]["max_cv_pct"] = 5.0
        spec["search"].update({
            "bayesian_sequential": bayesian_enabled,
            "bayesian_min_blocks": int(measurement.get("bayesian_min_blocks", 2)),
            "bayesian_max_blocks": min(
                int(measurement.get("bayesian_max_blocks", 6)),
                budgeted_max_repetitions,
            ),
            "bayesian_accept_probability": float(
                measurement.get("bayesian_accept_probability", 0.95)
            ),
            "bayesian_reject_probability": float(
                measurement.get("bayesian_reject_probability", 0.05)
            ),
            "bayesian_prior_mean_pct": float(
                history_prior.get("mean_improvement_pct")
                if isinstance(history_prior, dict)
                else measurement.get("bayesian_prior_mean_pct", 0.0)
            ),
            "bayesian_prior_strength": float(
                min(0.25, max(0.01, int(history_prior.get("samples", 1)) * 0.02))
                if isinstance(history_prior, dict)
                else measurement.get("bayesian_prior_strength", 0.01)
            ),
            "history_prior": deepcopy(history_prior),
        })
        if adaptive_extra_trials:
            spec["search"].update({
                "adaptive_confirmation_cv_pct": float(
                    measurement.get("adaptive_confirmation_cv_pct", 5.0)
                ),
                "adaptive_confirmation_max_repetitions": min(
                    configured_adaptive_repetitions, budgeted_max_repetitions
                ),
                "adaptive_confirmation_min_measurement_seconds": float(
                    measurement.get("adaptive_confirmation_min_measurement_seconds", 30.0)
                ),
            })
        spec["search"]["confirmation_budget_guard"] = {
            "remaining_trials_at_entry": remaining_trials,
            "required_initial_trials": required,
            "adaptive_extra_trials": adaptive_extra_trials,
            "maximum_paired_blocks": budgeted_max_repetitions,
            "policy": "Bayesian/adaptive windows may not exceed the residual trial budget",
        }
        spec["benchmark"]["flush_cache"] = True
    if reference_only:
        spec["search"]["min_confirm_repetitions"] = 1
        reference_prompts = int(long_reference["num_prompts"])
        spec["benchmark"]["num_prompts"] = reference_prompts
        spec["benchmark"]["flush_cache"] = True
        if spec["benchmark"].get("dataset_name") == "generated-shared-prefix":
            groups = max(1, int(spec["benchmark"]["gsp_num_groups"]))
            spec["benchmark"]["gsp_prompts_per_group"] = max(
                1, math.ceil(reference_prompts / groups)
            )
            spec["benchmark"]["num_prompts"] = (
                groups * spec["benchmark"]["gsp_prompts_per_group"]
            )
    return spec


def confirmation_candidate_pool(
    screen: dict[str, Any], interaction: dict[str, Any] | None,
    priors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select the confirmation nominee across atomic and combined screens.

    Interaction is an additional experiment, not a replacement for its input
    screen.  A composition can be slower than its strongest constituent, so
    confirmation must nominate the best accepted candidate from both result
    sets rather than blindly preferring the most recent stage.
    """
    screen_aggregates = screen.get("aggregates", [])
    interaction_aggregates = (
        interaction.get("aggregates", []) if isinstance(interaction, dict) else []
    )
    if not screen_aggregates:
        return interaction or screen
    screen_candidates = list(screen_aggregates[1:])
    interaction_candidates = list(interaction_aggregates[1:])
    for item in screen_candidates:
        item.setdefault("measurement_tier", "screening")
        item.setdefault("measurement_tier_rank", 1)
    interaction_default_tier = (
        interaction_aggregates[0].get("measurement_tier", "screening")
        if interaction_aggregates else "screening"
    )
    interaction_default_rank = (
        interaction_aggregates[0].get("measurement_tier_rank", 1)
        if interaction_aggregates else 1
    )
    for item in interaction_candidates:
        item.setdefault("measurement_tier", interaction_default_tier)
        item.setdefault("measurement_tier_rank", interaction_default_rank)
    candidates = [*screen_candidates, *interaction_candidates]

    def tier_rank(item: dict[str, Any]) -> int:
        value = item.get("measurement_tier_rank", 1)
        return int(value) if isinstance(value, int) else 1

    def changed_settings(item: dict[str, Any]) -> dict[str, Any]:
        baseline = screen_aggregates[0]
        changes = {
            f"config:{key}": value
            for key, value in item.get("config", {}).items()
            if baseline.get("config", {}).get(key) != value
        }
        changes.update({
            f"env:{key}": value
            for key, value in item.get("env", {}).items()
            if baseline.get("env", {}).get(key) != value
        })
        return changes

    def parent_relative_comparison(item: dict[str, Any]) -> dict[str, Any] | None:
        """Require generated compositions to improve on their best direct parent.

        A composition that only beats the original baseline can retain a
        redundant flag.  Generated ``combine-*`` candidates therefore need a
        measured strict-subset parent and must clear the same practical gain
        threshold relative to the strongest direct parent before confirmation.
        """
        if not str(item.get("configuration_name", "")).startswith("combine-"):
            return None
        changes = changed_settings(item)
        if len(changes) < 2:
            return None
        comparison = item.get("comparison", {})
        metric = comparison.get("objective_metric")
        direction = comparison.get("direction") or METRIC_DIRECTIONS.get(metric)
        resource_scope = comparison.get("resource_scope", "per_service")
        global_minimum = float(comparison.get("minimum_improvement_pct", 0.0))
        # The task threshold applies to the complete configuration versus the
        # baseline. Reusing it for each newly added flag systematically favors
        # simple configurations late in the search. Use a smaller explicit
        # parsimony threshold for the incremental edge instead.
        minimum = min(0.5, max(0.1, global_minimum * 0.25))
        candidate_value = resource_adjusted_metric_value(
            item, metric, resource_scope
        )
        parents: list[tuple[int, float, dict[str, Any]]] = []
        for parent in candidates:
            if parent is item:
                continue
            if tier_rank(parent) != tier_rank(item):
                continue
            parent_changes = changed_settings(parent)
            if not parent_changes or len(parent_changes) >= len(changes):
                continue
            if any(changes.get(key) != value for key, value in parent_changes.items()):
                continue
            parent_value = resource_adjusted_metric_value(
                parent, metric, resource_scope
            )
            if not isinstance(parent_value, (int, float)) or isinstance(parent_value, bool):
                continue
            if not parent.get("stable") or not parent.get("all_repetitions_slo_passed"):
                continue
            parents.append((len(parent_changes), float(parent_value), parent))
        if not parents or not isinstance(candidate_value, (int, float)) or isinstance(candidate_value, bool):
            return {
                "accepted": False,
                "reason": "no stable measured direct-parent evidence is available",
                "minimum_improvement_pct": minimum,
                "global_minimum_improvement_pct": global_minimum,
                "objective_metric": metric,
                "resource_scope": resource_scope,
                "direction": direction,
            }
        direct_depth = max(depth for depth, _, _ in parents)
        direct = [row for row in parents if row[0] == direct_depth]
        if direction == "minimize":
            _, parent_value, parent = min(direct, key=lambda row: row[1])
            delta = parent_value - float(candidate_value)
        else:
            _, parent_value, parent = max(direct, key=lambda row: row[1])
            delta = float(candidate_value) - parent_value
        improvement = None if parent_value == 0 else delta / abs(parent_value) * 100.0
        accepted = improvement is not None and improvement >= minimum
        return {
            "accepted": accepted,
            "reason": (
                "composition clears the parent-relative practical-gain threshold"
                if accepted else
                "composition does not improve enough over its strongest measured direct parent"
            ),
            "parent_configuration_name": parent.get("configuration_name"),
            "parent_config": deepcopy(parent.get("config", {})),
            "parent_env": deepcopy(parent.get("env", {})),
            "parent_objective": parent_value,
            "candidate_objective": float(candidate_value),
            "improvement_pct": improvement,
            "minimum_improvement_pct": minimum,
            "global_minimum_improvement_pct": global_minimum,
            "objective_metric": metric,
            "resource_scope": resource_scope,
            "direction": direction,
            "direct_parent_change_count": direct_depth,
        }

    composition_parent_gates: list[dict[str, Any]] = []
    for item in candidates:
        gate = parent_relative_comparison(item)
        if gate is None:
            continue
        item["parent_relative_comparison"] = gate
        item["composition_parent_accepted"] = bool(gate["accepted"])
        composition_parent_gates.append({
            "configuration_name": item.get("configuration_name"),
            **deepcopy(gate),
        })
        if not gate["accepted"]:
            reasons = item.setdefault("rejection_reasons", [])
            if "composition_parent_improvement_below_minimum" not in reasons:
                reasons.append("composition_parent_improvement_below_minimum")
    accepted = [
        item for item in candidates
        if item.get("screening_accepted")
        and item.get("composition_parent_accepted", True)
        and isinstance(item.get("comparison", {}).get("improvement_pct"), (int, float))
    ]
    positive_probe = [
        item for item in candidates
        if item.get("stable")
        and item.get("all_repetitions_slo_passed")
        and item.get("comparison", {}).get("secondary_regressions_passed")
        and item.get("composition_parent_accepted", True)
        and isinstance(item.get("comparison", {}).get("improvement_pct"), (int, float))
        and item["comparison"]["improvement_pct"] > 0
    ]
    eligible = accepted or positive_probe
    highest_evidence_tier = max(
        (tier_rank(item) for item in eligible), default=1
    )
    eligible = [
        item for item in eligible if tier_rank(item) == highest_evidence_tier
    ]
    use_interaction_baseline = (
        highest_evidence_tier >= 2 and bool(interaction_aggregates)
    )
    selected_baseline = (
        interaction_aggregates[0]
        if use_interaction_baseline else screen_aggregates[0]
    )
    merged = deepcopy(interaction if use_interaction_baseline else screen)
    merged["aggregates"] = [deepcopy(selected_baseline), *candidates]
    ranked_candidates = sorted(
        eligible,
        key=lambda item: float(item["comparison"]["improvement_pct"]),
        reverse=True,
    )
    unique_candidates: list[dict[str, Any]] = []
    seen = set()
    for item in ranked_candidates:
        signature = json.dumps(
            {"config": item.get("config", {}), "env": item.get("env", {})}, sort_keys=True
        )
        if signature in seen:
            continue
        seen.add(signature)
        prior = configuration_history_prior(
            priors or {}, item.get("config", {}), item.get("env", {})
        )
        if prior is not None:
            item["history_prior"] = prior
        unique_candidates.append(item)
    merged["confirmation_candidates"] = unique_candidates[:3]
    merged["screening_winner"] = (
        merged["confirmation_candidates"][0] if merged["confirmation_candidates"] else None
    )
    if merged["screening_winner"] is not None and not accepted:
        merged["screening_winner"]["noise_probe_candidate"] = True
    merged["confirmation_candidate_policy"] = (
        "select only the highest available measurement tier, then rank up to three unique, "
        "SLO-valid candidates and confirm each independently against the same-tier baseline "
        "while remaining budget permits; generated compositions must "
        "first clear the practical-gain threshold relative to their strongest measured direct "
        "parent; exact-compatible history only sets a weak Bayesian prior and never creates a candidate"
    )
    merged["confirmation_measurement_tier_rank"] = highest_evidence_tier
    merged["composition_parent_gates"] = composition_parent_gates
    return merged


def champion_augmentation_round_outcome(
    champion_name: str | None,
    composition_input: dict[str, Any],
    round_result: dict[str, Any],
    priors: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the measured round decision and whether champion advanced."""
    decision = confirmation_candidate_pool(composition_input, round_result, priors)
    current_round_candidates = {
        item.get("configuration_name")
        for collection in ("aggregates", "results")
        for item in round_result.get(collection, [])
        if isinstance(item, dict) and item.get("kind") == "candidate"
        and isinstance(item.get("configuration_name"), str)
    }
    accepted_edges = {
        item.get("configuration_name")
        for item in decision.get("composition_parent_gates", [])
        if item.get("accepted")
        and item.get("configuration_name") in current_round_candidates
    }
    winner = decision.get("screening_winner")
    winner_name = (
        winner.get("configuration_name") if isinstance(winner, dict) else None
    )
    advanced = (
        isinstance(winner_name, str)
        and winner_name != champion_name
        and winner_name in current_round_candidates
        and winner_name in accepted_edges
    )
    return decision, {
        "champion_in": champion_name,
        "accepted_edges": sorted(
            name for name in accepted_edges if isinstance(name, str)
        ),
        "champion_out": winner_name if advanced else champion_name,
        "advanced": advanced,
    }


def best_unconfirmed_performance_candidate(
    decision_input: dict[str, Any], confirmation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Preserve the best measured non-baseline candidate for an honest report."""
    candidates: list[dict[str, Any]] = []
    if isinstance(confirmation, dict):
        candidates.extend(
            item for item in confirmation.get("aggregates", [])
            if isinstance(item, dict) and item.get("kind") == "candidate"
        )
    candidates.extend(
        item for item in decision_input.get("confirmation_candidates", [])
        if isinstance(item, dict)
    )
    eligible = []
    for item in candidates:
        comparison = item.get("comparison", {})
        improvement = comparison.get("improvement_pct")
        if (
            not isinstance(improvement, (int, float))
            or isinstance(improvement, bool)
            or improvement <= 0
            or not item.get("stable", True)
            or not item.get("all_repetitions_slo_passed", True)
            or comparison.get("secondary_regressions_passed") is False
        ):
            continue
        eligible.append(item)
    if not eligible:
        return None
    selected = deepcopy(max(
        eligible,
        key=lambda item: float(item.get("comparison", {}).get("improvement_pct", 0)),
    ))
    selected["evidence_state"] = "unconfirmed_performance_candidate"
    return selected


def bottleneck_summary(screen: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    aggregates = screen.get("aggregates", [])
    if not aggregates:
        return {
            "classification": "screening_unavailable",
            "typical_prefill_batch_tokens": expected_prefill_tokens(task["workload"]) * task["workload"]["max_concurrency"],
            "baseline_metrics": {},
            "chunked_prefill_evidence": [],
            "explanation": "parameter-screening baseline did not complete; inspect the recorded trial failure before interpreting bottlenecks",
        }
    baseline = aggregates[0]
    candidates = screen["aggregates"][1:]
    chunk_rows = [row for row in candidates if row["configuration_name"].startswith("chunked_prefill_size-")]
    boundary = expected_prefill_tokens(task["workload"]) * task["workload"]["max_concurrency"]
    evidence = []
    for row in chunk_rows:
        value = row["config"].get("chunked_prefill_size")
        improvement = row.get("comparison", {}).get("improvement_pct")
        evidence.append({"chunked_prefill_size": value, "improvement_pct": improvement})
    below = [item for item in evidence if item["chunked_prefill_size"] < boundary]
    at_or_above = [item for item in evidence if item["chunked_prefill_size"] >= boundary]
    fragmentation = bool(
        below
        and at_or_above
        and min((item["improvement_pct"] or 0) for item in below) < -3
        and max((item["improvement_pct"] or 0) for item in at_or_above) > -2
    )
    return {
        "classification": "excessive_prefill_fragmentation" if fragmentation else "no_single_dominant_bottleneck",
        "typical_prefill_batch_tokens": boundary,
        "baseline_metrics": baseline["metrics"],
        "chunked_prefill_evidence": evidence,
        "explanation": (
            "chunk sizes below the typical concurrent prefill batch boundary regress throughput; keep the chunk at or above the boundary"
            if fragmentation
            else "screening did not isolate a dominant bottleneck; preserve the confirmed recommendation and profile only if SLO or optimization targets remain unmet"
        ),
    }


def operator_escalation_plan(profile: dict[str, Any]) -> dict[str, Any]:
    """Bound kernel work to a trace-proven hotspot and state its possible payoff."""
    diagnosis = profile["diagnosis"]
    top = diagnosis.get("top_kernel_families") or diagnosis.get("top_kernels", [])
    if not top:
        return {"required": False, "reason": "nsys did not identify a CUDA kernel"}
    kernel = top[0]
    kernel_gpu_share = float(kernel.get("time_pct") or 0) / 100
    if kernel_gpu_share < 0.25:
        return {
            "required": False,
            "reason": "no CUDA operator family accounts for at least 25% of GPU-active time",
            "top_kernel_family": kernel,
        }
    # Amdahl's law applies only to the measured GPU execution slice here.
    gpu_execution_upper_bound = (1 / (1 - kernel_gpu_share + kernel_gpu_share / 2) - 1) * 100
    ncu = profile.get("tool", {}).get("ncu", {})
    return {
        "required": True,
        "top_kernel_family": kernel,
        "evidence": {
            "family_share_of_gpu_active_pct": round(kernel_gpu_share * 100, 3),
            "scope": "CUDA kernel execution time only; excludes model loading and must not be converted to end-to-end gain without request-level attribution",
        },
        "two_x_kernel_speedup_gpu_execution_upper_bound_pct": round(gpu_execution_upper_bound, 3),
        "end_to_end_upper_bound_pct": None,
        "next_step": (
            "run a shape-matched Nsight Compute capture and the corresponding SGLang kernel microbenchmark before proposing a code change"
            if ncu.get("performance_counter_access") else
            "Nsight Compute is installed but GPU counters are not accessible. Enable driver-level profiler permissions, then run a shape-matched microbenchmark; do not auto-edit a kernel."
        ),
        "ncu": ncu,
    }


def moe_kernel_optimization_plan(
    task: dict[str, Any], discovery: dict[str, Any], profile: dict[str, Any]
) -> dict[str, Any]:
    runtime = profile.get("runtime_observations", {})
    moe = runtime.get("moe", {}) if isinstance(runtime, dict) else {}
    if not discovery.get("model", {}).get("is_moe"):
        return {"status": "not_applicable", "reason": "model is not MoE"}
    if not moe.get("missing_tuned_config"):
        return {"status": "not_needed", "reason": "SGLang did not report a missing fused MoE config"}

    summaries = [
        runtime.get("decode", {}).get("running_requests", {}),
        runtime.get("prefill", {}).get("new_tokens", {}),
    ]
    batch_sizes: set[int] = set()
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        for percentile in ("p50", "p95"):
            value = summary.get(percentile)
            if isinstance(value, (int, float)) and value > 0:
                batch_sizes.add(max(1, min(8192, int(round(value)))))
    if not batch_sizes:
        batch_sizes.add(max(1, int(task["workload"]["max_concurrency"])))

    repo = Path(task["repository"])
    standard_tuner = repo / "benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py"
    separate_tuner = repo / "benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py"
    effective = profile.get("effective_server_config", {})
    tp_size = int(effective.get("tp_size", discovery["derived"]["minimum_tp_size"]) or 1)
    ep_size = int(effective.get("ep_size", 1) or 1)
    quantization = str(discovery.get("model", {}).get("weight_quantization") or discovery.get("model", {}).get("quantization") or "").lower()
    tuner_dtype = "fp8_w8a8" if "fp8" in quantization else "auto"
    topk_ids_dir = task.get("kernel_tuning", {}).get("topk_ids_dir")
    requires_down = bool(moe.get("requires_down_kernel_config"))
    tuner = separate_tuner if requires_down else standard_tuner
    if requires_down:
        commands = []
        if topk_ids_dir:
            # The separate tuner writes both the normal and `_down` files in
            # one invocation. Its single-batch path only prints results, so
            # intentionally omit --batch-size to use its paired-file writer.
            commands = [[
                task["python"], str(tuner), "--model", task["model_path"],
                "--tp-size", str(tp_size), "--ep-size", str(ep_size),
                "--dtype", tuner_dtype, "--topk-ids-dir", str(topk_ids_dir), "--tune",
            ]]
    else:
        commands = [
            [
                task["python"], str(tuner), "--model", task["model_path"],
                "--tp-size", str(tp_size), "--ep-size", str(ep_size),
                "--dtype", tuner_dtype, "--batch-size", str(batch_size), "--tune",
            ]
            for batch_size in sorted(batch_sizes)
        ]
    moe_share = float(profile.get("diagnosis", {}).get("shares_pct", {}).get("moe_kernels") or 0)
    return {
        "status": "candidate_required",
        "priority": "high",
        "reason": (
            "missing tuned config and MoE kernels account for material GPU execution time"
            if moe_share >= 15
            else "the active MoE path used a generic fallback config; the aggregate trace is dominated by another phase, so tune observed MoE shapes and verify the result end to end"
        ),
        "missing_config_files": moe.get("missing_config_files", []),
        "requires_down_kernel_config": moe.get("requires_down_kernel_config", False),
        "tuning_mode": "separate_up_down" if requires_down else "standard_up_only",
        "topk_ids_dir": str(topk_ids_dir) if topk_ids_dir else None,
        "observed_moe_kernel_share_pct": moe_share,
        "shape_matched_batch_sizes": sorted(batch_sizes),
        "tuner_available": tuner.is_file(),
        "standard_tuner": str(standard_tuner),
        "separate_tuner": str(separate_tuner),
        "tuner_commands": commands,
        "application_policy": (
            "write generated JSON under the private run directory, set SGLANG_MOE_CONFIG_DIR only for a candidate trial, "
            "and retain it only after end-to-end SLO-valid A/B confirmation; `_down` warnings require paired up/down files"
        ),
    }


def prepare_local_ray_compat(tuner: Path, output_dir: Path) -> dict[str, Any]:
    """Provide the small synchronous Ray surface used by SGLang's MoE tuner.

    Ray is an optional benchmark dependency, not an SGLang serving dependency.
    Keeping this shim inside the private run directory avoids modifying the
    user's Python environment while preserving the official tuner logic.
    """
    source = tuner.read_text(encoding="utf-8", errors="replace")
    supported = {"remote", "init", "available_resources", "get_gpu_ids", "get", "experimental"}
    used = set(re.findall(r"\bray\.([A-Za-z_][A-Za-z0-9_]*)", source))
    unsupported = sorted(used - supported)
    if unsupported:
        return {
            "status": "unsupported",
            "reason": "installed tuner uses Ray APIs outside the local compatibility surface",
            "unsupported_ray_apis": unsupported,
        }

    root = output_dir / "ray-compat"
    package = root / "ray"
    experimental = package / "experimental"
    experimental.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(
        """class _RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class _ActorHandle:
    def __init__(self, instance):
        self.instance = instance

    def __getattr__(self, name):
        value = getattr(self.instance, name)
        return _RemoteMethod(value) if callable(value) else value


class _RemoteClass:
    def __init__(self, actor_class):
        self.actor_class = actor_class

    def remote(self, *args, **kwargs):
        return _ActorHandle(self.actor_class(*args, **kwargs))


def remote(*args, **kwargs):
    def decorate(actor_class):
        return _RemoteClass(actor_class)
    if len(args) == 1 and isinstance(args[0], type) and not kwargs:
        return decorate(args[0])
    return decorate


def init(*args, **kwargs):
    return {"local_mode": True}


def available_resources():
    return {"GPU": 1}


def get_gpu_ids():
    return [0]


def get(values):
    return values
""",
        encoding="utf-8",
    )
    (experimental / "__init__.py").write_text("", encoding="utf-8")
    (experimental / "tqdm_ray.py").write_text(
        """try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
""",
        encoding="utf-8",
    )
    return {
        "status": "ready",
        "mode": "local_single_gpu_compat",
        "pythonpath": str(root),
        "supported_ray_apis": sorted(supported),
    }


_MOE_CONFIG_FILENAME = re.compile(
    r"^E=\d+,N=\d+,device_name=[^,]*(?:,dtype=[^,]+)?"
    r"(?:,block_shape=\[[0-9]+,\s*[0-9]+\])?"
    r"(?:,per_channel_quant=True)?(?:_down)?\.json$"
)


def validate_moe_config_artifact(path: Path) -> tuple[bool, str | None]:
    """Validate the file contract consumed by SGLang's current MoE loader."""
    if not _MOE_CONFIG_FILENAME.fullmatch(path.name):
        return False, f"unexpected fused MoE config filename: {path.name}"
    try:
        values = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"cannot parse fused MoE config {path.name}: {exc}"
    if not isinstance(values, dict) or not values or not all(
        isinstance(key, str) and key.isdigit() and isinstance(value, dict)
        for key, value in values.items()
    ):
        return False, (
            f"invalid fused MoE config schema in {path.name}; expected JSON mapping "
            "batch-token M to kernel config"
        )
    required = {"BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "num_warps", "num_stages"}
    for batch_size, config in values.items():
        if not required.issubset(config):
            return False, f"config {path.name} batch {batch_size} is missing kernel tile fields"
    return True, None


def execute_moe_kernel_tuning(
    task: dict[str, Any], plan: dict[str, Any], output_dir: Path,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Generate isolated Triton MoE configs for a later end-to-end A/B trial."""
    mode = task.get("kernel_tuning", {}).get("mode", "detect_only")
    if mode == "auto":
        mode = "detect_only"
    if plan.get("status") != "candidate_required" or mode == "disabled":
        return {"status": "not_run", "reason": plan.get("reason", "kernel tuning disabled")}
    if mode == "detect_only":
        return {
            "status": "deferred",
            "reason": "kernel autotuning is opt-in and excluded from the normal deployment search budget",
            "plan": plan,
        }
    if not plan.get("tuner_available"):
        return {"status": "unavailable", "reason": "installed SGLang checkout has no fused MoE tuner", "plan": plan}
    if plan.get("requires_down_kernel_config") and not plan.get("topk_ids_dir"):
        return {
            "status": "blocked",
            "reason": (
                "SGLang requested a paired _down fused MoE config; the separate tuner requires "
                "topk_ids_dir from the official top-k capture workflow. Standard tuning cannot produce a deployable result."
            ),
            "plan": plan,
        }

    settings = task.get("kernel_tuning", {})
    max_batch_sizes = max(1, int(settings.get("max_batch_sizes", 4)))
    commands = list(plan.get("tuner_commands", []))[:max_batch_sizes]
    timeout_sec = float(settings.get("timeout_minutes", 120)) * 60
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    for key, value in task.get("env", {}).items():
        environment[key] = str(value)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    repo_python = str(Path(task["repository"]) / "python")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = repo_python if not existing_pythonpath else f"{repo_python}{os.pathsep}{existing_pythonpath}"

    ray_probe = subprocess.run(
        [task["python"], "-c", "import ray"],
        capture_output=True, text=True, timeout=30, check=False, env=environment,
    )
    ray_runtime = {"status": "ready", "mode": "installed_ray"}
    if ray_probe.returncode != 0:
        ray_runtime = prepare_local_ray_compat(Path(commands[0][1]), output_dir)
        if ray_runtime.get("status") != "ready":
            return {
                "status": "unavailable",
                "reason": ray_runtime.get("reason", "Ray is unavailable for the installed tuner"),
                "ray_runtime": ray_runtime,
                "plan": plan,
            }
        environment["PYTHONPATH"] = (
            f"{ray_runtime['pythonpath']}{os.pathsep}{environment['PYTHONPATH']}"
        )

    version = subprocess.run(
        [task["python"], "-c", "import triton; print(triton.__version__)"],
        capture_output=True, text=True, timeout=30, check=False, env=environment,
    )
    if version.returncode != 0 or not version.stdout.strip():
        return {"status": "unavailable", "reason": "cannot determine installed Triton version", "plan": plan}
    version_dir = f"triton_{version.stdout.strip().replace('.', '_')}"
    merged: dict[str, dict[str, Any]] = {}
    command_results: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, command in enumerate(commands):
        remaining = timeout_sec - (time.monotonic() - started)
        if remaining <= 0:
            return {"status": "timeout", "commands": command_results, "ray_runtime": ray_runtime, "plan": plan}
        batch_dir = output_dir / f"batch-{index:02d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            progress.emit(
                "optional-moe tune",
                f"starting shape group {index + 1}/{len(commands)}; artifacts={batch_dir}",
                completed=index, total=len(commands),
            )
        try:
            log_path = batch_dir / "tuner.log"
            with log_path.open("w", encoding="utf-8") as tuner_log:
                process = subprocess.Popen(
                    command, cwd=batch_dir, env=environment,
                    stdout=tuner_log, stderr=subprocess.STDOUT, text=True,
                )
                command_started = time.monotonic()
                last_heartbeat = command_started
                while process.poll() is None:
                    now = time.monotonic()
                    if now - command_started >= remaining:
                        process.kill()
                        process.wait(timeout=10)
                        raise subprocess.TimeoutExpired(command, remaining)
                    if progress and now - last_heartbeat >= 30:
                        progress.emit(
                            "optional-moe tune",
                            f"shape group {index + 1}/{len(commands)} still running; "
                            f"elapsed={now - command_started:.0f}s, log={log_path}",
                            completed=index, total=len(commands),
                        )
                        last_heartbeat = now
                    time.sleep(1)
            result = subprocess.CompletedProcess(command, process.returncode, "", "")
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "reason": f"fused MoE tuner exceeded {timeout_sec / 60:g} minutes",
                "commands": command_results, "ray_runtime": ray_runtime,
                "plan": plan,
            }
        command_results.append({"command": command, "returncode": result.returncode, "directory": str(batch_dir)})
        if result.returncode != 0:
            return {
                "status": "failed", "reason": f"fused MoE tuner exited with code {result.returncode}",
                "commands": command_results, "ray_runtime": ray_runtime, "plan": plan,
            }
        generated = list(batch_dir.glob("E=*.json"))
        if not generated:
            return {
                "status": "failed", "reason": "fused MoE tuner produced no config JSON",
                "commands": command_results, "ray_runtime": ray_runtime, "plan": plan,
            }
        names = {path.name for path in generated}
        if plan.get("requires_down_kernel_config"):
            up = {name for name in names if not name.endswith("_down.json")}
            down = {name.removesuffix("_down.json") for name in names if name.endswith("_down.json")}
            if not up or not down or not (up & down):
                return {
                    "status": "failed",
                    "reason": "separate fused MoE tuner did not produce matching up/down config files",
                    "generated_files": sorted(names),
                    "commands": command_results,
                    "ray_runtime": ray_runtime,
                    "plan": plan,
                }
        if progress:
            progress.emit(
                "optional-moe tune",
                f"completed shape group {index + 1}/{len(commands)}; generated={len(generated)} files",
                completed=index + 1, total=len(commands),
            )
        for path in generated:
            valid, reason = validate_moe_config_artifact(path)
            if not valid:
                return {
                    "status": "failed",
                    "reason": reason,
                    "commands": command_results,
                    "ray_runtime": ray_runtime,
                    "plan": plan,
                }
            values = load_json(path)
            merged.setdefault(path.name, {}).update(values)

    config_root = output_dir / "candidate-config-root"
    target_dir = config_root / "configs" / version_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in merged.items():
        write_json(target_dir / filename, values)
    return {
        "status": "completed",
        "elapsed_sec": time.monotonic() - started,
        "config_root": str(config_root),
        "generated_files": sorted(str(path) for path in target_dir.glob("*.json")),
        "paired_config_complete": (
            not plan.get("requires_down_kernel_config")
            or any(path.name.endswith("_down.json") for path in target_dir.glob("*.json"))
        ),
        "commands": command_results,
        "ray_runtime": ray_runtime,
        "plan": plan,
        "validation_policy": "candidate config is not deployable until the normal end-to-end screening and confirmation gates pass",
    }


def final_server_command(spec: dict[str, Any], recommendation: dict[str, Any]) -> list[str]:
    trial = {"config": recommendation["config"], "name": "recommended"}
    placeholder = Path(spec["scope"]["output_dir"]) / "DEPLOYMENT_COMMAND"
    return command_manifest(spec, trial, placeholder)["server"]


def reproducible_server_command(
    spec: dict[str, Any], recommendation: dict[str, Any], resolved: dict[str, Any],
) -> list[str]:
    """Pin performance-critical resolved settings in addition to measured deltas."""
    config = {
        key: value for key, value in resolved.items()
        if value is not None
    }
    config.update(recommendation.get("config", recommendation))
    return final_server_command(spec, {"config": config})


def host_deployment_plan(
    task: dict[str, Any], discovery: dict[str, Any],
    recommendation: dict[str, Any] | None, command: list[str] | None,
) -> dict[str, Any]:
    """Describe what was measured and avoid presenting one service as host-optimal."""
    if not isinstance(recommendation, dict) or not isinstance(command, list):
        return {
            "status": "unavailable", "confirmed_scope": "none",
            "measured_host_aggregate": False,
        }
    config = recommendation.get("config", recommendation)
    per_service = 1
    for parameter in ("tp_size", "pp_size", "dp_size"):
        value = config.get(parameter, 1)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            per_service *= value
    identifiers = selected_gpu_identifiers(task, discovery.get("hardware", {}))
    replica_count = len(identifiers) // max(1, per_service)
    groups = [
        identifiers[index * per_service:(index + 1) * per_service]
        for index in range(replica_count)
    ]
    base_port = int(task.get("port", 31000))
    commands = []
    for index, group in enumerate(groups):
        argv = list(command)
        if "--port" in argv:
            argv[argv.index("--port") + 1] = str(base_port + index)
        commands.append({
            "replica": index,
            "gpu_group": group,
            "environment": {"CUDA_VISIBLE_DEVICES": ",".join(group)},
            "command": argv,
        })
    return {
        "status": (
            "single_service_matches_selected_host"
            if replica_count == 1 and per_service == len(identifiers)
            else "replica_layout_unverified"
        ),
        "confirmed_scope": "single_service",
        "measured_host_aggregate": False,
        "selected_gpu_count": len(identifiers),
        "gpus_per_service": per_service,
        "possible_replica_count": replica_count,
        "unassigned_gpus": identifiers[replica_count * per_service:],
        "replica_commands": commands,
        "policy": (
            "per-service performance is confirmed; multi-replica host throughput is not claimed "
            "until a shared-host multi-service benchmark measures contention and routing"
        ),
    }


def build_plan(
    task: dict[str, Any], progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    if progress:
        progress.emit("plan", "validating the task contract", completed=0, total=6)
    errors = validate_task(task)
    if errors:
        raise ValueError("; ".join(errors))
    task = materialize_runtime_task(task)
    if progress:
        progress.emit(
            "plan", "discovering hardware, model, Cookbook and live ServerArgs",
            completed=1, total=6,
        )
    discovery = discover(task, progress=progress)
    if progress:
        progress.emit("plan", "building the bounded Nsight profile", completed=2, total=6)
    baseline_profile_spec = profile_spec(task, discovery)
    errors = execution_errors(baseline_profile_spec)
    if errors:
        raise ValueError("generated profiling spec is invalid: " + "; ".join(errors))
    initial_plan = cookbook_initial_search_plan(task, discovery)
    if progress:
        progress.emit(
            "plan", "routing Cookbook/topology candidates and parameter evolution",
            completed=3, total=6,
        )
    allocated_gpus = selected_gpus(task, discovery["hardware"])
    homogeneous_gpu_pool = len({
        (str(gpu.get("name")), int(gpu.get("memory_mib", 0))) for gpu in allocated_gpus
    }) <= 1
    requested_parallel_trials = min(
        int(task.get("parallel_trials", 1)), len(allocated_gpus)
    )
    if progress:
        progress.emit(
            "plan", "loading exact-compatible history and GPU scheduling limits",
            completed=4, total=6,
        )
    history_evidence = history_search_evidence(task, discovery)
    budget_allocation = task_trial_budget(task)
    if progress:
        progress.emit(
            "plan", "allocating discovery/refinement/confirmation budgets",
            completed=5, total=6,
        )
    result = {
        "schema_version": 4,
        "execution_enabled": False,
        "discovery": discovery,
        "deployment_policy": deployment_policy(task),
        "resource_scheduling": {
            "detected_selected_gpus": len(allocated_gpus),
            "max_gpus": int(task.get("max_gpus", len(allocated_gpus))),
            "max_parallel_trials": requested_parallel_trials,
            "homogeneous_gpu_pool": homogeneous_gpu_pool,
            "allocation_policy": (
                "pack independent one-pass trials onto disjoint GPU sets sized from TP/PP/DP"
            ),
            "profile_pipeline_eligible": bool(
                task.get("deployment_mode") == "offline_throughput"
                and not task.get("slo")
                and task.get("profile_dir") is None
                and requested_parallel_trials > 1
                and homogeneous_gpu_pool
                and discovery["derived"]["minimum_tp_size"] == 1
            ),
        },
        "startup_acceleration": startup_acceleration_plan(
            task, discovery, initial_plan
        ),
        "history": history_evidence,
        "parameter_evolution": compact_evolution_summary(
            discovery.get("parameter_evolution", {})
        ),
        "budget_allocation": budget_allocation,
        "knowledge_preflight": {
            "order": "hardware and model inventory -> official cookbook and hardware references -> local CLI/checkpoint compatibility -> initial bundle benchmark",
            "cookbook": discovery.get("cookbook"),
            "hardware_reference_urls": (task.get("knowledge") or {}).get("hardware_reference_urls", []),
        },
        "calibration": {
            "required": (task.get("calibration") or {}).get("enabled", True),
            "target_workload": deepcopy(task["workload"]),
            "concurrency_points": (
                ["runtime_resolved"]
                if task["workload"].get("runtime_capacity_pending")
                else calibration_concurrencies(task)
            ),
            "policy": (
                "for adaptive online SLO tasks, start from SGLang's runtime-resolved max_running_requests "
                "rather than a task concurrency hint; calibrate capacity only for diagnosis, then "
                "revalidate the final recommendation against the selected SLO-safe workload"
            ),
        },
        "profiling": {
            "required": True,
            "tool": "nsight_systems" if discovery["hardware"]["vendor"] == "nvidia" else "pytorch_or_rpd",
            "capture_policy": "warm up first, then capture benchmark only with framework profiling control",
            "spec": baseline_profile_spec,
        },
        "search": {
            "initial_bundles_before_profiling": True,
            "generated_after_profiling": True,
            "parameter_catalog_count": discovery["parameter_catalog"]["parameter_count"],
            "initial_plan": initial_plan,
            "budget_allocation": budget_allocation,
            "parameter_evolution": compact_evolution_summary(
                discovery.get("parameter_evolution", {})
            ),
            "policy": "benchmark cookbook-derived, locally compatible startup bundles first; profile the best SLO-valid initial configuration at calibrated analysis load; then screen profiler-driven deltas and revalidate on the target workload",
        },
        "confirmation": {
            "automatic": True,
            "repetitions": effective_confirmation_repetitions(task),
            "adaptive_max_repetitions": int(
                (task.get("measurement") or {}).get(
                    "adaptive_confirmation_max_repetitions",
                    effective_confirmation_repetitions(task),
                )
            ),
            "adaptive_trigger_cv_pct": (task.get("measurement") or {}).get(
                "adaptive_confirmation_cv_pct"
            ),
            "adaptive_min_measurement_seconds": (task.get("measurement") or {}).get(
                "adaptive_confirmation_min_measurement_seconds"
            ),
            "server_sessions": 2,
            "policy": (
                "start with two 15-second benchmark windows per configuration; if objective "
                "CV exceeds 5%, add one 30-second window per configuration; reuse resident "
                "servers and keep measurement fidelity independent of search intensity"
            ),
        },
    }
    if progress:
        progress.emit(
            "plan", "plan generated and execution spec validated",
            completed=6, total=6,
        )
    return result


def resume_task_mismatches(requested: dict[str, Any], recorded: dict[str, Any]) -> list[str]:
    """Reject reuse when fields that determine benchmark evidence changed."""
    mismatches: list[str] = []
    optional_objects = {
        "env", "slo", "objective", "parameter_evolution", "measurement",
        "calibration", "quality", "knowledge", "capability_overrides", "deployment",
    }
    for key in (
        "name", "repository", "python", "model_path", "output_dir",
        "deployment_mode", "experiment_mode", "env", "slo", "objective",
        "parallel_trials", "max_gpus",
        "parameter_evolution", "measurement", "calibration", "quality",
        "knowledge", "capability_overrides", "deployment",
    ):
        requested_value = requested.get(key) or {} if key in optional_objects else requested.get(key)
        recorded_value = recorded.get(key) or {} if key in optional_objects else recorded.get(key)
        if requested_value != recorded_value:
            mismatches.append(
                f"{key}: requested {requested_value!r}, recorded {recorded_value!r}"
            )
    requested_workload = requested.get("workload") or {}
    recorded_workload = recorded.get("workload") or {}
    for key in (
        "input_tokens", "output_tokens", "num_prompts", "request_rate",
        "max_concurrency", "prefix_reuse_ratio", "shared_prefix", "dataset",
    ):
        if requested_workload.get(key) != recorded_workload.get(key):
            mismatches.append(
                f"workload.{key}: requested {requested_workload.get(key)!r}, "
                f"recorded {recorded_workload.get(key)!r}"
            )
    return mismatches


def reusable_stage_spec_mismatches(
    expected: dict[str, Any], recorded: dict[str, Any]
) -> list[str]:
    """Compare fields that determine the meaning of cached stage evidence."""
    mismatches: list[str] = []
    for key in (
        "framework", "model", "hardware", "workload", "slo", "objective",
        "deployment_mode", "benchmark", "experiment_fingerprint",
        "optimizer_contract",
    ):
        if expected.get(key) != recorded.get(key):
            mismatches.append(key)
    for key in (
        "baseline", "strategy", "space", "explicit_configurations",
        "include_baseline", "reference_baseline",
    ):
        if expected.get("search", {}).get(key) != recorded.get("search", {}).get(key):
            mismatches.append(f"search.{key}")
    return mismatches


def run_autopilot(
    task: dict[str, Any], progress_reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    progress = progress_reporter or ProgressReporter()
    progress.emit("setup", "validating task, hardware, model, and installed SGLang parameters")
    errors = validate_task(task)
    if errors:
        raise ValueError("; ".join(errors))
    requested_task = deepcopy(task)
    task = materialize_runtime_task(task)
    progress.emit("setup", "discovering accelerators and validating NVIDIA execution support")
    hardware = parse_nvidia_inventory() or parse_amd_inventory()
    if hardware is None:
        raise RuntimeError("no supported NVIDIA or AMD accelerator inventory available")
    if hardware.get("vendor") != "nvidia":
        raise RuntimeError(
            "automatic AMD profiling requires the RPD/PyTorch executor, which is not yet implemented"
        )
    progress.emit("setup", "checking Nsight Systems availability")
    nsys = run_readonly(["nsys", "--version"], timeout=30)
    if nsys.get("returncode") != 0:
        raise RuntimeError(
            "Nsight Systems (nsys) is required for inferopt run; install it or fix PATH before starting GPU trials"
        )
    resume_run_dir = task.get("resume_run_dir")
    resumed = resume_run_dir is not None
    if resumed:
        root = Path(str(resume_run_dir)).expanduser().resolve()
        expected_parent = Path(task["output_dir"]).expanduser().resolve()
        if root.parent != expected_parent:
            raise ValueError("resume_run_dir must be an immediate child of task.output_dir")
        recorded_task = load_json(root / "task.json")
        mismatches = resume_task_mismatches(task, recorded_task)
        if mismatches:
            raise ValueError("resume task does not match recorded evidence: " + "; ".join(mismatches))
        runtime_overrides = {
            key: task[key] for key in ("resume_run_dir", "profile_dir") if key in task
        }
        task = recorded_task
        task.update(runtime_overrides)
        cookbook_snapshot = load_json(root / "cookbook-snapshot.json")
        plan = load_json(root / "plan.json")
        write_json(root / "resume.json", {
            "resumed_at": utc_now(),
            "profile_dir": task.get("profile_dir"),
            "policy": "reuse only completed, task-compatible stage artifacts",
        })
        progress.emit("setup", f"resuming compatible artifacts in {root}")
    else:
        root = Path(task["output_dir"]).expanduser() / (
            f"{task['name']}-autopilot-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
        root.mkdir(parents=True, exist_ok=False)
        os.chmod(root, 0o700)
        progress.emit("setup", "resolving the local or downloadable SGLang Cookbook snapshot")
        task, cookbook_snapshot = provision_cookbook_snapshot(
            task, root,
            progress=lambda phase, message: progress.emit(
                f"cookbook {phase}", message
            ),
        )
        write_json(root / "cookbook-snapshot.json", cookbook_snapshot)
        progress.emit("setup", "building the hardware/model/workload-aware execution plan")
        plan = build_plan(task, progress=progress)
        write_json(root / "task.json", task)
        write_json(root / "plan.json", plan)
        progress.emit("setup", f"ready; artifacts: {root}")
    started = time.monotonic()
    is_reference_baseline_mode = (
        task.get("deployment_mode") == "offline_throughput" and not task.get("slo")
    )
    calibration_path = root / "calibration.json"
    if resumed and calibration_path.is_file():
        calibration = load_json(calibration_path)
        for key in ("completed_trials", "approx_gpu_hours", "selected_analysis_concurrency", "points"):
            if key not in calibration:
                raise RuntimeError(f"cannot resume invalid calibration.json: missing {key}")
        progress.emit(
            "capacity",
            f"reused {calibration['completed_trials']} completed calibration trials",
        )
    elif is_reference_baseline_mode:
        progress.emit(
            "capacity",
            "skipped for offline no-SLO mode; the parameter screen will measure the single unprofiled baseline",
        )
        calibration = {
            "policy": deployment_policy(task),
            "target_concurrency": task["workload"]["max_concurrency"],
            "points": [],
            "selected_analysis_concurrency": task["workload"]["max_concurrency"],
            "stopped_before_requested_cap": False,
            "strategy": "skipped_offline_without_slo",
            "approx_gpu_hours": 0.0,
            "completed_trials": 0,
            "reason": "client concurrency is unbounded and SGLang retains its resolved admission capacity",
        }
    else:
        progress.emit("capacity", "measuring the baseline SLO-safe concurrency curve")
        calibration = run_calibration(task, plan["discovery"], root, progress)
    write_json(calibration_path, calibration)
    if not is_reference_baseline_mode and not any(point.get("valid_for_analysis") for point in calibration["points"]):
        progress.emit("capacity", "no valid baseline point; stopping before Cookbook, profiling, and parameter search")
        raise RuntimeError(
            "baseline capacity calibration failed; inspect " + str(root / "calibration.json")
        )
    used_trials_before_profile = calibration["completed_trials"]
    used_gpu_hours_before_profile = calibration["approx_gpu_hours"]
    cookbook_initial_path = root / "cookbook-initial-plan.json"
    cookbook_initial = (
        load_json(cookbook_initial_path)
        if resumed and cookbook_initial_path.is_file()
        else cookbook_initial_search_plan(task, plan["discovery"])
    )
    write_json(root / "cookbook-initial-plan.json", cookbook_initial)
    # Capacity calibration is a deployment decision, not merely a hint for
    # profiling. Every subsequent stage must measure the same SLO-safe
    # concurrency selected from the baseline curve.
    execution_task = task_at_calibrated_concurrency(task, calibration)
    initial_screen: dict[str, Any] | None = None
    raw_baseline = {"tp_size": plan["discovery"]["derived"]["minimum_tp_size"]}
    initial_candidate = deepcopy(raw_baseline)
    resumed_initial_path = root / "cookbook-initial.json"
    if resumed and resumed_initial_path.is_file():
        initial_screen = load_json(resumed_initial_path)
        if initial_screen.get("stop_reason") not in {
            "completed_search", "strong_candidate_early_stop",
            "consecutive_failure_budget_exhausted",
        }:
            raise RuntimeError("cannot resume incomplete cookbook-initial.json")
        used_trials_before_profile += int(initial_screen.get("completed_trials", 0))
        used_gpu_hours_before_profile += float(initial_screen.get("approx_gpu_hours", 0))
        resumed_candidate = fastest_slo_valid_configuration(
            initial_screen, execution_task["objective"]
        )
        if resumed_candidate is not None:
            initial_candidate = resumed_candidate
        else:
            initial_baseline = next(
                (
                    item for item in initial_screen.get("aggregates", [])
                    if item.get("kind") == "baseline"
                ),
                None,
            )
            if initial_baseline is not None:
                initial_candidate = deepcopy(initial_baseline["config"])
        progress.emit(
            "cookbook",
            f"reused {initial_screen.get('completed_trials', 0)} completed initial-screen trials",
        )
    # The pre-profile screen may contain model-cookbook bundles, topology
    # candidates such as TP=2/4, or both.  Do not accidentally skip a valid
    # multi-GPU comparison merely because this model has no cookbook profile.
    # Offline no-SLO execution still needs its model-native Cookbook and
    # topology screen.  Skipping this stage used to leave MTP/NEXTN and TP
    # candidates unmeasured precisely in the throughput mode where they can
    # matter most.  The measured SGLang-default configuration remains the
    # final comparison baseline; this is only an early compatibility screen.
    # An offline no-SLO candidate must be measured only after the unbounded
    # baseline profile has exposed SGLang's admission capacity.  Running the
    # Cookbook stage before that point used its small bootstrap window (for
    # example 20 or 40 requests), which is not saturated throughput evidence.
    if initial_screen is None and not is_reference_baseline_mode and (
        cookbook_initial["cookbook_candidate_bundles"]
        or cookbook_initial["ranked_parameter_groups"]
    ):
        cookbook_trial_budget = initial_cookbook_trial_budget(task, used_trials_before_profile)
        cookbook_initial["allocated_trial_budget"] = cookbook_trial_budget
        if cookbook_trial_budget < 2:
            cookbook_initial["deferred_reason"] = (
                "insufficient trial budget after reserving Nsight profiling, the mode-specific "
                "profiler-directed parameter screen, interaction search, and confirmation"
            )
            write_json(root / "cookbook-initial-plan.json", cookbook_initial)
        else:
            progress.emit(
                "cookbook",
                f"screening Cookbook bundles with {cookbook_trial_budget} trials reserved for this exploratory stage",
            )
        elapsed_minutes = (time.monotonic() - started) / 60
        if cookbook_trial_budget >= 2:
            initial_spec = screening_spec(
                execution_task, plan["discovery"], cookbook_initial,
                remaining_gpu_hours=float(task["budget"]["max_gpu_hours"]) - used_gpu_hours_before_profile,
                remaining_wall_minutes=float(task["budget"]["max_wall_time_minutes"]) - elapsed_minutes,
                remaining_trials=cookbook_trial_budget,
                confirmation_reserve_trials=0,
            )
            errors = execution_errors(initial_spec)
            if errors:
                raise ValueError("generated cookbook initial spec is invalid: " + "; ".join(errors))
            write_json(root / "cookbook-initial-spec.json", initial_spec)
            initial_screen = execute_with_progress(initial_spec, progress, "cookbook screening")
            write_json(root / "cookbook-initial.json", initial_screen)
            used_trials_before_profile += initial_screen["completed_trials"]
            used_gpu_hours_before_profile += initial_screen["approx_gpu_hours"]
            initial_candidate = fastest_slo_valid_configuration(initial_screen, execution_task["objective"])
            if initial_candidate is None:
                initial_baseline = next(
                    (item for item in initial_screen.get("aggregates", []) if item.get("kind") == "baseline"), None
                )
                if initial_baseline is not None:
                    initial_candidate = deepcopy(initial_baseline["config"])
    analysis_task = deepcopy(execution_task)
    analysis_task["workload"]["max_concurrency"] = calibration["selected_analysis_concurrency"]
    profile_task = analysis_task
    pipeline_executor: ThreadPoolExecutor | None = None
    pipeline_future: Future[dict[str, Any]] | None = None
    pipeline_spec: dict[str, Any] | None = None
    pipeline_error: str | None = None
    selected = selected_gpus(execution_task, hardware)
    selected_identifiers = selected_gpu_identifiers(execution_task, hardware)
    homogeneous_pool = len({
        (str(gpu.get("name")), int(gpu.get("memory_mib", 0))) for gpu in selected
    }) == 1
    requested_workers = min(
        int(execution_task.get("parallel_trials", 1)), len(selected_identifiers)
    )
    pipeline_eligible = (
        is_reference_baseline_mode
        and task.get("profile_dir") is None
        and not (resumed and (root / "profile" / "nsys-diagnosis.json").is_file())
        and requested_workers > 1
        and len(selected_identifiers) > 1
        and homogeneous_pool
        and int(plan["discovery"]["derived"]["minimum_tp_size"]) == 1
        and not execution_task["workload"].get("runtime_capacity_pending", False)
    )
    if pipeline_eligible:
        base_port = int(task.get("port", 31000))
        auxiliary_port = base_port + max(16, requested_workers)
        if auxiliary_port + requested_workers > 65535:
            auxiliary_port = max(1024, base_port - max(16, requested_workers))
        profile_task = task_on_gpus(
            analysis_task, [selected_identifiers[0]], port=base_port, parallel_trials=1
        )
        spare_task = task_on_gpus(
            execution_task,
            selected_identifiers[1:requested_workers],
            port=auxiliary_port,
            parallel_trials=requested_workers - 1,
        )
        preprofile_plan = preprofile_search_plan(spare_task, plan["discovery"])
        preprofile_budget = initial_cookbook_trial_budget(task, used_trials_before_profile)
        preprofile_plan["allocated_trial_budget"] = preprofile_budget
        write_json(root / "preprofile-parallel-plan.json", preprofile_plan)
        if preprofile_budget >= 2:
            candidate_spec = screening_spec(
                spare_task,
                plan["discovery"],
                preprofile_plan,
                remaining_gpu_hours=float(task["budget"]["max_gpu_hours"]),
                remaining_wall_minutes=float(task["budget"]["max_wall_time_minutes"]),
                remaining_trials=preprofile_budget,
                confirmation_reserve_trials=0,
                baseline=raw_baseline,
            )
            pipeline_spec = single_gpu_preprofile_spec(candidate_spec, task)
        if pipeline_spec is not None:
            pipeline_spec["execution"]["process_wide_child_reaping"] = False
            errors = execution_errors(pipeline_spec)
            if errors:
                raise ValueError("generated parallel preprofile spec is invalid: " + "; ".join(errors))
            write_json(root / "preprofile-parallel-spec.json", pipeline_spec)
            progress.emit(
                "pipeline",
                f"GPU {selected_identifiers[0]} will capture Nsys while "
                f"{len(selected_identifiers[1:requested_workers])} spare GPUs screen workload priors",
            )
            pipeline_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="inferopt-preprofile"
            )
            pipeline_future = pipeline_executor.submit(
                execute_with_progress, pipeline_spec, progress, "preprofile screening"
            )
    profiled_initial_configuration = deepcopy(initial_candidate)
    analysis_profile_spec = profile_spec(
        profile_task, plan["discovery"], profiled_initial_configuration
    )
    if pipeline_spec is not None:
        analysis_profile_spec["execution"]["process_wide_child_reaping"] = False
    write_json(root / "analysis-profile-spec.json", analysis_profile_spec)
    resumed_profile_dir = root / "profile"
    reused_profile = task.get("profile_dir") is not None or (
        resumed and (resumed_profile_dir / "nsys-diagnosis.json").is_file()
    )
    def nsys_progress(event: dict[str, Any]) -> None:
        progress.emit(
            "nsys " + str(event.get("phase", "working")),
            str(event.get("message", "working")),
            completed=event.get("completed"),
            total=event.get("total"),
        )
    try:
        if reused_profile:
            selected_profile_dir = (
                Path(task["profile_dir"]).expanduser()
                if task.get("profile_dir") is not None else resumed_profile_dir
            )
            progress.emit("nsys", "reusing a completed compatible Nsight Systems profile")
            profiling = diagnose_existing(
                selected_profile_dir, progress=nsys_progress
            )
            mismatches = profile_matches_task(profiling, analysis_profile_spec)
            if mismatches:
                raise RuntimeError("cannot reuse profile: " + "; ".join(mismatches))
        else:
            profile_gpu_note = (
                f"GPU {selected_identifiers[0]} is loading the Nsys-profiled server; "
                "Nsight startup and CUDA Graph capture can lag ordinary screening workers"
                if pipeline_spec is not None and selected_identifiers else
                "capturing and analyzing a bounded serving-only Nsight Systems trace"
            )
            progress.emit("nsys", profile_gpu_note)
            profiling = run_profile(
                analysis_profile_spec, root / "profile", progress=nsys_progress
            )
    finally:
        if pipeline_future is not None:
            try:
                initial_screen = pipeline_future.result()
                write_json(root / "preprofile-parallel.json", initial_screen)
            except (OSError, RuntimeError, ValueError) as exc:
                pipeline_error = str(exc)
                write_json(root / "preprofile-parallel-error.json", {"error": pipeline_error})
                progress.emit("pipeline", f"spare-GPU preprofile screen failed; continuing without it: {exc}")
            finally:
                assert pipeline_executor is not None
                pipeline_executor.shutdown(wait=True)
        if initial_screen is not None and pipeline_spec is not None:
            used_trials_before_profile += initial_screen["completed_trials"]
            used_gpu_hours_before_profile += initial_screen["approx_gpu_hours"]
            candidate = fastest_slo_valid_configuration(initial_screen, execution_task["objective"])
            if candidate is not None:
                initial_candidate = candidate
    preprofile_baseline = next(
        (
            item.get("metrics") for item in (initial_screen or {}).get("aggregates", [])
            if isinstance(item, dict) and item.get("kind") == "baseline"
            and isinstance(item.get("metrics"), dict)
        ),
        None,
    )
    profiling = annotate_profile_comparability(
        profiling, calibration, baseline_metrics=preprofile_baseline
    )
    write_json(root / "nsys-diagnosis.json", profiling)
    if profiling["status"].get("state") != "completed":
        raise RuntimeError("required baseline profiling did not complete")
    if not profiling["diagnosis"].get("top_kernels"):
        raise RuntimeError("required nsys trace contains no parsed CUDA kernels")
    if is_reference_baseline_mode and execution_task["workload"].get("runtime_capacity_pending"):
        observed_capacity = observed_admission_capacity(profiling)
        practical_capacity = observed_practical_capacity(profiling, execution_task)
        if observed_capacity is None:
            raise RuntimeError(
                "the unbounded baseline completed but SGLang did not expose an admission capacity; "
                "inspect profile/server-info.json and runtime-observations.json"
            )
        if practical_capacity is None:
            practical_capacity = observed_capacity
        for derived_task in (execution_task, analysis_task):
            derived_task["workload"]["max_concurrency"] = practical_capacity
            derived_task["workload"].pop("runtime_capacity_pending", None)
            derived_task["workload"]["observed_admission_capacity"] = observed_capacity
            derived_task["workload"]["observed_practical_capacity"] = practical_capacity
            derived_task["workload"]["offline_saturation_request_floor"] = (
                practical_capacity * offline_saturation_waves(derived_task)
            )
        calibration["selected_analysis_concurrency"] = practical_capacity
        calibration["observed_admission_capacity"] = observed_capacity
        calibration["observed_practical_capacity"] = practical_capacity
        write_json(root / "runtime-capacity.json", {
            "source": "unbounded_baseline_profile",
            "max_running_requests": observed_capacity,
            "practical_request_capacity": practical_capacity,
            "offline_saturation_request_floor": practical_capacity * offline_saturation_waves(execution_task),
            "policy": (
                "max_running_requests is an upper bound; request windows use shape-aware KV capacity"
            ),
        })
    search_plan = diagnosed_search_plan(analysis_task, plan["discovery"], profiling)
    search_plan["startup_acceleration"] = startup_acceleration_plan(
        analysis_task, plan["discovery"], search_plan, profiling
    )
    history_evidence = plan.get("history") or history_search_evidence(
        task, plan["discovery"]
    )
    apply_history_priors_to_search_plan(search_plan, history_evidence)
    search_plan["budget_allocation"] = deepcopy(plan["budget_allocation"])
    search_plan["raw_sglang_baseline"] = deepcopy(raw_baseline)
    search_plan["preprofile_seed"] = {
        "config": deepcopy(initial_candidate),
        "requires_target_workload_confirmation": initial_candidate != raw_baseline,
        "policy": (
            "the exploratory seed remains a candidate against the original SGLang-default "
            "baseline; it is never silently adopted as the final baseline"
        ),
    }
    search_plan["screening_priority_order"] = core_serving_parameter_order(
        analysis_task, plan["discovery"], search_plan
    )
    search_plan["screening_candidate_limit"] = MODE_CANDIDATE_LIMITS.get(
        normalized_experiment_mode(task), 12
    )
    search_plan["screening_early_stop"] = {
        "minimum_successful_candidates": {
            "fast": 3, "balanced": 6, "max": 12,
        }.get(normalized_experiment_mode(task), 6),
        "strong_improvement_pct": 3.0,
        "note": (
            "1% remains the deployment acceptance threshold; 3% only permits early search "
            "termination after the required successful mechanism coverage"
        ),
    }
    search_plan["screening_selection_policy"] = (
        "rank by expected end-to-end impact from workload shape, Nsys GPU-active kernel shares, "
        "SGLang queue/cache/graph logs, topology, and resolved defaults; cover one representative "
        "value per high-impact mechanism before local value refinement"
    )
    # Normal modes already screened Cookbook bundles before profiling.  In
    # offline no-SLO mode they are deliberately deferred until the profile
    # reports admission capacity, so their request windows are saturated and
    # directly comparable to profiler-routed candidates.
    deferred_cookbook_bundles = deepcopy(cookbook_initial["cookbook_candidate_bundles"])
    if not search_plan.get("mtp_relevance", {}).get("relevant", True):
        deferred_cookbook_bundles = [
            bundle for bundle in deferred_cookbook_bundles
            if "speculative_algorithm" not in bundle.get("config", {})
        ]
    search_plan["cookbook_candidate_bundles"] = (
        deferred_cookbook_bundles if is_reference_baseline_mode else []
    )
    if initial_screen is not None:
        search_plan["previously_evaluated_configurations"] = [
            deepcopy(item["config"])
            for item in initial_screen.get("aggregates", [])
            if item.get("completed_repetitions", 0) > 0 and isinstance(item.get("config"), dict)
        ]
    moe_tuning_plan = moe_kernel_optimization_plan(analysis_task, plan["discovery"], profiling)
    kernel_tuning = {
        "status": "not_run",
        "reason": "fused MoE kernel autotuning is a separate opt-in operation and is never executed by inferopt run",
    }
    write_json(root / "kernel-tuning.json", kernel_tuning)
    write_json(root / "search-plan.json", search_plan)
    progress.emit("search", "screening profiler- and workload-selected parameter changes")
    elapsed_minutes = (time.monotonic() - started) / 60
    profile_gpu_hours = (
        0.0 if reused_profile else profiling.get("accelerator_elapsed_sec", profiling["elapsed_sec"]) / 3600
        * configuration_accelerator_count(analysis_profile_spec, profiled_initial_configuration)
    )
    used_before_screen = used_gpu_hours_before_profile + profile_gpu_hours
    remaining_gpu_hours = float(task["budget"]["max_gpu_hours"]) - used_before_screen
    remaining_wall_minutes = float(task["budget"]["max_wall_time_minutes"]) - elapsed_minutes
    if remaining_gpu_hours <= 0 or remaining_wall_minutes <= 0:
        raise RuntimeError("profiling exhausted the experiment budget before parameter screening")
    screen_spec = screening_spec(
        execution_task, plan["discovery"], search_plan,
        remaining_gpu_hours=remaining_gpu_hours,
        remaining_wall_minutes=remaining_wall_minutes,
        remaining_trials=int(task["budget"]["max_trials"]) - used_trials_before_profile - (0 if reused_profile else 1),
        baseline=raw_baseline,
        anchor=initial_candidate,
    )
    preserved_reference = (
        measured_reference_baseline(initial_screen)
        if initial_screen is not None and pipeline_spec is not None
        else None
    )
    if preserved_reference is not None:
        apply_reference_baseline(screen_spec, preserved_reference)
    errors = execution_errors(screen_spec)
    if errors:
        raise ValueError("generated screening spec is invalid: " + "; ".join(errors))
    screening_spec_path = root / "screening-spec.json"
    screening_result_path = root / "screening.json"
    search_policy_changed = False
    reuse_screening = resumed and screening_result_path.is_file()
    if reuse_screening:
        if not screening_spec_path.is_file():
            raise RuntimeError("cannot resume screening.json without screening-spec.json")
        spec_mismatches = reusable_stage_spec_mismatches(
            screen_spec, load_json(screening_spec_path)
        )
        if spec_mismatches:
            if "optimizer_contract" in spec_mismatches:
                reuse_screening = False
                search_policy_changed = True
                progress.emit(
                    "search",
                    "search-policy version changed; preserving raw profile but rerunning screening",
                )
            else:
                raise RuntimeError(
                    "cannot reuse screening evidence after spec changes: "
                    + ", ".join(spec_mismatches)
                )
    if reuse_screening:
        screen = load_json(screening_result_path)
        if screen.get("stop_reason") not in {
            "completed_search", "strong_candidate_early_stop",
            "consecutive_failure_budget_exhausted",
        }:
            raise RuntimeError("cannot resume incomplete screening.json")
        progress.emit(
            "search",
            f"reused {screen.get('completed_trials', 0)} completed screening trials",
        )
    else:
        write_json(screening_spec_path, screen_spec)
        screen = execute_with_progress(screen_spec, progress, "parameter screening")
    screen_stage_completed_trials = screen["completed_trials"]
    screen_stage_gpu_hours = screen["approx_gpu_hours"]
    screen_stage_elapsed_sec = float(screen.get("elapsed_sec", 0) or 0)
    if preserved_reference is not None and initial_screen is not None:
        screen = merge_screening_evidence(screen_spec, screen, initial_screen)
    screen_baseline_metrics = next(
        (
            item.get("metrics") for item in screen.get("aggregates", [])
            if isinstance(item, dict) and item.get("kind") == "baseline"
            and isinstance(item.get("metrics"), dict)
        ),
        None,
    )
    if isinstance(screen_baseline_metrics, dict):
        profiling = annotate_profile_comparability(
            profiling, calibration, baseline_metrics=screen_baseline_metrics
        )
        write_json(root / "nsys-diagnosis.json", profiling)
        search_plan = refresh_search_plan_after_baseline(
            analysis_task, plan["discovery"], profiling, search_plan
        )
    if isinstance(search_plan.get("candidate_registry"), dict):
        search_plan["candidate_registry"] = update_registry_from_aggregates(
            search_plan["candidate_registry"], screen.get("aggregates", [])
        )
        search_plan["mechanism_outcomes_after_screen"] = adaptive_followup_schedule(
            search_plan["candidate_registry"],
            budget=0,
            minimum_improvement_pct=float(task["objective"].get("min_improvement_pct", 0)),
            anchor_config=raw_baseline,
            evaluated_configurations=[
                item.get("config", {}) for item in screen.get("aggregates", [])
            ],
        )["mechanism_outcomes"]
        write_json(root / "candidate-registry-screening.json", search_plan["candidate_registry"])
        write_json(root / "search-plan.json", search_plan)
    write_json(screening_result_path, screen)
    used_trials = screen_stage_completed_trials
    used_gpu_hours = used_before_screen + screen_stage_gpu_hours
    elapsed_minutes = (time.monotonic() - started) / 60
    remaining_trials = int(task["budget"]["max_trials"]) - used_trials_before_profile - (0 if reused_profile else 1) - used_trials
    remaining_gpu_hours = float(task["budget"]["max_gpu_hours"]) - used_gpu_hours
    remaining_wall_minutes = float(task["budget"]["max_wall_time_minutes"]) - elapsed_minutes
    interaction: dict[str, Any] | None = None
    refinement: dict[str, Any] | None = None
    composition: dict[str, Any] | None = None
    interaction_error: str | None = None
    interaction_plan: dict[str, Any] | None = None
    refinement_plan: dict[str, Any] | None = None
    composition_plan: dict[str, Any] | None = None
    composition_rounds: list[dict[str, Any]] = []
    composition_plans: list[dict[str, Any]] = []
    composition_round_evidence: list[dict[str, Any]] = []
    attempted_parameter_candidates = [
        row for row in screen.get("results", []) if row.get("kind") == "candidate"
    ]
    executed_parameter_candidates = [row for row in attempted_parameter_candidates if row.get("ok")]
    mandatory_capacity_parameters = (
        ["mem_fraction_static", "max_prefill_tokens"]
        if reference_baseline_mode(task)
        else []
    )
    executed_parameters = {
        parameter
        for row in executed_parameter_candidates
        for parameter in (row.get("config") or {})
        if parameter != "tp_size"
    }
    missing_mandatory_capacity_parameters = [
        parameter for parameter in mandatory_capacity_parameters
        if parameter not in executed_parameters
    ]
    registry_coverage = (
        search_plan.get("candidate_registry", {}).get("coverage", {})
        if isinstance(search_plan.get("candidate_registry"), dict) else {}
    )
    lifecycle_coverage = registry_coverage.get("mechanism_lifecycle", {})
    required_classes = list(
        lifecycle_coverage.get("scheduled")
        or search_plan.get("mechanism_schedule", {}).get("scheduled_mechanisms", [])
    )
    executed_classes = list(lifecycle_coverage.get("executed", []))
    measured_classes = list(lifecycle_coverage.get("measured", []))
    missing_classes = sorted(set(required_classes) - set(measured_classes))
    unavailable_classes = sorted(set(executed_classes) - set(measured_classes))
    coverage_floor_met = len(measured_classes) >= min(
        required_mechanism_coverage(task), len(required_classes)
    )
    parameter_search = {
        "planned_trials": screen.get("planned_trials", 0),
        "executed_trials": screen_stage_completed_trials,
        "preprofile_executed_trials": (
            initial_screen.get("completed_trials", 0)
            if initial_screen is not None and pipeline_spec is not None else 0
        ),
        "attempted_parameter_candidates": len(attempted_parameter_candidates),
        "executed_parameter_candidates": len(executed_parameter_candidates),
        "failed_parameter_candidates": len(attempted_parameter_candidates) - len(executed_parameter_candidates),
        "selection_evidence": deepcopy(screen_spec["search"].get("selection_evidence", [])),
        "positive_screening_signals": [
            {
                "configuration_name": item.get("configuration_name"),
                "config": deepcopy(item.get("config", {})),
                "improvement_pct": item.get("comparison", {}).get(
                    "improvement_pct"
                ),
                "measurement_tier": item.get(
                    "measurement_tier", "screening"
                ),
                "evidence_state": "positive_screening_signal",
            }
            for item in screen.get("aggregates", [])[1:]
            if isinstance(item.get("comparison", {}).get("improvement_pct"), (int, float))
            and item["comparison"]["improvement_pct"] > 0
            and item.get("all_repetitions_slo_passed")
        ],
        "required_parameter_breadth": required_mechanism_coverage(task),
        "required_distinct_mechanisms": len(required_classes),
        "required_mechanism_classes": required_classes,
        "executed_distinct_mechanisms": executed_classes,
        "measured_distinct_mechanisms": measured_classes,
        "missing_mechanism_classes": missing_classes,
        "unavailable_mechanism_classes": unavailable_classes,
        "blocking_missing_mechanism_classes": [],
        "mandatory_capacity_parameters": mandatory_capacity_parameters,
        "missing_mandatory_capacity_parameters": missing_mandatory_capacity_parameters,
        "coverage_floor_met": coverage_floor_met,
        "sufficient_evidence": (
            coverage_floor_met and not missing_mandatory_capacity_parameters
        ),
        "coverage_policy": (
            "mandatory capacity controls and the mode-specific measured-mechanism floor block deployment; "
            "attempted optional mechanisms that are unavailable remain reportable gaps but do not veto an otherwise confirmed result"
        ),
    }
    if not executed_parameter_candidates:
        interaction_error = (
            "no deployment parameter candidate completed with comparable benchmark evidence"
        )
    elif screen["stop_reason"] in {"completed_search", "strong_candidate_early_stop"}:
        try:
            refinement_plan = interaction_spec(
                execution_task, plan["discovery"], search_plan, screen,
                remaining_trials, remaining_gpu_hours, remaining_wall_minutes,
                phase="refinement",
                candidate_slot_limit={
                    "fast": 2, "balanced": 6, "max": 12,
                }.get(normalized_experiment_mode(task), 6),
            )
            if refinement_plan is not None:
                progress.emit("refinement", "refining values for trigger-matched positive mechanisms")
                errors = execution_errors(refinement_plan)
                if errors:
                    raise ValueError("generated refinement spec is invalid: " + "; ".join(errors))
                refinement_spec_path = root / "refinement-spec.json"
                refinement_path = root / "refinement.json"
                if resumed and not search_policy_changed and refinement_path.is_file():
                    if not refinement_spec_path.is_file():
                        raise RuntimeError(
                            "cannot resume refinement.json without refinement-spec.json"
                        )
                    spec_mismatches = reusable_stage_spec_mismatches(
                        refinement_plan, load_json(refinement_spec_path)
                    )
                    if spec_mismatches:
                        raise RuntimeError(
                            "cannot reuse refinement evidence after spec changes: "
                            + ", ".join(spec_mismatches)
                        )
                    refinement = load_json(refinement_path)
                    if refinement.get("stop_reason") not in {
                        "completed_search", "strong_candidate_early_stop",
                        "consecutive_failure_budget_exhausted",
                    }:
                        raise RuntimeError("cannot resume incomplete refinement.json")
                    progress.emit(
                        "refinement",
                        f"reused {refinement.get('completed_trials', 0)} completed trials",
                    )
                else:
                    write_json(refinement_spec_path, refinement_plan)
                    refinement = execute_with_progress(
                        refinement_plan, progress, "parameter refinement"
                    )
                write_json(root / "refinement.json", refinement)
                used_trials += refinement["completed_trials"]
                used_gpu_hours += refinement["approx_gpu_hours"]
                elapsed_minutes = (time.monotonic() - started) / 60
                remaining_trials -= refinement["completed_trials"]
                remaining_gpu_hours = float(task["budget"]["max_gpu_hours"]) - used_gpu_hours
                remaining_wall_minutes = float(task["budget"]["max_wall_time_minutes"]) - elapsed_minutes
            # Screening and refinement use the same accurate saturation
            # window. Repeated champion-augmentation rounds add one compatible
            # peer at a time until no parent-relative gain remains or the
            # confirmation reserve is reached.
            composition_seed = merge_stage_evidence(screen, refinement)
            composition_input = deepcopy(composition_seed)
            round_index = 1
            augmentation_stop_reason = "no_compatible_positive_peer"
            while remaining_trials > int(
                (search_plan.get("budget_allocation") or task_trial_budget(task))["planned"]["confirmation"]
            ):
                composition_plan = interaction_spec(
                    execution_task, plan["discovery"], search_plan, composition_input,
                    remaining_trials, remaining_gpu_hours, remaining_wall_minutes,
                    phase="composition",
                )
                if composition_plan is None:
                    augmentation_stop_reason = "no_new_compatible_augmentation"
                    break
                composition_plans.append(composition_plan)
                champion_name = composition_plan.get("search", {}).get(
                    "augmentation_champion"
                )
                progress.emit(
                    "composition",
                    f"champion augmentation round {round_index}: {champion_name}",
                )
                errors = execution_errors(composition_plan)
                if errors:
                    raise ValueError(
                        "generated composition spec is invalid: " + "; ".join(errors)
                    )
                suffix = "" if round_index == 1 else f"-{round_index}"
                composition_spec_path = root / f"composition-spec{suffix}.json"
                composition_path = root / f"composition{suffix}.json"
                if resumed and not search_policy_changed and composition_path.is_file():
                    if not composition_spec_path.is_file():
                        raise RuntimeError(
                            f"cannot resume {composition_path.name} without "
                            f"{composition_spec_path.name}"
                        )
                    spec_mismatches = reusable_stage_spec_mismatches(
                        composition_plan, load_json(composition_spec_path)
                    )
                    if spec_mismatches:
                        raise RuntimeError(
                            "cannot reuse composition evidence after spec changes: "
                            + ", ".join(spec_mismatches)
                        )
                    round_result = load_json(composition_path)
                    if round_result.get("stop_reason") not in {
                        "completed_search", "strong_candidate_early_stop",
                        "consecutive_failure_budget_exhausted",
                    }:
                        raise RuntimeError(
                            f"cannot resume incomplete {composition_path.name}"
                        )
                    progress.emit(
                        "composition",
                        f"reused round {round_index} with "
                        f"{round_result.get('completed_trials', 0)} completed trials",
                    )
                else:
                    write_json(composition_spec_path, composition_plan)
                    round_result = execute_with_progress(
                        composition_plan, progress,
                        f"champion augmentation {round_index}",
                    )
                write_json(composition_path, round_result)
                composition_rounds.append(round_result)
                used_trials += round_result["completed_trials"]
                used_gpu_hours += round_result["approx_gpu_hours"]
                elapsed_minutes = (time.monotonic() - started) / 60
                remaining_trials -= round_result["completed_trials"]
                remaining_gpu_hours = (
                    float(task["budget"]["max_gpu_hours"]) - used_gpu_hours
                )
                remaining_wall_minutes = (
                    float(task["budget"]["max_wall_time_minutes"]) - elapsed_minutes
                )

                _, round_outcome = champion_augmentation_round_outcome(
                    champion_name, composition_input, round_result,
                    search_plan.get("history", {}).get("priors"),
                )
                composition_round_evidence.append({
                    "round": round_index,
                    "tested_candidates": deepcopy(
                        composition_plan.get("search", {}).get(
                            "champion_augmentation_candidates", []
                        )
                    ),
                    "completed_trials": round_result.get("completed_trials", 0),
                    **round_outcome,
                    "remaining_trials": remaining_trials,
                })
                composition_input = merge_stage_evidence(
                    composition_input, round_result
                )
                if isinstance(search_plan.get("candidate_registry"), dict):
                    search_plan["candidate_registry"] = update_registry_from_aggregates(
                        search_plan["candidate_registry"],
                        round_result.get("aggregates", []),
                    )
                if not round_outcome["advanced"]:
                    augmentation_stop_reason = "no_parent_relative_improvement"
                    break
                augmentation_stop_reason = "budget_exhausted_after_champion_update"
                round_index += 1

            if composition_rounds:
                composition = merge_stage_evidence(
                    composition_seed, *composition_rounds,
                    include_screen_usage=False,
                )
                composition["elapsed_sec"] = sum(
                    float(item.get("elapsed_sec", 0) or 0)
                    for item in composition_rounds
                )
                composition["augmentation_rounds"] = deepcopy(
                    composition_round_evidence
                )
                composition["augmentation_stop_reason"] = augmentation_stop_reason
                write_json(root / "composition-rounds.json", {
                    "rounds": composition_round_evidence,
                    "stop_reason": augmentation_stop_reason,
                    "merged": composition,
                })
            interaction = merge_stage_evidence(
                screen, refinement, *composition_rounds,
                include_screen_usage=False,
            )
            if isinstance(search_plan.get("candidate_registry"), dict):
                search_plan["candidate_registry"] = update_registry_from_aggregates(
                    search_plan["candidate_registry"], interaction.get("aggregates", [])
                )
                write_json(
                    root / "candidate-registry-interaction.json",
                    search_plan["candidate_registry"],
                )
                write_json(root / "search-plan.json", search_plan)
            interaction_plan = composition_plan or refinement_plan
            write_json(root / "interaction.json", interaction)
        except (ValueError, RuntimeError) as exc:
            interaction_error = str(exc)
    interaction_candidate_rows = [
        row for row in (interaction or {}).get("results", [])
        if row.get("kind") == "candidate"
    ]
    all_parameter_candidate_rows = [
        *attempted_parameter_candidates, *interaction_candidate_rows,
    ]
    parameter_search.update({
        "attempted_parameter_candidates": len(all_parameter_candidate_rows),
        "executed_parameter_candidates": sum(
            1 for row in all_parameter_candidate_rows if row.get("ok")
        ),
        "failed_parameter_candidates": sum(
            1 for row in all_parameter_candidate_rows if not row.get("ok")
        ),
        "stage_candidate_counts": {
            "screening": len(attempted_parameter_candidates),
            "refinement": sum(
                1 for row in (refinement or {}).get("results", [])
                if row.get("kind") == "candidate"
            ),
            "composition": sum(
                1 for row in (composition or {}).get("results", [])
                if row.get("kind") == "candidate"
            ),
        },
        "continued_exploration_candidates": deepcopy(
            (refinement_plan or {}).get("search", {}).get(
                "mechanism_followup", {}
            ).get("continued_exploration", [])
        ),
        "measurement_tiers": {
            "screening": screen_spec.get("search", {}).get(
                "measurement_tier", "standard"
            ),
            "refinement": (refinement_plan or {}).get("search", {}).get(
                "measurement_tier"
            ),
            "composition": (composition_plan or {}).get("search", {}).get(
                "measurement_tier"
            ),
        },
        "champion_augmentation": {
            "initial_champion": (
                composition_plans[0].get("search", {}).get("augmentation_champion")
                if composition_plans else None
            ),
            "final_champion": (
                composition_round_evidence[-1].get("champion_out")
                if composition_round_evidence else None
            ),
            "rounds": deepcopy(composition_round_evidence),
            "round_count": len(composition_round_evidence),
            "generated_candidates": [
                name for plan_value in composition_plans
                for name in plan_value.get("search", {}).get(
                    "champion_augmentation_candidates", []
                )
            ],
            "budget_omitted": sum(
                int(plan_value.get("search", {}).get(
                    "budget_omitted_combinations", 0
                ) or 0)
                for plan_value in composition_plans
            ),
            "stop_reason": (
                composition.get("augmentation_stop_reason")
                if isinstance(composition, dict) else None
            ),
            "policy": (
                composition_plans[0].get("search", {}).get(
                    "champion_augmentation_policy"
                ) if composition_plans else None
            ),
        },
    })
    confirmation: dict[str, Any] | None = None
    confirmation_attempts: list[dict[str, Any]] = []
    confirmation_spec_paths: list[Path] = []
    selected_confirmation_spec: Path | None = None
    confirmation_error: str | None = None
    decision_input = confirmation_candidate_pool(
        screen, interaction, search_plan.get("history", {}).get("priors")
    )
    confirmation_cache_path = root / "confirmation.json"
    if resumed and not search_policy_changed and confirmation_cache_path.is_file():
        cached_confirmation = load_json(confirmation_cache_path)
        cached_attempts = cached_confirmation.get("attempts", [])
        cached_selected = cached_confirmation.get("selected")
        if not isinstance(cached_attempts, list) or not isinstance(cached_selected, dict):
            raise RuntimeError("cannot resume invalid confirmation.json")
        confirmation_attempts = cached_attempts
        confirmation = cached_selected
        for rank, attempt in enumerate(confirmation_attempts, 1):
            used_trials += int(attempt.get("completed_trials", 0))
            used_gpu_hours += float(attempt.get("approx_gpu_hours", 0))
            if attempt.get("run_dir") == confirmation.get("run_dir"):
                candidate_spec_path = root / f"confirmation-spec-{rank}.json"
                if candidate_spec_path.is_file():
                    selected_confirmation_spec = candidate_spec_path
        remaining_trials = int(task["budget"]["max_trials"]) - used_trials_before_profile - (
            0 if reused_profile else 1
        ) - used_trials
        remaining_gpu_hours = float(task["budget"]["max_gpu_hours"]) - used_gpu_hours
        remaining_wall_minutes = float(task["budget"]["max_wall_time_minutes"]) - (
            time.monotonic() - started
        ) / 60
        progress.emit(
            "confirmation",
            f"reused {sum(int(item.get('completed_trials', 0)) for item in confirmation_attempts)} completed confirmation windows",
        )
    if confirmation is None and executed_parameter_candidates and decision_input["stop_reason"] in {
        "completed_search", "strong_candidate_early_stop",
    }:
        candidates = decision_input.get("confirmation_candidates") or [
            decision_input.get("screening_winner")
        ]
        minimum_confirmation_trials = 2 * int(
            (execution_task.get("measurement") or {}).get("bayesian_min_blocks", 2)
        )
        confirmation_candidate_limit = min(
            len(candidates),
            max(1, remaining_trials // max(1, minimum_confirmation_trials)),
        )
        candidates = candidates[:confirmation_candidate_limit]
        decision_input["confirmation_allocation"] = {
            "candidate_limit": confirmation_candidate_limit,
            "minimum_trials_per_candidate": minimum_confirmation_trials,
            "remaining_trials_at_entry": remaining_trials,
            "history_prior_policy": "weak exact-compatible Bayesian prior; no history candidate trials",
        }
        for rank, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict):
                continue
            candidate_input = deepcopy(decision_input)
            candidate_input["screening_winner"] = candidate
            try:
                confirm_spec = confirmation_spec(
                    execution_task,
                    plan["discovery"],
                    candidate_input,
                    remaining_trials,
                    remaining_gpu_hours,
                    remaining_wall_minutes,
                )
                spec_path = root / f"confirmation-spec-{rank}.json"
                write_json(spec_path, confirm_spec)
                progress.emit(
                    "confirmation",
                    f"confirming ranked candidate {rank}/{len(candidates)} against the baseline",
                )
                attempt = execute_with_progress(confirm_spec, progress, f"confirmation {rank}")
                confirmation_attempts.append(attempt)
                confirmation_spec_paths.append(spec_path)
                used_trials += attempt["completed_trials"]
                used_gpu_hours += attempt["approx_gpu_hours"]
                elapsed_minutes = (time.monotonic() - started) / 60
                remaining_trials -= attempt["completed_trials"]
                remaining_gpu_hours = float(task["budget"]["max_gpu_hours"]) - used_gpu_hours
                remaining_wall_minutes = float(task["budget"]["max_wall_time_minutes"]) - elapsed_minutes
            except (ValueError, RuntimeError) as exc:
                confirmation_error = str(exc)
                break
        if confirmation_attempts:
            confirmed_attempts = [
                item for item in confirmation_attempts
                if item.get("recommendation_status") == "confirmed_candidate"
            ]
            confirmation = max(
                confirmed_attempts or confirmation_attempts,
                key=lambda item: float(
                    (item.get("recommended_configuration") or {}).get("comparison", {}).get(
                        "improvement_pct", float("-inf")
                    )
                ),
            )
            selected_confirmation_spec = confirmation_spec_paths[
                confirmation_attempts.index(confirmation)
            ]
            write_json(root / "confirmation.json", {
                "attempts": confirmation_attempts,
                "selected": confirmation,
            })
    if executed_parameter_candidates:
        decision = confirmation or decision_input
    else:
        decision = {
            **decision_input,
            "recommended_configuration": None,
            "recommendation_status": "insufficient_parameter_evidence",
            "recommendation_reason": (
                "no deployment parameter candidate completed; inspect failed candidates or increase the trial budget"
            ),
        }
    performance_candidate = best_unconfirmed_performance_candidate(
        decision_input, confirmation
    )
    if (
        performance_candidate is not None
        and decision.get("recommendation_status") != "confirmed_candidate"
    ):
        decision["provisional_configuration"] = performance_candidate
    if not parameter_search["sufficient_evidence"]:
        provisional = performance_candidate or decision.get("recommended_configuration")
        blocking_reasons = [
            *parameter_search.get("blocking_missing_mechanism_classes", []),
            *parameter_search.get("missing_mandatory_capacity_parameters", []),
        ]
        if not parameter_search.get("coverage_floor_met", False):
            blocking_reasons.append(
                f"measured_mechanism_floor<{parameter_search['required_parameter_breadth']}"
            )
        decision = {
            **decision,
            "provisional_configuration": provisional,
            "recommended_configuration": None,
            "recommendation_status": "insufficient_optimization_evidence",
            "recommendation_reason": (
                "the run did not complete mandatory optimization evidence: "
                + ", ".join(blocking_reasons or ["required parameter breadth or capacity controls"])
            ),
        }
    quality_gate = recommendation_quality_gate(task, decision.get("recommended_configuration"))
    if (
        decision.get("recommendation_status") == "confirmed_candidate"
        and quality_gate["required"]
        and not quality_gate["passed"]
    ):
        provisional = decision.get("recommended_configuration")
        decision = {
            **decision,
            "provisional_configuration": provisional,
            "recommended_configuration": None,
            "recommendation_status": "confirmed_performance_candidate_quality_unverified",
            "recommendation_reason": quality_gate["reason"],
        }
    recommendation = decision.get("recommended_configuration")
    deployment_spec = (
        load_json(selected_confirmation_spec) if selected_confirmation_spec is not None else screen_spec
    )
    baseline_evidence = next(
        (
            item for item in (confirmation or {}).get("aggregates", [])
            if isinstance(item, dict) and item.get("kind") == "baseline"
        ),
        deepcopy(screen.get("aggregates", [{}])[0]),
    )
    minimal_deploy_command = (
        final_server_command(deployment_spec, recommendation)
        if recommendation is not None else None
    )
    deploy_command = (
        reproducible_server_command(
            deployment_spec, recommendation, search_plan.get("resolved_baseline", {})
        )
        if recommendation is not None else None
    )
    baseline_deploy_command_minimal = (
        final_server_command(screen_spec, baseline_evidence)
        if isinstance(baseline_evidence, dict) and baseline_evidence.get("config") else None
    )
    baseline_deploy_command_reproducible = (
        reproducible_server_command(
            screen_spec, baseline_evidence, search_plan.get("resolved_baseline", {})
        )
        if isinstance(baseline_evidence, dict) and baseline_evidence.get("config") else None
    )
    host_plan = host_deployment_plan(
        execution_task, plan["discovery"], recommendation, deploy_command
    )
    total_gpu_hours = used_gpu_hours
    budget_allocation = deepcopy(plan["budget_allocation"])
    budget_used = {
        "discovery": (
            used_trials_before_profile + (0 if reused_profile else 1)
            + screen_stage_completed_trials
        ),
        "refinement": int(interaction.get("completed_trials", 0)) if interaction else 0,
        "confirmation": sum(
            int(attempt.get("completed_trials", 0)) for attempt in confirmation_attempts
        ),
    }
    budget_total_used = sum(budget_used.values())
    budget_accounting = {
        **budget_allocation,
        "used": budget_used,
        "used_percentages": {
            key: round(value / max(1, budget_total_used) * 100, 2)
            for key, value in budget_used.items()
        },
        "unused_trials": max(
            0, int(task["budget"]["max_trials"]) - budget_total_used
        ),
    }
    economics_summary = cost_per_token_summary(
        task, decision, plan["discovery"]["derived"]["visible_gpu_count"], total_gpu_hours
    )
    if isinstance(search_plan.get("candidate_registry"), dict):
        posterior_bottleneck = posterior_bottleneck_report(
            search_plan.get("canonical_bottleneck", {}),
            search_plan["candidate_registry"],
            minimum_improvement_pct=float(
                task.get("objective", {}).get("min_improvement_pct", 0)
            ),
        )
        search_plan["canonical_bottleneck"] = posterior_bottleneck
        registry = CandidateRegistry.from_dict(search_plan["candidate_registry"])
        registry.canonical_bottleneck = deepcopy(posterior_bottleneck)
        search_plan["candidate_registry"] = registry.to_dict()
    stage_elapsed_seconds = {
        "calibration": float((calibration or {}).get("elapsed_sec", 0) or 0),
        "cookbook_initial": float((initial_screen or {}).get("elapsed_sec", 0) or 0),
        "profiling": float((profiling or {}).get("elapsed_sec", 0) or 0),
        "screening": screen_stage_elapsed_sec,
        "refinement": float((refinement or {}).get("elapsed_sec", 0) or 0),
        "composition": float((composition or {}).get("elapsed_sec", 0) or 0),
        "confirmation": sum(
            float(item.get("elapsed_sec", 0) or 0)
            for item in confirmation_attempts if isinstance(item, dict)
        ),
    }
    current_process_elapsed_sec = time.monotonic() - started
    final = {
        "schema_version": 3,
        "experiment_mode": normalized_experiment_mode(task),
        "run_dir": str(root),
        "completed_at": utc_now(),
        "elapsed_sec": current_process_elapsed_sec,
        "current_process_elapsed_sec": current_process_elapsed_sec,
        "logical_experiment_elapsed_sec": sum(stage_elapsed_seconds.values()),
        "stage_elapsed_seconds": stage_elapsed_seconds,
        "discovery": plan["discovery"],
        "deployment_policy": plan["deployment_policy"],
        "calibration": calibration,
        "cookbook_preflight": {
            "candidate_bundles": cookbook_initial["cookbook_candidate_bundles"],
            "excluded_bundles": cookbook_initial["cookbook_bundle_exclusions"],
            "hardware_prior_candidates": cookbook_initial.get(
                "cookbook_hardware_prior_candidates", []
            ),
            "topology_candidates": cookbook_initial["ranked_parameter_groups"],
            "policy": cookbook_initial["policy"],
        },
        "cookbook_initial_screen": initial_screen,
        "cookbook_snapshot": cookbook_snapshot,
        "cookbook_automatic_behaviors": cookbook_automatic_behavior_report(
            plan["discovery"], profiling
        ),
        "profiled_initial_configuration": profiled_initial_configuration,
        "post_preprofile_anchor_configuration": initial_candidate,
        "raw_sglang_baseline": raw_baseline,
        "requested_workload": requested_task["workload"],
        "execution_workload": execution_task["workload"],
        "analysis_workload": analysis_task["workload"],
        "requested_slo": task["slo"],
        "measurement_policy": analysis_task.get("measurement", {}),
        "profiling": profiling,
        "profiling_reused": reused_profile,
        "resume_evidence": {
            "resumed": resumed,
            "search_policy_changed": search_policy_changed,
            "profile_reused": reused_profile,
            "screening_reused": reuse_screening,
            "policy": (
                "raw profiles may survive optimizer-policy changes; screening, refinement, "
                "composition, and confirmation may not"
            ),
        },
        "parallel_pipeline": {
            # Preprofile overlap and post-profile screening are distinct.  A
            # run may profile serially and still screen candidates on every
            # selected GPU, so do not use the former as a proxy for the latter.
            "enabled": pipeline_spec is not None,
            "error": pipeline_error,
            "profile_gpu": selected_identifiers[0] if selected_identifiers else None,
            "screening_gpus": selected_identifiers[:requested_workers],
            "screening_parallel_workers": requested_workers,
            "screening_gpu_allocation": "exclusive",
            "policy": (
                "Nsys and independent workload-prior screening overlap; trace-routed candidates are deduplicated afterward"
                if pipeline_spec is not None else
                "serial Nsys profiling followed by exclusive-GPU parallel screening"
            ),
        },
        "startup_acceleration": deepcopy(
            search_plan.get("startup_acceleration", {})
        ),
        "search_plan": search_plan,
        "bottleneck_class": search_plan.get("canonical_bottleneck", {}).get("primary"),
        "bottleneck_classification": search_plan.get("bottleneck_classification"),
        "canonical_bottleneck": search_plan.get("canonical_bottleneck"),
        "candidate_registry": search_plan.get("candidate_registry"),
        "mechanism_search": {
            "discovery": search_plan.get("mechanism_schedule"),
            "outcomes_after_screen": search_plan.get("mechanism_outcomes_after_screen"),
        },
        "parameter_evolution": search_plan.get(
            "parameter_evolution",
            compact_evolution_summary(
                plan["discovery"].get("parameter_evolution", {})
            ),
        ),
        "parameter_search": parameter_search,
        "budget_accounting": budget_accounting,
        "screening": screen,
        "refinement": refinement,
        "composition": composition,
        "composition_rounds": composition_rounds,
        "champion_augmentation_rounds": composition_round_evidence,
        "composition_parent_gates": deepcopy(
            decision_input.get("composition_parent_gates", [])
        ),
        "interaction": interaction,
        "interaction_error": interaction_error,
        "confirmation": confirmation,
        "confirmation_attempts": confirmation_attempts,
        "confirmation_error": confirmation_error,
        "bottleneck": {
            "nsys": profiling["diagnosis"],
            "screening_mechanism": bottleneck_summary(screen, execution_task),
            "operator_escalation": operator_escalation_plan(profiling),
        },
        "kernel_optimization": {
            "fused_moe": moe_tuning_plan,
            "fused_moe_execution": kernel_tuning,
        },
        "recommendation_status": decision.get("recommendation_status"),
        "recommendation_reason": decision.get("recommendation_reason"),
        "recommended_configuration": recommendation,
        "provisional_configuration": decision.get("provisional_configuration"),
        "safe_baseline_configuration": baseline_evidence,
        "quality_gate": quality_gate,
        "economics": economics_summary,
        "deployment_command": deploy_command,
        "deployment_command_minimal": minimal_deploy_command,
        "deployment_command_reproducible": deploy_command,
        "safe_baseline_deployment_command_minimal": baseline_deploy_command_minimal,
        "safe_baseline_deployment_command_reproducible": baseline_deploy_command_reproducible,
        "host_deployment_plan": host_plan,
        "deployment_environment": (
            {
                **execution_task.get("env", {}),
                **(recommendation.get("env", {}) if isinstance(recommendation, dict) else {}),
            }
            if recommendation is not None else {}
        ),
        "deployable": recommendation is not None and decision.get("recommendation_status") in {
            "confirmed_candidate", "retain_confirmed_baseline"
        },
        "total_completed_trials": used_trials_before_profile + (0 if reused_profile else 1) + used_trials,
        "total_approx_gpu_hours": total_gpu_hours,
    }
    history = task.get("history", {}) if isinstance(task.get("history"), dict) else {}
    if history.get("enabled", True):
        try:
            database = history_database_path(task)
            trial_ingestion = ingest_trial_history(database, final, task)
            _, history_components = history_compatibility_fingerprint(
                task, final["discovery"]
            )
            contract_ingestion = save_parameter_contract(
                database,
                build_parameter_contract(
                    final["discovery"]["parameter_catalog"],
                    final["discovery"].get("framework"),
                ),
                history_components["framework_fingerprint"],
            )
            evolution_evidence = ingest_parameter_evidence(
                database, final, task
            )
            final["history_ingestion"] = {
                "status": "completed",
                **trial_ingestion,
                "parameter_contract": contract_ingestion,
                "parameter_evidence": evolution_evidence,
            }
        except (OSError, ValueError, sqlite3.Error) as exc:
            final["history_ingestion"] = {
                "status": "failed", "reason": str(exc),
                "database": str(history_database_path(task)),
            }
    else:
        final["history_ingestion"] = {"status": "disabled"}
    write_json(root / "final.json", final)
    progress.emit(
        "complete",
        f"{final['recommendation_status']}; completed {final['total_completed_trials']} trials. "
        f"Result: {root / 'final.json'}",
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--task", required=True)
    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--task", required=True)
    discover_parser.add_argument("--output")
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--task", required=True)
    plan_parser.add_argument("--output")
    search_parser = subparsers.add_parser("search-plan")
    search_parser.add_argument("--task", required=True)
    search_parser.add_argument("--profile", required=True)
    search_parser.add_argument("--output")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--task", required=True)
    run_parser.add_argument("--yes", action="store_true")
    run_parser.add_argument("--output")
    args = parser.parse_args()
    try:
        task = load_json(args.task)
        if args.command == "validate":
            errors = validate_task(task)
            dump_json({"valid": not errors, "errors": errors}, None)
            return 0 if not errors else 2
        if args.command == "discover":
            errors = validate_task(task)
            if errors:
                raise ValueError("; ".join(errors))
            dump_json(discover(task), args.output)
            return 0
        if args.command == "plan":
            dump_json(build_plan(task), args.output)
            return 0
        if args.command == "search-plan":
            errors = validate_task(task)
            if errors:
                raise ValueError("; ".join(errors))
            profile = load_json(args.profile)
            if not isinstance(profile.get("diagnosis"), dict):
                raise ValueError("profile must contain a completed nsys diagnosis")
            dump_json(diagnosed_search_plan(task, discover(task), profile), args.output)
            return 0
        if args.command == "run":
            if not args.yes:
                raise ValueError("run requires --yes after reviewing the one-shot plan")
            dump_json(run_autopilot(task), args.output)
            return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
