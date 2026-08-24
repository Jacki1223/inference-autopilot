"""Private SQLite history for compatible InferOpt trials and warm starts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from statistics import median
from typing import Any


SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def open_store(path: str | Path) -> sqlite3.Connection:
    database = Path(path).expanduser()
    if not database.is_absolute() or database == Path("/"):
        raise ValueError("history database must be an absolute non-root path")
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
          run_dir TEXT PRIMARY KEY,
          compatibility_fingerprint TEXT NOT NULL,
          completed_at TEXT,
          recommendation_status TEXT,
          objective_metric TEXT NOT NULL,
          model_fingerprint TEXT NOT NULL,
          hardware_fingerprint TEXT NOT NULL,
          workload_fingerprint TEXT NOT NULL,
          framework_fingerprint TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS runs_compatibility_idx
          ON runs(compatibility_fingerprint);
        CREATE TABLE IF NOT EXISTS trials (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_dir TEXT NOT NULL REFERENCES runs(run_dir) ON DELETE CASCADE,
          stage TEXT NOT NULL,
          configuration_name TEXT NOT NULL,
          config_hash TEXT NOT NULL,
          config_json TEXT NOT NULL,
          objective_value REAL,
          improvement_pct REAL,
          slo_passed INTEGER,
          ok INTEGER NOT NULL,
          metrics_json TEXT,
          UNIQUE(run_dir, stage, configuration_name, config_hash)
        );
        CREATE INDEX IF NOT EXISTS trials_config_idx
          ON trials(config_hash);
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    connection.commit()
    return connection


def compatibility_components(
    task: dict[str, Any], discovery: dict[str, Any]
) -> dict[str, str]:
    model = discovery.get("model", {})
    hardware = discovery.get("hardware", {})
    framework = discovery.get("framework", {})
    selected = hardware.get("gpus", [])
    model_payload = {
        "config_sha256": model.get("config_sha256"),
        "checkpoint_name": model.get("checkpoint_name"),
        "weight_bytes": model.get("weight_bytes"),
        "weight_quantization": model.get("weight_quantization"),
        "architectures": model.get("architectures"),
    }
    hardware_payload = {
        "vendor": hardware.get("vendor"),
        "gpus": [
            {
                "name": gpu.get("canonical_name", gpu.get("name")),
                "memory_mib": gpu.get("memory_mib"),
                "compute_capability": gpu.get("compute_capability"),
            }
            for gpu in selected
        ],
        "topology": discovery.get("topology_class"),
    }
    workload = task.get("workload", {})
    workload_payload = {
        "deployment_mode": task.get("deployment_mode"),
        "input_tokens": workload.get("input_tokens"),
        "output_tokens": workload.get("output_tokens"),
        "prefix_reuse_ratio": workload.get("prefix_reuse_ratio", 0),
        "dataset": workload.get("dataset", {"name": "synthetic"}),
        "request_rate": workload.get("request_rate", "inf"),
        "objective": task.get("objective"),
        "slo": task.get("slo"),
    }
    framework_payload = {
        "git_commit": framework.get("git_commit"),
        "server_args_sha256": framework.get("server_args_sha256"),
        "parameter_catalog_server_args_sha256": discovery.get(
            "parameter_catalog", {}
        ).get("parameter_contract", {}).get("server_args_sha256"),
    }
    return {
        "model_fingerprint": hash_payload(model_payload),
        "hardware_fingerprint": hash_payload(hardware_payload),
        "workload_fingerprint": hash_payload(workload_payload),
        "framework_fingerprint": hash_payload(framework_payload),
    }


def compatibility_fingerprint(
    task: dict[str, Any], discovery: dict[str, Any]
) -> tuple[str, dict[str, str]]:
    components = compatibility_components(task, discovery)
    return hash_payload(components), components


def _stage_rows(final: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    value = final.get(stage)
    return value.get("results", []) if isinstance(value, dict) else []


def ingest_final(
    path: str | Path, final: dict[str, Any], task: dict[str, Any]
) -> dict[str, Any]:
    fingerprint, components = compatibility_fingerprint(task, final["discovery"])
    run_dir = str(final["run_dir"])
    objective_metric = str(task["objective"]["metric"])
    connection = open_store(path)
    try:
        connection.execute(
            """INSERT OR REPLACE INTO runs(
                 run_dir, compatibility_fingerprint, completed_at,
                 recommendation_status, objective_metric, model_fingerprint,
                 hardware_fingerprint, workload_fingerprint, framework_fingerprint
               ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_dir, fingerprint, final.get("completed_at"),
                final.get("recommendation_status"), objective_metric,
                components["model_fingerprint"], components["hardware_fingerprint"],
                components["workload_fingerprint"], components["framework_fingerprint"],
            ),
        )
        connection.execute("DELETE FROM trials WHERE run_dir = ?", (run_dir,))
        inserted = 0
        for stage in ("screening", "interaction", "confirmation"):
            aggregates = {
                item.get("configuration_name"): item
                for item in (final.get(stage) or {}).get("aggregates", [])
                if isinstance(item, dict)
            }
            for row in _stage_rows(final, stage):
                if not isinstance(row, dict):
                    continue
                config = row.get("config", {})
                aggregate = aggregates.get(row.get("configuration_name"), {})
                metrics = row.get("metrics", {})
                comparison = aggregate.get("comparison", {})
                connection.execute(
                    """INSERT OR REPLACE INTO trials(
                         run_dir, stage, configuration_name, config_hash,
                         config_json, objective_value, improvement_pct,
                         slo_passed, ok, metrics_json
                       ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_dir, stage, str(row.get("configuration_name", row.get("name"))),
                        hash_payload(config), canonical_json(config),
                        metrics.get(objective_metric), comparison.get("improvement_pct"),
                        int(bool(row.get("slo", {}).get("passed", False))),
                        int(bool(row.get("ok", False))), canonical_json(metrics),
                    ),
                )
                inserted += 1
        connection.commit()
        return {
            "database": str(Path(path).expanduser()),
            "compatibility_fingerprint": fingerprint,
            "inserted_trials": inserted,
        }
    finally:
        connection.close()


def warm_start_candidates(
    path: str | Path, fingerprint: str, *, limit: int = 5
) -> list[dict[str, Any]]:
    database = Path(path).expanduser()
    if not database.is_file() or limit <= 0:
        return []
    connection = open_store(database)
    try:
        rows = connection.execute(
            """SELECT t.config_hash, t.config_json, t.improvement_pct,
                      t.slo_passed, t.ok, t.run_dir
               FROM trials t JOIN runs r ON r.run_dir = t.run_dir
               WHERE r.compatibility_fingerprint = ?
                 AND t.improvement_pct IS NOT NULL
                 AND t.ok = 1 AND t.slo_passed = 1
                 AND t.config_json != '{}'
            """,
            (fingerprint,),
        ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(str(row["config_hash"]), []).append(row)
        ranked = []
        for config_hash, samples in grouped.items():
            improvements = [float(row["improvement_pct"]) for row in samples]
            ranked.append({
                "name": f"history-{config_hash[:12]}",
                "config": json.loads(str(samples[0]["config_json"])),
                "history_score_pct": median(improvements),
                "history_samples": len(improvements),
                "source_runs": sorted({str(row["run_dir"]) for row in samples}),
                "reason": "strictly compatible historical configuration",
                "priority": "high",
            })
        ranked.sort(
            key=lambda item: (item["history_score_pct"], item["history_samples"]),
            reverse=True,
        )
        return ranked[:limit]
    finally:
        connection.close()
