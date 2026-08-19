import json
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import autopilot
import autotune
import inferopt
import inferopt_cli
import profile_sglang
import sglang_runtime


class ProgressReporterTests(unittest.TestCase):
    def test_progress_bar_reports_exact_stage_fraction(self):
        self.assertEqual(
            autopilot.ProgressReporter.bar(6, 11, width=10),
            "[#####-----] 6/11 54%",
        )

    def test_started_and_finished_trials_advance_progress(self):
        reporter = autopilot.ProgressReporter()
        with mock.patch.object(reporter, "emit") as emit:
            reporter.trial("screen", {
                "event": "trial_started", "trial_index": 2, "trial_count": 4,
                "trial_name": "candidate",
            })
            self.assertEqual(emit.call_args.kwargs, {"completed": 1, "total": 4})
            reporter.trial("screen", {
                "event": "trial_finished", "trial_index": 2, "trial_count": 4,
                "trial_name": "candidate", "ok": True, "metrics": {}, "slo_passed": True,
            })
            self.assertEqual(emit.call_args.kwargs, {"completed": 2, "total": 4})


class LocalCookbookKnowledgeTests(unittest.TestCase):
    def make_checkout(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        repository = Path(temporary.name)
        recipe = repository / "docs" / "cookbook" / "autoregressive" / "Qwen" / "Qwen3.5.mdx"
        recipe.parent.mkdir(parents=True)
        recipe.write_text(
            "# Qwen3.5\n\n"
            "```bash\n"
            "SGLANG_ENABLE_SPEC_V2=1 python -m sglang.launch_server \\\n"
            "  --model-path Qwen/Qwen3.5-27B \\\n"
            "  --tp 8 \\\n"
            "  --speculative-algo NEXTN \\\n"
            "  --speculative-num-steps 3 \\\n"
            "  --mamba-radix-cache-strategy extra_buffer \\\n"
            "  --enable-mixed-chunk\n"
            "```\n",
            encoding="utf-8",
        )
        return temporary, repository

    def test_local_checkout_recipes_are_parsed_without_network(self):
        temporary, repository = self.make_checkout()
        self.addCleanup(temporary.cleanup)
        unrelated = repository / "docs" / "cookbook" / "autoregressive" / "Other" / "MiniCPM.mdx"
        unrelated.parent.mkdir(parents=True)
        unrelated.write_text("# MiniCPM\nQwen3.5 is mentioned only for comparison.\n", encoding="utf-8")
        evidence = autopilot.local_cookbook_evidence(
            {"repository": str(repository)},
            {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]},
        )
        self.assertEqual(evidence["status"], "available")
        self.assertEqual(len(evidence["documents"]), 1)
        self.assertEqual(evidence["documents"][0]["path"], "autoregressive/Qwen/Qwen3.5.mdx")
        recipe = evidence["recipes"][0]
        self.assertEqual(recipe["config"]["tp_size"], 8)
        self.assertEqual(recipe["config"]["speculative_algorithm"], "NEXTN")
        self.assertEqual(recipe["config"]["speculative_num_steps"], 3)
        self.assertEqual(recipe["config"]["mamba_radix_cache_strategy"], "extra_buffer")
        self.assertEqual(recipe["config"]["page_size"], 64)
        self.assertIn("checkpoint.has_mtp_weights", recipe["requirements"])

    def test_local_recipe_becomes_profile_candidate_when_no_builtin_match_exists(self):
        temporary, repository = self.make_checkout()
        self.addCleanup(temporary.cleanup)
        task = {"repository": str(repository), "allow_download": False, "knowledge": {}}
        model = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}
        evidence = autopilot.cookbook_evidence(task, model)
        self.assertEqual(evidence["status"], "available")
        bundles = evidence["model_profile"]["initial_bundles"]
        parsed = next(bundle for bundle in bundles if bundle["name"].startswith("cookbook-qwen3.5"))
        self.assertNotIn("tp_size", parsed["config"])
        self.assertEqual(parsed["config"]["speculative_algorithm"], "NEXTN")

    def test_series_page_does_not_apply_a_different_size_variant(self):
        temporary, repository = self.make_checkout()
        self.addCleanup(temporary.cleanup)
        evidence = autopilot.local_cookbook_evidence(
            {"repository": str(repository)},
            {
                "checkpoint_name": "Qwen3.5-7B",
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
            },
        )
        self.assertEqual(evidence["status"], "available")
        self.assertEqual(evidence["recipes"], [])
        self.assertIn("does not match the local checkpoint size", evidence["excluded_recipes"][0]["reason"])


class CapabilityCircuitBreakerTests(unittest.TestCase):
    def test_mtp_dependency_failure_skips_remaining_mtp_candidates(self):
        trials = [
            {
                "name": "mtp-first", "configuration_name": "mtp-first", "repeat_index": 0,
                "kind": "candidate", "config": {"speculative_algorithm": "EAGLE"},
            },
            {
                "name": "mtp-cache", "configuration_name": "mtp-cache", "repeat_index": 0,
                "kind": "candidate", "config": {
                    "speculative_algorithm": "EAGLE", "page_size": 64,
                },
            },
            {
                "name": "cache", "configuration_name": "cache", "repeat_index": 0,
                "kind": "candidate", "config": {"page_size": 64},
            },
        ]
        spec = {
            "execution": {"require_accelerator": False},
            "budget": {"max_wall_time_minutes": 1, "max_gpu_hours": 1},
            "hardware": {"gpus_per_host": 1},
        }
        calls: list[str] = []

        def fake_run_trial(_spec, trial, trial_dir, _remaining):
            calls.append(trial["name"])
            trial_dir.mkdir(parents=True, exist_ok=True)
            if trial["name"] == "mtp-first":
                (trial_dir / "server.log").write_text(
                    "ModuleNotFoundError: No module named 'cutlass'\n", encoding="utf-8"
                )
                return {
                    "ok": False,
                    "status": {"state": "failed", "detail": "server exited", "failure_class": "dependency_missing"},
                }
            return {
                "ok": True,
                "summary": {"metrics": {}, "slo": {"passed": True}},
                "status": {"state": "completed"},
            }

        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            run_dir.mkdir()
            with mock.patch.object(autotune, "enable_child_subreaper", return_value=False), \
                 mock.patch.object(autotune, "execution_errors", return_value=[]), \
                 mock.patch.object(autotune, "prepare_run", return_value=(run_dir, trials)), \
                 mock.patch.object(autotune, "run_trial", side_effect=fake_run_trial), \
                 mock.patch.object(autotune, "decision_report", return_value={"aggregates": []}):
                result = autotune.execute(spec)

        self.assertEqual(calls, ["mtp-first", "cache"])
        self.assertEqual(result["completed_trials"], 2)
        self.assertEqual(result["skipped_capability_trials"][0]["name"], "mtp-cache")
        self.assertEqual(result["disabled_capabilities"][0]["reason"], "missing Python module: cutlass")


class AdaptiveEarlyStopTests(unittest.TestCase):
    def test_strong_gain_stops_only_after_successful_coverage(self):
        trials = [
            {
                "name": "baseline", "configuration_name": "baseline", "repeat_index": 0,
                "kind": "baseline", "config": {"tp_size": 1},
            },
            *[
                {
                    "name": f"candidate-{index}",
                    "configuration_name": f"candidate-{index}",
                    "repeat_index": 0,
                    "kind": "candidate",
                    "config": {"page_size": index},
                }
                for index in range(1, 6)
            ],
        ]
        spec = {
            "execution": {"require_accelerator": False},
            "budget": {"max_wall_time_minutes": 1, "max_gpu_hours": 1},
            "hardware": {"gpus_per_host": 1},
            "objective": {
                "metric": "request_throughput_rps",
                "direction": "maximize",
                "min_improvement_pct": 1.0,
            },
            "slo": {},
            "search": {
                "repetitions": 1,
                "min_confirm_repetitions": 1,
                "max_cv_pct": 10.0,
                "require_all_slo_pass": True,
                "min_successful_candidates_before_early_stop": 3,
                "early_stop_improvement_pct": 3.0,
            },
        }
        calls: list[str] = []

        def fake_run_trial(_spec, trial, trial_dir, _remaining):
            calls.append(trial["name"])
            trial_dir.mkdir(parents=True, exist_ok=True)
            throughput = 100.0 if trial["kind"] == "baseline" else 104.0
            return {
                "ok": True,
                "summary": {
                    "metrics": {"request_throughput_rps": throughput},
                    "slo": {"passed": True},
                },
                "status": {"state": "completed"},
            }

        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            run_dir.mkdir()
            with patch.object(autotune, "enable_child_subreaper", return_value=False), \
                 patch.object(autotune, "execution_errors", return_value=[]), \
                 patch.object(autotune, "prepare_run", return_value=(run_dir, trials)), \
                 patch.object(autotune, "run_trial", side_effect=fake_run_trial):
                result = autotune.execute(spec)

        self.assertEqual(calls, ["baseline", "candidate-1", "candidate-2", "candidate-3"])
        self.assertEqual(result["stop_reason"], "strong_candidate_early_stop")
        self.assertEqual(result["screening_winner"]["comparison"]["minimum_improvement_pct"], 1.0)

    def test_backend_dependency_failure_disables_matching_backend_family(self):
        trial = {
            "config": {"prefill_attention_backend": "flashinfer"},
            "name": "flashinfer", "configuration_name": "flashinfer",
        }
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(
                "Traceback (most recent call last):\n"
                "ModuleNotFoundError: No module named 'cutlass'\n",
                encoding="utf-8",
            )
            reason = autotune.capability_failure_reason(
                trial,
                {"failure_class": "dependency_missing", "detail": "startup failed"},
                log,
            )
        self.assertEqual(reason["family"], "prefill_attention_backend:flashinfer")
        self.assertEqual(reason["reason"], "missing Python module: cutlass")


class ResourceAccountingTests(unittest.TestCase):
    def test_two_gpu_confirmation_keeps_two_servers_resident_and_alternates_windows(self):
        class FakeProcess:
            returncode = None

            def __init__(self, pid):
                self.pid = pid

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as temp:
            spec = {
                "name": "resident-ab", "mode": "execute", "framework": "sglang",
                "repository": "/tmp", "model": {"path": "/tmp/model"},
                "hardware": {"gpus_per_host": 2},
                "execution": {
                    "python": sys.executable, "host": "127.0.0.1", "port": 31000,
                    "startup_timeout_sec": 10, "benchmark_timeout_sec": 10,
                    "shutdown_timeout_sec": 1, "require_accelerator": False,
                    "env": {"CUDA_VISIBLE_DEVICES": "0,1"},
                },
                "benchmark": {
                    "dataset_name": "random-ids", "num_prompts": 10,
                    "random_input_len": 8, "random_output_len": 4,
                    "max_concurrency": 2, "warmup_requests": 1,
                    "min_measurement_seconds": 0, "min_tail_samples": 0,
                    "seed": 1, "flush_cache": True,
                },
                "search": {
                    "strategy": "explicit_configurations", "baseline": {"tp_size": 1},
                    "explicit_configurations": [{
                        "name": "winner", "config": {"tp_size": 1, "page_size": 16},
                    }],
                    "include_baseline": True, "repetitions": 2,
                    "min_confirm_repetitions": 2, "reuse_server_across_repetitions": True,
                },
                "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
                "slo": {}, "budget": {
                    "max_trials": 4, "max_gpu_hours": 1, "max_wall_time_minutes": 1,
                },
                "scope": {"output_dir": temp},
            }
            summaries = [
                {"metrics": {"request_throughput_rps": 10.0}, "slo": {"passed": True},
                 "measurement_validity": {"duration_gate_passed": True, "tail_sample_gate_passed": True}}
                for _ in range(4)
            ]
            with patch.object(autotune, "execution_errors", return_value=[]), \
                 patch.object(autotune, "inventory", return_value={}), \
                 patch.object(autotune, "reserve_worker_ports", return_value=[31000, 31001]), \
                 patch.object(autotune, "wait_port_available"), \
                 patch.object(autotune.subprocess, "Popen", side_effect=[FakeProcess(1), FakeProcess(2)]) as popen, \
                 patch.object(autotune, "wait_ready", return_value=(True, None)), \
                 patch.object(autotune.subprocess, "run", return_value=mock.Mock(returncode=0)) as bench, \
                 patch.object(autotune, "summarize_jsonl", side_effect=summaries), \
                 patch.object(autotune, "stop_owned_process", return_value={"method": "terminated"}), \
                 patch.object(autotune, "decision_report", return_value={"aggregates": []}):
                result = autotune.execute(spec)

        self.assertEqual(popen.call_count, 2)
        self.assertEqual(bench.call_count, 4)
        self.assertTrue(result["resident_ab"])
        self.assertEqual(result["measurement_order"], ["baseline", "winner", "winner", "baseline"])
        self.assertEqual(result["planned_server_sessions"], 2)

    def test_noisy_two_gpu_confirmation_adds_matched_30_second_pair(self):
        class FakeProcess:
            returncode = None

            def __init__(self, pid):
                self.pid = pid

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as temp:
            spec = {
                "name": "adaptive-resident-ab", "mode": "execute", "framework": "sglang",
                "repository": "/tmp", "model": {"path": "/tmp/model"},
                "hardware": {"gpus_per_host": 2},
                "execution": {
                    "python": sys.executable, "host": "127.0.0.1", "port": 31000,
                    "startup_timeout_sec": 10, "benchmark_timeout_sec": 10,
                    "shutdown_timeout_sec": 1, "require_accelerator": False,
                    "env": {"CUDA_VISIBLE_DEVICES": "0,1"},
                },
                "benchmark": {
                    "dataset_name": "random-ids", "num_prompts": 10,
                    "random_input_len": 8, "random_output_len": 4,
                    "max_concurrency": 2, "warmup_requests": 1,
                    "min_measurement_seconds": 15, "min_tail_samples": 0,
                    "seed": 1, "flush_cache": True,
                },
                "search": {
                    "strategy": "explicit_configurations", "baseline": {"tp_size": 1},
                    "explicit_configurations": [{
                        "name": "winner", "config": {"tp_size": 1, "page_size": 16},
                    }],
                    "include_baseline": True, "repetitions": 2,
                    "min_confirm_repetitions": 2, "reuse_server_across_repetitions": True,
                    "adaptive_confirmation_cv_pct": 5,
                    "adaptive_confirmation_max_repetitions": 3,
                    "adaptive_confirmation_min_measurement_seconds": 30,
                },
                "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
                "slo": {}, "budget": {
                    "max_trials": 6, "max_gpu_hours": 1, "max_wall_time_minutes": 1,
                },
                "scope": {"output_dir": temp},
            }
            summaries = [
                {
                    "metrics": {"request_throughput_rps": value},
                    "slo": {"passed": True},
                    "measurement_validity": {
                        "duration_gate_passed": True, "tail_sample_gate_passed": True,
                    },
                }
                for value in (10.0, 20.0, 10.0, 20.0, 15.0, 15.0)
            ]
            with patch.object(autotune, "execution_errors", return_value=[]), \
                 patch.object(autotune, "inventory", return_value={}), \
                 patch.object(autotune, "reserve_worker_ports", return_value=[31000, 31001]), \
                 patch.object(autotune, "wait_port_available"), \
                 patch.object(autotune.subprocess, "Popen", side_effect=[FakeProcess(1), FakeProcess(2)]), \
                 patch.object(autotune, "wait_ready", return_value=(True, None)), \
                 patch.object(autotune.subprocess, "run", return_value=mock.Mock(returncode=0)) as bench, \
                 patch.object(autotune, "summarize_jsonl", side_effect=summaries), \
                 patch.object(autotune, "stop_owned_process", return_value={"method": "terminated"}), \
                 patch.object(autotune, "decision_report", return_value={"aggregates": []}):
                result = autotune.execute(spec)

        self.assertEqual(bench.call_count, 6)
        self.assertEqual(
            result["measurement_order"],
            ["baseline", "winner", "winner", "baseline", "baseline", "winner"],
        )
        self.assertTrue(result["adaptive_confirmation"]["triggered"])
        adaptive_rows = [
            row for row in result["results"]
            if row["status"]["adaptive_confirmation_window"]
        ]
        self.assertEqual(len(adaptive_rows), 2)
        self.assertTrue(all(
            row["measurement_validity"]["minimum_duration_sec"] == 30
            for row in adaptive_rows
        ))

    def test_adaptive_calibration_binary_search_reuses_one_loaded_server(self):
        class FakeProcess:
            pid = 4322
            returncode = None

            def poll(self):
                return None

        spec = {
            "repository": "/tmp", "model": {"path": "/tmp/model"},
            "execution": {
                "python": sys.executable, "host": "127.0.0.1", "port": 31000,
                "startup_timeout_sec": 10, "benchmark_timeout_sec": 10,
                "shutdown_timeout_sec": 1, "env": {},
            },
            "benchmark": {
                "dataset_name": "random-ids", "num_prompts": 64,
                "random_input_len": 8, "random_output_len": 4,
                "max_concurrency": 64, "auto_max_concurrency": True,
                "warmup_requests": 1, "min_measurement_seconds": 0, "seed": 1,
                "flush_cache": True,
                "calibration_session": {
                    "strategy": "adaptive_slo", "concurrencies": [64],
                    "min_concurrency": 1, "max_steps": 4,
                    "request_waves": 5, "requested_concurrency": 64,
                    "initial_unbounded_probe": True,
                },
            },
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "slo": {"p99_e2e_latency_ms": 1000},
        }
        trial = {
            "name": "baseline", "configuration_name": "baseline", "repeat_index": 0,
            "kind": "baseline", "config": {}, "env": {},
        }
        summaries = [
            {"metrics": {"request_throughput_rps": 1.0}, "slo": {"passed": passed},
             "measurement_validity": {"duration_sec": 1.0, "duration_gate_passed": True}}
            for passed in (False, True, False, True)
        ]
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(autotune, "wait_port_available"), \
             patch.object(autotune.subprocess, "Popen", return_value=FakeProcess()) as popen, \
             patch.object(autotune, "wait_ready", return_value=(True, None)), \
             patch.object(autotune, "resolved_server_capacity", return_value={
                 "max_running_requests": 512, "source": "/server_info",
             }), \
             patch.object(autotune.subprocess, "run", return_value=mock.Mock(returncode=0)) as bench, \
             patch.object(autotune, "summarize_jsonl", side_effect=summaries), \
             patch.object(autotune, "stop_owned_process", return_value={"method": "terminated"}), \
             patch.object(autotune, "summarize_sglang_log", return_value={}):
            result = autotune.run_trial(spec, trial, Path(temp) / "trial", 60)

        commands = [call.args[0] for call in bench.call_args_list]
        self.assertNotIn("--max-concurrency", commands[0])
        measured = [
            512,
            *[int(command[command.index("--max-concurrency") + 1]) for command in commands[1:]],
        ]
        self.assertEqual(measured, [512, 256, 384, 320])
        self.assertEqual(popen.call_count, 1)
        self.assertEqual([item["calibration_concurrency"] for item in result["summaries"]], measured)

    def test_run_trial_reuses_one_server_for_repeated_measurement_windows(self):
        class FakeProcess:
            pid = 4321
            returncode = None

            def poll(self):
                return None

        spec = {
            "repository": "/tmp",
            "model": {"path": "/tmp/model"},
            "execution": {
                "python": sys.executable, "host": "127.0.0.1", "port": 31000,
                "startup_timeout_sec": 10, "benchmark_timeout_sec": 10,
                "shutdown_timeout_sec": 1, "env": {},
            },
            "benchmark": {
                "dataset_name": "random-ids", "num_prompts": 10,
                "random_input_len": 8, "random_output_len": 4,
                "max_concurrency": 2, "warmup_requests": 1,
                "min_measurement_seconds": 0, "seed": 1,
            },
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "slo": {},
        }
        trial = {
            "name": "baseline-resident", "configuration_name": "baseline",
            "repeat_index": 0, "repeat_indices": [0, 1],
            "kind": "baseline", "config": {}, "env": {},
        }
        summaries = [
            {"metrics": {"request_throughput_rps": 10.0 + index}, "slo": {"passed": True},
             "measurement_validity": {"duration_sec": 1.0, "duration_gate_passed": True}}
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(autotune, "wait_port_available"), \
             patch.object(autotune.subprocess, "Popen", return_value=FakeProcess()) as popen, \
             patch.object(autotune, "wait_ready", return_value=(True, None)), \
             patch.object(autotune.subprocess, "run", return_value=mock.Mock(returncode=0)) as bench, \
             patch.object(autotune, "summarize_jsonl", side_effect=summaries), \
             patch.object(autotune, "stop_owned_process", return_value={"method": "terminated"}), \
             patch.object(autotune, "summarize_sglang_log", return_value={}):
            result = autotune.run_trial(spec, trial, Path(temp) / "trial", 60)

        self.assertTrue(result["ok"])
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(bench.call_count, 2)
        self.assertEqual(len(result["summaries"]), 2)
        self.assertEqual(result["status"]["measurement_windows"], 2)

    def test_single_gpu_noisy_confirmation_extends_resident_service_once(self):
        class FakeProcess:
            pid = 4323
            returncode = None

            def poll(self):
                return None

        spec = {
            "repository": "/tmp", "model": {"path": "/tmp/model"},
            "execution": {
                "python": sys.executable, "host": "127.0.0.1", "port": 31000,
                "startup_timeout_sec": 10, "benchmark_timeout_sec": 10,
                "shutdown_timeout_sec": 1, "env": {},
            },
            "benchmark": {
                "dataset_name": "random-ids", "num_prompts": 10,
                "random_input_len": 8, "random_output_len": 4,
                "max_concurrency": 2, "warmup_requests": 1,
                "min_measurement_seconds": 15, "seed": 1,
            },
            "search": {
                "adaptive_confirmation_cv_pct": 5,
                "adaptive_confirmation_max_repetitions": 3,
                "adaptive_confirmation_min_measurement_seconds": 30,
            },
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "slo": {},
        }
        trial = {
            "name": "baseline-resident", "configuration_name": "baseline",
            "repeat_index": 0, "repeat_indices": [0, 1],
            "kind": "baseline", "config": {}, "env": {},
        }
        summaries = [
            {
                "metrics": {"request_throughput_rps": value}, "slo": {"passed": True},
                "measurement_validity": {
                    "duration_sec": duration, "duration_gate_passed": True,
                },
            }
            for value, duration in ((10.0, 15.0), (20.0, 15.0), (15.0, 30.0))
        ]
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(autotune, "wait_port_available"), \
             patch.object(autotune.subprocess, "Popen", return_value=FakeProcess()), \
             patch.object(autotune, "wait_ready", return_value=(True, None)), \
             patch.object(autotune.subprocess, "run", return_value=mock.Mock(returncode=0)) as bench, \
             patch.object(autotune, "summarize_jsonl", side_effect=summaries), \
             patch.object(autotune, "stop_owned_process", return_value={"method": "terminated"}), \
             patch.object(autotune, "summarize_sglang_log", return_value={}):
            result = autotune.run_trial(spec, trial, Path(temp) / "trial", 60)

        self.assertTrue(result["ok"])
        self.assertEqual(bench.call_count, 3)
        self.assertEqual(len(result["summaries"]), 3)
        self.assertTrue(result["status"]["adaptive_confirmation"]["triggered"])
        self.assertEqual(
            result["summaries"][2]["measurement_validity"]["minimum_duration_sec"], 30
        )

    def test_resident_confirmation_counts_windows_without_reloading_each_window(self):
        trials = [
            {"name": "baseline-resident", "configuration_name": "baseline", "repeat_index": 0,
             "repeat_indices": [0, 1], "kind": "baseline", "config": {"tp_size": 1}, "env": {}},
            {"name": "winner-resident", "configuration_name": "winner", "repeat_index": 0,
             "repeat_indices": [0, 1], "kind": "candidate",
             "config": {"tp_size": 1, "page_size": 16}, "env": {}},
        ]
        spec = {
            "execution": {"require_accelerator": False},
            "budget": {"max_wall_time_minutes": 1, "max_gpu_hours": 1},
            "hardware": {"gpus_per_host": 1},
            "search": {"repetitions": 2, "min_confirm_repetitions": 2},
        }
        calls = []

        def fake_run_trial(_spec, trial, trial_dir, _remaining):
            trial_dir.mkdir(parents=True, exist_ok=True)
            calls.append(trial["configuration_name"])
            summaries = [
                {"repeat_index": repeat_index, "metrics": {"request_throughput_rps": 10 + repeat_index},
                 "slo": {"passed": True}, "measurement_validity": {"duration_gate_passed": True}}
                for repeat_index in trial["repeat_indices"]
            ]
            return {"ok": True, "summary": summaries[0], "summaries": summaries,
                    "status": {"state": "completed"}}

        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            run_dir.mkdir()
            with patch.object(autotune, "enable_child_subreaper", return_value=False), \
                 patch.object(autotune, "execution_errors", return_value=[]), \
                 patch.object(autotune, "prepare_run", return_value=(run_dir, trials)), \
                 patch.object(autotune, "run_trial", side_effect=fake_run_trial), \
                 patch.object(autotune, "decision_report", return_value={"aggregates": []}):
                result = autotune.execute(spec)

        self.assertEqual(calls, ["baseline", "winner"])
        self.assertEqual(result["planned_server_sessions"], 2)
        self.assertEqual(result["completed_server_sessions"], 2)
        self.assertEqual(result["planned_trials"], 4)
        self.assertEqual(result["completed_trials"], 4)
        self.assertEqual(
            [(row["configuration_name"], row["repeat_index"]) for row in result["results"]],
            [("baseline", 0), ("baseline", 1), ("winner", 0), ("winner", 1)],
        )

    def test_trial_gpu_hours_follow_the_trial_parallelism_not_visibility(self):
        spec = {
            "execution": {"env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}},
            "hardware": {"gpus_per_host": 4},
        }
        self.assertEqual(autotune.configuration_accelerator_count(spec, {"tp_size": 1}), 1)
        self.assertEqual(autotune.configuration_accelerator_count(spec, {"tp_size": 2}), 2)
        self.assertEqual(autotune.configuration_accelerator_count(spec, {"tp_size": 2, "dp_size": 2}), 4)
        self.assertEqual(autotune.configuration_accelerator_count(spec, {"tp_size": 4, "pp_size": 2}), 4)

    def test_single_gpu_screening_candidates_use_exclusive_parallel_workers(self):
        trials = [
            {"name": "baseline", "configuration_name": "baseline", "repeat_index": 0,
             "kind": "baseline", "config": {"tp_size": 1}, "env": {}},
            *[
                {"name": f"candidate-{index}", "configuration_name": f"candidate-{index}",
                 "repeat_index": 0, "kind": "candidate", "config": {"page_size": index}, "env": {}}
                for index in range(1, 5)
            ],
        ]
        spec = {
            "execution": {
                "require_accelerator": False, "parallel_trials": 4,
                "port": 31000, "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            },
            "budget": {"max_wall_time_minutes": 1, "max_gpu_hours": 1},
            "hardware": {"gpus_per_host": 4},
            "search": {"repetitions": 1, "min_confirm_repetitions": 1},
        }
        calls = []

        def fake_run_trial(_spec, trial, trial_dir, _remaining):
            trial_dir.mkdir(parents=True, exist_ok=True)
            calls.append((trial["name"], trial.get("env", {}).get("CUDA_VISIBLE_DEVICES"), trial.get("_port")))
            return {
                "ok": True,
                "summary": {"metrics": {}, "slo": {"passed": True}},
                "status": {"state": "completed"},
            }

        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            run_dir.mkdir()
            with patch.object(autotune, "enable_child_subreaper", return_value=False), \
                 patch.object(autotune, "execution_errors", return_value=[]), \
                 patch.object(autotune, "prepare_run", return_value=(run_dir, trials)), \
                 patch.object(autotune, "run_trial", side_effect=fake_run_trial), \
                 patch.object(autotune, "reserve_worker_ports", return_value=[31000, 31001, 31002, 31003, 31004]), \
                 patch.object(autotune, "decision_report", return_value={"aggregates": []}):
                result = autotune.execute(spec)

        assigned = {name: (gpu, port) for name, gpu, port in calls}
        self.assertEqual(assigned["baseline"], ("0", 31000))
        self.assertEqual(
            {assigned[f"candidate-{index}"] for index in range(1, 4)},
            {("1", 31001), ("2", 31002), ("3", 31003)},
        )
        self.assertIn(assigned["candidate-4"][0], {"0", "1", "2", "3"})
        self.assertEqual(assigned["candidate-4"][1], 31004)
        self.assertEqual(result["completed_trials"], 5)
        self.assertTrue(all(row["env"] == {} for row in result["results"]))

    def test_parallel_workers_choose_the_first_free_contiguous_port_range(self):
        with patch.object(
            autotune, "port_available_now", side_effect=lambda _host, port: port in {31002, 31003, 31004}
        ):
            self.assertEqual(
                autotune.reserve_worker_ports("127.0.0.1", 31000, 3),
                [31002, 31003, 31004],
            )

    def test_parallel_batch_packs_two_tp2_trials_on_four_gpus(self):
        trials = [
            {"name": "tp2-a", "configuration_name": "tp2-a", "repeat_index": 0,
             "kind": "candidate", "config": {"tp_size": 2}, "env": {}},
            {"name": "tp2-b", "configuration_name": "tp2-b", "repeat_index": 0,
             "kind": "candidate", "config": {"tp_size": 2}, "env": {}},
            {"name": "tp1-c", "configuration_name": "tp1-c", "repeat_index": 0,
             "kind": "candidate", "config": {"tp_size": 1}, "env": {}},
        ]
        spec = {
            "execution": {
                "parallel_trials": 4, "port": 31000,
                "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            },
            "hardware": {"gpus_per_host": 4},
            "search": {"repetitions": 1},
        }
        batch = autotune.parallel_candidate_batch(spec, trials, 0, {})
        self.assertEqual([item["name"] for item in batch], ["tp2-a", "tp2-b", "tp1-c"])
        placements = {}

        def fake_run_trial(_spec, trial, trial_dir, _remaining):
            trial_dir.mkdir(parents=True, exist_ok=True)
            placements[trial["name"]] = trial["env"]["CUDA_VISIBLE_DEVICES"]
            return {
                "ok": True,
                "summary": {"metrics": {}, "slo": {"passed": True}},
                "status": {"state": "completed"},
            }

        with tempfile.TemporaryDirectory() as temp, \
             patch.object(autotune, "run_trial", side_effect=fake_run_trial), \
             patch.object(autotune, "reserve_worker_ports", return_value=[31000, 31001, 31002]):
            autotune.parallel_screening_batch(
                spec, batch, 0, 4, Path(temp), 60, 3600, time.monotonic(), 0,
            )
        self.assertEqual({placements["tp2-a"], placements["tp2-b"]}, {"0,1", "2,3"})
        self.assertIn(placements["tp1-c"], {"0", "1", "2", "3"})

    def test_parallel_queue_backfills_a_gpu_before_the_straggler_finishes(self):
        trials = [
            {"name": name, "configuration_name": name, "repeat_index": 0,
             "kind": "candidate", "config": {"tp_size": 1}, "env": {}}
            for name in ("slow", "fast", "backfill")
        ]
        spec = {
            "execution": {
                "parallel_trials": 2, "port": 31000,
                "env": {"CUDA_VISIBLE_DEVICES": "0,1"},
            },
            "hardware": {"gpus_per_host": 2},
            "search": {"repetitions": 1},
        }
        backfill_started = threading.Event()

        def fake_run_trial(_spec, trial, trial_dir, _remaining):
            trial_dir.mkdir(parents=True, exist_ok=True)
            if trial["name"] == "slow":
                self.assertTrue(backfill_started.wait(2))
            elif trial["name"] == "backfill":
                backfill_started.set()
            return {
                "ok": True,
                "summary": {"metrics": {}, "slo": {"passed": True}},
                "status": {"state": "completed"},
            }

        with tempfile.TemporaryDirectory() as temp, \
             patch.object(autotune, "run_trial", side_effect=fake_run_trial), \
             patch.object(autotune, "reserve_worker_ports", return_value=[31000, 31001, 31002]):
            results = autotune.parallel_screening_batch(
                spec, trials, 0, 2, Path(temp), 60, 3600, time.monotonic(), 0,
            )
        self.assertTrue(backfill_started.is_set())
        self.assertEqual(len(results), 3)


class HardwarePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = autopilot.load_hardware_catalog()

    def inventory(self, vendor, name, memory_mib=96 * 1024, count=1, topology=""):
        return {
            "vendor": vendor,
            "gpus": [
                {"index": index, "name": name, "memory_mib": memory_mib}
                for index in range(count)
            ],
            "topology": topology,
        }

    def test_matches_distinct_vendor_profiles_and_variants(self):
        cases = [
            ("nvidia", "NVIDIA H20", "nvidia-hopper-datacenter", "H20"),
            ("nvidia", "NVIDIA B200", "nvidia-blackwell-datacenter", "B200"),
            ("nvidia", "NVIDIA L40S", "nvidia-ada-pcie", "L40S"),
            ("amd", "AMD Instinct MI325X", "amd-cdna3-instinct", "MI325X"),
        ]
        for vendor, name, profile_id, variant_name in cases:
            with self.subTest(name=name):
                profile = autopilot.match_hardware_profile(
                    self.inventory(vendor, name), self.catalog
                )
                self.assertEqual(profile["id"], profile_id)
                self.assertEqual(profile["matched_variant"]["name"], variant_name)

    def test_topology_classifies_single_nvlink_and_pcie(self):
        self.assertEqual(
            autopilot.topology_class(self.inventory("nvidia", "H20")), "single-gpu"
        )
        self.assertEqual(
            autopilot.topology_class(
                self.inventory("nvidia", "H100", count=2, topology="GPU0 NV4 GPU1")
            ),
            "nvlink-or-nvswitch",
        )
        self.assertEqual(
            autopilot.topology_class(
                self.inventory("nvidia", "L40S", count=2, topology="GPU0 PHB GPU1")
            ),
            "pcie",
        )

    def test_visibility_selects_actual_devices(self):
        inventory = {
            "vendor": "nvidia",
            "gpus": [
                {"index": 0, "uuid": "GPU-H200", "name": "H200", "memory_mib": 141 * 1024},
                {"index": 1, "uuid": "GPU-L40S", "name": "L40S", "memory_mib": 48 * 1024},
            ],
        }
        task = {"env": {"CUDA_VISIBLE_DEVICES": "1"}}
        self.assertEqual(autopilot.selected_gpus(task, inventory)[0]["name"], "L40S")
        task["env"]["CUDA_VISIBLE_DEVICES"] = "GPU-H200"
        self.assertEqual(autopilot.selected_gpus(task, inventory)[0]["name"], "H200")
        task["env"]["CUDA_VISIBLE_DEVICES"] = "1,GPU-H200"
        self.assertEqual(
            [gpu["name"] for gpu in autopilot.selected_gpus(task, inventory)],
            ["L40S", "H200"],
        )
        task["max_gpus"] = 1
        self.assertEqual(
            [gpu["name"] for gpu in autopilot.selected_gpus(task, inventory)], ["L40S"]
        )
        self.assertEqual(autopilot.selected_gpu_identifiers(task, inventory), ["1"])
        task.pop("max_gpus")
        task["env"]["CUDA_VISIBLE_DEVICES"] = "9"
        with self.assertRaisesRegex(ValueError, "does not match discovered"):
            autopilot.selected_gpus(task, inventory)

    def test_minimum_tp_uses_visible_memory(self):
        task = {"env": {"CUDA_VISIBLE_DEVICES": "0,1"}}
        inventory = self.inventory("nvidia", "L40S", memory_mib=48 * 1024, count=2)
        model = {"weight_bytes": 70 * 1024**3}
        self.assertEqual(autopilot.minimum_tp(task, inventory, model), 2)

    def test_minimum_tp_rejects_head_incompatible_layout(self):
        task = {"env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}}
        inventory = self.inventory("nvidia", "H800", memory_mib=80 * 1024, count=4)
        model = {
            "weight_bytes": 220 * 1024**3,
            "num_attention_heads": 30,
            "num_key_value_heads": 6,
        }
        with self.assertRaisesRegex(ValueError, "attention heads, and KV heads"):
            autopilot.minimum_tp(task, inventory, model)
        self.assertTrue(autopilot.kv_heads_support_tp(8, 16))
        self.assertFalse(autopilot.kv_heads_support_tp(6, 4))

    def test_multi_gpu_deployment_feasibility_reports_legal_tp(self):
        task = {"env": {}, "workload": {"input_tokens": 256, "output_tokens": 64, "max_concurrency": 4}}
        inventory = self.inventory("nvidia", "H800", memory_mib=80 * 1024, count=4)
        model = {
            "weight_bytes": 220 * 1024**3,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
        }
        result = autopilot.deployment_feasibility(task, inventory, model)
        self.assertEqual(result["status"], "deployable_as_is")
        self.assertEqual(result["minimum_tp_size"], 4)

    def test_chunk_candidates_follow_workload_boundary(self):
        task = {"workload": {"input_tokens": 256, "max_concurrency": 4}}
        self.assertEqual(
            autopilot.chunk_candidates(task, framework_default=8192),
            [512, 1024, 2048, 4096, 8192],
        )
        task["workload"]["prefix_reuse_ratio"] = 0.75
        self.assertEqual(
            autopilot.chunk_candidates(task, framework_default=8192),
            [256, 512, 1024, 2048, 4096, 8192],
        )
        task["workload"].pop("prefix_reuse_ratio")
        task["workload"]["input_tokens"] = 4096
        self.assertEqual(
            autopilot.chunk_candidates(task, framework_default=8192),
            [4096, 8192, 16384, 32768],
        )

    def test_chunk_memory_filter_excludes_h800_16k_for_235b_fp8(self):
        task = {
            "env": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            "workload": {"input_tokens": 16384, "output_tokens": 256, "max_concurrency": 8},
        }
        discovery = {
            "hardware": self.inventory("nvidia", "NVIDIA H800", memory_mib=81559, count=4),
            "framework": {"chunk_activation_reserve_mib_per_token": 1.5},
            "model": {"weight_bytes": 236426193880},
            "derived": {"minimum_tp_size": 4},
        }
        feasible, excluded = autopilot.chunk_memory_feasibility(
            task,
            discovery,
            {"tp_size": 4, "chunked_prefill_size": 8192, "mem_fraction_static": 0.819},
            [4096, 8192, 12288, 16384],
        )
        self.assertEqual(feasible, [4096, 8192, 12288])
        self.assertEqual(excluded[0]["chunked_prefill_size"], 16384)
        self.assertAlmostEqual(excluded[0]["predicted_mem_fraction_static"], 0.668, places=3)

    def test_single_gpu_feasibility_is_explicit_about_headroom(self):
        task = {
            "env": {"CUDA_VISIBLE_DEVICES": "0"},
            "workload": {"input_tokens": 256, "output_tokens": 64, "max_concurrency": 4},
            "deployment": {"allow_model_variant_recommendations": True},
            "quality": {},
        }
        model = {
            "weight_bytes": 60 * 1024**3,
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }
        result = autopilot.single_gpu_feasibility(task, self.inventory("nvidia", "H20"), model)
        self.assertEqual(result["status"], "deployable_as_is")
        self.assertGreater(result["estimated_headroom_gib"], 0)
        model["weight_bytes"] = 80 * 1024**3
        result = autopilot.single_gpu_feasibility(task, self.inventory("nvidia", "H20"), model)
        self.assertEqual(result["status"], "requires_parallel_or_variant")
        self.assertTrue(result["recommendations"])

    def test_noninteractive_init_generates_a_valid_task(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            model = root_path / "model"
            repository = root_path / "sglang"
            output_dir = root_path / "runs"
            model.mkdir()
            repository.mkdir()
            args = type("Args", (), {
                "non_interactive": True,
                "repository": str(repository),
                "python": sys.executable,
                "model_path": str(model),
                "output_dir": str(output_dir),
                "name": "smoke",
                "deployment_mode": "online_latency",
                "input_tokens": "256",
                "output_tokens": "64",
                "max_concurrency": "4",
                "concurrency_points": "1,2,4",
                "shared_prefix_tokens": "192",
                "experiment_mode": "fast",
                "cuda_visible_devices": "0",
            })()
            task = inferopt_cli.init_task(args)
            self.assertEqual(autopilot.validate_task(task), [])
            self.assertEqual(task["calibration"]["min_concurrency"], 1)
            self.assertEqual(task["calibration"]["max_concurrency"], 4)
            self.assertEqual(task["calibration"]["strategy"], "adaptive")
            self.assertEqual(task["calibration"]["concurrencies"], [1, 2, 4])
            self.assertEqual(task["workload"]["shared_prefix"]["system_prompt_tokens"], 192)
            self.assertEqual(task["experiment_mode"], "fast")
            self.assertEqual(task["max_gpus"], 1)
            self.assertEqual(task["parallel_trials"], 1)
            self.assertEqual(task["search_depth"], "evidence_guided")
            self.assertEqual(task["measurement"]["min_measurement_seconds"], 15)
            self.assertEqual(task["measurement"]["adaptive_confirmation_cv_pct"], 5)
            self.assertEqual(task["measurement"]["adaptive_confirmation_max_repetitions"], 3)
            self.assertEqual(task["workload"]["num_prompts"], 40)
            self.assertEqual(task["measurement"]["min_measurement_requests"], 40)
            self.assertEqual(task["measurement"]["confirmation_requests"], 20)
            self.assertEqual(task["measurement"]["warmup_requests"], 8)
            self.assertEqual(task["measurement"]["p99_request_waves"], 10)
            self.assertEqual(task["slo"], {})
            self.assertNotIn("kernel_tuning", task)

            args.cuda_visible_devices = "all"
            all_gpu_task = inferopt_cli.init_task(args)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", all_gpu_task["env"])

            args.shared_prefix_tokens = None
            args.experiment_mode = None
            args.max_concurrency = None
            args.concurrency_points = None
            online_task = inferopt_cli.init_task(args)
            self.assertEqual(online_task["workload"]["max_concurrency"], 8)
            self.assertEqual(online_task["experiment_mode"], "balanced")
            self.assertEqual(online_task["workload"]["num_prompts"], 40)
            self.assertEqual(online_task["measurement"]["confirmation_requests"], 20)
            self.assertEqual(online_task["measurement"]["warmup_requests"], 8)
            args.deployment_mode = "offline_throughput"
            offline_task = inferopt_cli.init_task(args)
            self.assertNotIn("max_concurrency", offline_task["workload"])
            self.assertTrue(offline_task["workload"]["unbounded_client_concurrency"])
            self.assertEqual(offline_task["workload"]["initial_backlog_requests"], 40)
            self.assertFalse(offline_task["calibration"]["enabled"])
            self.assertEqual(offline_task["workload"]["num_prompts"], 40)
            self.assertEqual(offline_task["measurement"]["min_measurement_requests"], 40)
            self.assertEqual(offline_task["measurement"]["confirmation_requests"], 20)
            self.assertEqual(offline_task["slo"], {})
            self.assertEqual(offline_task["objective"]["metric"], "total_throughput_tps")

    def test_experiment_modes_change_search_budget_not_measurement_fidelity(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            model = root_path / "model"
            repository = root_path / "sglang"
            model.mkdir()
            repository.mkdir()
            base = {
                "non_interactive": True, "repository": str(repository),
                "python": sys.executable, "model_path": str(model),
                "output_dir": str(root_path / "runs"), "name": "mode-contract",
                "deployment_mode": "online_latency", "input_tokens": "1024",
                "output_tokens": "128", "max_concurrency": "16",
                "concurrency_points": None, "shared_prefix_tokens": None,
                "cuda_visible_devices": "0", "p99_ttft_ms": "1000",
            }
            tasks = {}
            for mode in ("fast", "balanced", "max"):
                args = type("Args", (), {**base, "experiment_mode": mode})()
                tasks[mode] = inferopt_cli.init_task(args)
            measurement = tasks["fast"]["measurement"]
            self.assertEqual(tasks["balanced"]["measurement"], measurement)
            self.assertEqual(tasks["max"]["measurement"], measurement)
            self.assertEqual(tasks["fast"]["confirmation_repetitions"], 2)
            self.assertEqual(tasks["max"]["confirmation_repetitions"], 2)
            self.assertLess(tasks["fast"]["budget"]["max_trials"], tasks["max"]["budget"]["max_trials"])
            self.assertEqual(tasks["max"]["search_depth"], "thorough")

    def test_legacy_rigorous_mode_is_normalized_to_max_by_init(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            model = root_path / "model"
            repository = root_path / "sglang"
            model.mkdir()
            repository.mkdir()
            args = type("Args", (), {
                "non_interactive": True, "repository": str(repository),
                "python": sys.executable, "model_path": str(model),
                "output_dir": str(root_path / "runs"), "name": "legacy-mode",
                "deployment_mode": "online_latency", "input_tokens": "256",
                "output_tokens": "64", "max_concurrency": "8",
                "concurrency_points": None, "shared_prefix_tokens": None,
                "experiment_mode": "rigorous", "cuda_visible_devices": "0",
            })()
            task = inferopt_cli.init_task(args)
            self.assertEqual(task["experiment_mode"], "max")
            self.assertEqual(autopilot.validate_task(task), [])

    def test_init_accepts_budget_and_confirmation_overrides(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            model = root_path / "model"
            repository = root_path / "sglang"
            model.mkdir()
            repository.mkdir()
            args = type("Args", (), {
                "non_interactive": True,
                "repository": str(repository), "python": sys.executable,
                "model_path": str(model), "output_dir": str(root_path / "runs"),
                "name": "budget", "deployment_mode": "offline_throughput",
                "input_tokens": "4096", "output_tokens": "128", "max_concurrency": None,
                "concurrency_points": None, "shared_prefix_tokens": None,
                "experiment_mode": "balanced", "cuda_visible_devices": "all",
                "max_trials": 17, "max_gpu_hours": 6.0,
                "max_wall_time_minutes": 180.0, "confirmation_repetitions": 2,
            })()
            task = inferopt_cli.init_task(args)
            self.assertEqual(task["budget"], {
                "max_trials": 17, "max_gpu_hours": 6.0, "max_wall_time_minutes": 180.0,
            })
            self.assertEqual(task["confirmation_repetitions"], 2)

    def test_init_accepts_real_custom_jsonl_workload(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            model = root_path / "model"
            repository = root_path / "sglang"
            dataset = root_path / "requests.jsonl"
            model.mkdir()
            repository.mkdir()
            rows = [
                json.dumps({"conversations": [
                    {"content": f"question {index}"}, {"content": "answer"},
                ]})
                for index in range(64)
            ]
            dataset.write_text("\n".join(rows) + "\n", encoding="utf-8")
            args = type("Args", (), {
                "non_interactive": True,
                "repository": str(repository), "python": sys.executable,
                "model_path": str(model), "output_dir": str(root_path / "runs"),
                "name": "custom", "deployment_mode": "offline_throughput",
                "input_tokens": "1024", "output_tokens": "128", "max_concurrency": None,
                "concurrency_points": None, "shared_prefix_tokens": None,
                "experiment_mode": "fast", "cuda_visible_devices": "all",
                "dataset_name": "custom", "dataset_path": str(dataset),
                "apply_chat_template": True,
            })()
            task = inferopt_cli.init_task(args)
            self.assertEqual(task["workload"]["dataset"], {
                "name": "custom", "path": str(dataset.resolve()), "apply_chat_template": True,
            })
            self.assertNotIn("shared_prefix", task["workload"])
            self.assertEqual(autopilot.validate_task(task), [])

    def test_init_accepts_explicit_online_slo_limits(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            model = root_path / "model"
            repository = root_path / "sglang"
            model.mkdir()
            repository.mkdir()
            args = type("Args", (), {
                "non_interactive": True,
                "repository": str(repository), "python": sys.executable,
                "model_path": str(model), "output_dir": str(root_path / "runs"),
                "name": "slo", "deployment_mode": "online_latency",
                "input_tokens": "256", "output_tokens": "64", "max_concurrency": "4",
                "concurrency_points": None, "shared_prefix_tokens": None,
                "experiment_mode": "fast", "cuda_visible_devices": "0",
                "p99_e2e_latency_ms": "1500", "p99_ttft_ms": "0",
                "p99_tpot_ms": "75",
            })()
            task = inferopt_cli.init_task(args)
            self.assertEqual(task["slo"], {
                "p99_e2e_latency_ms": 1500.0,
                "p99_tpot_ms": 75.0,
            })

    def test_init_accepts_average_latency_limits_as_one_metric_family(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            model = root_path / "model"
            repository = root_path / "sglang"
            model.mkdir()
            repository.mkdir()
            args = type("Args", (), {
                "non_interactive": True,
                "repository": str(repository), "python": sys.executable,
                "model_path": str(model), "output_dir": str(root_path / "runs"),
                "name": "avg-slo", "deployment_mode": "online_latency",
                "input_tokens": "256", "output_tokens": "64", "max_concurrency": "4",
                "concurrency_points": None, "shared_prefix_tokens": None,
                "experiment_mode": "fast", "cuda_visible_devices": "0",
                "latency_slo_statistic": "avg", "avg_e2e_latency_ms": "1500",
                "avg_ttft_ms": "0", "avg_tpot_ms": "75",
            })()
            task = inferopt_cli.init_task(args)
            self.assertEqual(task["slo"], {
                "mean_e2e_latency_ms": 1500.0,
                "mean_tpot_ms": 75.0,
            })
            self.assertEqual(autopilot.validate_task(task), [])

    def test_init_rejects_mixed_average_and_p99_latency_limits(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            model = root_path / "model"
            repository = root_path / "sglang"
            model.mkdir()
            repository.mkdir()
            args = type("Args", (), {
                "non_interactive": True,
                "repository": str(repository), "python": sys.executable,
                "model_path": str(model), "output_dir": str(root_path / "runs"),
                "name": "mixed-slo", "deployment_mode": "online_latency",
                "input_tokens": "256", "output_tokens": "64", "max_concurrency": "4",
                "concurrency_points": None, "shared_prefix_tokens": None,
                "experiment_mode": "fast", "cuda_visible_devices": "0",
                "p99_e2e_latency_ms": "1500", "avg_ttft_ms": "500",
            })()
            with self.assertRaisesRegex(ValueError, "either p99 or avg"):
                inferopt_cli.init_task(args)

    def test_concurrency_points_accept_comma_or_space_separators(self):
        self.assertEqual(inferopt_cli.parse_concurrency_points("1, 4,16"), [1, 4, 16])
        self.assertEqual(inferopt_cli.parse_concurrency_points("1 4 16"), [1, 4, 16])
        self.assertEqual(inferopt_cli.visibility_environment("all"), {})
        self.assertEqual(
            inferopt_cli.visibility_environment("0, 2,3"),
            {"CUDA_VISIBLE_DEVICES": "0,2,3"},
        )

    def test_doctor_compares_explicit_local_model_variants(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            model = root_path / "model"
            variant = root_path / "variant"
            repository = root_path / "sglang"
            model.mkdir()
            variant.mkdir()
            repository.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (variant / "config.json").write_text("{}", encoding="utf-8")
            task = {
                "name": "doctor-variants",
                "repository": str(repository),
                "python": sys.executable,
                "model_path": str(model),
                "output_dir": str(root_path / "runs"),
                "workload": {"input_tokens": 256, "output_tokens": 64, "max_concurrency": 4, "num_prompts": 512},
                "slo": {"p99_ttft_ms": 1000},
                "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
                "budget": {"max_trials": 9, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
                "model_variants": [{"name": "fp8", "model_path": str(variant), "quantization": "fp8"}],
            }
            original_nvidia = autopilot.parse_nvidia_inventory
            original_amd = autopilot.parse_amd_inventory
            original_framework = autopilot.framework_evidence
            try:
                autopilot.parse_nvidia_inventory = lambda: self.inventory("nvidia", "H20")
                autopilot.parse_amd_inventory = lambda: None
                autopilot.framework_evidence = lambda _: {"launch_server_help_available": True}
                result = inferopt_cli.doctor(task)
            finally:
                autopilot.parse_nvidia_inventory = original_nvidia
                autopilot.parse_amd_inventory = original_amd
                autopilot.framework_evidence = original_framework
            self.assertEqual(len(result["local_model_variants"]), 1)
            self.assertEqual(result["local_model_variants"][0]["quality_gate"]["state"], "missing_evaluation_dataset")


class ValidationTests(unittest.TestCase):
    def test_offline_throughput_does_not_apply_implicit_latency_regression_gate(self):
        base = {"metrics": {"request_throughput_rps": 100.0, "p99_ttft_ms": 100.0}}
        candidate = {"metrics": {"request_throughput_rps": 102.0, "p99_ttft_ms": 200.0}}
        spec = {
            "objective": {
                "metric": "request_throughput_rps", "direction": "maximize",
                "min_improvement_pct": 1, "max_regression_pct": 5,
            },
            "slo": {},
            "deployment_mode": "offline_throughput",
        }
        self.assertTrue(inferopt.compare(base, candidate, spec)["accepted"])
        spec["deployment_mode"] = "online_latency"
        self.assertFalse(inferopt.compare(base, candidate, spec)["accepted"])

    def test_historical_failure_cache_requires_matching_fingerprint_and_definitive_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                "repository": str(root / "repo"),
                "model_path": str(root / "model"),
                "output_dir": str(root / "runs"),
                "env": {"CUDA_VISIBLE_DEVICES": "0"},
                "workload": {"input_tokens": 256, "output_tokens": 64, "max_concurrency": 4},
            }
            discovery = {
                "framework": {"git_commit": "abc"},
                "model": {"weight_bytes": 123},
                "hardware": {"gpus": [{"index": 0, "name": "H20", "memory_mib": 98304}]},
            }
            fingerprint = autopilot.experiment_fingerprint(task, discovery)

            def prior(name, value, failure_class, observed_fingerprint):
                stage = root / "runs" / "stages" / name
                stage.mkdir(parents=True)
                autotune.write_json(stage / "spec.json", {
                    "experiment_fingerprint": observed_fingerprint,
                    "search": {"baseline": {"tp_size": 1}},
                })
                autotune.write_json(stage / "results.json", [{
                    "ok": False, "kind": "candidate",
                    "config": {"tp_size": 1, "page_size": value},
                    "status": {"failure_class": failure_class},
                }])

            prior("transient", 1, "port_conflict", fingerprint)
            prior("definitive", 16, "oom", fingerprint)
            prior("foreign", 32, "oom", "different")
            self.assertEqual(
                autopilot.known_failed_candidates(task, discovery),
                {("page_size", "16")},
            )

    def test_steady_state_retry_expands_generated_shared_prefix_requests(self):
        command = [
            "python3", "-m", "sglang.bench_serving", "--num-prompts", "512",
            "--gsp-num-groups", "8", "--gsp-prompts-per-group", "64",
        ]
        effective = autotune.increase_benchmark_request_count(command, 967)
        self.assertEqual(effective, 968)
        self.assertEqual(command[command.index("--num-prompts") + 1], "967")
        self.assertEqual(command[command.index("--gsp-prompts-per-group") + 1], "121")

    def valid_task(self):
        return {
            "name": "test",
            "repository": "/tmp",
            "python": sys.executable,
            "model_path": "/tmp",
            "workload": {
                "input_tokens": 256,
                "output_tokens": 64,
                "max_concurrency": 4,
                "num_prompts": 100,
            },
            "slo": {"p99_ttft_ms": 1000},
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "budget": {"max_trials": 8, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "output_dir": "/tmp/runs",
            "confirmation_repetitions": 3,
        }

    def test_requires_enough_trials_for_screen_and_confirmation(self):
        task = self.valid_task()
        task["budget"]["max_trials"] = 8
        self.assertIn(
            "budget.max_trials must be at least 9 for profiling, screening, and confirmation",
            autopilot.validate_task(task),
        )

    def test_task_rejects_unsafe_paths_names_ports_and_environment_before_execution(self):
        task = self.valid_task()
        task.update({
            "name": "../../escape",
            "output_dir": "/",
            "port": 80,
            "env": {"UNSAFE_SHELL_HOOK": "value"},
        })
        errors = autopilot.validate_task(task)
        self.assertIn("name must be a safe 1-64 character identifier", errors)
        self.assertIn("output_dir must not be the filesystem root", errors)
        self.assertIn("port must be an integer between 1024 and 65535", errors)
        self.assertIn("env contains unsupported key: UNSAFE_SHELL_HOOK", errors)

    def test_null_measurement_uses_defaults_for_backward_compatibility(self):
        task = self.valid_task()
        task["measurement"] = None
        self.assertNotIn("measurement must be an object", autopilot.validate_task(task))

    def test_online_mode_accepts_no_slo_constraints(self):
        task = self.valid_task()
        task["budget"]["max_trials"] = 9
        task["slo"] = {}
        self.assertEqual(autopilot.validate_task(task), [])

    def test_benchmark_flag_contract_follows_real_and_unbounded_workloads(self):
        custom = self.valid_task()
        custom["workload"]["dataset"] = {
            "name": "custom", "path": "/tmp/requests.jsonl",
            "apply_chat_template": True,
        }
        custom_flags = autopilot.required_benchmark_cli_flags(
            custom, {"context_length": 32768}
        )
        self.assertTrue({
            "--dataset-path", "--apply-chat-template", "--sharegpt-context-len",
            "--max-concurrency",
        }.issubset(custom_flags))
        self.assertNotIn("--tokenize-prompt", custom_flags)

        offline = self.valid_task()
        offline.update({"deployment_mode": "offline_throughput", "slo": {}})
        offline_flags = autopilot.required_benchmark_cli_flags(offline, {})
        self.assertIn("--flush-cache", offline_flags)
        self.assertIn("--tokenize-prompt", offline_flags)
        self.assertNotIn("--max-concurrency", offline_flags)

    def test_empty_slo_is_an_objective_only_pass(self):
        self.assertEqual(
            inferopt.slo_results({"metrics": {}}, {"slo": {}}),
            {"passed": True, "checks": []},
        )
        self.assertNotIn(
            "slo must contain at least one hard constraint",
            inferopt.validate_spec({"slo": {}}),
        )

    def test_catalog_binding_renders_current_sglang_flag(self):
        bindings = {
            "fp8_gemm_runner_backend": {
                "primary_flag": "--fp8-gemm-runner-backend",
                "action": "_StoreAction",
                "value_type": "str",
                "choices": ["auto", "flashinfer"],
            },
        }
        self.assertEqual(
            autotune.parameter_args({"fp8_gemm_runner_backend": "flashinfer"}, bindings),
            ["--fp8-gemm-runner-backend", "flashinfer"],
        )

    def test_empty_screen_has_structured_bottleneck_result(self):
        result = autopilot.bottleneck_summary(
            {"aggregates": []},
            {"workload": {"input_tokens": 256, "max_concurrency": 4}},
        )
        self.assertEqual(result["classification"], "screening_unavailable")

    def test_incomplete_parameter_search_is_not_rendered_as_a_recommendation(self):
        report = inferopt_cli.markdown_report({
            "run_dir": "/tmp/run",
            "recommendation_status": "insufficient_parameter_evidence",
            "deployable": False,
            "recommended_configuration": None,
            "deployment_command": None,
            "parameter_search": {"executed_parameter_candidates": 0, "sufficient_evidence": False},
        })
        self.assertIn("No deployment command is recommended", report)
        self.assertNotIn("## Deployment Command", report)
        self.assertIn("Executed parameter candidates: `0`", report)

    def test_report_does_not_show_screening_rejection_for_confirmed_candidate(self):
        candidate = {
            "kind": "candidate",
            "config": {"tp_size": 4, "cuda_graph_max_bs_decode": 16},
            "env": {},
            "metrics": {"total_throughput_tps": 110.0},
            "comparison": {"improvement_pct": 10.0},
            "rejection_reasons": ["baseline_not_confirmed"],
        }
        report = inferopt_cli.markdown_report({
            "run_dir": "/tmp/run",
            "recommendation_status": "confirmed_candidate",
            "deployable": True,
            "recommended_configuration": {**candidate, "confirmed": True},
            "screening": {"aggregates": [{"kind": "baseline"}, candidate]},
        })
        self.assertIn("Final confirmation: `confirmed_candidate`", report)
        self.assertNotIn("baseline_not_confirmed", report)

    def test_report_distinguishes_serial_profiling_from_parallel_screening(self):
        report = inferopt_cli.markdown_report({
            "run_dir": "/tmp/run",
            "parallel_pipeline": {
                "enabled": False,
                "profile_gpu": "0",
                "screening_gpus": ["0", "1", "2", "3"],
                "screening_parallel_workers": 4,
                "screening_gpu_allocation": "exclusive",
                "policy": "serial Nsys profiling followed by exclusive-GPU parallel screening",
            },
        })
        self.assertIn("Nsys/preprofile overlap enabled: `False`", report)
        self.assertIn("Screening GPU pool: `['0', '1', '2', '3']`", report)
        self.assertIn("Maximum concurrent screening workers: `4`", report)

    def test_report_marks_offline_no_slo_workload_as_unbounded(self):
        report = inferopt_cli.markdown_report({
            "run_dir": "/tmp/run",
            "deployment_policy": {"mode": "offline_throughput"},
            "requested_slo": {},
            "analysis_workload": {
                "input_tokens": 16384,
                "output_tokens": 128,
                "initial_backlog_requests": 32,
                "max_concurrency": 1,
            },
            "calibration": {"selected_analysis_concurrency": 45},
        })
        self.assertIn("Client concurrency: `unbounded`", report)
        self.assertNotIn("Selected SLO-safe execution concurrency", report)

    def test_report_records_local_cookbook_provenance(self):
        report = inferopt_cli.markdown_report({
            "run_dir": "/tmp/run",
            "discovery": {"cookbook": {"local_checkout": {
                "status": "available",
                "documents": [{
                    "path": "autoregressive/Qwen/Qwen3.5.mdx",
                    "commit": "abc123",
                    "sha256": "digest",
                }],
                "recipes": [{"name": "cookbook-qwen3.5-2"}],
                "excluded_recipes": [{
                    "name": "cookbook-qwen3.5-397b",
                    "documented_model": "qwen3.5-397b-a17b",
                    "reason": "documented checkpoint variant does not match the local checkpoint size",
                }],
            }}},
            "cookbook_preflight": {
                "candidate_bundles": [{"name": "cookbook-qwen3.5-2"}],
                "excluded_bundles": [{
                    "name": "cookbook-qwen3.5-amd",
                    "reason": "cookbook recipe requires AMD GPU support",
                }],
            },
        })
        self.assertIn("## Cookbook Knowledge", report)
        self.assertIn("Qwen3.5.mdx", report)
        self.assertIn("SGLang commit `abc123`", report)
        self.assertIn("Cookbook TP/PP/DP/EP values", report)
        self.assertIn("qwen3.5-397b-a17b", report)
        self.assertIn("## Cookbook Qualification", report)
        self.assertIn("requires AMD GPU support", report)

    def test_report_explains_nsys_denominators_and_timing_policy(self):
        report = inferopt_cli.markdown_report({
            "run_dir": "/tmp/run",
            "profiling": {"diagnosis": {
                "primary_bottleneck": "attention",
                "profiling_run_performance_comparable": False,
                "gpu_timeline_active_pct": 65.0,
                "gpu_timeline_gap_pct": 35.0,
                "shares_pct": {"attention_kernels": 93.6, "moe_kernels": 6.4},
                "top_kernels": [{"name": "flash_attention", "time_pct": 93.6}],
                "top_cuda_apis": [{"name": "cudaLaunchKernel", "time_pct": 80.0}],
            }},
        })
        self.assertIn("shares of total GPU kernel time", report)
        self.assertIn("only GPU kernel shares influence routing", report)
        self.assertIn("occupancy", report)

    def test_report_exposes_moe_tuning_only_as_a_separate_opt_in_command(self):
        report = inferopt_cli.markdown_report({
            "run_dir": "/tmp/run",
            "recommendation_status": "retain_confirmed_baseline",
            "deployable": True,
            "kernel_optimization": {
                "fused_moe": {
                    "status": "candidate_required",
                    "priority": "high",
                    "reason": "missing config",
                    "shape_matched_batch_sizes": [8, 8192],
                },
                "fused_moe_execution": {
                    "status": "not_run",
                    "reason": "separate opt-in operation",
                },
            },
        })
        self.assertIn("inferopt tune-moe", report)
        self.assertIn("--yes", report)
        self.assertIn("not part of `inferopt run`", report)
        self.assertIn("generated_config_deployable=true", report)

    def test_optional_moe_command_runs_end_to_end_gate_after_generation(self):
        task = self.valid_task()
        task["budget"]["max_trials"] = 12
        profile = {"run_dir": "/tmp/profile", "runtime_observations": {}}
        completed = {
            "recommended_configuration": {"config": {"tp_size": 1}},
            "deployment_environment": {},
        }
        discovery = {
            "derived": {"minimum_tp_size": 1},
            "model": {"is_moe": True},
            "hardware": {"gpus": []},
            "parameter_catalog": {"parameters": []},
        }
        winner = {
            "configuration_name": "fused-moe-autotuned-config",
            "confirmed": True,
            "config": {"tp_size": 1},
            "env": {"SGLANG_MOE_CONFIG_DIR": "/tmp/generated"},
        }
        with (
            mock.patch.object(inferopt, "load_json", side_effect=[profile, completed]),
            mock.patch.object(autopilot, "discover", return_value=discovery),
            mock.patch.object(autopilot, "moe_kernel_optimization_plan", return_value={"status": "candidate_required"}),
            mock.patch.object(autopilot, "execute_moe_kernel_tuning", return_value={
                "status": "completed", "config_root": "/tmp/generated"
            }),
            mock.patch.object(autopilot, "explicit_configuration_spec", return_value={"validation": True}) as build_validation,
            mock.patch.object(autopilot, "execute_with_progress", return_value={"winner": winner}),
            mock.patch.object(autopilot, "final_server_command", return_value=["python", "-m", "sglang.launch_server"]),
            mock.patch.object(autopilot.ProgressReporter, "emit"),
        ):
            result = inferopt_cli.tune_moe(
                task, "/tmp/profile.json", "/tmp/final.json", "/tmp/output", 30, 2
            )
        self.assertTrue(result["generated_config_deployable"])
        self.assertEqual(result["deployment_environment"]["SGLANG_MOE_CONFIG_DIR"], "/tmp/generated")
        self.assertEqual(build_validation.call_args.kwargs["repetitions"], 3)

    def test_calibration_is_geometric_and_budget_aware(self):
        task = self.valid_task()
        task["calibration"] = {"strategy": "full_curve"}
        task["budget"]["max_trials"] = 30
        self.assertEqual(autopilot.calibration_concurrencies(task), [4, 8, 16, 32, 64])
        task["budget"]["max_trials"] = 11
        self.assertEqual(autopilot.calibration_concurrencies(task), [4, 8])

    def test_adaptive_calibration_starts_at_the_target_only(self):
        task = self.valid_task()
        task["budget"]["max_trials"] = 30
        self.assertEqual(autopilot.calibration_concurrencies(task), [4])

    def test_calibration_uses_scaled_task_measurement_not_fixed_512_requests(self):
        task = self.valid_task()
        task["workload"]["max_concurrency"] = 8
        task["measurement"] = {"warmup_requests": 32, "min_measurement_requests": 128, "min_measurement_seconds": 20}
        discovery = {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}}
        spec = autopilot.calibration_spec(task, discovery, 2, 1, 30)
        self.assertGreaterEqual(spec["benchmark"]["num_prompts"], 20)
        self.assertEqual(spec["benchmark"]["p99_request_waves"], 10)

    def test_screening_uses_bounded_fidelity_before_confirmation(self):
        task = self.valid_task()
        task["measurement"] = {
            "warmup_requests": 64,
            "min_measurement_requests": 1024,
            "min_measurement_seconds": 45,
        }
        discovery = {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}}
        spec = autopilot.build_execution_spec(
            task, discovery, stage_name="screen", baseline={"tp_size": 1},
            space={}, max_trials=1, repetitions=1, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        self.assertEqual(spec["benchmark"]["num_prompts"], 40)
        self.assertEqual(spec["benchmark"]["p99_request_waves"], 10)
        self.assertEqual(spec["benchmark"]["warmup_requests"], 16)
        self.assertEqual(spec["benchmark"]["min_measurement_seconds"], 15.0)

    def test_offline_screening_uses_short_nomination_window(self):
        task = self.valid_task()
        task["deployment_mode"] = "offline_throughput"
        task["slo"] = {}
        task["workload"]["max_concurrency"] = 64
        task["measurement"] = {
            "warmup_requests": 256,
            "min_measurement_requests": 512,
            "min_measurement_seconds": 45,
        }
        discovery = {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}}
        spec = autopilot.build_execution_spec(
            task, discovery, stage_name="screen", baseline={"tp_size": 1},
            space={}, max_trials=1, repetitions=1, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        self.assertEqual(spec["benchmark"]["num_prompts"], 320)
        self.assertEqual(spec["benchmark"]["warmup_requests"], 32)
        self.assertEqual(spec["benchmark"]["min_measurement_seconds"], 10.0)

    def test_offline_calibration_caps_warmup_and_requests(self):
        task = self.valid_task()
        task["deployment_mode"] = "offline_throughput"
        task["slo"] = {}
        task["workload"]["max_concurrency"] = 64
        task["measurement"] = {
            "warmup_requests": 256,
            "min_measurement_requests": 512,
            "min_measurement_seconds": 45,
        }
        discovery = {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}}
        spec = autopilot.calibration_spec(task, discovery, 64, 1, 30)
        self.assertEqual(spec["benchmark"]["num_prompts"], 320)
        self.assertEqual(spec["benchmark"]["warmup_requests"], 32)

    def test_offline_screen_uses_five_concurrency_waves_and_scales_with_load(self):
        task = self.valid_task()
        task["deployment_mode"] = "offline_throughput"
        task["slo"] = {}
        task["workload"].update({
            "input_tokens": 16384, "output_tokens": 256, "max_concurrency": 64,
        })
        task["measurement"] = {
            "warmup_requests": 256,
            "min_measurement_requests": 512,
            "min_measurement_seconds": 45,
        }
        discovery = {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}}
        spec = autopilot.build_execution_spec(
            task, discovery, stage_name="screen", baseline={"tp_size": 1},
            space={}, max_trials=1, repetitions=1, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        self.assertEqual(spec["benchmark"]["num_prompts"], 320)
        self.assertEqual(spec["benchmark"]["warmup_requests"], 32)
        self.assertTrue(spec["benchmark"]["unbounded_concurrency"])

        task["workload"]["max_concurrency"] = 8
        lower_load = autopilot.build_execution_spec(
            task, discovery, stage_name="screen", baseline={"tp_size": 1},
            space={}, max_trials=1, repetitions=1, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        self.assertEqual(lower_load["benchmark"]["num_prompts"], 40)
        self.assertEqual(lower_load["benchmark"]["warmup_requests"], 8)

    def test_shared_prefix_screening_uses_bounded_window_before_confirmation(self):
        task = self.valid_task()
        task["experiment_mode"] = "fast"
        task["workload"].update({
            "input_tokens": 4096,
            "output_tokens": 128,
            "max_concurrency": 8,
            "num_prompts": 1024,
            "shared_prefix": {
                "groups": 8,
                "prompts_per_group": 128,
                "system_prompt_tokens": 2048,
                "question_tokens": 2048,
                "ordered": True,
            },
        })
        task["measurement"] = {
            "warmup_requests": 32,
            "min_measurement_requests": 512,
            "min_measurement_seconds": 45,
        }
        discovery = {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}}
        spec = autopilot.build_execution_spec(
            task, discovery, stage_name="screen", baseline={"tp_size": 1},
            space={}, max_trials=1, repetitions=1, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        self.assertEqual(spec["benchmark"]["num_prompts"], 80)
        self.assertEqual(spec["benchmark"]["gsp_prompts_per_group"], 10)
        self.assertEqual(spec["benchmark"]["warmup_requests"], 16)
        self.assertEqual(spec["benchmark"]["min_measurement_seconds"], 15.0)

    def test_fast_screening_uses_a_small_nomination_window(self):
        task = self.valid_task()
        task["experiment_mode"] = "fast"
        task["workload"]["max_concurrency"] = 8
        task["measurement"] = {
            "warmup_requests": 32,
            "min_measurement_requests": 512,
            "min_measurement_seconds": 45,
        }
        discovery = {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}}
        spec = autopilot.build_execution_spec(
            task, discovery, stage_name="screen", baseline={"tp_size": 1},
            space={}, max_trials=1, repetitions=1, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        self.assertEqual(spec["benchmark"]["num_prompts"], 80)
        self.assertEqual(spec["benchmark"]["p99_request_waves"], 10)
        self.assertEqual(spec["benchmark"]["warmup_requests"], 16)
        self.assertEqual(spec["benchmark"]["min_measurement_seconds"], 15.0)

    def test_explicit_calibration_range_starts_at_one_and_includes_the_cap(self):
        task = self.valid_task()
        task["budget"]["max_trials"] = 30
        task["calibration"] = {
            "strategy": "full_curve", "min_concurrency": 1,
            "max_concurrency": 50, "max_steps": 7,
        }
        self.assertEqual(autopilot.calibration_concurrencies(task), [1, 2, 4, 8, 16, 32, 50])

    def test_explicit_calibration_points_are_preserved(self):
        task = self.valid_task()
        task["budget"]["max_trials"] = 9
        task["workload"]["max_concurrency"] = 32
        task["calibration"] = {"concurrencies": [1, 4, 12, 32]}
        self.assertEqual(autopilot.validate_task(task), [])
        self.assertEqual(autopilot.calibration_concurrencies(task), [1, 4, 12, 32])

    def test_confirmation_rejects_exhausted_gpu_budget(self):
        task = self.valid_task()
        discovery = {"model": {}, "hardware": {"vendor": "nvidia", "gpus": []}}
        screen = {
            "aggregates": [{"config": {"tp_size": 1}}],
            "screening_winner": None,
        }
        with self.assertRaisesRegex(ValueError, "GPU-hour budget exhausted"):
            autopilot.confirmation_spec(task, discovery, screen, 3, 0, 10)

    def test_backend_error_beats_warmup_oom_wording(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(
                "reducing max_m to avoid out of memory\nValueError: not enough values to unpack\n",
                encoding="utf-8",
            )
            self.assertEqual(
                autotune.classify_failure(log, Path(directory) / "benchmark.log", "benchmark exited"),
                "backend_incompatible",
            )

    def test_sglang_static_memory_error_beats_unrelated_missing_module(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(
                "ModuleNotFoundError: No module named 'optional_backend'\n"
                "ValueError: Loaded weights leave no GPU memory for KV cache under "
                "--mem-fraction-static=0.666. Raise --mem-fraction-static above 0.715.\n",
                encoding="utf-8",
            )
            self.assertEqual(
                autotune.classify_failure(log, Path(directory) / "benchmark.log", "server exited"),
                "memory_infeasible",
            )

    def test_sigkill_is_not_misclassified_as_optional_dependency_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text("warning: No module named 'optional_backend'\n", encoding="utf-8")
            self.assertEqual(
                autotune.classify_failure(
                    log, Path(directory) / "benchmark.log",
                    "server exited during startup with code -9",
                ),
                "process_killed",
            )

    def test_quantized_checkpoint_does_not_emit_implicit_dtype_override(self):
        task = self.valid_task()
        discovery = {
            "derived": {"minimum_tp_size": 1},
            "model": {
                "is_moe": True,
                "dtype": "bfloat16",
                "checkpoint_dtype": "bfloat16",
                "quantization": "fp8",
                "weight_quantization": "fp8",
                "context_length": 32768,
            },
            "hardware": {"gpus": []},
            "parameter_catalog": {"parameters": []},
        }
        spec = autopilot.build_execution_spec(
            task, discovery, stage_name="confirm", baseline={"tp_size": 1},
            space={}, max_trials=1, repetitions=1, remaining_gpu_hours=1,
            remaining_wall_minutes=30,
        )
        manifest = autotune.command_manifest(
            spec,
            {"config": {"tp_size": 1}},
            Path("/tmp/unused-trial"),
        )
        self.assertNotIn("--dtype", manifest["server"])
        self.assertNotIn("--quantization", manifest["server"])
        self.assertEqual(spec["model"]["detected_weight_quantization"], "fp8")

    def test_explicit_dtype_override_is_preserved(self):
        task = self.valid_task()
        task["model"] = {"dtype": "float16"}
        discovery = {
            "derived": {"minimum_tp_size": 1},
            "model": {"is_moe": False, "dtype": "bfloat16"},
            "hardware": {"gpus": []},
            "parameter_catalog": {"parameters": []},
        }
        spec = autopilot.build_execution_spec(
            task, discovery, stage_name="confirm", baseline={"tp_size": 1},
            space={}, max_trials=1, repetitions=1, remaining_gpu_hours=1,
            remaining_wall_minutes=30,
        )
        self.assertEqual(spec["model"]["dtype"], "float16")

    def test_profile_comparability_uses_unprofiled_target_concurrency(self):
        profiling = {
            "benchmark": {"metrics": {"request_throughput_rps": 0.91}},
            "diagnosis": {},
        }
        calibration = {
            "selected_analysis_concurrency": 8,
            "points": [
                {
                    "concurrency": 8,
                    "valid_for_analysis": True,
                    "metrics": {"request_throughput_rps": 1.41},
                }
            ],
        }
        result = autopilot.annotate_profile_comparability(profiling, calibration)
        self.assertFalse(result["diagnosis"]["profiling_run_performance_comparable"])
        self.assertAlmostEqual(result["diagnosis"]["profile_throughput_regression_pct"], 35.461, places=3)

    def test_steady_state_duration_uses_sglang_aggregate_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.jsonl"
            result.write_text(
                '{"completed": 256, "duration": 29.34, "request_throughput": 8.72}\n',
                encoding="utf-8",
            )
            summary = autotune.summarize_jsonl(
                result,
                {
                    "benchmark": {"min_measurement_seconds": 30},
                    "slo": {},
                    "objective": {"metric": "request_throughput_rps"},
                },
            )
            validity = summary["measurement_validity"]
            self.assertEqual(validity["purpose"], "sample-validity gate only; not an SLO or optimization objective")
            self.assertEqual(validity["request_count"], 256)
            self.assertEqual(validity["duration_source"], "sglang_result_duration")
            self.assertAlmostEqual(validity["duration_sec"], 29.34)
            self.assertFalse(validity["duration_gate_passed"])

    def test_p99_slo_requires_explicit_tail_sample_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.jsonl"
            result.write_text(
                '{"completed": 256, "duration": 60, "request_throughput": 4.2, "p99_e2e_latency_ms": 900}\n',
                encoding="utf-8",
            )
            summary = autotune.summarize_jsonl(
                result,
                {
                    "benchmark": {"min_measurement_seconds": 30, "min_tail_samples": 5},
                    "slo": {"p99_e2e_latency_ms": 1000},
                    "objective": {"metric": "request_throughput_rps"},
                },
            )
            validity = summary["measurement_validity"]
            self.assertTrue(validity["duration_gate_passed"])
            self.assertFalse(validity["tail_sample_gate_passed"])
            self.assertEqual(validity["minimum_request_count_for_tail"], 500)

            result.write_text(
                '{"completed": 500, "duration": 60, "request_throughput": 8.3, "p99_e2e_latency_ms": 900}\n',
                encoding="utf-8",
            )
            summary = autotune.summarize_jsonl(
                result,
                {
                    "benchmark": {"min_measurement_seconds": 30, "min_tail_samples": 5},
                    "slo": {"p99_e2e_latency_ms": 1000},
                    "objective": {"metric": "request_throughput_rps"},
                },
            )
            self.assertTrue(summary["measurement_validity"]["tail_sample_gate_passed"])

    def test_p99_evidence_uses_ten_waves_of_actual_concurrency(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.jsonl"
            spec = {
                "benchmark": {
                    "min_measurement_seconds": 30,
                    "max_concurrency": 8,
                    "p99_request_waves": 10,
                },
                "slo": {"p99_e2e_latency_ms": 1000},
                "objective": {"metric": "request_throughput_rps"},
            }
            result.write_text(
                '{"completed": 79, "duration": 60, "request_throughput": 4.2, "p99_e2e_latency_ms": 700}\n',
                encoding="utf-8",
            )
            insufficient = autotune.summarize_jsonl(result, spec)["measurement_validity"]
            self.assertFalse(insufficient["tail_sample_gate_passed"])
            self.assertEqual(insufficient["minimum_request_count_for_tail"], 80)
            self.assertEqual(insufficient["tail_requirement_reason"], "concurrency_waves")

            result.write_text(
                '{"completed": 160, "duration": 60, "request_throughput": 4.2, "p99_e2e_latency_ms": 950}\n',
                encoding="utf-8",
            )
            measured = autotune.summarize_jsonl(
                result, spec, effective_concurrency=16
            )["measurement_validity"]
            self.assertTrue(measured["tail_sample_gate_passed"])
            self.assertEqual(measured["measurement_concurrency"], 16)
            self.assertEqual(measured["minimum_request_count_for_tail"], 160)

    def test_explicit_configuration_matrix_preserves_combined_configs(self):
        spec = {
            "budget": {"max_trials": 3},
            "search": {
                "strategy": "explicit_configurations",
                "baseline": {"tp_size": 1},
                "explicit_configurations": [{
                    "name": "combined-cache-and-graph",
                    "config": {"tp_size": 1, "page_size": 16, "cuda_graph_max_bs_decode": 8},
                }],
            },
        }
        matrix = autotune.candidate_matrix(spec)
        self.assertEqual([item["name"] for item in matrix], ["baseline", "combined-cache-and-graph"])
        self.assertEqual(matrix[1]["config"]["page_size"], 16)
        self.assertEqual(matrix[1]["config"]["cuda_graph_max_bs_decode"], 8)

    def test_explicit_configuration_matrix_preserves_isolated_environment(self):
        spec = {
            "budget": {"max_trials": 2},
            "search": {
                "strategy": "explicit_configurations",
                "baseline": {"tp_size": 4},
                "explicit_configurations": [{
                    "name": "fused-moe-autotuned-config",
                    "config": {"tp_size": 4},
                    "env": {"SGLANG_MOE_CONFIG_DIR": "/tmp/run/moe-config"},
                }],
            },
        }
        matrix = autotune.candidate_matrix(spec)
        self.assertEqual(len(matrix), 2)
        self.assertEqual(matrix[0]["env"], {})
        self.assertEqual(
            matrix[1]["env"],
            {"SGLANG_MOE_CONFIG_DIR": "/tmp/run/moe-config"},
        )

    def test_shared_prefix_workload_uses_native_sglang_dataset(self):
        workload = self.valid_task()["workload"]
        workload["shared_prefix"] = {
            "groups": 8,
            "prompts_per_group": 64,
            "system_prompt_tokens": 192,
            "question_tokens": 64,
            "ordered": True,
        }
        benchmark = autopilot.shared_prefix_benchmark(workload)
        self.assertEqual(benchmark["dataset_name"], "generated-shared-prefix")
        self.assertEqual(benchmark["num_prompts"], 512)
        self.assertEqual(benchmark["gsp_system_prompt_len"], 192)
        self.assertEqual(benchmark["gsp_question_len"], 64)
        spec = {
            "execution": {"python": sys.executable, "host": "127.0.0.1", "port": 31000},
            "benchmark": {"max_concurrency": 4, "warmup_requests": 32, "seed": 1, **benchmark},
            "model": {"path": "/tmp/model"},
            "objective": {"metric": "request_throughput_rps"},
        }
        manifest = autotune.command_manifest(spec, {"config": {}, "name": "baseline"}, Path("/tmp/trial"))
        self.assertIn("generated-shared-prefix", manifest["benchmark"])
        self.assertIn("--gsp-system-prompt-len", manifest["benchmark"])
        self.assertNotIn("--tokenize-prompt", manifest["benchmark"])

    def test_manifest_uses_discovered_current_benchmark_module(self):
        spec = {
            "execution": {
                "python": sys.executable, "host": "127.0.0.1", "port": 31000,
                "benchmark_module": "sglang.benchmark.serving",
            },
            "benchmark": {"num_prompts": 8, "max_concurrency": 1, "warmup_requests": 1},
            "model": {"path": "/tmp/model"},
            "objective": {"metric": "request_throughput_rps"},
        }
        manifest = autotune.command_manifest(
            spec, {"config": {}, "name": "baseline"}, Path("/tmp/trial")
        )
        module_index = manifest["benchmark"].index("-m") + 1
        self.assertEqual(manifest["benchmark"][module_index], "sglang.benchmark.serving")

    def test_offline_without_slo_omits_client_concurrency_cap(self):
        task = self.valid_task()
        task["deployment_mode"] = "offline_throughput"
        task["slo"] = {}
        task["workload"]["max_concurrency"] = 64
        discovery = {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}}
        spec = autopilot.build_execution_spec(
            task, discovery, stage_name="screen", baseline={"tp_size": 1},
            space={}, max_trials=1, repetitions=1, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        self.assertTrue(spec["benchmark"]["unbounded_concurrency"])
        manifest = autotune.command_manifest(spec, {"config": {"tp_size": 1}, "name": "baseline"}, Path("/tmp/trial"))
        self.assertNotIn("--max-concurrency", manifest["benchmark"])

        task["slo"] = {"p99_e2e_latency_ms": 1000}
        bounded = autopilot.build_execution_spec(
            task, discovery, stage_name="screen", baseline={"tp_size": 1},
            space={}, max_trials=1, repetitions=1, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        bounded_manifest = autotune.command_manifest(
            bounded, {"config": {"tp_size": 1}, "name": "baseline"}, Path("/tmp/trial")
        )
        self.assertFalse(bounded["benchmark"]["unbounded_concurrency"])
        self.assertFalse(bounded["benchmark"]["auto_max_concurrency"])
        self.assertIn("--max-concurrency", bounded_manifest["benchmark"])

    def test_slo_calibration_starts_at_resolved_capacity_then_binary_searches(self):
        task = self.valid_task()
        task.update({"deployment_mode": "offline_throughput", "slo": {"p99_e2e_latency_ms": 1000}})
        task["workload"]["max_concurrency"] = 64
        task["calibration"] = {"strategy": "adaptive", "min_concurrency": 1, "max_steps": 4}
        task["budget"] = {"max_trials": 13, "max_gpu_hours": 1, "max_wall_time_minutes": 30}
        discovery = {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}}
        calls = []

        def fake_execute(spec):
            calls.append(spec)
            status = {
                "resolved_client_max_concurrency": 512,
                "resolved_server_max_running_requests": 512,
                "resolved_capacity_source": "/server_info",
            }
            return {
                "run_dir": "/tmp/run", "stop_reason": "completed_search", "approx_gpu_hours": 0,
                "completed_server_sessions": 1,
                "results": [
                    {"kind": "baseline", "ok": True, "status": status,
                     "calibration_concurrency": concurrency,
                     "effective_num_prompts": concurrency * 5,
                     "slo": {"passed": concurrency <= 320}, "metrics": {}}
                    for concurrency in (512, 256, 384, 320)
                ],
            }

        with patch.object(autopilot, "execute", side_effect=fake_execute):
            result = autopilot.run_calibration(task, discovery, Path("/tmp"))
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["benchmark"]["auto_max_concurrency"])
        self.assertTrue(calls[0]["benchmark"]["unbounded_concurrency"])
        self.assertTrue(calls[0]["benchmark"]["calibration_session"]["initial_unbounded_probe"])
        self.assertEqual(calls[0]["benchmark"]["calibration_session"]["strategy"], "adaptive_slo")
        self.assertEqual(calls[0]["benchmark"]["calibration_session"]["max_steps"], 4)
        self.assertEqual(calls[0]["benchmark"]["calibration_session"]["request_waves"], 10)
        self.assertEqual(result["points"][0]["concurrency"], 512)
        self.assertEqual(result["points"][0]["requested_concurrency"], 64)
        self.assertEqual(result["selected_analysis_concurrency"], 320)
        self.assertEqual(result["server_sessions"], 1)

    def test_calibrated_slo_concurrency_is_the_post_calibration_workload(self):
        task = self.valid_task()
        task["workload"]["max_concurrency"] = 64
        task["measurement"] = {
            "confirmation_requests": 640,
            "p99_request_waves": 10,
        }
        calibrated = autopilot.task_at_calibrated_concurrency(
            task, {"selected_analysis_concurrency": 12}
        )
        self.assertEqual(calibrated["workload"]["max_concurrency"], 12)
        self.assertEqual(calibrated["measurement"]["confirmation_requests"], 120)
        self.assertEqual(task["workload"]["max_concurrency"], 64)
        self.assertEqual(task["measurement"]["confirmation_requests"], 640)

        task["slo"] = {}
        unbounded = autopilot.task_at_calibrated_concurrency(
            task, {"selected_analysis_concurrency": 12}
        )
        self.assertEqual(unbounded["workload"]["max_concurrency"], 64)

    def test_report_explains_concurrency_scaled_p99_request_floor(self):
        report = inferopt_cli.markdown_report({
            "run_dir": "/tmp/run",
            "requested_slo": {"p99_ttft_ms": 1000},
            "analysis_workload": {
                "input_tokens": 256, "output_tokens": 64, "max_concurrency": 8,
            },
            "calibration": {"selected_analysis_concurrency": 8},
            "measurement_policy": {"p99_request_waves": 10},
            "confirmation": {
                "planned_trials": 6, "planned_server_sessions": 2,
                "adaptive_confirmation": {
                    "enabled": True, "triggered": True,
                    "trigger_cv_pct": 5, "completed_repetitions": 3,
                },
            },
        })
        self.assertIn("Concurrency waves per measured window: `10`", report)
        self.assertIn("Selected-concurrency request floor: `80`", report)
        self.assertIn("lower statistical confidence", report)
        self.assertIn("Adaptive noise extension triggered: `True`", report)

    def test_auto_capacity_probe_omits_static_cap_from_initial_manifest(self):
        spec = {
            "execution": {"python": sys.executable, "host": "127.0.0.1", "port": 31000},
            "benchmark": {
                "dataset_name": "random-ids", "num_prompts": 64,
                "max_concurrency": 64, "auto_max_concurrency": True,
                "warmup_requests": 1, "seed": 1,
            },
            "model": {"path": "/tmp/model"},
            "objective": {"metric": "request_throughput_rps"},
        }
        manifest = autotune.command_manifest(
            spec, {"config": {}, "name": "baseline"}, Path("/tmp/trial")
        )
        self.assertNotIn("--max-concurrency", manifest["benchmark"])

    def test_custom_dataset_manifest_uses_native_sglang_flags(self):
        spec = {
            "execution": {"python": sys.executable, "host": "127.0.0.1", "port": 31000},
            "benchmark": {
                "dataset_name": "custom", "dataset_path": "/tmp/requests.jsonl",
                "apply_chat_template": True, "sharegpt_context_len": 32768,
                "num_prompts": 64, "max_concurrency": 8,
                "warmup_requests": 8, "seed": 1,
            },
            "model": {"path": "/tmp/model"},
            "objective": {"metric": "request_throughput_rps"},
        }
        manifest = autotune.command_manifest(
            spec, {"config": {}, "name": "baseline"}, Path("/tmp/trial")
        )["benchmark"]
        self.assertEqual(manifest[manifest.index("--dataset-name") + 1], "custom")
        self.assertEqual(manifest[manifest.index("--dataset-path") + 1], "/tmp/requests.jsonl")
        self.assertIn("--apply-chat-template", manifest)
        self.assertEqual(manifest[manifest.index("--sharegpt-context-len") + 1], "32768")

    def test_shared_prefix_defaults_to_ordered_groups(self):
        workload = self.valid_task()["workload"]
        workload["shared_prefix"] = {
            "groups": 2,
            "prompts_per_group": 4,
            "system_prompt_tokens": 192,
            "question_tokens": 64,
        }
        self.assertTrue(autopilot.shared_prefix_benchmark(workload)["gsp_ordered"])

    def test_interaction_uses_stable_positive_subthreshold_seeds(self):
        task = self.valid_task()
        task["confirmation_repetitions"] = 2
        task["budget"] = {"max_trials": 12, "max_gpu_hours": 1, "max_wall_time_minutes": 30}
        baseline = {"kind": "baseline", "config": {"tp_size": 1}}
        seed = lambda name, config, gain: {
            "kind": "candidate", "configuration_name": name, "config": config,
            "stable": True, "all_repetitions_slo_passed": True,
            "screening_accepted": False,
            "comparison": {"improvement_pct": gain, "secondary_regressions_passed": True},
        }
        spec = autopilot.interaction_spec(
            task,
            {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}},
            {"aggregates": [baseline, seed("graph", {"tp_size": 1, "cuda_graph_max_bs_decode": 8}, 0.5), seed("admission", {"tp_size": 1, "max_running_requests": 16}, 0.3)]},
            remaining_trials=8, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        self.assertIsNotNone(spec)
        combined = spec["search"]["explicit_configurations"][0]["config"]
        self.assertEqual(combined["cuda_graph_max_bs_decode"], 8)
        self.assertEqual(combined["max_running_requests"], 16)

    def test_reference_baseline_does_not_create_a_baseline_trial(self):
        spec = {
            "search": {
                "strategy": "explicit_configurations",
                "baseline": {"tp_size": 4},
                "include_baseline": False,
                "reference_baseline": {
                    "config": {"tp_size": 4},
                    "metrics": {"total_throughput_tps": 100.0},
                },
                "explicit_configurations": [{
                    "name": "graph", "config": {"tp_size": 4, "cuda_graph_max_bs_decode": 16},
                }],
                "repetitions": 1,
            },
            "budget": {"max_trials": 1},
        }
        matrix = autotune.candidate_matrix(spec)
        self.assertEqual([item["name"] for item in matrix], ["graph"])
        self.assertTrue(all(item["kind"] == "candidate" for item in matrix))

    def test_offline_confirmation_runs_only_the_selected_candidate(self):
        task = self.valid_task()
        task.update({"deployment_mode": "offline_throughput", "slo": {}})
        task["confirmation_repetitions"] = 2
        screen = {
            "aggregates": [{
                "kind": "baseline", "config": {"tp_size": 1}, "env": {},
                "metrics": {"request_throughput_rps": 10.0},
                "confirmation_reference": {
                    "metrics": {"request_throughput_rps": 9.5},
                    "measurement_validity": {"duration_sec": 45.0},
                    "num_prompts": 50,
                    "dataset_name": "random-ids",
                },
            }],
            "screening_winner": {
                "config": {"tp_size": 1, "enable_mixed_chunk": True}, "env": {},
            },
        }
        spec = autopilot.confirmation_spec(
            task,
            {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}},
            screen, remaining_trials=3, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        self.assertFalse(spec["search"]["include_baseline"])
        self.assertEqual(spec["search"]["repetitions"], 1)
        self.assertEqual(spec["search"]["min_confirm_repetitions"], 1)
        self.assertEqual(spec["benchmark"]["num_prompts"], 50)
        self.assertEqual(
            spec["search"]["reference_baseline"]["metrics"]["request_throughput_rps"],
            9.5,
        )
        self.assertEqual([item["name"] for item in autotune.candidate_matrix(spec)], ["selected-candidate"])
        self.assertFalse(any(
            "min_confirm_repetitions" in error for error in autotune.execution_errors(spec)
        ))

    def test_balanced_slo_confirmation_uses_two_resident_server_sessions(self):
        task = self.valid_task()
        task.update({"deployment_mode": "online_latency", "experiment_mode": "balanced"})
        task["confirmation_repetitions"] = 2
        screen = {
            "aggregates": [{
                "kind": "baseline", "config": {"tp_size": 1}, "env": {},
                "metrics": {"request_throughput_rps": 10.0},
            }],
            "screening_winner": {
                "config": {"tp_size": 1, "enable_mixed_chunk": True}, "env": {},
            },
        }
        spec = autopilot.confirmation_spec(
            task,
            {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []},
             "parameter_catalog": {"parameters": []}},
            screen, remaining_trials=4, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        self.assertEqual(spec["search"]["repetitions"], 2)
        self.assertEqual(spec["search"]["min_confirm_repetitions"], 2)
        self.assertTrue(spec["search"]["reuse_server_across_repetitions"])
        self.assertTrue(spec["benchmark"]["flush_cache"])
        sessions = autotune.measurement_plan(spec)
        self.assertEqual(len(sessions), 2)
        self.assertEqual([item["configuration_name"] for item in sessions], ["baseline", "selected-candidate"])
        self.assertTrue(all(item["repeat_indices"] == [0, 1] for item in sessions))
        self.assertFalse(any(
            "reuse_server_across_repetitions" in error
            for error in autotune.execution_errors(spec)
        ))

    def test_generated_slo_confirmation_reserves_adaptive_third_pair(self):
        task = self.valid_task()
        task.update({"deployment_mode": "online_latency", "experiment_mode": "balanced"})
        task["confirmation_repetitions"] = 2
        task["measurement"] = {
            "min_measurement_seconds": 15,
            "adaptive_confirmation_cv_pct": 5,
            "adaptive_confirmation_max_repetitions": 3,
            "adaptive_confirmation_min_measurement_seconds": 30,
        }
        screen = {
            "aggregates": [{
                "kind": "baseline", "config": {"tp_size": 1}, "env": {},
                "metrics": {"request_throughput_rps": 10.0},
            }],
            "screening_winner": {
                "config": {"tp_size": 1, "enable_mixed_chunk": True}, "env": {},
            },
        }
        spec = autopilot.confirmation_spec(
            task,
            {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []},
             "parameter_catalog": {"parameters": []}},
            screen, remaining_trials=6, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        self.assertEqual(spec["budget"]["max_trials"], 6)
        self.assertEqual(spec["search"]["repetitions"], 2)
        self.assertEqual(spec["search"]["max_cv_pct"], 5.0)
        self.assertEqual(spec["search"]["adaptive_confirmation_cv_pct"], 5.0)
        self.assertEqual(spec["search"]["adaptive_confirmation_max_repetitions"], 3)
        self.assertEqual(
            spec["search"]["adaptive_confirmation_min_measurement_seconds"], 30.0
        )

    def test_offline_screen_captures_matched_half_size_confirmation_reference(self):
        task = self.valid_task()
        task.update({"deployment_mode": "offline_throughput", "slo": {}, "experiment_mode": "balanced"})
        task["measurement"] = {
            "warmup_requests": 32,
            "min_measurement_requests": 512,
            "min_measurement_seconds": 45,
            "confirmation_requests": 256,
        }
        task["budget"] = {"max_trials": 12, "max_gpu_hours": 1, "max_wall_time_minutes": 30}
        discovery = {
            "derived": {"minimum_tp_size": 1}, "model": {},
            "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []},
        }
        search_plan = {"ranked_parameter_groups": [{
            "parameter": "enable_mixed_chunk", "family": "scheduler", "values": [True],
        }]}
        spec = autopilot.screening_spec(
            task, discovery, search_plan, remaining_trials=6,
        )
        self.assertEqual(spec["benchmark"]["baseline_reference_num_prompts"], 256)
        self.assertEqual(spec["benchmark"]["num_prompts"], 256)
        self.assertEqual(spec["benchmark"]["min_measurement_seconds"], 45.0)
        self.assertTrue(spec["benchmark"]["flush_cache"])
        self.assertEqual(
            spec["benchmark"]["baseline_reference_min_measurement_seconds"], 45.0
        )

    def test_preprofile_screen_keeps_only_single_gpu_candidates(self):
        task = self.valid_task()
        task.update({"deployment_mode": "offline_throughput", "slo": {}})
        task["measurement"] = {"confirmation_requests": 32, "min_measurement_seconds": 10}
        spec = {
            "hardware": {"gpus_per_host": 3},
            "execution": {"env": {"CUDA_VISIBLE_DEVICES": "1,2,3"}},
            "benchmark": {},
            "budget": {"max_trials": 3},
            "search": {
                "strategy": "explicit_configurations",
                "baseline": {"tp_size": 1},
                "include_baseline": True,
                "repetitions": 1,
                "explicit_configurations": [
                    {"name": "single", "config": {"tp_size": 1, "chunked_prefill_size": 4096}},
                    {"name": "tp2", "config": {"tp_size": 2}},
                ],
            },
        }
        filtered = autopilot.single_gpu_preprofile_spec(spec, task)
        self.assertIsNotNone(filtered)
        self.assertEqual(
            [item["name"] for item in filtered["search"]["explicit_configurations"]],
            ["single"],
        )
        self.assertEqual(filtered["benchmark"]["baseline_reference_num_prompts"], 32)

    def test_reference_baseline_turns_followup_into_candidate_only_screen(self):
        result = {
            "run_dir": "/tmp/preprofile",
            "results": [{
                "kind": "baseline", "ok": True, "config": {"tp_size": 1}, "env": {},
                "metrics": {"total_throughput_tps": 100.0},
                "confirmation_reference": {"metrics": {"total_throughput_tps": 101.0}},
            }],
        }
        reference = autopilot.measured_reference_baseline(result)
        self.assertEqual(reference["metrics"]["total_throughput_tps"], 101.0)
        spec = {
            "search": {"include_baseline": True},
            "budget": {"max_trials": 4},
            "benchmark": {"baseline_reference_num_prompts": 64},
        }
        autopilot.apply_reference_baseline(spec, reference)
        self.assertFalse(spec["search"]["include_baseline"])
        self.assertEqual(spec["search"]["min_confirm_repetitions"], 1)
        self.assertEqual(spec["budget"]["max_trials"], 3)
        self.assertNotIn("baseline_reference_num_prompts", spec["benchmark"])

    def test_interaction_preserves_environment_and_builds_cumulative_config(self):
        task = self.valid_task()
        task["confirmation_repetitions"] = 2
        baseline = {"kind": "baseline", "config": {"tp_size": 1}, "env": {}}

        def seed(name, config, gain, env=None):
            return {
                "kind": "candidate", "configuration_name": name, "config": config,
                "env": env or {}, "stable": True,
                "all_repetitions_slo_passed": True, "screening_accepted": False,
                "comparison": {"improvement_pct": gain, "secondary_regressions_passed": True},
            }

        screen = {"aggregates": [
            baseline,
            seed(
                "moe", {"tp_size": 1, "moe_runner_backend": "triton"}, 0.8,
                {"SGLANG_MOE_CONFIG_DIR": "/tmp/config"},
            ),
            seed("chunk", {"tp_size": 1, "chunked_prefill_size": 4096}, 0.6),
            seed("graph", {"tp_size": 1, "cuda_graph_max_bs_decode": 8}, 0.4),
        ]}
        spec = autopilot.interaction_spec(
            task,
            {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}},
            screen, remaining_trials=9, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        configurations = spec["search"]["explicit_configurations"]
        self.assertEqual(configurations[0]["env"]["SGLANG_MOE_CONFIG_DIR"], "/tmp/config")
        self.assertIn("chunked_prefill_size", configurations[-1]["config"])
        self.assertIn("cuda_graph_max_bs_decode", configurations[-1]["config"])

    def test_interaction_merges_candidates_with_a_shared_seed(self):
        task = self.valid_task()
        task["confirmation_repetitions"] = 2
        baseline = {"kind": "baseline", "config": {"tp_size": 1}, "env": {}}

        def seed(name, config, gain):
            return {
                "kind": "candidate", "configuration_name": name, "config": config,
                "stable": True, "all_repetitions_slo_passed": True,
                "screening_accepted": False,
                "comparison": {"improvement_pct": gain, "secondary_regressions_passed": True},
            }

        screen = {"aggregates": [
            baseline,
            seed("seed", {"tp_size": 1, "page_size": 64}, 0.7),
            seed("seed-chunk", {"tp_size": 1, "page_size": 64, "chunked_prefill_size": 4096}, 0.5),
        ]}
        spec = autopilot.interaction_spec(
            task,
            {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}},
            screen, remaining_trials=7, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        self.assertIsNone(spec)

    def test_offline_interaction_covers_each_accepted_compatible_combination(self):
        task = self.valid_task()
        task.update({"deployment_mode": "offline_throughput", "slo": {}})
        task["confirmation_repetitions"] = 2
        baseline = {
            "kind": "baseline", "config": {"tp_size": 4}, "env": {},
            "metrics": {"total_throughput_tps": 100.0},
        }

        def accepted(name, config, gain):
            return {
                "kind": "candidate", "configuration_name": name, "config": config,
                "env": {}, "stable": True, "all_repetitions_slo_passed": True,
                "screening_accepted": True,
                "comparison": {"improvement_pct": gain, "secondary_regressions_passed": True},
            }

        screen = {"aggregates": [
            baseline,
            accepted("graph", {"tp_size": 4, "cuda_graph_max_bs_decode": 16}, 6.7),
            accepted("memory", {"tp_size": 4, "mem_fraction_static": 0.839}, 2.8),
            accepted("mixed", {"tp_size": 4, "enable_mixed_chunk": True}, 1.3),
        ]}
        spec = autopilot.interaction_spec(
            task,
            {"derived": {"minimum_tp_size": 4}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}},
            screen, remaining_trials=6, remaining_gpu_hours=1, remaining_wall_minutes=30,
        )
        self.assertIsNotNone(spec)
        self.assertEqual(spec["search"]["candidate_slots"], 5)
        self.assertEqual(spec["search"]["compatible_combinations"], 4)
        self.assertEqual(spec["search"]["budget_omitted_combinations"], 0)
        configs = [item["config"] for item in spec["search"]["explicit_configurations"]]
        self.assertIn(
            {"tp_size": 4, "cuda_graph_max_bs_decode": 16, "mem_fraction_static": 0.839},
            configs,
        )
        self.assertIn(
            {"tp_size": 4, "cuda_graph_max_bs_decode": 16, "enable_mixed_chunk": True},
            configs,
        )
        self.assertIn(
            {
                "tp_size": 4, "cuda_graph_max_bs_decode": 16,
                "mem_fraction_static": 0.839, "enable_mixed_chunk": True,
            },
            configs,
        )
        self.assertFalse(spec["search"]["include_baseline"])
        self.assertEqual(spec["budget"]["max_trials"], 4)
        self.assertFalse(any(
            error.startswith("unsupported search field:")
            for error in autotune.execution_errors(spec)
        ))


class NsysAnalysisTests(unittest.TestCase):
    def test_routing_stats_reuse_existing_sqlite_and_skip_detailed_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "baseline.nsys-rep"
            sqlite = root / "baseline.sqlite"
            report.touch()
            sqlite.touch()
            commands = []

            def fake_run(command, **_kwargs):
                commands.append(command)
                return {
                    "returncode": 0,
                    "stdout": "Time (%),Total Time (ns),Name\n1.0,1,kernel\n",
                    "stderr": "",
                }

            with patch.object(profile_sglang, "run_command", side_effect=fake_run):
                reports, statuses = profile_sglang.collect_stats(report, root)
            self.assertEqual(set(reports), set(profile_sglang.NSYS_ROUTING_REPORTS))
            self.assertEqual(set(statuses), set(profile_sglang.NSYS_ROUTING_REPORTS))
            self.assertTrue(all(command[-1] == str(sqlite) for command in commands))
            self.assertFalse(any("--force-export=true" in command for command in commands))

    def test_routing_summary_diagnosis_does_not_claim_timeline_evidence(self):
        diagnosis = profile_sglang.analyze_reports({
            "cuda_gpu_kern_sum": [{"Time (%)": "90", "Name": "gemm_kernel"}],
            "cuda_api_sum": [],
            "cuda_gpu_mem_time_sum": [],
        })
        self.assertEqual(diagnosis["evidence_quality"], "nsys_routing_summaries")

    def test_csv_parser_skips_nsys_preamble(self):
        rows = profile_sglang.parse_csv(
            "Using existing SQLite export\nTime (%),Total Time (ns),Instances,Name\n60.0,600,2,my_gemm\n"
        )
        self.assertEqual(rows[0]["Name"], "my_gemm")

    def test_diagnosis_routes_low_gpu_activity_to_host_stall(self):
        reports = {
            "cuda_gpu_trace": [
                {"Start (ns)": "0", "Duration (ns)": "10", "Name": "kernel_a"},
                {"Start (ns)": "90", "Duration (ns)": "10", "Name": "kernel_b"},
            ],
            "cuda_gpu_kern_sum": [{"Time (%)": "90", "Name": "gemm_kernel"}],
            "cuda_api_sum": [{"Time (%)": "90", "Name": "cudaEventSynchronize"}],
            "cuda_gpu_mem_time_sum": [],
            "cuda_kern_exec_sum": [],
        }
        diagnosis = profile_sglang.analyze_reports(reports)
        self.assertEqual(diagnosis["primary_bottleneck"], "host_or_scheduler_stall")
        self.assertAlmostEqual(diagnosis["gpu_timeline_active_pct"], 20.0)

    def test_diagnosis_classifies_attention(self):
        reports = {
            "cuda_gpu_trace": [{"Start (ns)": "0", "Duration (ns)": "100", "Name": "attention"}],
            "cuda_gpu_kern_sum": [{"Time (%)": "70", "Name": "flash_fwd_attention"}],
            "cuda_api_sum": [],
            "cuda_gpu_mem_time_sum": [],
            "cuda_kern_exec_sum": [],
        }
        self.assertEqual(
            profile_sglang.analyze_reports(reports)["primary_bottleneck"], "attention"
        )

    def test_flash_attention_cutlass_kernel_is_not_classified_as_gemm(self):
        reports = {
            "cuda_gpu_trace": [{"Start (ns)": "0", "Duration (ns)": "100"}],
            "cuda_gpu_kern_sum": [
                {"Time (%)": "94.8", "Name": "void cutlass::device_kernel<flash::FlashAttnFwdSm90>()"},
                {"Time (%)": "5.2", "Name": "cutlass_gemm_kernel"},
            ],
            "cuda_api_sum": [],
            "cuda_gpu_mem_time_sum": [],
            "cuda_kern_exec_sum": [],
        }
        diagnosis = profile_sglang.analyze_reports(reports)
        self.assertEqual(diagnosis["shares_pct"]["attention_kernels"], 94.8)
        self.assertEqual(diagnosis["shares_pct"]["gemm_kernels"], 5.2)
        self.assertEqual(diagnosis["primary_bottleneck"], "attention")

    def test_effective_server_config_reads_nested_cuda_graph_limits(self):
        config = profile_sglang.effective_server_config({
            "available": True,
            "value": {"server_args": {
                "model_path": "/tmp/model",
                "tp_size": 4,
                "chunked_prefill_size": 8192,
                "cuda_graph_max_bs_decode": None,
                "cuda_graph_config": {
                    "decode": {"max_bs": 512},
                    "prefill": {"max_bs": 32},
                },
            }},
        })
        self.assertEqual(config["cuda_graph_max_bs_decode"], 512)
        self.assertEqual(config["cuda_graph_max_bs_prefill"], 32)

    def test_scheduler_log_extracts_cache_and_graph_evidence(self):
        text = """[2026-08-14 08:17:30] Decode batch, #running-req: 4, #full token: 1051, full token usage: 0.20, mamba num: 16, mamba usage: 0.20, cuda graph: True, gen throughput (token/s): 453.12, #queue-req: 0
[2026-08-14 08:17:30] Prefill batch, #new-seq: 3, #new-token: 256, #cached-token: 576, full token usage: 0.20, mamba usage: 0.20, #running-req: 1, #queue-req: 2, #pending-token: 64, cuda graph: False, input throughput (token/s): 107534.31
Using default MoE kernel config. Performance might be sub-optimal! Config file not found at /tmp/fused_moe_triton/configs/triton_3_4_0/E=128,N=384,device_name=NVIDIA_H800,dtype=fp8_w8a8,block_shape=[128, 128].json, you can create it
Using MoE kernel config with down_moe=False. Config file not found at /tmp/fused_moe_triton/configs/triton_3_4_0/E=128,N=384,device_name=NVIDIA_H800,dtype=fp8_w8a8,block_shape=[128, 128]_down.json, you can create it"""
        summary = sglang_runtime.summarize_sglang_log(text)
        self.assertEqual(summary["decode"]["cuda_graph_coverage_pct"], 100.0)
        self.assertEqual(summary["prefill"]["cached_token_share_pct"], 69.23076923076923)
        self.assertEqual(summary["prefill"]["queue_nonempty_batch_pct"], 100.0)
        self.assertTrue(summary["moe"]["missing_tuned_config"])
        self.assertEqual(summary["moe"]["missing_config_count"], 2)
        self.assertTrue(summary["moe"]["requires_down_kernel_config"])

    def test_scheduler_log_extracts_new_moe_runner_config_path(self):
        text = (
            "[TP0] Using default MoE kernel config. Performance might be sub-optimal! "
            "Config file not found at /opt/sglang/layers/moe/moe_runner/triton_utils/configs/"
            "triton_3_6_0/E=128,N=384,device_name=NVIDIA_H800,dtype=fp8_w8a8,"
            "block_shape=[128, 128].json, you can create them with https://github.com/sgl-project/sglang\n"
            "[TP0] Using MoE kernel config with down_moe=False. Performance might be sub-optimal! "
            "Config file not found at /opt/sglang/layers/moe/moe_runner/triton_utils/configs/"
            "triton_3_6_0/E=128,N=384,device_name=NVIDIA_H800,dtype=fp8_w8a8,"
            "block_shape=[128, 128]_down.json, you can create them"
        )
        summary = sglang_runtime.summarize_sglang_log(text)
        self.assertTrue(summary["moe"]["missing_tuned_config"])
        self.assertEqual(summary["moe"]["missing_config_count"], 2)
        self.assertTrue(summary["moe"]["requires_down_kernel_config"])


class SearchRoutingTests(unittest.TestCase):
    def discovery(self, *, is_moe=True, gpu_count=1, minimum_tp_size=1, architecture="hopper"):
        names = {
            "chunked_prefill_size": ("scheduler", None, None),
            "max_running_requests": ("scheduler", None, None),
            "mem_fraction_static": ("memory_cache", 0.8, None),
            "page_size": ("memory_cache", 1, None),
            "schedule_conservativeness": ("scheduler", 1.0, None),
            "prefill_attention_backend": ("kernel_backend", None, ["fa3", "flashinfer", "triton"]),
            "moe_runner_backend": ("moe", "auto", ["auto", "deep_gemm", "flashinfer_trtllm", "triton"]),
            "ep_size": ("moe", 1, None),
            "num_continuous_decode_steps": ("scheduler", 1, None),
            "scheduler_recv_interval": ("cpu_frontend", 1, None),
            "tokenizer_worker_num": ("cpu_frontend", 1, None),
            "cuda_graph_max_bs_decode": ("cuda_graph", None, None),
            "disable_radix_cache": ("memory_cache", False, None),
            "enable_mixed_chunk": ("scheduler", False, None),
            "enable_mscclpp": ("communication", False, None),
            "disable_custom_all_reduce": ("communication", False, None),
            "enable_dp_attention": ("parallelism", False, None),
            "mamba_radix_cache_strategy": ("hybrid_mamba", "auto", ["auto", "no_buffer", "extra_buffer"]),
        }
        parameters = [
            {
                "dest": name,
                "family": family,
                "default": default,
                "choices": choices,
                "deprecated": False,
                "primary_flag": "--" + name.replace("_", "-"),
                "help": name,
            }
            for name, (family, default, choices) in names.items()
        ]
        return {
            "hardware": {"vendor": "nvidia"},
            "hardware_profile": {"architecture": architecture},
            "topology_class": "nvlink-or-nvswitch" if gpu_count > 1 else "single-gpu",
            "model": {"is_moe": is_moe},
            "derived": {
                "visible_gpu_count": gpu_count,
                "minimum_tp_size": minimum_tp_size,
                "typical_prefill_batch_tokens": 1024,
            },
            "parameter_catalog": {"parameters": parameters},
        }

    def task(self):
        return {"workload": {
            "input_tokens": 256, "output_tokens": 64, "max_concurrency": 4,
            "num_prompts": 100, "prefix_reuse_ratio": 0,
        }}

    def routed(self, primary, shares=None, **kwargs):
        profile = {"diagnosis": {"primary_bottleneck": primary, "shares_pct": shares or {}}}
        task = self.task()
        task["search_depth"] = "evidence_guided"
        return autopilot.diagnosed_search_plan(task, self.discovery(**kwargs), profile)

    def test_attention_routes_attention_backend(self):
        plan = self.routed("attention", {"attention_kernels": 60})
        self.assertEqual(plan["ranked_parameter_groups"][0]["parameter"], "prefill_attention_backend")

    def test_attention_backend_search_excludes_resolved_active_backend(self):
        task = self.task()
        task["search_depth"] = "evidence_guided"
        profile = {
            "diagnosis": {"primary_bottleneck": "attention", "shares_pct": {"attention_kernels": 94.8}},
            "effective_server_config": {"attention_backend": "fa3"},
        }
        plan = autopilot.diagnosed_search_plan(task, self.discovery(), profile)
        attention = next(
            item for item in plan["ranked_parameter_groups"]
            if item["parameter"] == "prefill_attention_backend"
        )
        self.assertEqual(attention["values"], ["flashinfer", "triton"])
        self.assertIn("resolved_active_attention_backend=fa3", attention["evidence"])

    def test_distorted_profile_does_not_route_host_timing_parameters(self):
        task = self.task()
        task["search_depth"] = "evidence_guided"
        profile = {
            "diagnosis": {
                "primary_bottleneck": "host_or_scheduler_stall",
                "secondary_bottlenecks": ["cuda_synchronization"],
                "profiling_run_performance_comparable": False,
                "shares_pct": {"attention_kernels": 94.8},
            }
        }
        plan = autopilot.diagnosed_search_plan(task, self.discovery(), profile)
        names = [item["parameter"] for item in plan["ranked_parameter_groups"]]
        self.assertIn("prefill_attention_backend", names)
        self.assertNotIn("scheduler_recv_interval", names)
        self.assertNotIn("num_continuous_decode_steps", names)
        self.assertFalse(plan["profile_timing_comparable"])

    def test_moe_routes_moe_runner(self):
        plan = self.routed("moe_compute", {"moe_kernels": 55})
        self.assertEqual(plan["ranked_parameter_groups"][0]["parameter"], "moe_runner_backend")

    def test_missing_moe_config_routes_runner_without_aggregate_moe_hotspot(self):
        task = self.task()
        task["search_depth"] = "evidence_guided"
        profile = {
            "diagnosis": {"primary_bottleneck": "attention", "shares_pct": {"attention_kernels": 94.8}},
            "runtime_observations": {"moe": {
                "missing_tuned_config": True,
                "missing_config_count": 2,
            }},
        }
        plan = autopilot.diagnosed_search_plan(task, self.discovery(), profile)
        names = [item["parameter"] for item in plan["ranked_parameter_groups"]]
        self.assertIn("moe_runner_backend", names)
        self.assertTrue(plan["runtime_moe_config_missing"])

    def test_multi_gpu_moe_routes_dp_attention(self):
        plan = self.routed("moe_compute", {"moe_kernels": 55}, gpu_count=8, minimum_tp_size=8)
        names = [item["parameter"] for item in plan["ranked_parameter_groups"]]
        self.assertIn("enable_dp_attention", names)

    def test_qwen3_fp8_routes_only_legal_expert_parallel_sizes(self):
        task = self.task()
        task["search_depth"] = "evidence_guided"
        task["workload"]["max_concurrency"] = 16
        discovery = self.discovery(gpu_count=4, minimum_tp_size=4)
        discovery["model"].update({
            "model_type": "qwen3_moe",
            "moe_intermediate_size": 1536,
            "weight_block_size": [128, 128],
        })
        profile = {
            "diagnosis": {"primary_bottleneck": "moe_compute", "shares_pct": {"moe_kernels": 55}},
            "effective_server_config": {"tp_size": 4},
        }
        self.assertEqual(autopilot.supported_ep_sizes(discovery), [2, 4])
        plan = autopilot.diagnosed_search_plan(task, discovery, profile)
        ep = next(item for item in plan["ranked_parameter_groups"] if item["parameter"] == "ep_size")
        self.assertEqual(ep["values"], [2, 4])
        self.assertIn("ep_size", autopilot.core_serving_parameter_order(task, discovery, plan))

    def test_long_context_offline_orders_prefill_controls_before_topology(self):
        task = self.task()
        task["deployment_mode"] = "offline_throughput"
        task["workload"].update({"input_tokens": 16384, "output_tokens": 256})
        discovery = self.discovery(gpu_count=4, minimum_tp_size=4)
        plan = {
            "runtime_moe_config_missing": True,
            "profiler_evidence": {"primary_bottleneck": "attention", "shares_pct": {"attention_kernels": 94}},
            "ranked_parameter_groups": [
                {"parameter": "chunked_prefill_size", "family": "scheduler", "values": [12288]},
                {"parameter": "enable_mixed_chunk", "family": "scheduler", "values": [True]},
                {"parameter": "prefill_attention_backend", "family": "kernel_backend", "values": ["flashinfer"]},
                {"parameter": "ep_size", "family": "moe", "values": [2, 4]},
            ],
        }
        order = autopilot.core_serving_parameter_order(task, discovery, plan)
        self.assertLess(order.index("chunked_prefill_size"), order.index("prefill_attention_backend"))
        self.assertLess(order.index("enable_mixed_chunk"), order.index("ep_size"))

    def test_screening_keeps_all_legal_ep_degrees_ahead_of_sensitivity_values(self):
        task = self.task()
        task.update({
            "confirmation_repetitions": 2,
            "budget": {"max_trials": 12, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {}, "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp", "name": "test", "output_dir": "/tmp/runs",
        })
        discovery = self.discovery(gpu_count=4, minimum_tp_size=4)
        plan = {
            "runtime_moe_config_missing": True,
            "profiler_evidence": {"primary_bottleneck": "moe_compute", "shares_pct": {"moe_kernels": 55}},
            "ranked_parameter_groups": [
                {"parameter": "ep_size", "family": "moe", "values": [2, 4]},
                {"parameter": "num_continuous_decode_steps", "family": "scheduler", "values": [2, 4]},
            ],
        }
        spec = autopilot.screening_spec(
            task, discovery, plan, remaining_trials=8, confirmation_reserve_trials=0,
        )
        names = [item["name"] for item in spec["search"]["explicit_configurations"]]
        self.assertIn("ep_size-2", names)
        self.assertIn("ep_size-4", names)

    def test_confirmation_pool_keeps_stronger_atomic_candidate_over_weaker_combination(self):
        baseline = {"kind": "baseline", "config": {"tp_size": 4}}
        atomic = {
            "kind": "candidate", "configuration_name": "chunk", "config": {"tp_size": 4, "chunked_prefill_size": 4096},
            "screening_accepted": True, "comparison": {"improvement_pct": 6.0},
        }
        combined = {
            "kind": "candidate", "configuration_name": "chunk-mem", "config": {"tp_size": 4, "chunked_prefill_size": 4096, "mem_fraction_static": 0.839},
            "screening_accepted": True, "comparison": {"improvement_pct": 3.2},
        }
        pool = autopilot.confirmation_candidate_pool(
            {"aggregates": [baseline, atomic], "screening_winner": atomic},
            {"aggregates": [baseline, combined], "screening_winner": combined},
        )
        self.assertEqual(pool["screening_winner"]["configuration_name"], "chunk")

    def test_ep_requires_published_fp8_shape_information(self):
        discovery = self.discovery(gpu_count=4, minimum_tp_size=4)
        discovery["model"].update({"moe_intermediate_size": 1536})
        self.assertEqual(autopilot.supported_ep_sizes(discovery), [])

    def test_communication_routes_collectives_only_on_multi_gpu(self):
        plan = self.routed("communication", {"communication_kernels": 40}, gpu_count=8)
        names = [item["parameter"] for item in plan["ranked_parameter_groups"]]
        self.assertIn("enable_mscclpp", names)
        self.assertIn("disable_custom_all_reduce", names)

    def test_host_stall_routes_scheduler_controls(self):
        plan = self.routed("host_or_scheduler_stall")
        names = [item["parameter"] for item in plan["ranked_parameter_groups"][:2]]
        self.assertEqual(names, ["num_continuous_decode_steps", "scheduler_recv_interval"])

    def test_chunk_search_uses_resolved_default_and_workload_boundary(self):
        profile = {
            "diagnosis": {"primary_bottleneck": "host_or_scheduler_stall", "shares_pct": {}},
            "effective_server_config": {"chunked_prefill_size": 8192},
        }
        task = self.task()
        task["search_depth"] = "evidence_guided"
        plan = autopilot.diagnosed_search_plan(task, self.discovery(), profile)
        chunk = next(item for item in plan["ranked_parameter_groups"] if item["parameter"] == "chunked_prefill_size")
        self.assertEqual(chunk["values"], [4096, 2048, 1024, 512])
        self.assertIn("resolved_sglang_default=8192", chunk["evidence"])
        self.assertEqual(
            plan["chunked_prefill_strategy"]["strategy"],
            "throughput_amortization_first",
        )

    def test_long_context_chunk_search_includes_whole_request_anchor(self):
        task = self.task()
        task["workload"].update({
            "input_tokens": 16384,
            "max_concurrency": 16,
            "prefix_reuse_ratio": 0.75,
        })
        self.assertIn(16384, autopilot.chunk_candidates(task, framework_default=8192))

    def test_profile_spec_bounds_shared_prefix_requests_and_warmup(self):
        task = {
            "workload": {
                "input_tokens": 16384,
                "output_tokens": 256,
                "max_concurrency": 64,
                "num_prompts": 2048,
                "prefix_reuse_ratio": 0.75,
                "shared_prefix": {
                    "groups": 8,
                    "prompts_per_group": 256,
                    "system_prompt_tokens": 12288,
                    "question_tokens": 4096,
                },
            },
            "measurement": {
                "warmup_requests": 256,
                "min_measurement_requests": 2048,
                "min_measurement_seconds": 45,
            },
            "budget": {"max_trials": 10, "max_gpu_hours": 3, "max_wall_time_minutes": 360},
            "slo": {},
            "objective": {"metric": "total_throughput_tps", "direction": "maximize"},
            "name": "profile-bound", "repository": "/tmp", "python": sys.executable,
            "model_path": "/tmp", "output_dir": "/tmp/runs",
        }
        discovery = self.discovery(gpu_count=4, minimum_tp_size=4)
        spec = autopilot.profile_spec(task, discovery)
        benchmark = spec["benchmark"]
        self.assertEqual(benchmark["num_prompts"], 128)
        self.assertEqual(benchmark["gsp_prompts_per_group"], 16)
        self.assertEqual(benchmark["warmup_requests"], 128)
        self.assertEqual(benchmark["min_measurement_seconds"], 20.0)

    def test_chunk_search_prioritizes_uncached_suffix_under_latency_pressure(self):
        profile = {
            "diagnosis": {"primary_bottleneck": "attention", "shares_pct": {}},
            "effective_server_config": {"chunked_prefill_size": 8192},
            "runtime_observations": {"prefill": {"queue_nonempty_batch_pct": 20.0}},
        }
        task = self.task()
        task.update({
            "deployment_mode": "online_latency",
            "slo": {"p99_ttft_ms": 1000},
        })
        task["workload"].update({
            "input_tokens": 16384,
            "max_concurrency": 8,
            "prefix_reuse_ratio": 0.75,
        })
        discovery = self.discovery()
        discovery["derived"]["typical_prefill_batch_tokens"] = 32768
        plan = autopilot.diagnosed_search_plan(task, discovery, profile)
        chunk = next(
            item for item in plan["ranked_parameter_groups"]
            if item["parameter"] == "chunked_prefill_size"
        )
        self.assertEqual(chunk["values"][0], 4096)
        self.assertEqual(plan["chunked_prefill_strategy"]["strategy"], "latency_interleaving_first")

    def test_screening_balances_parameter_families(self):
        task = self.task()
        task.update({
            "confirmation_repetitions": 2,
            "budget": {"max_trials": 10, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {"p99_ttft_ms": 1000},
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp", "name": "test", "output_dir": "/tmp/runs",
        })
        search_plan = {
            "ranked_parameter_groups": [
                {"parameter": "num_continuous_decode_steps", "family": "scheduler", "values": [2, 4]},
                {"parameter": "moe_runner_backend", "family": "moe", "values": ["deep_gemm"]},
                {"parameter": "disable_radix_cache", "family": "memory_cache", "values": [True]},
            ]
        }
        spec = autopilot.screening_spec(task, self.discovery(), search_plan, remaining_trials=9)
        self.assertEqual(spec["search"]["strategy"], "explicit_configurations")
        self.assertEqual(
            [item["name"] for item in spec["search"]["explicit_configurations"]],
            ["num_continuous_decode_steps-2", "moe_runner_backend-deep_gemm", "disable_radix_cache-true"],
        )

    def test_screening_skips_values_equal_to_resolved_runtime_defaults(self):
        task = self.task()
        task.update({
            "confirmation_repetitions": 2,
            "budget": {"max_trials": 12, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {}, "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp", "name": "test", "output_dir": "/tmp/runs",
        })
        plan = {
            "resolved_baseline": {"max_running_requests": 16, "cuda_graph_max_bs_decode": 8},
            "ranked_parameter_groups": [
                {"parameter": "max_running_requests", "family": "scheduler", "values": [16, 32]},
                {"parameter": "cuda_graph_max_bs_decode", "family": "cuda_graph", "values": [8]},
            ],
        }
        spec = autopilot.screening_spec(
            task, self.discovery(), plan, remaining_trials=6, confirmation_reserve_trials=0,
        )
        names = [item["name"] for item in spec["search"]["explicit_configurations"]]
        self.assertIn("max_running_requests-32", names)
        self.assertNotIn("max_running_requests-16", names)
        self.assertNotIn("cuda_graph_max_bs_decode-8", names)

    def test_screening_keeps_raw_baseline_when_retesting_preprofile_seed(self):
        task = self.task()
        task.update({
            "confirmation_repetitions": 2,
            "budget": {"max_trials": 8, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {}, "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp", "name": "test", "output_dir": "/tmp/runs",
        })
        spec = autopilot.screening_spec(
            task, self.discovery(),
            {"ranked_parameter_groups": []},
            baseline={"tp_size": 1},
            anchor={"tp_size": 1, "page_size": 64},
            remaining_trials=4,
            confirmation_reserve_trials=0,
        )
        self.assertEqual(spec["search"]["baseline"], {"tp_size": 1})
        self.assertEqual(
            spec["search"]["explicit_configurations"][0],
            {"name": "preprofile-seed", "config": {"tp_size": 1, "page_size": 64}},
        )

    def test_balanced_screen_reserves_an_interaction_trial(self):
        task = self.task()
        task.update({
            "experiment_mode": "balanced", "search_depth": "evidence_guided",
            "confirmation_repetitions": 2,
            "budget": {"max_trials": 12, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {}, "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp", "name": "test", "output_dir": "/tmp/runs",
        })
        plan = {
            "ranked_parameter_groups": [
                {"parameter": "max_running_requests", "family": "scheduler", "values": [16]},
                {"parameter": "cuda_graph_max_bs_decode", "family": "cuda_graph", "values": [8]},
            ],
        }
        spec = autopilot.screening_spec(task, self.discovery(), plan, remaining_trials=12)
        # The screen needs only its baseline plus two atomic probes. The unused
        # four trials remain available for a composition and confirmation.
        self.assertEqual(spec["budget"]["max_trials"], 3)

    def test_moe_config_environment_is_candidate_only_not_global(self):
        task = self.task()
        task.update({
            "confirmation_repetitions": 2,
            "budget": {"max_trials": 8, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {},
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp/model",
            "name": "test", "output_dir": "/tmp/runs", "env": {"CUDA_VISIBLE_DEVICES": "0"},
        })
        plan = {
            "ranked_parameter_groups": [],
            "ranked_configuration_bundles": [{
                "name": "fused-moe-autotuned-config",
                "config": {"tp_size": 1},
                "env": {"SGLANG_MOE_CONFIG_DIR": "/tmp/run/moe-config"},
                "priority": "high",
            }],
        }
        spec = autopilot.screening_spec(task, self.discovery(), plan, remaining_trials=8)
        self.assertNotIn("SGLANG_MOE_CONFIG_DIR", spec["execution"]["env"])
        candidate = spec["search"]["explicit_configurations"][0]
        self.assertEqual(candidate["env"]["SGLANG_MOE_CONFIG_DIR"], "/tmp/run/moe-config")

    def test_qwen35_uses_its_own_nextn_cookbook_profile(self):
        model = {"architectures": ["Qwen3_5ForConditionalGeneration"], "has_mtp_weights": True}
        self.assertEqual(
            autopilot.inferred_cookbook_url(model),
            "https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5",
        )
        evidence = autopilot.cookbook_evidence({"allow_download": False}, model)
        profile = evidence["model_profile"]
        self.assertEqual(profile["name"], "qwen3.5-hybrid-mtp")
        mtp = next(bundle for bundle in profile["initial_bundles"] if "mtp-nextn" in bundle["name"])
        self.assertEqual(mtp["config"]["speculative_algorithm"], "NEXTN")

    def test_qwen3_moe_uses_qwen3_cookbook_without_inventing_mtp(self):
        model = {
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "has_mtp_weights": False,
        }
        self.assertEqual(
            autopilot.inferred_cookbook_url(model),
            "https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3",
        )
        evidence = autopilot.cookbook_evidence({"allow_download": False}, model)
        profile = evidence["model_profile"]
        self.assertEqual(profile["name"], "qwen3-moe")
        self.assertEqual(profile["initial_bundles"], [])
        self.assertEqual(profile["speculative_policy"], "requires_explicit_compatible_draft_model")

    def test_qwen3_cookbook_identity_ignores_qwen35_navigation_text(self):
        model = {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe"}
        with patch.object(autopilot, "fetch_reference", return_value={
            "url": "https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3",
            "status": "fetched",
            "text": "Navigation: Qwen3.5 Qwen3.6. Qwen3 MoE expert parallelism.",
        }):
            evidence = autopilot.cookbook_evidence({"allow_download": True}, model)
        self.assertEqual(evidence["model_profile"]["name"], "qwen3-moe")

    def test_cookbook_snapshot_records_matching_markdown_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            checkout = Path(root)
            (checkout / ".git").mkdir()
            recipe = checkout / "qwen35.md"
            recipe.write_text("# Qwen3.5\n--speculative-algorithm NEXTN\n", encoding="utf-8")
            evidence = autopilot.cookbook_snapshot_evidence(
                checkout, {"architectures": ["Qwen3_5ForConditionalGeneration"]},
            )
        self.assertEqual(evidence["status"], "available")
        self.assertEqual(evidence["matched_markdown"][0]["path"], "qwen35.md")
        self.assertEqual(len(evidence["matched_markdown"][0]["sha256"]), 64)

    def test_supported_tp_sizes_require_attention_and_kv_head_divisibility(self):
        discovery = self.discovery(gpu_count=4)
        discovery["model"].update({"num_attention_heads": 24, "num_key_value_heads": 4})
        self.assertEqual(autopilot.supported_tp_sizes(discovery), [1, 2, 4])
        discovery["model"]["num_key_value_heads"] = 3
        self.assertEqual(autopilot.supported_tp_sizes(discovery), [1])

    def test_screening_advances_tensor_parallelism_from_preprofile_winner(self):
        task = self.task()
        task.update({
            "confirmation_repetitions": 2,
            "budget": {"max_trials": 12, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {"p99_ttft_ms": 1000},
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp", "name": "test", "output_dir": "/tmp/runs",
        })
        discovery = self.discovery(gpu_count=4)
        discovery["model"].update({"num_attention_heads": 24, "num_key_value_heads": 4})
        discovery["parameter_catalog"]["parameters"].append({
            "dest": "tp_size", "family": "parallelism", "default": 1, "choices": None,
            "deprecated": False, "primary_flag": "--tp-size", "help": "tp size",
        })
        spec = autopilot.screening_spec(
            task, discovery,
            {"ranked_parameter_groups": [{"parameter": "tp_size", "family": "parallelism", "values": [2, 4]}]},
            baseline={"tp_size": 2}, remaining_trials=8,
        )
        names = [item["name"] for item in spec["search"]["explicit_configurations"]]
        self.assertIn("tp_size-4", names)
        self.assertNotIn("tp_size-2", names)

    def test_initial_screen_covers_all_tp_candidates_then_model_capability(self):
        task = self.task()
        task.update({
            "confirmation_repetitions": 2,
            "budget": {"max_trials": 12, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {"p99_ttft_ms": 1000},
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp", "name": "test", "output_dir": "/tmp/runs",
        })
        discovery = self.discovery(gpu_count=4)
        discovery["model"].update({"num_attention_heads": 24, "num_key_value_heads": 4})
        discovery["parameter_catalog"]["parameters"].append({
            "dest": "tp_size", "family": "parallelism", "default": 1, "choices": None,
            "deprecated": False, "primary_flag": "--tp-size", "help": "tp size",
        })
        spec = autopilot.screening_spec(
            task, discovery,
            {
                "phase": "cookbook_initialization",
                "ranked_parameter_groups": [{"parameter": "tp_size", "family": "parallelism", "values": [2, 4]}],
                "cookbook_candidate_bundles": [
                    {"name": "cache", "config": {"page_size": 64}},
                    {"name": "mtp", "config": {"speculative_algorithm": "NEXTN"}},
                ],
            },
            remaining_trials=4, confirmation_reserve_trials=0,
        )
        names = [item["name"] for item in spec["search"]["explicit_configurations"]]
        self.assertEqual(names, ["tp_size-2", "tp_size-4", "mtp"])

    def test_core_serving_controls_precede_dependent_bundles(self):
        task = self.task()
        task["workload"].update({"input_tokens": 4096, "prefix_reuse_ratio": 0.5})
        task.update({
            "confirmation_repetitions": 2,
            "budget": {"max_trials": 20, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {"p99_ttft_ms": 1000},
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp", "name": "test", "output_dir": "/tmp/runs",
        })
        ranked = [
            {"parameter": name, "family": family, "values": [value]}
            for name, family, value in [
                ("mem_fraction_static", "memory_cache", 0.77),
                ("max_running_requests", "scheduler", 16),
                ("cuda_graph_max_bs_decode", "cuda_graph", 8),
                ("moe_runner_backend", "moe", "triton"),
                ("page_size", "memory_cache", 16),
                ("scheduler_recv_interval", "cpu_frontend", 2),
                ("num_continuous_decode_steps", "scheduler", 2),
                ("schedule_policy", "scheduler", "lpm"),
                ("enable_mixed_chunk", "scheduler", True),
                ("chunked_prefill_size", "scheduler", 16384),
            ]
        ]
        search_plan = {
            "profiler_evidence": {
                "primary_bottleneck": "host_or_scheduler_stall",
                "gpu_timeline_active_pct": 45,
                "shares_pct": {"moe_kernels": 35},
            },
            "ranked_parameter_groups": ranked,
            "ranked_configuration_bundles": [{"name": "dependent", "config": {"page_size": 64}}],
        }
        spec = autopilot.screening_spec(task, self.discovery(), search_plan, remaining_trials=16)
        names = [item["name"] for item in spec["search"]["explicit_configurations"]]
        self.assertEqual(names, [
            "chunked_prefill_size-16384", "enable_mixed_chunk-true",
            "cuda_graph_max_bs_decode-8", "schedule_policy-lpm",
            "num_continuous_decode_steps-2", "page_size-16",
            "moe_runner_backend-triton", "mem_fraction_static-0.77",
        ])
        self.assertEqual(spec["search"]["candidate_limit"], 9)
        self.assertNotIn("max_running_requests-16", names)
        self.assertNotIn("dependent", names)

    def test_balanced_screen_refines_high_impact_values_after_six_mechanisms(self):
        task = self.task()
        task["workload"].update({"input_tokens": 16384, "output_tokens": 256})
        task.update({
            "experiment_mode": "balanced",
            "confirmation_repetitions": 1,
            "budget": {"max_trials": 20, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {},
            "objective": {
                "metric": "total_throughput_tps", "direction": "maximize",
                "min_improvement_pct": 1,
            },
            "deployment_mode": "offline_throughput",
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp",
            "name": "test", "output_dir": "/tmp/runs",
        })
        ranked = [
            {"parameter": "chunked_prefill_size", "family": "scheduler", "values": [4096, 8192, 16384]},
            {"parameter": "enable_mixed_chunk", "family": "scheduler", "values": [True]},
            {"parameter": "cuda_graph_max_bs_decode", "family": "cuda_graph", "values": [8, 16]},
            {"parameter": "page_size", "family": "memory_cache", "values": [16, 32]},
            {"parameter": "schedule_conservativeness", "family": "scheduler", "values": [0.3, 0.6]},
            {"parameter": "num_continuous_decode_steps", "family": "scheduler", "values": [2, 4]},
            {"parameter": "moe_runner_backend", "family": "moe", "values": ["triton"]},
            {"parameter": "mem_fraction_static", "family": "memory_cache", "values": [0.77]},
        ]
        spec = autopilot.screening_spec(
            task,
            self.discovery(),
            {"ranked_parameter_groups": ranked},
            remaining_trials=14,
        )
        names = [item["name"] for item in spec["search"]["explicit_configurations"]]
        self.assertEqual(len(names), 9)
        self.assertIn("chunked_prefill_size-4096", names[:6])
        self.assertIn("chunked_prefill_size-8192", names[6:])
        self.assertIn("chunked_prefill_size-16384", names[6:])
        self.assertNotIn("moe_runner_backend-triton", names)

    def test_post_profile_screen_skips_identical_preprofile_topology(self):
        task = self.task()
        task.update({
            "confirmation_repetitions": 2,
            "budget": {"max_trials": 12, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp",
            "name": "test", "output_dir": "/tmp/runs",
            "slo": {},
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
        })
        discovery = self.discovery(is_moe=False)
        discovery["derived"].update({"visible_gpu_count": 4, "minimum_tp_size": 1})
        discovery["model"].update({"num_attention_heads": 32, "num_key_value_heads": 8})
        discovery["parameter_catalog"]["parameters"].append({
            "dest": "tp_size", "family": "parallelism", "default": 1,
            "choices": None, "deprecated": False, "primary_flag": "--tp-size", "help": "tp size",
        })
        plan = {
            "ranked_parameter_groups": [
                {"parameter": "tp_size", "family": "parallelism", "values": [2, 4]},
                {"parameter": "num_continuous_decode_steps", "family": "scheduler", "values": [2]},
            ],
            "previously_evaluated_configurations": [{"tp_size": 2}, {"tp_size": 4}],
        }
        spec = autopilot.screening_spec(
            task, discovery, plan, remaining_trials=8,
            baseline={"tp_size": 4}, confirmation_reserve_trials=0,
        )
        names = [item["name"] for item in spec["search"]["explicit_configurations"]]
        self.assertNotIn("tp_size-2", names)
        self.assertIn("num_continuous_decode_steps-2", names)

    def test_cookbook_budget_preserves_post_profile_parameter_trials(self):
        task = self.task()
        task.update({
            "confirmation_repetitions": 2,
            "budget": {"max_trials": 14, "max_gpu_hours": 1, "max_wall_time_minutes": 90},
        })
        self.assertEqual(autopilot.initial_cookbook_trial_budget(task, 3), 2)

    def test_active_decode_graph_skips_graph_tuning(self):
        task = self.task()
        task["search_depth"] = "evidence_guided"
        profile = {
            "diagnosis": {"primary_bottleneck": "host_or_scheduler_stall", "shares_pct": {}},
            "prometheus": {"selected_samples": ['sglang:cuda_graph_passes_total{mode="decode_cuda_graph"} 42']},
        }
        plan = autopilot.diagnosed_search_plan(task, self.discovery(), profile)
        names = [item["parameter"] for item in plan["ranked_parameter_groups"]]
        self.assertNotIn("cuda_graph_max_bs_decode", names)

    def test_scheduler_log_graph_coverage_skips_graph_tuning(self):
        task = self.task()
        task["search_depth"] = "evidence_guided"
        profile = {
            "diagnosis": {"primary_bottleneck": "host_or_scheduler_stall", "shares_pct": {}},
            "runtime_observations": {"decode": {"cuda_graph_coverage_pct": 100.0}},
        }
        plan = autopilot.diagnosed_search_plan(task, self.discovery(), profile)
        names = [item["parameter"] for item in plan["ranked_parameter_groups"]]
        self.assertNotIn("cuda_graph_max_bs_decode", names)

    def test_oversized_graph_limit_is_tuned_even_when_runtime_coverage_is_complete(self):
        task = self.task()
        task["search_depth"] = "evidence_guided"
        profile = {
            "diagnosis": {"primary_bottleneck": "attention", "shares_pct": {"attention_kernels": 94.8}},
            "runtime_observations": {"decode": {"cuda_graph_coverage_pct": 100.0}},
            "effective_server_config": {"cuda_graph_max_bs_decode": 512},
        }
        plan = autopilot.diagnosed_search_plan(task, self.discovery(), profile)
        graph = next(
            item for item in plan["ranked_parameter_groups"]
            if item["parameter"] == "cuda_graph_max_bs_decode"
        )
        self.assertEqual(graph["values"], [4, 8])
        self.assertIn("resolved_cuda_graph_max_bs_decode=512", graph["evidence"])

    def test_moe_tuning_plan_uses_observed_shapes_and_fast_mode_defers_cold_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            tuner = repo / "benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py"
            tuner.parent.mkdir(parents=True)
            tuner.write_text("# test tuner\n", encoding="utf-8")
            task = {
                **self.task(),
                "repository": str(repo),
                "python": sys.executable,
                "model_path": "/tmp/model",
                "experiment_mode": "fast",
                "kernel_tuning": {"mode": "auto"},
            }
            discovery = self.discovery(gpu_count=4, minimum_tp_size=4)
            discovery["model"]["weight_quantization"] = "fp8"
            profile = {
                "diagnosis": {"shares_pct": {"attention_kernels": 94.8, "moe_kernels": 0.0}},
                "effective_server_config": {"tp_size": 4, "ep_size": 1},
                "runtime_observations": {
                    "moe": {"missing_tuned_config": True, "missing_config_files": ["E=128.json"]},
                    "decode": {"running_requests": {"p50": 4, "p95": 8}},
                    "prefill": {"new_tokens": {"p50": 4096, "p95": 8192}},
                },
            }
            plan = autopilot.moe_kernel_optimization_plan(task, discovery, profile)
            self.assertEqual(plan["shape_matched_batch_sizes"], [4, 8, 4096, 8192])
            self.assertEqual(plan["priority"], "high")
            self.assertIn("fp8_w8a8", plan["tuner_commands"][0])
            result = autopilot.execute_moe_kernel_tuning(task, plan, repo / "output")
            self.assertEqual(result["status"], "deferred")

    def test_moe_down_config_requires_separate_topk_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            standard = repo / "benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py"
            separate = repo / "benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py"
            standard.parent.mkdir(parents=True)
            standard.write_text("# standard tuner\n", encoding="utf-8")
            separate.write_text("# separate tuner\n", encoding="utf-8")
            task = {
                **self.task(),
                "repository": str(repo),
                "python": sys.executable,
                "model_path": "/tmp/model",
                "kernel_tuning": {"mode": "execute"},
            }
            discovery = self.discovery(gpu_count=4, minimum_tp_size=4)
            discovery["model"]["weight_quantization"] = "fp8"
            profile = {
                "diagnosis": {"shares_pct": {"moe_kernels": 40.0}},
                "effective_server_config": {"tp_size": 4, "ep_size": 1},
                "runtime_observations": {
                    "moe": {
                        "missing_tuned_config": True,
                        "requires_down_kernel_config": True,
                        "missing_config_files": ["E=128,N=256,device_name=NVIDIA_H800,dtype=fp8_w8a8_down.json"],
                    }
                },
            }
            plan = autopilot.moe_kernel_optimization_plan(task, discovery, profile)
            self.assertEqual(plan["tuning_mode"], "separate_up_down")
            self.assertEqual(plan["tuner_commands"], [])
            result = autopilot.execute_moe_kernel_tuning(task, plan, repo / "output")
            self.assertEqual(result["status"], "blocked")
            self.assertIn("topk_ids_dir", result["reason"])

    def test_moe_config_artifact_matches_loader_filename_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / (
                "E=128,N=256,device_name=NVIDIA_H800,dtype=fp8_w8a8,"
                "block_shape=[128, 128]_down.json"
            )
            path.write_text(json.dumps({"1": {
                "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2,
            }}), encoding="utf-8")
            self.assertEqual(autopilot.validate_moe_config_artifact(path), (True, None))
            bad = path.with_name("not-a-sglang-config.json")
            bad.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            self.assertFalse(autopilot.validate_moe_config_artifact(bad)[0])

    def test_moe_tuner_local_ray_compat_is_isolated_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tuner = root / "tuner.py"
            tuner.write_text(
                "import ray\n"
                "from ray.experimental.tqdm_ray import tqdm\n"
                "@ray.remote(num_gpus=1)\n"
                "class Worker: pass\n"
                "ray.init(); ray.get([]); ray.get_gpu_ids(); ray.available_resources()\n",
                encoding="utf-8",
            )
            result = autopilot.prepare_local_ray_compat(tuner, root / "artifacts")
            self.assertEqual(result["status"], "ready")
            compat_root = Path(result["pythonpath"])
            self.assertTrue((compat_root / "ray/__init__.py").is_file())
            self.assertTrue((compat_root / "ray/experimental/tqdm_ray.py").is_file())

            tuner.write_text("import ray\nray.cluster_resources()\n", encoding="utf-8")
            unsupported = autopilot.prepare_local_ray_compat(tuner, root / "unsupported")
            self.assertEqual(unsupported["status"], "unsupported")
            self.assertEqual(unsupported["unsupported_ray_apis"], ["cluster_resources"])

    def test_prefill_queue_promotes_mixed_chunk_in_evidence_guided_mode(self):
        task = self.task()
        task["workload"]["input_tokens"] = 4096
        task["search_depth"] = "evidence_guided"
        profile = {
            "diagnosis": {"primary_bottleneck": "host_or_scheduler_stall", "shares_pct": {}},
            "runtime_observations": {"prefill": {"queue_nonempty_batch_pct": 26.0}},
            "effective_server_config": {"chunked_prefill_size": 8192},
        }
        plan = autopilot.diagnosed_search_plan(task, self.discovery(is_moe=False), profile)
        mixed = next(item for item in plan["ranked_parameter_groups"] if item["parameter"] == "enable_mixed_chunk")
        self.assertEqual(mixed["values"], [True])
        self.assertIn("workload_trace_coverage", mixed["tiers"])

    def test_operator_escalation_uses_timeline_aware_amdahl_bound(self):
        plan = autopilot.operator_escalation_plan({
            "tool": {"ncu": {"available": True, "performance_counter_access": False}},
            "diagnosis": {
                "gpu_timeline_active_pct": 20,
                "top_kernels": [{"name": "fused_moe", "time_pct": 50}],
            },
        })
        self.assertTrue(plan["required"])
        self.assertAlmostEqual(plan["two_x_kernel_speedup_gpu_execution_upper_bound_pct"], 33.333, places=3)
        self.assertIsNone(plan["end_to_end_upper_bound_pct"])

    def test_underdriven_workload_skips_scheduler_and_admission_tuning(self):
        task = self.task()
        task["search_depth"] = "evidence_guided"
        task["workload"]["request_rate"] = 1.0
        profile = {
            "diagnosis": {"primary_bottleneck": "host_or_scheduler_stall", "shares_pct": {}},
            "prometheus": {"selected_samples": [
                'sglang:num_queue_reqs{engine_type="unified"} 0',
                'sglang:token_usage{engine_type="unified"} 0.21',
                'sglang:cuda_graph_passes_total{mode="decode_cuda_graph"} 42',
            ]},
        }
        plan = autopilot.diagnosed_search_plan(task, self.discovery(is_moe=False), profile)
        names = [item["parameter"] for item in plan["ranked_parameter_groups"]]
        self.assertNotIn("num_continuous_decode_steps", names)
        self.assertNotIn("max_running_requests", names)
        self.assertTrue(plan["workload_assessment"]["underdriven"])

    def test_closed_loop_queue_empty_does_not_suppress_scheduler_tuning(self):
        task = self.task()
        task["search_depth"] = "evidence_guided"
        profile = {
            "diagnosis": {"primary_bottleneck": "host_or_scheduler_stall", "shares_pct": {}},
            "prometheus": {"selected_samples": [
                'sglang:num_queue_reqs{engine_type="unified"} 0',
                'sglang:token_usage{engine_type="unified"} 0.21',
            ]},
        }
        plan = autopilot.diagnosed_search_plan(task, self.discovery(is_moe=False), profile)
        names = [item["parameter"] for item in plan["ranked_parameter_groups"]]
        self.assertIn("num_continuous_decode_steps", names)
        self.assertFalse(plan["workload_assessment"]["underdriven"])

    def test_hybrid_extra_buffer_is_an_atomic_post_profile_bundle(self):
        task = self.task()
        task["workload"]["prefix_reuse_ratio"] = 0.5
        discovery = self.discovery(is_moe=False)
        discovery["model"]["is_hybrid"] = True
        profile = {"diagnosis": {"primary_bottleneck": "mixed_gpu_compute", "shares_pct": {}}}
        plan = autopilot.diagnosed_search_plan(task, discovery, profile)
        bundle = next(item for item in plan["ranked_configuration_bundles"] if item["name"] == "hybrid-mamba-extra-buffer-page-64")
        self.assertEqual(bundle["config"], {"mamba_radix_cache_strategy": "extra_buffer", "page_size": 64})

    def test_thorough_online_mode_does_not_automatically_tune_admission_ceiling(self):
        profile = {"diagnosis": {"primary_bottleneck": "mixed_gpu_compute", "shares_pct": {}}}
        plan = autopilot.diagnosed_search_plan(self.task(), self.discovery(is_moe=False), profile)
        names = [item["parameter"] for item in plan["ranked_parameter_groups"]]
        self.assertNotIn("max_running_requests", names)
        self.assertIn("num_continuous_decode_steps", names)
        self.assertIn("cuda_graph_max_bs_decode", names)
        self.assertIn("page_size", names)
        self.assertIn("sensitivity", next(item for item in plan["ranked_parameter_groups"] if item["parameter"] == "page_size")["tiers"])

    def test_unbounded_offline_mode_excludes_explicit_admission_ceiling(self):
        task = self.task()
        task["deployment_mode"] = "offline_throughput"
        profile = {"diagnosis": {"primary_bottleneck": "mixed_gpu_compute", "shares_pct": {}}}
        plan = autopilot.diagnosed_search_plan(task, self.discovery(is_moe=False), profile)
        names = [item["parameter"] for item in plan["ranked_parameter_groups"]]
        self.assertNotIn("max_running_requests", names)
        self.assertIn("schedule_conservativeness", names)
        self.assertIn("num_continuous_decode_steps", names)
        self.assertEqual(plan["deployment_mode"], "offline_throughput")

    def test_parameter_audit_accounts_for_visible_parameters(self):
        task = self.task()
        profile = {"diagnosis": {"primary_bottleneck": "mixed_gpu_compute", "shares_pct": {}}}
        plan = autopilot.diagnosed_search_plan(task, self.discovery(is_moe=False), profile)
        audit = plan["parameter_audit"]
        entries = {item["parameter"]: item for item in audit["parameters"]}
        self.assertEqual(len(entries), len(self.discovery()["parameter_catalog"]["parameters"]))
        self.assertEqual(entries["page_size"]["state"], "selected")
        self.assertEqual(entries["enable_dp_attention"]["state"], "inapplicable")
        self.assertEqual(entries["enable_mscclpp"]["state"], "inapplicable")
        self.assertIn("selected", audit["summary"])

    def test_cookbook_mtp_bundle_is_an_atomic_initial_candidate(self):
        discovery = self.discovery(is_moe=True)
        discovery["model"].update({"is_hybrid": True, "has_mtp_weights": True, "mtp_weight_key_count": 19})
        discovery["cookbook"] = {
            "model_profile": {
                "name": "qwen3.6-hybrid-mtp",
                "requires_mtp_weights": True,
                "initial_bundles": [{
                    "name": "qwen36-mtp-eagle-3-1-4",
                    "config": {
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 3,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": 4,
                        "mamba_radix_cache_strategy": "extra_buffer",
                        "page_size": 64,
                    },
                }],
            },
        }
        for name in ("speculative_algorithm", "speculative_num_steps", "speculative_eagle_topk", "speculative_num_draft_tokens"):
            discovery["parameter_catalog"]["parameters"].append({
                "dest": name, "family": "speculative", "default": None, "choices": None,
                "deprecated": False, "primary_flag": "--" + name.replace("_", "-"), "help": name,
            })
        task = self.task()
        task.update({
            "confirmation_repetitions": 2,
            "budget": {"max_trials": 10, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {"p99_ttft_ms": 1000},
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp", "name": "test", "output_dir": "/tmp/runs",
        })
        plan = autopilot.cookbook_initial_search_plan(task, discovery)
        spec = autopilot.screening_spec(task, discovery, plan, remaining_trials=9)
        self.assertEqual(spec["search"]["strategy"], "explicit_configurations")
        configuration = next(
            item for item in spec["search"]["explicit_configurations"]
            if item["name"] == "qwen36-mtp-eagle-3-1-4"
        )
        config = configuration["config"]
        self.assertEqual(config["speculative_algorithm"], "EAGLE")
        self.assertEqual(config["speculative_num_steps"], 3)
        self.assertEqual(config["mamba_radix_cache_strategy"], "extra_buffer")

    def test_mtp_override_excludes_speculative_bundles_and_audits_them(self):
        discovery = self.discovery(is_moe=True)
        discovery["model"].update({"is_hybrid": True, "has_mtp_weights": True})
        discovery["cookbook"] = {
            "model_profile": {
                "initial_bundles": [{
                    "name": "cache", "config": {"page_size": 64},
                }],
            },
        }
        discovery["parameter_catalog"]["parameters"].append({
            "dest": "speculative_algorithm", "family": "speculative", "default": None,
            "choices": None, "deprecated": False,
            "primary_flag": "--speculative-algorithm", "help": "speculative algorithm",
        })
        task = self.task()
        task["capability_overrides"] = {"mtp": "disabled"}
        catalog = autopilot.catalog_index(discovery)
        audit = autopilot.parameter_audit(catalog, [], discovery, task)
        speculative = [item for item in audit["parameters"] if item["family"] == "speculative"]
        self.assertTrue(speculative)
        self.assertTrue(all(item["state"] == "inapplicable" for item in speculative))

    def test_preprofile_selection_uses_fastest_slo_valid_configuration(self):
        result = {
            "aggregates": [
                {
                    "kind": "baseline", "config": {"tp_size": 1},
                    "all_repetitions_slo_passed": True,
                    "metrics": {"request_throughput_rps": 10.0},
                },
                {
                    "kind": "candidate", "config": {"tp_size": 1, "page_size": 64},
                    "all_repetitions_slo_passed": True,
                    "metrics": {"request_throughput_rps": 11.0},
                },
                {
                    "kind": "candidate", "config": {"tp_size": 1, "page_size": 16},
                    "all_repetitions_slo_passed": False,
                    "metrics": {"request_throughput_rps": 20.0},
                },
            ],
        }
        self.assertEqual(
            autopilot.fastest_slo_valid_configuration(
                result, {"metric": "request_throughput_rps", "direction": "maximize"}
            ),
            {"tp_size": 1, "page_size": 64},
        )


if __name__ == "__main__":
    unittest.main()
