import json
import io
import os
import signal
import sys
import tempfile
import unittest
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
import bayesian
import trial_store
import optimization_rules
import parameter_evolution
import sglang_catalog


class CapabilityCircuitBreakerTests(unittest.TestCase):
    def test_logged_benchmark_runner_preserves_output_and_returncode(self):
        with tempfile.TemporaryDirectory() as root:
            log_path = Path(root) / "benchmark.log"
            with log_path.open("w", encoding="utf-8") as log:
                returncode = autotune.run_logged_subprocess(
                    [sys.executable, "-c", "print('benchmark-progress-ok')"],
                    cwd=root, env=dict(os.environ), log_handle=log, timeout=5,
                )
            content = log_path.read_text(encoding="utf-8")
        self.assertEqual(returncode, 0)
        self.assertIn("benchmark-progress-ok", content)

    def test_progress_reports_bayesian_update_without_failure_label(self):
        reporter = autopilot.ProgressReporter()
        emitted = []
        reporter.emit = lambda stage, message, **kwargs: emitted.append((stage, message))
        reporter.trial("confirmation", {
            "event": "bayesian_update", "trial_index": 4, "trial_count": 12,
            "trial_name": "baseline-r02",
            "posterior": {"blocks": 2, "action": "accept"},
        })
        self.assertEqual(emitted, [("confirmation", "Bayesian update after 2 paired blocks: accept")])

    def test_parallel_progress_counts_completed_trials_not_trial_indexes(self):
        reporter = autopilot.ProgressReporter()
        emitted = []
        reporter.emit = lambda stage, message, **kwargs: emitted.append(
            (stage, message, kwargs)
        )
        reporter.reset_stage("screen", 4)
        for index, name in ((1, "a"), (3, "c")):
            reporter.trial("screen", {
                "event": "trial_started", "trial_index": index,
                "trial_count": 4, "trial_name": name,
            })
        self.assertEqual(emitted[0][2]["completed"], 0)
        self.assertEqual(emitted[1][2]["completed"], 0)
        self.assertIn("active=2", emitted[1][1])
        reporter.trial("screen", {
            "event": "trial_finished", "trial_index": 3,
            "trial_count": 4, "trial_name": "c", "ok": True,
            "metrics": {}, "slo_passed": True,
        })
        self.assertEqual(emitted[-1][2]["completed"], 1)
        self.assertNotEqual(emitted[-1][2]["completed"], 3)

    def test_trial_phase_heartbeat_does_not_advance_completion(self):
        reporter = autopilot.ProgressReporter()
        emitted = []
        reporter.emit = lambda stage, message, **kwargs: emitted.append(
            (message, kwargs)
        )
        reporter.reset_stage("screen", 2)
        reporter.trial("screen", {
            "event": "trial_started", "trial_index": 1,
            "trial_count": 2, "trial_name": "candidate",
        })
        reporter.trial("screen", {
            "event": "trial_phase", "trial_index": 1,
            "trial_count": 2, "trial_name": "candidate",
            "phase": "server_startup", "message": "elapsed=30s",
        })
        self.assertEqual(emitted[-1][1]["completed"], 0)
        self.assertIn("server_startup", emitted[-1][0])

    def test_progress_is_written_to_stderr_so_json_stdout_stays_clean(self):
        reporter = autopilot.ProgressReporter()
        stderr = io.StringIO()
        stdout = io.StringIO()
        with mock.patch("sys.stderr", stderr), mock.patch("sys.stdout", stdout):
            reporter.emit("plan", "working", completed=1, total=2)
        self.assertIn("plan", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")

        json_stderr = io.StringIO()
        with mock.patch("sys.stderr", json_stderr):
            autopilot.ProgressReporter("json").emit(
                "screen", "working", completed=1, total=4
            )
        event = json.loads(json_stderr.getvalue())
        self.assertEqual(event["stage"], "screen")
        self.assertEqual(event["completed"], 1)
        silent = io.StringIO()
        with mock.patch("sys.stderr", silent):
            autopilot.ProgressReporter("none").emit("screen", "hidden")
        self.assertEqual(silent.getvalue(), "")

    def test_execution_schema_accepts_search_audit_metadata(self):
        spec = {
            "schema_version": 1,
            "name": "audit-fields",
            "repository": "/tmp",
            "model": {"path": "/tmp/model"},
            "execution": {"python": sys.executable, "host": "127.0.0.1", "port": 31000},
            "hardware": {"gpus_per_host": 1},
            "benchmark": {"dataset_name": "random-ids", "num_prompts": 4, "random_input_len": 8, "random_output_len": 4, "warmup_requests": 1, "seed": 1},
            "objective": {"metric": "request_throughput_rps", "direction": "maximize", "min_improvement_pct": 1, "max_regression_pct": 5},
            "slo": {},
            "search": {
                "strategy": "explicit_configurations", "baseline": {"tp_size": 1},
                "explicit_configurations": [], "repetitions": 1,
                "history_candidate_quota": 2,
                "history_candidates_selected": ["history-a"],
                "mandatory_mechanism_parameters": ["moe_runner_backend"],
                "mechanism_coverage_target": 2,
                "covered_submechanisms": ["kv_capacity", "kv_layout"],
                "high_magnitude_rule_parameter_floor": 2,
                "high_magnitude_rule_coverage": {"kv": ["page_size", "mem_fraction_static"]},
                "deferred_triggered_parameters": [],
                "compatibility_baseline": {"moe_runner_backend": "flashinfer_cutlass"},
                "compatibility_evidence": ["NVFP4 MoE runtime requirement"],
                "sibling_refinement_candidates": [],
                "sibling_refinement_policy": "bounded",
            },
            "budget": {"max_trials": 1, "max_gpu_hours": 1, "max_wall_time_minutes": 1},
        }
        self.assertFalse(any("unsupported search field" in error for error in autotune.execution_errors(spec)))

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
                {"index": 0, "name": "H200", "memory_mib": 141 * 1024},
                {"index": 1, "name": "L40S", "memory_mib": 48 * 1024},
            ],
        }
        task = {"env": {"CUDA_VISIBLE_DEVICES": "1"}}
        self.assertEqual(autopilot.selected_gpus(task, inventory)[0]["name"], "L40S")

    def test_minimum_tp_uses_visible_memory(self):
        task = {"env": {"CUDA_VISIBLE_DEVICES": "0,1"}}
        inventory = self.inventory("nvidia", "L40S", memory_mib=48 * 1024, count=2)
        model = {"weight_bytes": 70 * 1024**3}
        self.assertEqual(autopilot.minimum_tp(task, inventory, model), 2)

    def test_qwen35_mtp_num_hidden_layers_marks_integrated_mtp_weights(self):
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / "qwen35"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "mtp_num_hidden_layers": 1,
                    "layer_types": ["linear_attention", "full_attention"],
                },
            }), encoding="utf-8")
            inventory = autopilot.model_inventory(str(model))
        self.assertEqual(inventory["num_mtp_layers"], 1)
        self.assertTrue(inventory["has_mtp_weights"])

    def test_qwen35_nested_text_config_is_recognized_as_moe(self):
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / "qwen35-moe"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "num_experts": 512,
                    "moe_intermediate_size": 1024,
                },
            }), encoding="utf-8")
            inventory = autopilot.model_inventory(str(model))
        self.assertTrue(inventory["is_moe"])
        self.assertEqual(inventory["num_experts"], 512)
        self.assertEqual(inventory["moe_intermediate_size"], 1024)

    def test_nvfp4_moe_requires_cutlass_runtime_baseline(self):
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / "gemma4-nvfp4"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "num_experts": 128,
                "quantization_config": {
                    "quant_method": "modelopt", "quant_algo": "NVFP4",
                },
            }), encoding="utf-8")
            inventory = autopilot.model_inventory(str(model))
        discovery = {
            "model": inventory,
            "parameter_catalog": {"parameters": [{
                "dest": "moe_runner_backend", "deprecated": False, "cli_visible": True,
                "choices": ["flashinfer_cutlass", "flashinfer_trtllm", "triton"],
            }]},
        }
        constraints = autopilot.runtime_compatibility_constraints(discovery)
        self.assertEqual(inventory["quantization_algorithm"], "NVFP4")
        self.assertEqual(
            constraints["required_config"],
            {"moe_runner_backend": "flashinfer_cutlass"},
        )
        compatible, reason = autopilot.parameter_value_runtime_compatible(
            discovery, "moe_runner_backend", "flashinfer_trtllm"
        )
        self.assertFalse(compatible)
        self.assertIn("NVFP4", reason)

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
            self.assertEqual(task["calibration"]["concurrencies"], [1, 2, 4])
            self.assertEqual(task["workload"]["shared_prefix"]["system_prompt_tokens"], 192)
            self.assertEqual(task["experiment_mode"], "fast")
            self.assertEqual(task["search_depth"], "evidence_guided")
            self.assertEqual(task["measurement"]["min_measurement_seconds"], 15)
            self.assertEqual(task["slo"], {})
            self.assertEqual(task["parameter_evolution"]["mode"], "conservative")
            self.assertNotIn("exploration_budget_pct", task["parameter_evolution"])
            self.assertNotIn("economics", task)
            self.assertNotIn("hardware", task)

            args.shared_prefix_tokens = None
            args.experiment_mode = None
            args.max_concurrency = None
            args.concurrency_points = None
            online_task = inferopt_cli.init_task(args)
            self.assertEqual(online_task["workload"]["max_concurrency"], 8)
            self.assertEqual(online_task["experiment_mode"], "balanced")
            args.deployment_mode = "offline_throughput"
            offline_task = inferopt_cli.init_task(args)
            self.assertNotIn("max_concurrency", offline_task["workload"])
            self.assertTrue(offline_task["workload"]["unbounded_client_concurrency"])
            self.assertEqual(offline_task["slo"], {})

            args.parameter_evolution_mode = "experimental"
            args.parameter_evolution_budget_pct = "12.5"
            args.max_provisional_trials = 2
            experimental_task = inferopt_cli.init_task(args)
            self.assertEqual(
                experimental_task["parameter_evolution"]["mode"], "experimental"
            )
            self.assertEqual(
                experimental_task["parameter_evolution"]["exploration_budget_pct"], 12.5
            )
            self.assertEqual(
                experimental_task["parameter_evolution"]["max_provisional_trials"], 2
            )

            args.parameter_evolution_mode = "conservative"
            args.parameter_evolution_budget_pct = None
            args.max_provisional_trials = None
            args.enable_history = False
            disabled_history_task = inferopt_cli.init_task(args)
            self.assertEqual(disabled_history_task["history"], {"enabled": False})
            self.assertNotIn("economics", disabled_history_task)

            args.history_database = str(root_path / "ignored.sqlite3")
            with self.assertRaisesRegex(ValueError, "require --enable-history"):
                inferopt_cli.init_task(args)
            args.history_database = None
            args.currency = "CNY"
            with self.assertRaisesRegex(ValueError, "requires --cost-per-gpu-hour"):
                inferopt_cli.init_task(args)
            args.currency = None
            args.parameter_evolution_budget_pct = "10"
            with self.assertRaisesRegex(ValueError, "require --parameter-evolution-mode experimental"):
                inferopt_cli.init_task(args)
            args.parameter_evolution_budget_pct = None
            args.cost_per_gpu_hour = 20
            args.currency = "CNY"
            cost_task = inferopt_cli.init_task(args)
            self.assertEqual(cost_task["economics"], {
                "cost_per_gpu_hour": 20.0, "currency": "CNY",
            })
            args.cost_per_gpu_hour = None
            args.currency = None

            legacy_override = dict(task)
            legacy_override["hardware"] = {"canonical_gpu_model": "H800"}
            self.assertIn(
                "unsupported field: hardware",
                autopilot.validate_task(legacy_override),
            )

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

    def test_interactive_init_skips_disabled_history_evolution_and_cost_followups(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            model = root_path / "model"
            repository = root_path / "sglang"
            model.mkdir()
            repository.mkdir()
            args = type("Args", (), {
                "non_interactive": False,
                "repository": str(repository), "python": sys.executable,
                "model_path": str(model), "output_dir": str(root_path / "runs"),
                "name": "interactive", "deployment_mode": "online_latency",
                "input_tokens": "256", "output_tokens": "64",
                "dataset_name": "synthetic", "latency_slo_statistic": "",
                "max_concurrency": "4", "concurrency_points": "1,2,4",
                "shared_prefix_tokens": "0", "experiment_mode": "fast",
                "cuda_visible_devices": "all", "max_gpus": 1,
                "allow_download": True,
                "allow_kv_cache_precision_tuning": False,
                "enable_history": False,
                "parameter_evolution_mode": "conservative",
                "cost_per_gpu_hour": None,
            })()
            prompts = []

            def answer(prompt):
                prompts.append(prompt)
                return ""

            with mock.patch.object(sys.stdin, "isatty", return_value=True), \
                 mock.patch("builtins.input", side_effect=answer):
                task = inferopt_cli.init_task(args)
        self.assertEqual(task["history"], {"enabled": False})
        self.assertEqual(task["parameter_evolution"], {"mode": "conservative"})
        self.assertNotIn("economics", task)
        self.assertEqual(len(prompts), 1, prompts)
        self.assertIn("blank skips all cost questions", prompts[0])
        self.assertFalse(any("SQLite" in prompt for prompt in prompts))
        self.assertFalse(any("historical configurations" in prompt for prompt in prompts))
        self.assertFalse(any("provisional" in prompt for prompt in prompts))
        self.assertFalse(any("Currency" in prompt for prompt in prompts))
        self.assertFalse(any("Canonical" in prompt for prompt in prompts))

    def test_concurrency_points_accept_comma_or_space_separators(self):
        self.assertEqual(inferopt_cli.parse_concurrency_points("1, 4,16"), [1, 4, 16])
        self.assertEqual(inferopt_cli.parse_concurrency_points("1 4 16"), [1, 4, 16])

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
    def test_bayesian_measurement_plan_respects_trial_budget(self):
        spec = {
            "search": {
                "strategy": "explicit_configurations", "baseline": {"tp_size": 1},
                "explicit_configurations": [{
                    "name": "candidate", "config": {"tp_size": 1, "page_size": 16},
                }],
                "include_baseline": True, "repetitions": 2,
                "bayesian_sequential": True, "bayesian_max_blocks": 6,
            },
            "budget": {"max_trials": 5},
        }
        trials = autotune.measurement_plan(spec)
        self.assertEqual(len(trials), 4)
        self.assertEqual({item["repeat_index"] for item in trials}, {0, 1})

    def test_outlier_retry_is_disabled_for_bayesian_confirmation(self):
        spec = {
            "search": {"bayesian_sequential": True, "repetitions": 2, "outlier_retry_pct": 15},
        }
        self.assertFalse(autotune.outlier_retry_required(
            spec, {"name": "candidate"}, 100.0, 120.0
        ))

    def test_outlier_retry_is_allowed_for_extreme_one_pass_screen(self):
        spec = {
            "search": {"bayesian_sequential": False, "repetitions": 1, "outlier_retry_pct": 15},
        }
        self.assertTrue(autotune.outlier_retry_required(
            spec, {"name": "candidate"}, 100.0, 120.0
        ))

    def test_steady_state_retry_expands_generated_shared_prefix_requests(self):
        command = [
            "python3", "-m", "sglang.bench_serving", "--num-prompts", "512",
            "--gsp-num-groups", "8", "--gsp-prompts-per-group", "64",
        ]
        effective = autotune.increase_benchmark_request_count(command, 967)
        self.assertEqual(effective, 968)
        self.assertEqual(command[command.index("--num-prompts") + 1], "968")
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

    def test_null_measurement_uses_defaults_for_backward_compatibility(self):
        task = self.valid_task()
        task["measurement"] = None
        self.assertNotIn("measurement must be an object", autopilot.validate_task(task))

    def test_online_mode_accepts_no_slo_constraints(self):
        task = self.valid_task()
        task["budget"]["max_trials"] = 9
        task["slo"] = {}
        self.assertEqual(autopilot.validate_task(task), [])

    def test_empty_slo_is_an_objective_only_pass(self):
        self.assertEqual(
            inferopt.slo_results({"metrics": {}}, {"slo": {}}),
            {"passed": True, "checks": []},
        )
        self.assertNotIn(
            "slo must contain at least one hard constraint",
            inferopt.validate_spec({"slo": {}}),
        )

    def test_online_comparison_protects_declared_slo_and_error_not_every_metric(self):
        spec = {
            "deployment_mode": "online_latency",
            "objective": {
                "metric": "request_throughput_rps", "direction": "maximize",
                "min_improvement_pct": 1, "max_regression_pct": 5,
            },
            "slo": {"p99_e2e_latency_ms": 1000},
        }
        baseline = {"metrics": {
            "request_throughput_rps": 100, "p99_e2e_latency_ms": 500,
            "p99_ttft_ms": 100, "error_rate": 0,
        }}
        candidate = {"metrics": {
            "request_throughput_rps": 105, "p99_e2e_latency_ms": 600,
            "p99_ttft_ms": 150, "error_rate": 0,
        }}
        result = inferopt.compare(baseline, candidate, spec)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["protected_secondary_metrics"], ["error_rate"])

    def test_standalone_confirmation_defaults_to_two_repetitions(self):
        spec = {
            "deployment_mode": "online_latency", "slo": {},
            "objective": {
                "metric": "request_throughput_rps", "direction": "maximize",
                "min_improvement_pct": 1, "max_regression_pct": 5,
            },
            "search": {},
        }
        common = {
            "eligible_for_confirmation": True, "completed_repetitions": 2,
            "expected_repetitions": 2, "stable": True,
            "all_repetitions_slo_passed": True,
        }
        baseline = {
            **common, "configuration_name": "baseline", "kind": "baseline",
            "config": {}, "metrics": {"request_throughput_rps": 100, "error_rate": 0},
            "metric_samples": {"request_throughput_rps": [100.0, 100.1]},
        }
        candidate = {
            **common, "configuration_name": "candidate", "kind": "candidate",
            "config": {"chunked_prefill_size": 4096},
            "metrics": {"request_throughput_rps": 102, "error_rate": 0},
            "metric_samples": {"request_throughput_rps": [102.0, 102.1]},
        }
        _, _, winner = autotune.evaluate_aggregates([baseline, candidate], spec)
        self.assertIsNotNone(winner)

    def test_overlapping_two_sample_interval_reports_noise_limited(self):
        spec = {
            "deployment_mode": "online_latency", "slo": {},
            "objective": {
                "metric": "request_throughput_rps", "direction": "maximize",
                "min_improvement_pct": 0.1, "max_regression_pct": 5,
            },
            "search": {"min_confirm_repetitions": 2},
        }
        common = {
            "eligible_for_confirmation": True, "completed_repetitions": 2,
            "expected_repetitions": 2, "stable": True,
            "all_repetitions_slo_passed": True,
        }
        baseline = {
            **common, "configuration_name": "baseline", "kind": "baseline", "config": {},
            "metrics": {"request_throughput_rps": 100.5, "error_rate": 0},
            "metric_samples": {"request_throughput_rps": [100.0, 101.0]},
        }
        candidate = {
            **common, "configuration_name": "candidate", "kind": "candidate",
            "config": {"page_size": 64},
            "metrics": {"request_throughput_rps": 101.0, "error_rate": 0},
            "metric_samples": {"request_throughput_rps": [100.5, 101.5]},
        }
        aggregates, screening, confirmed = autotune.evaluate_aggregates(
            [baseline, candidate], spec
        )
        self.assertIsNone(confirmed)
        self.assertTrue(aggregates[1]["comparison"]["noise_limited"])
        recommendation, status, _ = autotune.deployment_recommendation(
            aggregates, screening, confirmed
        )
        self.assertIsNone(recommendation)
        self.assertEqual(status, "noise_limited")

    def test_welch_interval_rounds_fractional_df_down_conservatively(self):
        interval = autotune.objective_improvement_confidence_interval(
            [17634.63361272083, 17611.617676267026],
            [17902.327839340785, 17903.472965959714],
            "maximize",
        )
        self.assertIsNotNone(interval)
        self.assertEqual(interval["critical_value"], 12.706)
        self.assertAlmostEqual(interval["lower_pct"], 0.7568103968, places=6)

    def test_two_gpu_tp2_confirmation_uses_abba_not_grouped_sessions(self):
        spec = {
            "search": {
                "strategy": "explicit_configurations",
                "baseline": {"tp_size": 2},
                "explicit_configurations": [{
                    "name": "candidate",
                    "config": {"tp_size": 2, "page_size": 64},
                }],
                "include_baseline": True,
                "repetitions": 2,
                "reuse_server_across_repetitions": True,
            },
            "budget": {"max_trials": 4},
            "execution": {"env": {"CUDA_VISIBLE_DEVICES": "0,1"}},
            "hardware": {"gpus_per_host": 2},
        }
        self.assertFalse(autotune.resident_ab_eligible(spec))
        plan = autotune.measurement_plan(spec)
        self.assertEqual(
            [(item["configuration_name"], item["repeat_index"]) for item in plan],
            [("baseline", 0), ("candidate", 0), ("candidate", 1), ("baseline", 1)],
        )

    def test_fp8_performance_winner_fails_closed_without_quality_evaluation(self):
        task = {"quality": {"allow_kv_cache_precision_tuning": True}}
        winner = {"config": {"tp_size": 2, "kv_cache_dtype": "fp8_e5m2"}}
        gate = autopilot.recommendation_quality_gate(task, winner)
        self.assertTrue(gate["required"])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["state"], "quality_unverified")

    def test_confirmation_reserve_includes_adaptive_repetitions(self):
        task = {
            "confirmation_repetitions": 2,
            "measurement": {"adaptive_confirmation_max_repetitions": 3},
        }
        self.assertEqual(autopilot.confirmation_trial_reserve(task), 6)

    def test_reproducible_command_merges_resolved_runtime_settings(self):
        recommendation = {"config": {"tp_size": 2, "kv_cache_dtype": "fp8_e5m2"}}
        resolved = {
            "attention_backend": "flashinfer", "chunked_prefill_size": 8192,
            "mem_fraction_static": 0.809, "max_running_requests": None,
        }
        with mock.patch.object(
            autopilot, "final_server_command",
            side_effect=lambda _spec, value: value["config"],
        ):
            config = autopilot.reproducible_server_command({}, recommendation, resolved)
        self.assertEqual(config["attention_backend"], "flashinfer")
        self.assertEqual(config["chunked_prefill_size"], 8192)
        self.assertEqual(config["kv_cache_dtype"], "fp8_e5m2")
        self.assertNotIn("max_running_requests", config)

    def test_quality_unverified_report_emits_no_deployment_command(self):
        report = inferopt_cli.markdown_report({
            "run_dir": "/tmp/run",
            "recommendation_status": "confirmed_performance_candidate_quality_unverified",
            "recommendation_reason": "quality evaluation required",
            "deployable": False,
            "recommended_configuration": None,
            "provisional_configuration": {
                "config": {"tp_size": 2, "kv_cache_dtype": "fp8_e5m2"},
            },
            "deployment_command": None,
            "quality_gate": {
                "required": True, "passed": False, "state": "quality_unverified",
                "parameter": "kv_cache_dtype", "value": "fp8_e5m2",
                "reason": "quality evaluation required",
            },
        })
        self.assertIn("Performance-Only Candidate", report)
        self.assertIn("quality_unverified", report)
        self.assertNotIn("## Reproducible Deployment Command", report)

    def test_bayesian_sequential_accepts_clear_paired_gain(self):
        result = bayesian.sequential_decision_from_samples(
            [100.0, 100.2], [105.0, 105.2],
            objective_metric="request_throughput_rps", minimum_improvement_pct=1.0,
            min_blocks=2, max_blocks=6,
        )
        self.assertEqual(result["action"], "accept")
        self.assertGreater(result["probability_improvement_gt_minimum"], 0.99)

    def test_bayesian_sequential_accepts_four_percent_gain_after_two_blocks(self):
        result = bayesian.sequential_decision_from_samples(
            [17708.041164437564, 17712.389474802232],
            [18447.6938125855, 18464.725655383776],
            objective_metric="total_throughput_tps", minimum_improvement_pct=1.0,
            min_blocks=2, max_blocks=6,
        )
        self.assertEqual(result["action"], "accept")

    def test_bayesian_sequential_rejects_clear_loss(self):
        result = bayesian.sequential_decision_from_samples(
            [100.0, 100.2], [96.0, 96.2],
            objective_metric="request_throughput_rps", minimum_improvement_pct=1.0,
            min_blocks=2, max_blocks=6,
        )
        self.assertEqual(result["action"], "reject")

    def test_bayesian_sequential_requests_more_blocks_when_ambiguous(self):
        result = bayesian.sequential_decision_from_samples(
            [100.0, 101.0], [101.0, 100.0],
            objective_metric="request_throughput_rps", minimum_improvement_pct=1.0,
            min_blocks=2, max_blocks=6,
        )
        self.assertEqual(result["action"], "continue")

    def test_cost_per_token_uses_user_supplied_gpu_price(self):
        task = {"economics": {"cost_per_gpu_hour": 2.0, "currency": "USD"}}
        decision = {"aggregates": [
            {"kind": "baseline", "metrics": {"total_throughput_tps": 1000, "output_throughput_tps": 100, "error_rate": 0}},
        ], "recommended_configuration": {
            "metrics": {"total_throughput_tps": 1250, "output_throughput_tps": 125, "error_rate": 0},
        }}
        cost = autopilot.cost_per_token_summary(task, decision, 4, 2.0)
        self.assertTrue(cost["available"])
        self.assertAlmostEqual(cost["baseline"]["cost_per_million_total_tokens"], 2.2222222, places=6)
        self.assertAlmostEqual(cost["winner"]["cost_per_million_total_tokens"], 1.7777777, places=6)

    def test_roofline_reports_counter_permission_without_guessing_bound(self):
        result = profile_sglang.roofline_diagnosis({
            "available": True, "performance_counter_access": False,
        }, {"name": "kernel", "time_pct": 30})
        self.assertEqual(result["status"], "roofline_unavailable_permission")
        self.assertNotIn("classification", result)

    def test_roofline_classifies_memory_bound_from_ncu_metrics(self):
        result = profile_sglang.roofline_diagnosis(
            {"available": True, "performance_counter_access": True},
            {"name": "kernel"},
            {
                "achieved_flops": 4e13,
                "achieved_bandwidth_bytes_s": 2e12,
                "peak_flops": 1e14,
                "peak_bandwidth_bytes_s": 2.5e12,
            },
        )
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["classification"], "memory_bound")

    def test_history_warm_start_requires_exact_compatibility_fingerprint(self):
        with tempfile.TemporaryDirectory() as root:
            database = Path(root) / "history.sqlite3"
            connection = trial_store.open_store(database)
            connection.execute(
                """INSERT INTO runs VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("/run", "match", "now", "confirmed_candidate", "request_throughput_rps", "m", "h", "w", "f"),
            )
            connection.execute(
                """INSERT INTO trials(run_dir,stage,configuration_name,config_hash,config_json,objective_value,improvement_pct,slo_passed,ok,metrics_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                ("/run", "screening", "winner", "hash", '{"page_size":64}', 100, 3.0, 1, 1, '{}'),
            )
            connection.commit()
            connection.close()
            self.assertEqual(len(trial_store.warm_start_candidates(database, "match")), 1)
            self.assertEqual(trial_store.warm_start_candidates(database, "different"), [])

    def test_offline_reference_window_requires_runtime_capacity_and_five_waves(self):
        task = self.valid_task()
        task.update({"deployment_mode": "offline_throughput", "slo": {}})
        spec = {"benchmark": {
            "dataset_name": "random-ids", "num_prompts": 40,
            "min_measurement_seconds": 5,
        }}
        with self.assertRaisesRegex(ValueError, "observed practical/admission capacity"):
            autopilot.configure_offline_reference_window(spec, task)
        task["workload"]["observed_admission_capacity"] = 50
        task["workload"]["observed_practical_capacity"] = 18
        autopilot.configure_offline_reference_window(spec, task)
        self.assertEqual(spec["benchmark"]["num_prompts"], 90)
        self.assertEqual(spec["benchmark"]["saturation_capacity"], 18)
        self.assertEqual(spec["benchmark"]["saturation_waves"], 5)

    def test_offline_practical_capacity_uses_shape_aware_kv_tokens(self):
        task = self.valid_task()
        task["workload"].update({"input_tokens": 8192, "output_tokens": 128})
        profile = {"startup_capacity": {
            "max_running_requests": 2048,
            "max_total_tokens": 377277,
            "practical_request_capacity": 45,
        }}
        self.assertEqual(autopilot.observed_admission_capacity(profile), 2048)
        self.assertEqual(autopilot.observed_practical_capacity(profile, task), 45)
        task["deployment_mode"] = "offline_throughput"
        task["slo"] = {}
        task["workload"].update({
            "observed_admission_capacity": 2048,
            "observed_practical_capacity": 45,
        })
        self.assertEqual(autopilot.offline_saturation_request_count(task), 225)
        self.assertEqual(
            autopilot.offline_saturation_request_count(task, confirmation=True), 450
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

    def test_default_calibration_starts_at_declared_target(self):
        task = self.valid_task()
        task["budget"]["max_trials"] = 30
        self.assertEqual(autopilot.calibration_concurrencies(task), [4])
        task["budget"]["max_trials"] = 11
        self.assertEqual(autopilot.calibration_concurrencies(task), [4])

    def test_calibration_uses_scaled_task_measurement_not_fixed_512_requests(self):
        task = self.valid_task()
        task["workload"]["max_concurrency"] = 8
        task["measurement"] = {"warmup_requests": 32, "min_measurement_requests": 128, "min_measurement_seconds": 20}
        discovery = {"derived": {"minimum_tp_size": 1}, "model": {}, "hardware": {"gpus": []}, "parameter_catalog": {"parameters": []}}
        spec = autopilot.calibration_spec(task, discovery, 2, 1, 30)
        self.assertEqual(spec["benchmark"]["num_prompts"], 32)

    def test_explicit_calibration_range_starts_at_one_and_includes_the_cap(self):
        task = self.valid_task()
        task["budget"]["max_trials"] = 30
        task["calibration"] = {
            "strategy": "fixed_curve", "min_concurrency": 1,
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

    def test_nvfp4_moe_backend_exception_is_not_misclassified_as_process_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(
                "NotImplementedError: Unsupported moe_runner_backend for NVFP4 MoE. "
                "Use --moe-runner-backend flashinfer_cutlass instead.\n",
                encoding="utf-8",
            )
            detail = autotune.startup_failure_detail(
                "server exited during startup with code 137", log
            )
            self.assertIn("Unsupported moe_runner_backend", detail)
            self.assertEqual(
                autotune.classify_failure(
                    log, Path(directory) / "benchmark.log", detail
                ),
                "backend_incompatible",
            )

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


class NsysAnalysisTests(unittest.TestCase):
    def test_profile_command_runner_preserves_output_with_polling(self):
        result = profile_sglang.run_command([
            sys.executable, "-c", "print('progress-runner-ok')"
        ], timeout=5)
        self.assertEqual(result["returncode"], 0)
        self.assertIn("progress-runner-ok", result["stdout"])

    def test_nsys_stats_reports_each_known_substage(self):
        with tempfile.TemporaryDirectory() as root:
            report = Path(root) / "baseline.nsys-rep"
            report.write_bytes(b"report")
            events = []
            with mock.patch.object(profile_sglang, "run_command", return_value={
                "returncode": 0, "stdout": "Name,Total Time\nkernel,1\n",
                "stderr": "",
            }):
                parsed, statuses = profile_sglang.collect_stats(
                    report, Path(root), progress=events.append
                )
        self.assertEqual(len(statuses), len(profile_sglang.NSYS_ROUTING_REPORTS))
        self.assertEqual(events[0]["completed"], 0)
        self.assertEqual(events[-1]["completed"], len(profile_sglang.NSYS_ROUTING_REPORTS))
        self.assertEqual(events[-1]["total"], len(profile_sglang.NSYS_ROUTING_REPORTS))

    def test_profile_preserves_nohup_sighup_ignore(self):
        installed = []

        def inherited(signum):
            return signal.SIG_IGN if signum == signal.SIGHUP else signal.SIG_DFL

        with mock.patch.object(profile_sglang.signal, "getsignal", side_effect=inherited), \
             mock.patch.object(profile_sglang.signal, "signal", side_effect=lambda signum, handler: installed.append(signum)):
            previous = profile_sglang.install_profile_interrupt_handlers()
        self.assertNotIn(signal.SIGHUP, installed)
        self.assertNotIn(signal.SIGHUP, previous)

    def test_unbounded_profile_uses_kv_capacity_not_admission_ceiling(self):
        sizing = profile_sglang.bounded_profile_request_target(
            current_prompts=32,
            group_floor=4,
            admission_capacity=2048,
            token_capacity=377277,
            tokens_per_request=8192 + 128,
        )
        self.assertEqual(sizing["practical_request_capacity"], 45)
        self.assertEqual(sizing["target_prompts"], 135)
        self.assertEqual(sizing["policy"], "three_practical_kv_waves_capped_256")

    def test_unbounded_profile_fallback_is_hard_capped(self):
        sizing = profile_sglang.bounded_profile_request_target(
            current_prompts=32, group_floor=4,
            admission_capacity=2048, token_capacity=None,
            tokens_per_request=8320,
        )
        self.assertEqual(sizing["target_prompts"], 256)

    def test_profile_step_window_auto_stops_after_prefill_decode_transition(self):
        window = profile_sglang.bounded_profile_step_window(
            capture_prompts=135, output_tokens=128, practical_capacity=45,
        )
        self.assertEqual(window, {"start_step": 45, "steps": 64})
        short = profile_sglang.bounded_profile_step_window(
            capture_prompts=32, output_tokens=1, practical_capacity=45,
        )
        self.assertEqual(short, {"start_step": 0, "steps": 1})

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

    def test_kernel_family_aggregates_gdn_variants(self):
        reports = {
            "cuda_gpu_trace": [],
            "cuda_gpu_kern_sum": [
                {"Time (%)": "26.3", "Name": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"},
                {"Time (%)": "8.7", "Name": "chunk_gated_delta_rule_fwd_kkt_solve_kernel"},
                {"Time (%)": "7.6", "Name": "chunk_fwd_kernel_o"},
                {"Time (%)": "5.8", "Name": "recompute_w_u_fwd_kernel"},
                {"Time (%)": "29.1", "Name": "flash::FlashAttnFwdSm90"},
            ],
            "cuda_api_sum": [], "cuda_gpu_mem_time_sum": [], "cuda_kern_exec_sum": [],
        }
        families = profile_sglang.analyze_reports(reports)["top_kernel_families"]
        self.assertEqual(families[0]["name"], "gdn_delta_rule")
        self.assertAlmostEqual(families[0]["time_pct"], 48.4)


class OptimizationRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Deliberately synthetic: regression tests must not publish user
        # experiment records, model names, server paths, or measured artifacts.
        cls.replay = {
            "task": {
                "deployment_mode": "offline_throughput",
                "budget": {"max_trials": 36},
                "workload": {
                    "input_tokens": 16384, "output_tokens": 256,
                    "max_concurrency": 16, "prefix_reuse_ratio": 0.0,
                    "request_rate": "inf",
                },
            },
            "discovery": {
                "hardware": {
                    "vendor": "nvidia",
                    "gpus": [
                        {"index": index, "name": "Synthetic Hopper 80GB", "memory_mib": 81920}
                        for index in range(4)
                    ],
                },
                "hardware_profile": {"architecture": "hopper"},
                "model": {
                    "is_moe": True, "is_hybrid": True,
                    "has_mtp_weights": True, "weight_bytes": 200 * 1024**3,
                    "context_length": 262144,
                },
                "derived": {"visible_gpu_count": 4, "minimum_tp_size": 4},
            },
            "profile": {
                "diagnosis": {
                    "primary_bottleneck": "mixed_gpu_compute",
                    "profiling_run_performance_comparable": False,
                    "shares_pct": {
                        "attention_kernels": 30.0,
                        "moe_kernels": 0.0,
                        "communication_kernels": 3.0,
                    },
                    "top_kernel_families": [
                        {"name": "gdn_delta_rule", "time_pct": 50.0},
                        {"name": "flash_attention", "time_pct": 30.0},
                    ],
                },
                "benchmark": {"metrics": {
                    "mean_e2e_latency_ms": 40000.0,
                    "mean_ttft_ms": 32000.0,
                }},
                "effective_server_config": {
                    "tp_size": 4, "mem_fraction_static": 0.75,
                    "chunked_prefill_size": 8192,
                    "max_prefill_tokens": 16384,
                },
                "runtime_observations": {
                    "prefill": {
                        "queue_nonempty_batch_pct": 95.0,
                        "token_usage_ratio": {"p95": 0.85},
                    },
                    "decode": {"token_usage_ratio": {"p95": 0.85}},
                    "moe": {"missing_tuned_config": False},
                },
            },
        }

    def test_synthetic_hybrid_classifies_gdn_then_prefill_attention(self):
        value = optimization_rules.classify_bottleneck(
            self.replay["task"], self.replay["discovery"], self.replay["profile"]
        )
        self.assertEqual(value["primary"], "gdn_state_compute_bound")
        self.assertIn("prefill_attention_bound", value["secondary"])
        self.assertGreater(value["confidence"], 0.7)

    def test_declarative_rule_catalog_is_valid(self):
        self.assertEqual(optimization_rules.validate_rule_catalog(), [])

    def test_rule_matching_accepts_unmatched_hardware_profile(self):
        discovery = dict(self.replay["discovery"])
        discovery["hardware_profile"] = None
        classification = optimization_rules.classify_bottleneck(
            self.replay["task"], discovery, self.replay["profile"]
        )
        plan = optimization_rules.match_parameter_rules(
            self.replay["task"], discovery, self.replay["profile"], classification,
            {"mem_fraction_static", "page_size", "chunked_prefill_size"},
        )
        self.assertIsNone(plan["match_context"]["hardware"]["architecture"])
        self.assertIn("mem_fraction_static", plan["parameters"])

    def test_synthetic_hybrid_matches_specialized_mechanisms(self):
        classification = optimization_rules.classify_bottleneck(
            self.replay["task"], self.replay["discovery"], self.replay["profile"]
        )
        available = {
            "moe_runner_backend", "ep_size", "enable_dp_attention",
            "mamba_full_memory_ratio", "mamba_radix_cache_strategy", "page_size",
            "mem_fraction_static", "prefill_attention_backend", "attention_backend",
            "chunked_prefill_size", "max_prefill_tokens", "enable_mixed_chunk",
        }
        plan = optimization_rules.match_parameter_rules(
            self.replay["task"], self.replay["discovery"], self.replay["profile"],
            classification, available,
        )
        for parameter in (
            "moe_runner_backend", "mamba_full_memory_ratio", "page_size",
            "mem_fraction_static", "prefill_attention_backend", "chunked_prefill_size",
        ):
            self.assertIn(parameter, plan["parameters"])
        self.assertEqual(plan["parameters"]["mamba_full_memory_ratio"]["magnitude"], "high")

    def test_memory_values_use_vram_headroom(self):
        values, evidence = optimization_rules.dynamic_parameter_values(
            "mem_fraction_static", self.replay["task"], self.replay["discovery"],
            self.replay["profile"], 0.8,
        )
        self.assertEqual(evidence["strategy"], "vram_headroom_ladder")
        self.assertEqual(evidence["gpu_memory_mib"], 81920.0)
        self.assertGreater(max(values), 0.837)
        self.assertIn(0.72, values)

    def test_budget_defaults_to_sixty_twentyfive_fifteen(self):
        budget = optimization_rules.tiered_trial_budget(36)
        self.assertEqual(budget["planned"], {
            "discovery": 22, "refinement": 9, "confirmation": 5,
        })
        self.assertEqual(sum(budget["planned"].values()), 36)

    def test_history_becomes_prior_without_candidate_trial(self):
        priors = optimization_rules.history_priors([{
            "config": {"tp_size": 4, "mem_fraction_static": 0.837},
            "history_score_pct": 10.8,
            "history_samples": 2,
            "source_runs": ["run-a"],
        }])
        self.assertEqual(priors["candidate_trials_created"], 0)
        self.assertIn("mem_fraction_static", priors["parameter_priors"])
        exact = optimization_rules.configuration_history_prior(
            priors, {"tp_size": 4, "mem_fraction_static": 0.837}
        )
        self.assertEqual(exact["mean_improvement_pct"], 10.8)

    def test_search_plan_history_does_not_create_candidate_bundle(self):
        search_plan = {
            "ranked_configuration_bundles": [
                {"name": "history-old", "config": {"mem_fraction_static": 0.8}},
                {"name": "model-native", "config": {"page_size": 64}},
            ]
        }
        evidence = {"enabled": True, "candidates": [{
            "config": {"tp_size": 4, "mem_fraction_static": 0.837},
            "history_score_pct": 10.8, "history_samples": 2,
        }]}
        autopilot.apply_history_priors_to_search_plan(search_plan, evidence)
        self.assertEqual(
            [item["name"] for item in search_plan["ranked_configuration_bundles"]],
            ["model-native"],
        )
        self.assertTrue(search_plan["history"]["prior_only"])
        self.assertEqual(search_plan["history"]["priors"]["candidate_trials_created"], 0)

    def test_report_prints_classifier_triggers_and_budget(self):
        classification = optimization_rules.classify_bottleneck(
            self.replay["task"], self.replay["discovery"], self.replay["profile"]
        )
        final = {
            "run_dir": "/tmp/run", "recommendation_status": "retain_confirmed_baseline",
            "deployable": True, "recommended_configuration": {"config": {"tp_size": 4}},
            "profiling": self.replay["profile"],
            "search_plan": {
                "bottleneck_classification": classification,
                "parameter_match_order": [{
                    "parameter": "mem_fraction_static", "magnitude": "high",
                    "rule_ids": ["gdn_state_strong_preset"],
                }],
                "ranked_parameter_groups": [{
                    "parameter": "mem_fraction_static", "values": [0.8],
                    "reason": "trigger replay",
                }],
                "history": {"enabled": True, "priors": optimization_rules.history_priors([])},
            },
            "budget_accounting": {
                **optimization_rules.tiered_trial_budget(36),
                "used": {"discovery": 12, "refinement": 8, "confirmation": 4},
                "used_percentages": {"discovery": 50, "refinement": 33.33, "confirmation": 16.67},
                "unused_trials": 12,
            },
            "parameter_search": {}, "screening": {}, "bottleneck": {},
        }
        report = inferopt_cli.markdown_report(final)
        self.assertIn("## Bottleneck Classifier", report)
        self.assertIn("gdn_state_compute_bound", report)
        self.assertIn("## Trial Budget", report)
        self.assertIn("Trigger-matched parameter order", report)

    def test_scheduler_log_extracts_cache_and_graph_evidence(self):
        text = """[2026-08-14 08:17:30] Decode batch, #running-req: 4, #full token: 1051, full token usage: 0.20, mamba num: 16, mamba usage: 0.20, cuda graph: True, gen throughput (token/s): 453.12, #queue-req: 0
[2026-08-14 08:17:30] Prefill batch, #new-seq: 3, #new-token: 256, #cached-token: 576, full token usage: 0.20, mamba usage: 0.20, #running-req: 1, #queue-req: 2, #pending-token: 64, cuda graph: False, input throughput (token/s): 107534.31"""
        summary = sglang_runtime.summarize_sglang_log(text)
        self.assertEqual(summary["decode"]["cuda_graph_coverage_pct"], 100.0)
        self.assertEqual(summary["prefill"]["cached_token_share_pct"], 69.23076923076923)
        self.assertEqual(summary["prefill"]["queue_nonempty_batch_pct"], 100.0)

    def test_scheduler_log_extracts_mtp_acceptance_without_double_counting(self):
        summary = sglang_runtime.summarize_sglang_log(
            "Speculative acceptance rate: 0.72\n"
            "accepted draft tokens: 72\n"
            "drafted tokens: 100\n"
        )
        speculative = summary["speculative"]
        self.assertTrue(speculative["telemetry_available"])
        self.assertEqual(speculative["acceptance_rate_pct"]["p50"], 72.0)
        self.assertEqual(speculative["inferred_acceptance_rate_pct"], 72.0)


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

    def test_moe_routes_moe_runner(self):
        plan = self.routed("moe_compute", {"moe_kernels": 55})
        self.assertEqual(plan["ranked_parameter_groups"][0]["parameter"], "moe_runner_backend")

    def test_moe_routes_runner_without_moe_kernel_name(self):
        plan = self.routed("cpu_gpu_synchronization", {"attention_kernels": 30})
        names = [item["parameter"] for item in plan["ranked_parameter_groups"]]
        self.assertIn("moe_runner_backend", names)

    def test_moe_coverage_is_not_required_without_an_executable_candidate(self):
        task = self.task()
        discovery = self.discovery(is_moe=True)
        plan = {
            "ranked_parameter_groups": [{
                "parameter": "page_size", "family": "memory_cache", "values": [16],
            }],
            "cookbook_candidate_bundles": [],
            "ranked_configuration_bundles": [],
        }
        self.assertNotIn(
            "moe", autopilot.required_mechanism_classes(task, discovery, plan)
        )

    def test_multi_gpu_moe_routes_dp_attention(self):
        plan = self.routed("moe_compute", {"moe_kernels": 55}, gpu_count=8, minimum_tp_size=8)
        names = [item["parameter"] for item in plan["ranked_parameter_groups"]]
        self.assertIn("enable_dp_attention", names)

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
        self.assertEqual(chunk["values"], [256, 512, 4096])
        self.assertEqual(chunk["value_strategy"]["strategy"], "uncached_workload_boundary")
        self.assertIn("resolved_sglang_default=8192", chunk["evidence"])

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
        names = [item["name"] for item in spec["search"]["explicit_configurations"]]
        self.assertIn("num_continuous_decode_steps-2", names)
        self.assertIn("moe_runner_backend-deep_gemm", names)
        self.assertIn("disable_radix_cache-true", names)
        self.assertIn("num_continuous_decode_steps-4", names)

    def test_balanced_screen_keeps_two_high_value_siblings_in_same_submechanism(self):
        task = self.task()
        task.update({
            "confirmation_repetitions": 2,
            "experiment_mode": "balanced",
            "budget": {"max_trials": 12, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {},
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp",
            "name": "sibling-screen", "output_dir": "/tmp/runs",
        })
        search_plan = {
            "ranked_parameter_groups": [
                {
                    "parameter": "num_continuous_decode_steps", "family": "scheduler",
                    "submechanism": "scheduler_cadence", "values": [2],
                    "trigger_magnitude": "high",
                    "trigger": {"rule_ids": ["scheduler-amortization"]},
                },
                {
                    "parameter": "scheduler_recv_interval", "family": "scheduler",
                    "submechanism": "scheduler_cadence", "values": [0.0005],
                    "trigger_magnitude": "high",
                    "trigger": {"rule_ids": ["scheduler-amortization"]},
                },
                {
                    "parameter": "disable_radix_cache", "family": "memory_cache",
                    "submechanism": "prefix_cache", "values": [True],
                    "trigger_magnitude": "low", "trigger": {"rule_ids": ["control"]},
                },
            ],
            "trigger_rule_plan": {
                "matches": [{
                    "id": "scheduler-amortization", "magnitude": "high",
                    "parameters": ["num_continuous_decode_steps", "scheduler_recv_interval"],
                }],
                "strong_candidates": [],
            },
        }
        spec = autopilot.screening_spec(task, self.discovery(), search_plan, remaining_trials=10)
        names = [item["name"] for item in spec["search"]["explicit_configurations"]]
        self.assertIn("num_continuous_decode_steps-2", names)
        self.assertIn("scheduler_recv_interval-0.0005", names)
        self.assertEqual(
            spec["search"]["high_magnitude_rule_coverage"]["scheduler-amortization"],
            ["num_continuous_decode_steps", "scheduler_recv_interval"],
        )

    def test_priority_bundle_reserves_slot_without_displacing_parameter_breadth(self):
        task = self.task()
        task.update({
            "confirmation_repetitions": 2,
            "budget": {"max_trials": 10, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {}, "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
            "repository": "/tmp", "python": sys.executable, "model_path": "/tmp",
            "name": "breadth", "output_dir": "/tmp/runs", "experiment_mode": "balanced",
        })
        discovery = self.discovery(is_moe=True)
        discovery["model"]["is_hybrid"] = True
        search_plan = {
            "ranked_parameter_groups": [
                {"parameter": "page_size", "family": "memory_cache", "values": [16, 32, 64]},
                {"parameter": "prefill_attention_backend", "family": "kernel_backend", "values": ["triton"]},
                {"parameter": "moe_runner_backend", "family": "moe", "values": ["triton"]},
                {"parameter": "mem_fraction_static", "family": "memory_cache", "values": [0.82]},
            ],
            "cookbook_candidate_bundles": [{
                "name": "mamba", "config": {
                    "mamba_radix_cache_strategy": "extra_buffer", "page_size": 64,
                },
            }],
            "ranked_configuration_bundles": [],
        }
        spec = autopilot.screening_spec(task, discovery, search_plan, remaining_trials=10)
        names = [item["name"] for item in spec["search"]["explicit_configurations"]]
        self.assertEqual(names[0], "mamba")
        self.assertIn("page_size-16", names)
        self.assertIn("prefill_attention_backend-triton", names)
        self.assertIn("moe_runner_backend-triton", names)
        self.assertNotIn("page_size-32", names)

    def test_successive_refinement_expands_best_measured_parameter(self):
        task = self.task()
        task.update({
            "name": "adaptive-test", "repository": "/tmp", "python": sys.executable,
            "model_path": "/tmp", "output_dir": "/tmp/runs",
            "deployment_mode": "online_latency", "experiment_mode": "balanced",
            "confirmation_repetitions": 2, "parallel_trials": 1,
            "budget": {"max_trials": 12, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {},
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
        })
        discovery = self.discovery(is_moe=False)
        discovery["hardware"]["gpus"] = [{
            "index": 0, "name": "H100", "memory_mib": 80 * 1024,
        }]
        search_plan = {"ranked_parameter_groups": [{
            "parameter": "mem_fraction_static", "family": "memory_cache",
            "values": [0.77, 0.82, 0.87],
        }]}
        screen = {"aggregates": [
            {
                "configuration_name": "baseline", "kind": "baseline", "config": {"tp_size": 1},
                "metrics": {"request_throughput_rps": 100.0},
            },
            {
                "configuration_name": "mem_fraction_static-0.82", "kind": "candidate",
                "config": {"tp_size": 1, "mem_fraction_static": 0.82},
                "metrics": {"request_throughput_rps": 100.5}, "stable": True,
                "all_repetitions_slo_passed": True, "screening_accepted": False,
                "comparison": {"improvement_pct": 0.5, "secondary_regressions_passed": True},
            },
        ]}
        spec = autopilot.interaction_spec(
            task, discovery, search_plan, screen, 6, 1.0, 30.0
        )
        self.assertIsNotNone(spec)
        names = [item["name"] for item in spec["search"]["explicit_configurations"]]
        self.assertIn("refine-mem_fraction_static-0.77", names)

    def test_positive_parameter_promotes_unmeasured_trigger_sibling(self):
        task = self.task()
        task.update({
            "name": "sibling-refinement", "repository": "/tmp", "python": sys.executable,
            "model_path": "/tmp", "output_dir": "/tmp/runs",
            "deployment_mode": "online_latency", "experiment_mode": "balanced",
            "confirmation_repetitions": 2, "parallel_trials": 1,
            "budget": {"max_trials": 15, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {},
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
        })
        search_plan = {"ranked_parameter_groups": [
            {
                "parameter": "num_continuous_decode_steps", "family": "scheduler",
                "submechanism": "scheduler_cadence", "values": [2, 4],
                "trigger_magnitude": "high", "trigger": {"rule_ids": ["scheduler-amortization"]},
            },
            {
                "parameter": "scheduler_recv_interval", "family": "scheduler",
                "submechanism": "scheduler_cadence", "values": [0.0005, 0.001],
                "trigger_magnitude": "high", "trigger": {"rule_ids": ["scheduler-amortization"]},
            },
        ]}
        screen = {"aggregates": [
            {
                "configuration_name": "baseline", "kind": "baseline", "config": {"tp_size": 1},
                "metrics": {"request_throughput_rps": 100.0},
            },
            {
                "configuration_name": "num-continuous-2", "kind": "candidate",
                "config": {"tp_size": 1, "num_continuous_decode_steps": 2},
                "metrics": {"request_throughput_rps": 102.0}, "stable": True,
                "all_repetitions_slo_passed": True, "screening_accepted": True,
                "comparison": {"improvement_pct": 2.0, "secondary_regressions_passed": True},
            },
        ], "completed_trials": 2}
        spec = autopilot.interaction_spec(
            task, self.discovery(is_moe=False), search_plan, screen, 8, 1.0, 30.0,
            phase="refinement",
        )
        self.assertIsNotNone(spec)
        names = [item["name"] for item in spec["search"]["explicit_configurations"]]
        self.assertIn("refine-sibling-scheduler_recv_interval-0.0005", names)
        self.assertIn(
            "refine-sibling-scheduler_recv_interval-0.0005",
            spec["search"]["sibling_refinement_candidates"],
        )

    def test_composition_uses_refined_winner(self):
        task = self.task()
        task.update({
            "name": "two-stage", "repository": "/tmp", "python": sys.executable,
            "model_path": "/tmp", "output_dir": "/tmp/runs",
            "deployment_mode": "online_latency", "experiment_mode": "balanced",
            "confirmation_repetitions": 2, "parallel_trials": 1,
            "budget": {"max_trials": 16, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "slo": {}, "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
        })
        discovery = self.discovery(is_moe=False)
        discovery["hardware"]["gpus"] = [{"index": 0, "name": "H100", "memory_mib": 80 * 1024}]
        search_plan = {
            "budget_allocation": optimization_rules.tiered_trial_budget(16),
            "ranked_parameter_groups": [{
                "parameter": "mem_fraction_static", "family": "memory_cache",
                "values": [0.805, 0.9],
            }],
        }
        baseline = {
            "configuration_name": "baseline", "kind": "baseline", "config": {"tp_size": 1},
            "metrics": {"request_throughput_rps": 100.0},
        }
        memory = {
            "configuration_name": "memory-0805", "kind": "candidate",
            "config": {"tp_size": 1, "mem_fraction_static": 0.805},
            "metrics": {"request_throughput_rps": 105.0}, "stable": True,
            "all_repetitions_slo_passed": True, "screening_accepted": True,
            "comparison": {"improvement_pct": 5.0, "secondary_regressions_passed": True},
        }
        mixed = {
            "configuration_name": "mixed", "kind": "candidate",
            "config": {"tp_size": 1, "enable_mixed_chunk": True},
            "metrics": {"request_throughput_rps": 104.0}, "stable": True,
            "all_repetitions_slo_passed": True, "screening_accepted": True,
            "comparison": {"improvement_pct": 4.0, "secondary_regressions_passed": True},
        }
        screen = {"aggregates": [baseline, memory, mixed], "completed_trials": 3}
        refined = {
            "aggregates": [baseline, {
                **memory,
                "configuration_name": "memory-0900",
                "config": {"tp_size": 1, "mem_fraction_static": 0.9},
                "metrics": {"request_throughput_rps": 112.0},
                "comparison": {"improvement_pct": 12.0, "secondary_regressions_passed": True},
            }],
            "completed_trials": 1,
        }
        combined = autopilot.merge_stage_evidence(screen, refined)
        spec = autopilot.interaction_spec(
            task, discovery, search_plan, combined, 8, 1.0, 30.0,
            phase="composition",
        )
        configs = [item["config"] for item in spec["search"]["explicit_configurations"]]
        self.assertIn(
            {"tp_size": 1, "mem_fraction_static": 0.9, "enable_mixed_chunk": True},
            configs,
        )

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

    def test_operator_escalation_prefers_kernel_family(self):
        plan = autopilot.operator_escalation_plan({
            "tool": {"ncu": {"available": True, "performance_counter_access": False}},
            "diagnosis": {
                "top_kernels": [{"name": "flash_attention", "time_pct": 29.1}],
                "top_kernel_families": [{"name": "gdn_delta_rule", "time_pct": 48.4}],
            },
        })
        self.assertEqual(plan["top_kernel_family"]["name"], "gdn_delta_rule")

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

    def test_hybrid_model_gets_page_size_64_single_parameter_evidence(self):
        task = self.task()
        task["workload"]["input_tokens"] = 4096
        discovery = self.discovery(is_moe=False)
        discovery["model"]["is_hybrid"] = True
        profile = {"diagnosis": {"primary_bottleneck": "mixed_gpu_compute", "shares_pct": {}}}
        plan = autopilot.diagnosed_search_plan(task, discovery, profile)
        page = next(
            item for item in plan["ranked_parameter_groups"]
            if item["parameter"] == "page_size"
        )
        self.assertIn(64, page["values"])

    def test_low_decode_share_defers_mtp_but_not_mamba(self):
        task = self.task()
        discovery = self.discovery(is_moe=False)
        discovery["model"].update({"is_hybrid": True, "has_mtp_weights": True})
        discovery["cookbook"] = {"model_profile": {"initial_bundles": [
            {"name": "mtp", "config": {
                "speculative_algorithm": "EAGLE", "speculative_num_steps": 3,
                "speculative_eagle_topk": 1, "speculative_num_draft_tokens": 4,
            }},
            {"name": "mamba", "config": {
                "mamba_radix_cache_strategy": "extra_buffer", "page_size": 64,
            }},
        ]}}
        for name in (
            "speculative_algorithm", "speculative_num_steps",
            "speculative_eagle_topk", "speculative_num_draft_tokens",
        ):
            discovery["parameter_catalog"]["parameters"].append({
                "dest": name, "family": "speculative", "default": None,
                "choices": None, "deprecated": False,
                "primary_flag": "--" + name.replace("_", "-"), "help": name,
            })
        profile = {
            "diagnosis": {"primary_bottleneck": "attention", "shares_pct": {"attention_kernels": 40}},
            "benchmark": {"metrics": {"mean_e2e_latency_ms": 100, "mean_ttft_ms": 90}},
        }
        plan = autopilot.diagnosed_search_plan(task, discovery, profile)
        self.assertFalse(plan["mtp_relevance"]["relevant"])
        self.assertFalse(any(
            "speculative_algorithm" in bundle["config"]
            for bundle in plan["cookbook_candidate_bundles"]
        ))
        self.assertIn("mamba", autopilot.required_mechanism_classes(task, discovery, plan))
        self.assertNotIn("mtp", autopilot.required_mechanism_classes(task, discovery, plan))

    def test_fp8_kv_candidates_require_explicit_quality_opt_in(self):
        task = self.task()
        discovery = self.discovery(is_moe=False)
        discovery["hardware_profile"] = {"architecture": "blackwell", "precision": ["fp8"]}
        discovery["parameter_catalog"]["parameters"].append({
            "dest": "kv_cache_dtype", "family": "memory_cache", "default": "auto",
            "choices": ["auto", "fp8_e4m3", "fp8_e5m2"], "deprecated": False,
            "primary_flag": "--kv-cache-dtype", "help": "KV cache dtype",
        })
        profile = {"diagnosis": {"primary_bottleneck": "attention", "shares_pct": {"attention_kernels": 40}}}
        disabled = autopilot.diagnosed_search_plan(task, discovery, profile)
        self.assertNotIn("kv_cache_dtype", [
            item["parameter"] for item in disabled["ranked_parameter_groups"]
        ])
        task["quality"] = {"allow_kv_cache_precision_tuning": True}
        enabled = autopilot.diagnosed_search_plan(task, discovery, profile)
        kv = next(item for item in enabled["ranked_parameter_groups"] if item["parameter"] == "kv_cache_dtype")
        self.assertEqual(kv["values"], ["fp8_e4m3", "fp8_e5m2"])

    def test_thorough_online_mode_covers_sensitivity_families(self):
        profile = {"diagnosis": {"primary_bottleneck": "mixed_gpu_compute", "shares_pct": {}}}
        plan = autopilot.diagnosed_search_plan(self.task(), self.discovery(is_moe=False), profile)
        names = [item["parameter"] for item in plan["ranked_parameter_groups"]]
        self.assertNotIn("max_running_requests", names)
        self.assertIn("num_continuous_decode_steps", names)
        self.assertIn("cuda_graph_max_bs_decode", names)
        self.assertIn("page_size", names)
        self.assertIn("sensitivity", next(item for item in plan["ranked_parameter_groups"] if item["parameter"] == "page_size")["tiers"])

    def test_offline_mode_routes_capacity_controls_after_calibration(self):
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


class ParameterEvolutionTests(unittest.TestCase):
    @staticmethod
    def parameter(
        name, *, default=False, action="store_true", value_type=None,
        family="scheduler", help_text="Enable dynamic prefill chunk scheduling",
        choices=None,
    ):
        return {
            "dest": name,
            "flags": ["--" + name.replace("_", "-")],
            "primary_flag": "--" + name.replace("_", "-"),
            "default": default,
            "required": False,
            "nargs": None,
            "choices": choices,
            "value_type": value_type,
            "action": action,
            "help": help_text,
            "deprecated": False,
            "family": family,
            "cli_visible": True,
        }

    def test_contract_diff_tracks_add_remove_change_and_rename_hint(self):
        previous = parameter_evolution.build_parameter_contract({"parameters": [
            self.parameter("old_dynamic_chunk", help_text="Enable dynamic prefill chunk scheduling"),
            self.parameter("stable_flag", default=False),
        ]})
        current = parameter_evolution.build_parameter_contract({"parameters": [
            self.parameter("enable_dynamic_chunking", help_text="Enable dynamic prefill chunk scheduling"),
            self.parameter("stable_flag", default=True),
        ]})
        difference = parameter_evolution.diff_parameter_contract(previous, current)
        self.assertEqual(difference["status"], "changed")
        self.assertEqual(difference["added"], ["enable_dynamic_chunking"])
        self.assertEqual(difference["removed"], ["old_dynamic_chunk"])
        self.assertIn("default", difference["changed"]["stable_flag"])
        self.assertEqual(difference["rename_hints"][0]["removed"], "old_dynamic_chunk")

    def synthetic_analysis(self, previous=True, parameter=None, mode="experimental"):
        parameter = parameter or self.parameter("enable_dynamic_chunking")
        with tempfile.TemporaryDirectory() as root:
            repository = Path(root)
            source = repository / "python/sglang/srt/managers/scheduler.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "def configure(server_args):\n    return server_args.enable_dynamic_chunking\n",
                encoding="utf-8",
            )
            current_catalog = {"parameters": [
                self.parameter("stable_flag"), parameter,
            ]}
            discovery = {
                "parameter_catalog": current_catalog,
                "framework": {"git_commit": "new"},
                "cookbook": {"local_checkout": {
                    "recipes": [{
                        "name": "new-recipe",
                        "config": {},
                        "unrecognized_config": {parameter["dest"]: True},
                    }],
                    "tuning_tips": [],
                }},
                "hardware": {"vendor": "nvidia"},
                "model": {},
                "derived": {"visible_gpu_count": 1},
            }
            task = {
                "repository": str(repository),
                "experiment_mode": "balanced",
                "budget": {"max_trials": 20},
                "parameter_evolution": {
                    "mode": mode, "exploration_budget_pct": 10,
                    "minimum_confidence": 0.8,
                },
            }
            old = parameter_evolution.build_parameter_contract({"parameters": [
                self.parameter("stable_flag"),
            ]}) if previous else None
            return parameter_evolution.analyze_parameter_evolution(
                task, discovery, old, known_parameters={"stable_flag"}
            )

    def test_new_safe_parameter_becomes_bounded_provisional_candidate(self):
        evolution = self.synthetic_analysis()
        candidate = next(
            item for item in evolution["provisional_candidates"]
            if item["parameter"] == "enable_dynamic_chunking"
        )
        self.assertEqual(candidate["state"], "provisional")
        self.assertEqual(candidate["candidate_values"], [True])
        self.assertEqual(candidate["submechanism"], "prefill_chunking")
        self.assertGreaterEqual(candidate["confidence"], 0.8)
        self.assertEqual(evolution["exploration_budget"]["slots"], 1)

    def test_provisional_budget_never_runs_in_fast_or_conservative_mode(self):
        fast = parameter_evolution.provisional_trial_budget({
            "experiment_mode": "fast", "budget": {"max_trials": 48},
            "parameter_evolution": {"mode": "experimental", "exploration_budget_pct": 25},
        }, 20)
        conservative = parameter_evolution.provisional_trial_budget({
            "experiment_mode": "max", "budget": {"max_trials": 48},
            "parameter_evolution": {"mode": "conservative", "exploration_budget_pct": 25},
        }, 20)
        maximum = parameter_evolution.provisional_trial_budget({
            "experiment_mode": "max", "budget": {"max_trials": 48},
            "parameter_evolution": {"mode": "experimental", "exploration_budget_pct": 25},
        }, 20)
        self.assertEqual(fast["slots"], 0)
        self.assertEqual(conservative["slots"], 0)
        self.assertEqual(maximum["slots"], 6)

    def test_first_contract_is_baseline_not_mass_provisional_search(self):
        evolution = self.synthetic_analysis(previous=False)
        self.assertEqual(evolution["contract_diff"]["status"], "first_observation")
        self.assertEqual(evolution["provisional_candidates"], [])
        self.assertEqual(evolution["exploration_budget"]["slots"], 0)

    def test_new_choice_on_known_parameter_gets_provisional_exploration(self):
        with tempfile.TemporaryDirectory() as root:
            old = self.parameter(
                "attention_backend", default="a", action="_StoreAction",
                value_type="str", family="kernel_backend",
                help_text="Attention backend", choices=["a", "b"],
            )
            new = {**old, "choices": ["a", "b", "fa_new"]}
            previous = parameter_evolution.build_parameter_contract({"parameters": [old]})
            discovery = {
                "parameter_catalog": {"parameters": [new]},
                "framework": {"git_commit": "new"}, "cookbook": {},
                "hardware": {"vendor": "nvidia"}, "model": {},
                "derived": {"visible_gpu_count": 1},
            }
            task = {
                "repository": root, "experiment_mode": "balanced",
                "budget": {"max_trials": 20},
                "parameter_evolution": {"mode": "experimental"},
            }
            evolution = parameter_evolution.analyze_parameter_evolution(
                task, discovery, previous, known_parameters={"attention_backend"}
            )
        candidate = evolution["provisional_candidates"][0]
        self.assertEqual(candidate["parameter"], "attention_backend")
        self.assertEqual(candidate["candidate_values"], ["fa_new"])
        self.assertEqual(candidate["provisional_kind"], "choice_extension")

    def test_new_precision_choice_never_enters_generic_provisional_search(self):
        with tempfile.TemporaryDirectory() as root:
            old = self.parameter(
                "kv_cache_dtype", default="auto", action="_StoreAction",
                value_type="str", family="memory_cache",
                help_text="KV cache dtype and precision", choices=["auto"],
            )
            new = {**old, "choices": ["auto", "fp8_new"]}
            previous = parameter_evolution.build_parameter_contract({"parameters": [old]})
            discovery = {
                "parameter_catalog": {"parameters": [new]},
                "framework": {"git_commit": "new"}, "cookbook": {},
                "hardware": {"vendor": "nvidia"}, "model": {},
                "derived": {"visible_gpu_count": 1},
            }
            task = {
                "repository": root, "experiment_mode": "max",
                "budget": {"max_trials": 48},
                "parameter_evolution": {"mode": "experimental"},
            }
            evolution = parameter_evolution.analyze_parameter_evolution(
                task, discovery, previous, known_parameters={"kv_cache_dtype"}
            )
        self.assertEqual(evolution["provisional_candidates"], [])

    def test_control_plane_and_unbounded_numeric_parameters_never_execute(self):
        control = self.parameter(
            "admin_port", default=30001, action="_StoreAction", value_type="int",
            family="other", help_text="Admin API port between 1024 and 65535",
        )
        control_evolution = self.synthetic_analysis(parameter=control)
        control_item = next(
            item for item in control_evolution["parameters"]
            if item["parameter"] == "admin_port"
        )
        self.assertEqual(control_item["state"], "control_plane")
        numeric = self.parameter(
            "scheduler_magic", default=5, action="_StoreAction", value_type="int",
            help_text="Scheduler performance tuning value",
        )
        numeric_evolution = self.synthetic_analysis(parameter=numeric)
        numeric_item = next(
            item for item in numeric_evolution["parameters"]
            if item["parameter"] == "scheduler_magic"
        )
        self.assertEqual(numeric_item["state"], "unclassified")
        self.assertEqual(numeric_item["value_strategy"]["strategy"], "numeric_range_not_declared")

    def test_cookbook_hardware_requirement_blocks_wrong_vendor(self):
        with tempfile.TemporaryDirectory() as root:
            item = self.parameter("enable_dynamic_chunking")
            analysis = parameter_evolution.infer_parameter_semantics(
                item, root,
                {"local_checkout": {"recipes": [{
                    "name": "amd-only", "config": {},
                    "unrecognized_config": {"enable_dynamic_chunking": True},
                    "requirements": ["amd_gpu"],
                }], "tuning_tips": []}},
                known_parameters=set(), explicitly_added=True,
                indexed_source_paths=["python/sglang/srt/scheduler.py"],
                discovery={"hardware": {"vendor": "nvidia"}, "model": {}},
            )
        self.assertEqual(analysis["state"], "inapplicable")
        self.assertIn("requires AMD", analysis["reason"])

    def test_current_cookbook_scalar_can_bound_new_numeric_parameter(self):
        numeric = self.parameter(
            "scheduler_magic", default=5, action="_StoreAction", value_type="int",
            help_text="Scheduler performance tuning value",
        )
        with tempfile.TemporaryDirectory() as root:
            repository = Path(root)
            source = repository / "python/sglang/srt/managers/scheduler.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "def f(server_args): return server_args.scheduler_magic\n",
                encoding="utf-8",
            )
            discovery = {
                "parameter_catalog": {"parameters": [self.parameter("stable_flag"), numeric]},
                "framework": {"git_commit": "new"},
                "cookbook": {"local_checkout": {
                    "recipes": [{
                        "name": "new-value", "config": {},
                        "unrecognized_config": {"scheduler_magic": 8},
                    }], "tuning_tips": [],
                }},
                "hardware": {"vendor": "nvidia"}, "model": {},
                "derived": {"visible_gpu_count": 1},
            }
            task = {
                "repository": str(repository), "experiment_mode": "balanced",
                "budget": {"max_trials": 20},
                "parameter_evolution": {"mode": "experimental"},
            }
            previous = parameter_evolution.build_parameter_contract({
                "parameters": [self.parameter("stable_flag")]
            })
            evolution = parameter_evolution.analyze_parameter_evolution(
                task, discovery, previous, known_parameters={"stable_flag"}
            )
        item = next(
            value for value in evolution["parameters"]
            if value["parameter"] == "scheduler_magic"
        )
        self.assertEqual(item["candidate_values"], [8])
        self.assertEqual(
            item["value_strategy"]["strategy"],
            "current_cookbook_documented_scalars",
        )

    def test_cookbook_unknown_flag_is_evidence_only(self):
        command = (
            "sglang serve --model-path /tmp/model --enable-dynamic-chunking "
            "--chunked-prefill-size 4096"
        )
        trusted = autopilot.cookbook_command_config(command)
        complete = autopilot.cookbook_command_config(command, include_unrecognized=True)
        self.assertNotIn("enable_dynamic_chunking", trusted)
        self.assertTrue(complete["enable_dynamic_chunking"])
        self.assertEqual(complete["chunked_prefill_size"], 4096)
        evidence = parameter_evolution.cookbook_parameter_evidence({
            "local_checkout": {"recipes": [{
                "name": "atomic", "config": {"chunked_prefill_size": 4096},
                "unrecognized_config": {"enable_dynamic_chunking": True},
            }], "tuning_tips": []},
        }, "enable_dynamic_chunking")
        self.assertEqual(
            evidence[0]["companion_config"], {"chunked_prefill_size": 4096}
        )

    def test_static_catalog_fallback_extracts_literal_server_args(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "python/sglang/srt/server_args.py"
            source.parent.mkdir(parents=True)
            source.write_text(
                "class ServerArgs:\n"
                "    @staticmethod\n"
                "    def add_cli_args(parser):\n"
                "        parser.add_argument('--enable-dynamic-chunking', action='store_true', default=False, help='Enable dynamic chunk')\n"
                "        parser.add_argument('--new-backend', choices=['a','b'], default='a', help='Attention backend')\n",
                encoding="utf-8",
            )
            catalog = sglang_catalog.export_catalog_static(Path(root))
        self.assertEqual(catalog["extraction_mode"], "static_ast_fallback")
        self.assertEqual(catalog["parameter_count"], 2)
        by_name = {item["dest"]: item for item in catalog["parameters"]}
        self.assertEqual(by_name["new_backend"]["choices"], ["a", "b"])

    def test_runtime_catalog_uses_task_python_in_a_fresh_process(self):
        with tempfile.TemporaryDirectory() as root:
            repository = Path(root)
            package = repository / "python/sglang/srt"
            package.mkdir(parents=True)
            for init in (
                repository / "python/sglang/__init__.py",
                package / "__init__.py",
            ):
                init.write_text("", encoding="utf-8")
            (package / "server_args.py").write_text(
                "class ServerArgs:\n"
                "    @staticmethod\n"
                "    def add_cli_args(parser):\n"
                "        parser.add_argument('--fresh-parameter', action='store_true', default=False, help='Fresh process flag')\n",
                encoding="utf-8",
            )
            catalog = autopilot.parameter_catalog({
                "python": sys.executable, "repository": str(repository), "env": {},
            })
        self.assertEqual(catalog["extraction_mode"], "runtime_argparse")
        self.assertEqual(catalog["parameters"][0]["dest"], "fresh_parameter")

    def test_screening_reserves_provisional_slot_without_confirmation_budget(self):
        task = {
            "name": "evolution-screen", "repository": "/tmp", "python": sys.executable,
            "model_path": "/tmp", "output_dir": "/tmp/runs",
            "deployment_mode": "online_latency", "experiment_mode": "balanced",
            "confirmation_repetitions": 2, "parallel_trials": 1, "max_gpus": 1,
            "budget": {"max_trials": 14, "max_gpu_hours": 1, "max_wall_time_minutes": 30},
            "workload": {
                "input_tokens": 1024, "output_tokens": 64, "max_concurrency": 4,
                "num_prompts": 40, "request_rate": "inf",
            },
            "slo": {},
            "objective": {"metric": "request_throughput_rps", "direction": "maximize"},
        }
        parameters = [
            {**self.parameter(
                "mem_fraction_static", default=0.8, action="_StoreAction",
                value_type="float", family="memory_cache",
            )},
            self.parameter("enable_dynamic_chunking"),
        ]
        discovery = {
            "derived": {"minimum_tp_size": 1, "visible_gpu_count": 1},
            "model": {"is_moe": False},
            "hardware": {"vendor": "nvidia", "gpus": [{
                "index": 0, "name": "Synthetic GPU", "memory_mib": 81920,
            }]},
            "parameter_catalog": {"parameters": parameters},
        }
        search_plan = {
            "budget_allocation": optimization_rules.tiered_trial_budget(14),
            "ranked_parameter_groups": [
                {
                    "parameter": "mem_fraction_static", "family": "memory_cache",
                    "values": [0.82], "trigger_magnitude": "high",
                    "trigger": {"rule_ids": ["known"]},
                },
                {
                    "parameter": "enable_dynamic_chunking", "family": "scheduler",
                    "submechanism": "prefill_chunking", "values": [True],
                    "trigger_magnitude": "medium", "provisional": True,
                    "parameter_evolution": {
                        "state": "provisional",
                        "atomic_config": {"mem_fraction_static": 0.82},
                    },
                },
            ],
            "trigger_rule_plan": {"matches": [], "strong_candidates": []},
            "parameter_evolution": {"exploration_budget": {"slots": 1}},
            "resolved_baseline": {"mem_fraction_static": 0.8},
        }
        spec = autopilot.screening_spec(
            task, discovery, search_plan, remaining_trials=12
        )
        names = [item["name"] for item in spec["search"]["explicit_configurations"]]
        self.assertIn("provisional-enable_dynamic_chunking-true", names)
        self.assertEqual(
            spec["search"]["provisional_exploration_budget"]["selected_slots"], 1
        )
        self.assertEqual(
            spec["search"]["provisional_parameter_names"], ["enable_dynamic_chunking"]
        )
        provisional_config = next(
            item["config"] for item in spec["search"]["explicit_configurations"]
            if item["name"] == "provisional-enable_dynamic_chunking-true"
        )
        self.assertEqual(provisional_config["mem_fraction_static"], 0.82)
        provisional_position = names.index(
            "provisional-enable_dynamic_chunking-true"
        ) + 1
        self.assertGreaterEqual(
            spec["search"]["min_successful_candidates_before_early_stop"],
            provisional_position,
        )

    def test_provisional_failure_has_parameter_scoped_circuit_breaker(self):
        trial = {
            "name": "provisional-enable_dynamic_chunking-true",
            "configuration_name": "provisional-enable_dynamic_chunking-true",
            "config": {"enable_dynamic_chunking": True},
            "provisional_parameter": "enable_dynamic_chunking",
        }
        self.assertEqual(
            autotune.capability_family(trial),
            "provisional_parameter:enable_dynamic_chunking",
        )
        with tempfile.TemporaryDirectory() as root:
            reason = autotune.capability_failure_reason(
                trial, {"failure_class": "runtime", "detail": "smoke failed"},
                Path(root) / "server.log",
            )
        self.assertIsNotNone(reason)
        self.assertEqual(reason["failure_class"], "runtime")

        matrix = autotune.candidate_matrix({
            "budget": {"max_trials": 2},
            "search": {
                "strategy": "explicit_configurations", "include_baseline": False,
                "baseline": {}, "repetitions": 1,
                "explicit_configurations": [{
                    "name": "provisional-attention_backend-fa-new",
                    "config": {"attention_backend": "fa-new"},
                    "provisional_parameter": "attention_backend",
                    "provisional_state": "provisional",
                }],
            },
        })
        self.assertEqual(matrix[0]["provisional_parameter"], "attention_backend")
        matrix[0]["configuration_name"] = matrix[0]["name"]
        self.assertEqual(
            autotune.capability_family(matrix[0]),
            "provisional_parameter:attention_backend",
        )

    def test_provisional_smoke_is_bounded_and_uses_the_resident_server_command(self):
        benchmark = [
            sys.executable, "-m", "sglang.benchmark.serving",
            "--num-prompts", "225", "--warmup-requests", "8",
            "--output-file", "/tmp/full.jsonl",
        ]
        trial = {
            "configuration_name": "provisional-enable_dynamic_chunking-true",
        }
        spec = {
            "hardware": {"gpus_per_host": 1},
            "search": {"provisional_parameter_candidates": [
                "provisional-enable_dynamic_chunking-true"
            ]},
        }
        plan = autotune.provisional_smoke_plan(spec, trial, benchmark)
        self.assertEqual(plan["effective_num_prompts"], 4)
        self.assertEqual(plan["full_benchmark_num_prompts"], 225)
        self.assertEqual(
            plan["command"][plan["command"].index("--warmup-requests") + 1], "2"
        )
        self.assertIsNone(autotune.provisional_smoke_plan(
            {"search": {"provisional_parameter_candidates": []}}, trial, benchmark
        ))

    def test_contract_and_provisional_evidence_persist_in_sqlite(self):
        with tempfile.TemporaryDirectory() as root:
            database = Path(root) / "history.sqlite3"
            first = parameter_evolution.build_parameter_contract({"parameters": [
                self.parameter("stable_flag"),
            ]}, {"git_commit": "old"})
            second = parameter_evolution.build_parameter_contract({"parameters": [
                self.parameter("stable_flag"), self.parameter("enable_dynamic_chunking"),
            ]}, {"git_commit": "new"})
            trial_store.save_parameter_contract(database, first, "framework-old")
            trial_store.save_parameter_contract(database, second, "framework-new")
            previous = trial_store.latest_parameter_contract(
                database, exclude_contract_hash=second["contract_hash"]
            )
            self.assertEqual(previous["contract_hash"], first["contract_hash"])

            task = {
                "workload": {"input_tokens": 8, "output_tokens": 4},
                "deployment_mode": "online_latency", "objective": {
                    "metric": "request_throughput_rps", "direction": "maximize",
                }, "slo": {},
            }
            discovery = {
                "model": {}, "hardware": {"vendor": "nvidia", "gpus": []},
                "framework": {"git_commit": "new"}, "topology_class": "single-gpu",
                "parameter_catalog": {"parameter_contract": {}},
                "parameter_evolution": {"provisional_candidates": [{
                    "parameter": "enable_dynamic_chunking",
                }]},
            }
            final = {
                "run_dir": "/private/run-evolution", "completed_at": "now",
                "recommendation_status": "confirmed_candidate", "discovery": discovery,
                "raw_sglang_baseline": {"tp_size": 1},
                "screening": {
                    "aggregates": [{
                        "configuration_name": "provisional-enable_dynamic_chunking-true",
                        "comparison": {"improvement_pct": 4.0},
                    }],
                    "results": [{
                        "configuration_name": "provisional-enable_dynamic_chunking-true",
                        "name": "provisional-enable_dynamic_chunking-true", "ok": True,
                        "config": {"tp_size": 1, "enable_dynamic_chunking": True},
                        "metrics": {"request_throughput_rps": 104.0},
                        "slo": {"passed": True}, "status": {
                            "provisional_smoke": {"passed": True},
                        },
                    }],
                },
            }
            trial_store.ingest_final(database, final, task)
            evidence = trial_store.ingest_parameter_evidence(database, final, task)
            self.assertEqual(evidence["inserted_parameter_evidence"], 1)
            _, components = trial_store.compatibility_fingerprint(task, discovery)
            priors = trial_store.parameter_evidence_priors(
                database, components["framework_fingerprint"],
                {"enable_dynamic_chunking"},
            )
            self.assertEqual(
                priors["enable_dynamic_chunking"][0]["median_improvement_pct"], 4.0
            )
            mismatched = trial_store.parameter_evidence_priors(
                database, components["framework_fingerprint"],
                {"enable_dynamic_chunking"},
                compatibility_components={
                    **components, "hardware_fingerprint": "different-hardware",
                },
            )
            self.assertEqual(mismatched, {})

    def test_history_schema_migrates_existing_v1_database_in_place(self):
        import sqlite3

        with tempfile.TemporaryDirectory() as root:
            database = Path(root) / "history.sqlite3"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO metadata VALUES('schema_version', '1');
                CREATE TABLE runs(
                  run_dir TEXT PRIMARY KEY, compatibility_fingerprint TEXT NOT NULL,
                  completed_at TEXT, recommendation_status TEXT, objective_metric TEXT NOT NULL,
                  model_fingerprint TEXT NOT NULL, hardware_fingerprint TEXT NOT NULL,
                  workload_fingerprint TEXT NOT NULL, framework_fingerprint TEXT NOT NULL
                );
                CREATE TABLE trials(
                  id INTEGER PRIMARY KEY AUTOINCREMENT, run_dir TEXT NOT NULL,
                  stage TEXT NOT NULL, configuration_name TEXT NOT NULL,
                  config_hash TEXT NOT NULL, config_json TEXT NOT NULL,
                  objective_value REAL, improvement_pct REAL, slo_passed INTEGER,
                  ok INTEGER NOT NULL, metrics_json TEXT
                );
                """
            )
            connection.commit()
            connection.close()
            migrated = trial_store.open_store(database)
            tables = {
                row[0] for row in migrated.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            version = migrated.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()[0]
            migrated.close()
        self.assertIn("parameter_contracts", tables)
        self.assertIn("parameter_evidence", tables)
        self.assertEqual(version, "2")

    def test_report_exposes_contract_diff_and_exploration_budget(self):
        report = inferopt_cli.markdown_report({
            "recommendation_status": "insufficient_parameter_evidence",
            "deployable": False,
            "parameter_evolution": {
                "contract_hash": "abc", "diff_status": "changed",
                "added": ["enable_dynamic_chunking"], "removed": [],
                "changed_parameters": [], "state_counts": {"provisional": 1},
                "policy": {"mode": "experimental"},
                "exploration_budget": {"slots": 1},
                "selected_for_exploration": ["enable_dynamic_chunking"],
            },
        })
        self.assertIn("## SGLang Parameter Evolution", report)
        self.assertIn("enable_dynamic_chunking", report)
        self.assertIn("Confirmation budget is never reduced", report)


if __name__ == "__main__":
    unittest.main()
