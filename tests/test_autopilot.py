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

    def test_offline_reference_window_requires_runtime_capacity_and_five_waves(self):
        task = self.valid_task()
        task.update({"deployment_mode": "offline_throughput", "slo": {}})
        spec = {"benchmark": {
            "dataset_name": "random-ids", "num_prompts": 40,
            "min_measurement_seconds": 5,
        }}
        with self.assertRaisesRegex(ValueError, "observed_admission_capacity"):
            autopilot.configure_offline_reference_window(spec, task)
        task["workload"]["observed_admission_capacity"] = 50
        autopilot.configure_offline_reference_window(spec, task)
        self.assertEqual(spec["benchmark"]["num_prompts"], 250)
        self.assertEqual(spec["benchmark"]["saturation_waves"], 5)

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
        self.assertEqual(chunk["values"], [4096, 2048, 1024, 512])
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
        self.assertEqual(
            [item["name"] for item in spec["search"]["explicit_configurations"]],
            [
                "num_continuous_decode_steps-2",
                "moe_runner_backend-deep_gemm",
                "disable_radix_cache-true",
                "num_continuous_decode_steps-4",
            ],
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


if __name__ == "__main__":
    unittest.main()
