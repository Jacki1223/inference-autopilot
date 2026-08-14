# Task Specification

Use JSON for the first implementation. Unknown keys are allowed for engine-specific extensions, but the fields below define the stable contract.

```json
{
  "name": "qwen-serving-h100",
  "mode": "dry_run",
  "framework": "sglang",
  "repository": "/absolute/path/to/sglang",
  "model": {
    "path": "Qwen/Qwen3-32B",
    "revision": "pinned-revision",
    "dtype": "fp8",
    "architecture": "dense",
    "context_length": 32768
  },
  "hardware": {
    "hosts": 1,
    "gpus_per_host": 8,
    "gpu_model": "H100-SXM-80GB",
    "interconnect": "NVLink",
    "network": "none"
  },
  "workload": {
    "dataset": "/absolute/path/to/trace.jsonl",
    "arrival": "poisson",
    "request_rate": 20,
    "max_concurrency": 128,
    "num_prompts": 1000,
    "input_tokens": {"p50": 1024, "p95": 8192},
    "output_tokens": {"p50": 256, "p95": 1024},
    "prefix_reuse_ratio": 0.4,
    "structured_output_ratio": 0.2,
    "multimodal_ratio": 0.0
  },
  "slo": {
    "p99_ttft_ms": 1500,
    "p99_tpot_ms": 60,
    "min_request_throughput_rps": 18,
    "max_error_rate": 0.001
  },
  "objective": {
    "metric": "request_goodput_rps",
    "direction": "maximize",
    "min_improvement_pct": 5,
    "max_regression_pct": 2,
    "goodput_slo": {
      "max_ttft_ms": 1500,
      "max_tpot_ms": 60
    }
  },
  "budget": {
    "max_trials": 30,
    "max_gpu_hours": 16,
    "max_wall_time_minutes": 360,
    "max_consecutive_failures": 3
  },
  "scope": {
    "allow_launch": false,
    "allow_profiling": false,
    "allow_parameter_changes": true,
    "allow_code_changes": false,
    "allow_kernel_changes": false,
    "production": false,
    "output_dir": "/absolute/path/to/private-runs"
  }
}
```

## Required Fields

- `name`: stable experiment name.
- `mode`: `dry_run`, `shadow`, or `execute`.
- `framework`: serving engine name and version when known.
- `model.path`: local path or model identifier.
- `workload`: trace or statistically meaningful synthetic distribution.
- `slo`: at least one measurable hard constraint.
- `objective.metric`: one primary optimization metric.
- `objective.goodput_slo`: required when optimizing `request_goodput_rps`; requires detailed per-request benchmark output.
- `budget`: bounded trials and time.
- `scope.output_dir`: private artifact directory.

## Modes

- `dry_run`: inspect, validate, inventory, plan, and analyze existing results only.
- `shadow`: launch an isolated candidate deployment and replay non-production or mirrored traffic; never route user traffic.
- `execute`: run authorized experiments in the specified test environment.

## Metric Names

Supported comparison metrics include:

- `request_throughput_rps`
- `output_throughput_tps`
- `total_throughput_tps`
- `request_goodput_rps`
- `mean_ttft_ms`, `median_ttft_ms`, `p99_ttft_ms`
- `mean_tpot_ms`, `median_tpot_ms`, `p99_tpot_ms`
- `mean_itl_ms`, `median_itl_ms`, `p99_itl_ms`
- `mean_e2e_latency_ms`, `p99_e2e_latency_ms`
- `error_rate`

The analyzer maps common SGLang benchmark field names onto these canonical names.
To compute `request_goodput_rps`, run SGLang with detailed output enabled so the JSONL includes `duration`, `ttfts`, `itls`, and `errors`. Treat detailed records as sensitive because they may also contain generated text.
