<p align="center">
  <img src="assets/inference-autopilot-logo.svg" alt="Inference Autopilot" width="900">
</p>

<h1 align="center">Inference Autopilot</h1>

<p align="center">
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/English-Current-2563eb?style=for-the-badge"></a>
  <a href="README_zh.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-切换-7dd3fc?style=for-the-badge"></a>
</p>

<p align="center">
  <strong>Find a better SGLang deployment configuration on your own hardware, for your own workload.</strong>
</p>

<p align="center">
  <a href="https://github.com/Jacki1223/inference-autopilot/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/Jacki1223/inference-autopilot?color=4f46e5&label=release"></a>
  <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-2563eb">
  <a href="https://github.com/Jacki1223/inference-autopilot/actions/workflows/sglang-parameter-compat.yml"><img alt="SGLang parameter compatibility" src="https://github.com/Jacki1223/inference-autopilot/actions/workflows/sglang-parameter-compat.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-1e3a8a"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#how-it-works">How It Works</a> ·
  <a href="#results-and-artifacts">Results</a> ·
  <a href="https://github.com/Jacki1223/inference-autopilot/releases/latest">Download</a>
</p>

Inference Autopilot (`inferopt`) is a single-host (for now) optimization CLI for [SGLang](https://github.com/sgl-project/sglang). Give it a model, a GPU host, a representative workload, optional latency SLOs, and an experiment budget. It validates the deployment, benchmarks relevant configurations, diagnoses bottlenecks, and returns a reproducible launch command backed by measured evidence.

The result is deliberately bounded: it is the best configuration found for the recorded model, SGLang version, hardware, workload, and budget—not a claim of a universal optimum.

## What You Get

- A feasibility check before expensive GPU experiments begin.
- A measured comparison between the SGLang baseline and promising candidates.
- A recommended launch command that uses flags supported by the selected SGLang installation.
- A Markdown report explaining the bottleneck, tested changes, rejected candidates, SLO results, and final decision.
- Structured artifacts for reproducing or auditing the run.

Keeping the baseline is a valid outcome when no candidate produces a reliable improvement.

## How It Works

1. **Discover** — inspect the visible GPUs, model metadata, SGLang arguments, and deployment constraints.
2. **Measure** — warm up the service and establish a baseline with the requested workload and SLOs.
3. **Optimize** — profile the serving path and test a bounded set of applicable scheduling, cache, graph, backend, speculative decoding, and parallelism choices.
4. **Confirm** — remeasure the best candidate against the baseline and reject noisy, incorrect, or SLO-violating results.
5. **Report** — write the recommended command, comparison metrics, evidence, and artifacts.

Candidate selection is version- and workload-aware. InferOpt reads the actual SGLang argument surface instead of relying on a fixed list of flags, and it only spends the experiment budget on configurations that are applicable to the current run.

## Requirements

- Python 3.9 or newer.
- A local SGLang checkout or installation runnable by the selected Python interpreter.
- A locally available, SGLang-compatible model.
- NVIDIA GPUs for automatic tuning and profiling.
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`) available on `PATH`.

AMD hardware inventory and planning are supported, but automatic AMD profiling and tuning are not yet implemented. Nsight Compute is optional and is only needed for deeper, explicitly requested kernel analysis.

## Install

Install directly from GitHub without changing the existing SGLang, CUDA, PyTorch, or model environment:

```bash
python3 -m pip install \
  --no-deps \
  --no-build-isolation \
  "git+https://github.com/Jacki1223/inference-autopilot.git"
```

To replace an older installation, add `--force-reinstall`. For development from a source checkout, run `python3 -m pip install .`.

## Quick Start

Create a task interactively:

```bash
inferopt init --output task.json
```

Interactive prompts use the following conventions:

- A value in square brackets is the default. Press **Enter** to accept it; do not type the brackets. For example, `[balanced]` selects `balanced` when you press Enter.
- A prompt without a bracketed value requires input unless it explicitly says that a blank value is allowed.
- For `yes/no` prompts, enter `yes` or `no`; pressing Enter accepts the bracketed choice.
- Follow the format shown for lists. GPU indexes use commas with no spaces (for example, `0,1,2`), while concurrency points accept commas or spaces.
- Paths refer to files and directories on the GPU host where InferOpt will run.

Inspect the environment and generated experiment plan before starting GPU work:

```bash
inferopt doctor --task task.json --output doctor.json
inferopt plan --task task.json --output plan.json
```

After reviewing the plan, run the experiment and render the report:

```bash
inferopt run --task task.json --yes --output final.json
inferopt report --result final.json --output report.md
```

`doctor` and `plan` are read-only and do not start a model server. `run --yes` starts only the processes created for the current experiment and shows live progress for calibration, profiling, candidate trials, and confirmation.

## Configure a Run

The interactive `init` command asks for the model and SGLang paths, GPU selection, workload shape or dataset, objective, SLOs, and budget.

### Deployment objective

- `online_latency` optimizes latency and SLO-safe serving capacity.
- `offline_throughput` maximizes throughput, optionally under latency or error-rate constraints.

### Workload

InferOpt can benchmark fixed-shape synthetic requests, generated shared-prefix traffic, local custom JSONL conversations, or ShareGPT-format data. Real datasets stay local. Use traffic that resembles production; a recommendation is only as representative as the workload used to measure it.

### SLOs

Latency limits can use either `p99` or `avg` consistently across end-to-end latency, time to first token (TTFT), and time per output token (TPOT/ITL). Error-rate and throughput constraints are also supported. Leave latency limits unset for objective-only tuning.

### Experiment budget

- `fast` provides a narrow first pass.
- `balanced` is the default and covers the main applicable mechanisms.
- `max` explores more candidates and combinations.

All modes use the same correctness, SLO, and confirmation gates. The intensity changes search breadth, not the standard required for a recommendation. Trial count, GPU-hour, wall-time, and concurrent GPU use can also be capped explicitly.

For automation, start from [`assets/task.autopilot.example.json`](assets/task.autopilot.example.json) or generate a task once with `init`, then validate it before execution:

```bash
inferopt validate --task task.json
inferopt run --task task.json --yes --output final.json
```

## Results and Artifacts

The output directory contains:

- `final.json` — machine-readable decision, metrics, selected configuration, and deployment command.
- `report.md` — human-readable findings and recommendation.
- The resolved task and SGLang launch arguments.
- Benchmark outputs, server logs, profiler evidence, and rejected-trial reasons.

Run artifacts may contain model paths, workload details, and generated text. Keep the output directory private; generated artifacts are ignored by Git by default.

## Safety and Scope

Inference Autopilot is intended for authorized experiments on a single GPU host. It does not install packages at runtime, modify drivers or CUDA, edit SGLang or model weights, change kernels, deploy to production, or kill processes it did not create. Precision-changing candidates are opt-in and require a separate quality evaluation before deployment.

Multi-node search, production rollout orchestration, automatic operator changes, and full multimodal workload optimization are outside the current scope. See the [roadmap](ROADMAP.md) for planned work.

## Agent-Assisted Use

The CLI is fully standalone; no agent or Codex session is required. Environments that support skills can use [`SKILL.md`](SKILL.md) as an orchestration guide for collecting inputs, reviewing the plan, monitoring the run, and explaining the evidence. The CLI and its artifacts remain the source of performance decisions.

## Documentation

- [`SKILL.md`](SKILL.md) — end-to-end operational workflow.
- [`references/input-schema.md`](references/input-schema.md) — task fields and metrics.
- [`references/execution-schema.md`](references/execution-schema.md) — execution and artifact contracts.
- [`references/safety-policy.md`](references/safety-policy.md) — safety boundaries.
- [`references/sglang-adapter.md`](references/sglang-adapter.md) — SGLang integration details.
- [`ROADMAP.md`](ROADMAP.md) — current direction and planned scope.

## License

Licensed under the [Apache License 2.0](LICENSE).
