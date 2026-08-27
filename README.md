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

- Feasibility checks before expensive GPU experiments begin.
- Measured comparisons between the SGLang baseline and workload-specific candidates.
- A statistically confirmed recommendation, or an explicit decision to retain the baseline.
- A directly copyable launch command using flags supported by the installed SGLang version.
- Structured benchmark, log, profiler, and decision artifacts for audit and reuse.

## How It Works

1. **Understand the deployment** — inspect GPU memory and topology, checkpoint metadata, SGLang capabilities, official Cookbook evidence, workload shape, prefix locality, deployment mode, and SLOs. InferOpt reads the current `ServerArgs` contract on every run and rejects infeasible or incompatible configurations before spending GPU time.
2. **Establish a trustworthy baseline** — launch SGLang, warm the service, discover practical request capacity, and measure a steady-state window sized from concurrency or runtime capacity. Throughput, E2E latency, TTFT, TPOT/ITL, error rate, memory headroom, and SLO results are recorded together.
3. **Profile and diagnose** — capture a bounded serving-only Nsight Systems trace and combine it with SGLang startup/scheduler logs, cache and queue telemetry, CUDA Graph coverage, model structure, and workload evidence. Raw observations are reduced to canonical bottleneck classes that can safely activate optimization rules.
4. **Search by mechanism** — match the `(hardware, model, workload, SLO, bottleneck)` context against versioned rules, compatible Cookbook recipes, model-native MTP/Mamba features, and the live parameter contract. Search covers distinct mechanisms first, then refines promising values and compatible combinations instead of running a blind Cartesian grid.
5. **Adapt to results** — positive mechanisms receive value refinement; a failed backend can promote a compatible sibling; optional unavailable mechanisms are reported without invalidating unrelated confirmed gains. Startup-only controls such as SGLang 0.5.18+ Weight Cache are analyzed separately from steady-state throughput parameters and are never enabled when their daemon, topology, speculative-decoding, or capacity-pin constraints are unsafe.
6. **Confirm and report** — compare the best candidate with the baseline using resident ABBA windows, SLO and stability gates, confidence intervals, and Bayesian sequential evidence. Clear outcomes stop early; ambiguous outcomes consume reserved complete A/B pairs. The report separates the confirmed winner, the best unconfirmed candidate, and the safe baseline, then emits a complete copy-paste deployment command.

## Requirements

- Python 3.9 or newer.
- A local SGLang checkout or installation runnable by the selected Python interpreter.
- A locally available, SGLang-compatible model.
- NVIDIA GPUs for automatic tuning and profiling.
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`) available on `PATH`.

AMD hardware inventory and planning are supported, but automatic AMD profiling and tuning are not yet implemented. Nsight Compute is optional and requires GPU performance-counter permission for Roofline or kernel analysis.

## Install

Install or update directly from GitHub without changing the existing SGLang, CUDA, PyTorch, or model environment:

```bash
python3 -m pip install \
  --no-deps \
  --no-build-isolation \
  --force-reinstall \
  "git+https://github.com/Jacki1223/inference-autopilot.git"
```

For development from a source checkout, run `python3 -m pip install .`.

## Quick Start

Create a task interactively:

```bash
inferopt init --output task.json
```

Values in square brackets are defaults; press **Enter** to accept them. GPU indexes use commas without spaces, for example `0,1,2`. Paths refer to the GPU host where InferOpt runs.

Inspect the environment and generated plan before starting GPU work:

```bash
inferopt doctor --task task.json --output doctor.json
inferopt plan --task task.json --output plan.json
```

Run the experiment and render the report:

```bash
inferopt run --task task.json --yes --output final.json
inferopt report --result final.json --output report.md
```

`doctor` and `plan` are read-only. `run --yes` starts only processes owned by the current experiment and shows stage, candidate, GPU-worker, benchmark, and confirmation progress.

## Configure a Run

### Deployment objective

- `online_latency` optimizes SLO-safe latency and serving capacity.
- `offline_throughput` maximizes throughput, optionally under latency or error-rate constraints.

### Workload

InferOpt supports fixed-shape synthetic traffic, generated shared-prefix traffic, custom JSONL conversations, and ShareGPT-format data. Use traffic representative of production; recommendations are specific to the measured workload.

### SLOs

Latency limits use either `p99` or `avg` consistently across E2E latency, TTFT, and TPOT/ITL. Leave them unset for objective-only tuning.

### Experiment intensity

- `fast` is a narrow first pass. It covers three high-impact mechanisms, uses three saturated capacity waves per window, and may stop after a strong measured gain.
- `balanced` is the default. It covers more mechanisms, uses five saturated waves, and adds adaptive refinement and compatible combinations.
- `max` provides the widest bounded search and disables the strong-gain early stop.

All modes use the same correctness, SLO, and statistical acceptance gates. Intensity changes search breadth and measurement cost, not the evidence required to authorize a changed deployment command.

Trial budget is adaptive rather than a fixed percentage split. Discovery and refinement remain bounded, while confirmation reserves the minimum Bayesian blocks plus one complete A/B pair so an unresolved posterior can continue. Unused confirmation trials remain unspent when the winner is already clear.

Offline no-SLO runs omit client `--max-concurrency`, discover practical capacity from the loaded server, and derive request counts from that capacity. Fast uses three waves per window; balanced and max use five.

For non-interactive use, begin with [`assets/task.autopilot.example.json`](assets/task.autopilot.example.json):

```bash
cp assets/task.autopilot.example.json task.json
inferopt validate --task task.json
inferopt run --task task.json --yes --output final.json
```

## Results and Artifacts

The output directory contains:

- `final.json` — machine-readable metrics, evidence, decision, and deployment command.
- `report.md` — human-readable diagnosis, tested candidates, statistical decision, limitations, and a complete copy-paste launch command.
- Exact task, SGLang parameter contract, benchmark outputs, server logs, profile evidence, candidate registry, and rejected-trial reasons.
- Optional private SQLite history for exact-compatible future priors.

Run artifacts may contain model paths and workload details. Keep the output directory private; generated artifacts are ignored by Git by default.

## Safety and Scope

Inference Autopilot is intended for authorized single-host experiments. It does not install packages at runtime, modify drivers or CUDA, edit SGLang or model weights, change kernels automatically, deploy to production, or kill processes it does not own. Precision-changing candidates are opt-in and require separate quality evidence before deployment.

Multi-node search, production rollout orchestration, automatic kernel modification, and complete multimodal workload optimization are not yet implemented.

## Agent-Assisted Use

The CLI is fully standalone; no Agent or Codex session is required. Environments that support Skills can use [`SKILL.md`](SKILL.md) to collect inputs, review plans, monitor runs, and explain evidence. The CLI and recorded experiment artifacts remain the source of performance decisions.

## Documentation

- [`SKILL.md`](SKILL.md) — end-to-end operational workflow.
- [`references/input-schema.md`](references/input-schema.md) — task fields and metrics.
- [`references/execution-schema.md`](references/execution-schema.md) — execution and artifact contracts.
- [`references/safety-policy.md`](references/safety-policy.md) — safety boundaries.
- [`references/sglang-adapter.md`](references/sglang-adapter.md) — SGLang integration details.
- [`PARAMETER_EVOLUTION.md`](PARAMETER_EVOLUTION.md) — live parameter discovery and safety policy.

## License

Licensed under the [Apache License 2.0](LICENSE).
