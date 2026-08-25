# Inference Autopilot

Evidence-driven, bounded optimization for [SGLang](https://github.com/sgl-project/sglang) inference deployments.

Given a model, local hardware, workload, SLOs, and an experiment budget, Inference Autopilot discovers the installed SGLang parameter surface, checks deployment feasibility, runs repeatable benchmark trials, captures a representative Nsight Systems trace, parses SGLang runtime logs, and produces a data-backed deployment recommendation.

It is a standalone Python command-line tool. It does not require Codex, an LLM agent, SSH access, package installation, source modifications, or kernel changes to run its core workflow.

See [ROADMAP.md](ROADMAP.md) for the planned multi-GPU, multi-node, multimodal, operator-optimization, and continuous-regression work.

## What It Does

- Inspects the local GPU topology, model configuration and weight footprint.
- Reads the checked-out SGLang `server_args.py` and `sglang.launch_server --help` on every run, then freezes those real flags and types into every emitted launch command.
- Checks single-GPU feasibility and reports when a model needs quantization or multi-GPU parallelism. It never downloads or switches checkpoints automatically.
- Supports online latency and offline throughput objectives, with explicit E2E, TTFT, TPOT/ITL, error-rate, and throughput gates.
- Performs warmup, minimum-duration steady-state measurement, candidate screening, interleaved repeat confirmation, and noise/SLO gating.
- Reduces profile evidence to a confidence-bearing bottleneck classifier (`prefill_attention`, `decode_attention`, `MoE`, `GDN state`, `KV capacity`, communication, host/scheduler, or mixed/unknown) and prints the evidence in every report.
- Matches the `(bottleneck, workload, model, hardware)` tuple against versioned declarative trigger rules. Parameters that do not match an applicable rule cannot consume the search budget.
- Derives nonlinear value sets from live inputs: memory fractions use per-GPU VRAM/weight/activation headroom; prefill chunks and budgets use uncached workload length and context limits.
- Runs a mechanism-level coarse screen, then performs successive refinement around the best measured parameter neighborhoods and tests compatible positive combinations. It is not a fixed recipe menu or a blind Cartesian grid.
- Treats MTP and Mamba as model-native mechanisms: compatible Cookbook commands are measured together with bounded draft-depth and Mamba cache-memory variants, and acceptance telemetry is recorded when the installed SGLang revision emits it.
- Uses measured decode-latency share to decide whether MTP has enough end-to-end leverage; Mamba cache remains an independent hybrid-model mechanism.
- Supports explicitly authorized FP8 KV-cache performance candidates through `--allow-kv-cache-precision-tuning`. They remain disabled by default, and a winner stays non-deployable until a separate model-quality evaluation passes.
- Tunes CUDA Graph sizes only when runtime logs show incomplete graph coverage. A large resolved default is not automatically treated as a performance problem.
- Detects startup dependency and backend failures by capability family. After the first definitive MTP/EAGLE failure, it records the cause and skips remaining candidates in that family while continuing independent tuning work.
- For offline no-SLO work, first measures SGLang's unbounded admission capacity, then uses at least five capacity waves for screening. Short 20/40-request probes cannot be reported as saturated-throughput evidence.
- Allocates trial budget by tier (approximately 60% discovery, 25% refinement/composition, 15% confirmation). Unused earlier-tier trials flow forward and the report records planned versus used trials.
- Confirms a positive nominee with repeated baseline and candidate windows. Two-GPU TP=2 runs use ABBA service order; larger hosts alternate resident services. A conservative Welch-style 95% interval must clear the configured minimum gain, otherwise the result is `noise_limited` or `effect_size_uncertain`.
- Emits both a minimal command and a reproducible command that pins performance-critical resolved SGLang defaults.
- Establishes a warm steady-state serving window before a bounded Nsight Systems capture, samples workload-time metrics during that capture, then routes queueing, CPU/GPU overlap, cache, graph, communication, and kernel evidence into a second tuning stage.
- Writes structured artifacts, a reproducible launch command, rejected-trial evidence, and a Markdown report.

## What It Does Not Do

- It does not claim a global optimum. A result is the best configuration within the recorded SGLang version, tested parameter space, hardware, workload, budget, and acceptance gates.
- It does not modify drivers, CUDA packages, SGLang source, model weights, kernels, production services, or unowned processes.
- It does not make kernel changes automatically. Nsight Compute is used only after Nsight Systems has isolated a relevant kernel, and requires GPU performance-counter permission.
- Reports the top GPU kernel, its GPU-active share, an Amdahl upper bound, and the exact Nsight Compute or microbenchmark escalation when startup-parameter tuning reaches its measured ceiling.
- Stores private, structured trial evidence in SQLite. Exact-compatible history becomes a weak parameter/configuration prior; it never creates a candidate trial or consumes a discovery slot.
- Uses a paired Bayesian posterior during confirmation: clear wins stop early, clear losses stop early, and ambiguous effects extend through at most six ABBA blocks.
- Produces cost per million output, total, and SLO-valid tokens when the task provides `economics.cost_per_gpu_hour`; it never infers a price from GPU name.
- Reports Roofline classification only from shape-matched Nsight Compute counters. Without counter permission, the report explicitly says so instead of guessing memory- or compute-bound.
- Single-host execution is implemented now. Multi-node and production rollout orchestration are intentionally out of scope for the first release.

## Requirements

- Python 3.9 or newer.
- A local SGLang checkout or installation runnable by the selected Python interpreter.
- An NVIDIA or AMD GPU host for execution. Planning and validation can run without a GPU.
- A locally available model directory for execution.
- A SGLang-compatible benchmark entry point and an authorized output directory.
- `nsys` is optional but recommended. Nsight Compute is optional and may be blocked by driver-level performance-counter permissions.

The tool has no mandatory third-party Python runtime dependency of its own. SGLang and its GPU runtime remain dependencies of the target environment.

## Install

Install or replace an existing release directly from GitHub:

```bash
python3 -m pip install \
  --no-deps \
  --no-build-isolation \
  --force-reinstall \
  "git+https://github.com/Jacki1223/inference-autopilot.git"
```

`--no-deps` ensures this package installation does not change the existing SGLang, CUDA, PyTorch, or model-runtime environment. `--force-reinstall` replaces an older installed Inference Autopilot version.

For development from a source checkout:

```bash
python3 -m pip install .
```

For an isolated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install .
```

## Quick Start

Create a task interactively, then validate the host before allowing execution:

```bash
inferopt init --output task.json
inferopt doctor --task task.json --output doctor.json
inferopt plan --task task.json --output plan.json
inferopt run --task task.json --yes --output final.json
inferopt report --result final.json --output report.md
```

`run` displays live stage and trial progress, including capacity points, candidate names, completion status, request throughput, p99 E2E latency, and SLO status. Full results remain in the requested JSON file and private artifact directory.

During `init`, set **Shared prefix tokens** to the number of tokens common to requests in a prefix-cache workload. Set it to `0` when requests do not share a prefix. The value must be smaller than the input-token length.

`init` lets both deployment modes choose one latency statistic family, `p99` or `avg`, for optional E2E, TTFT, and TPOT limits. Leave the statistic blank, or enter `0` for every limit, to run without a latency SLO. Online mode starts from the declared target concurrency and only searches lower loads after an SLO failure. Offline no-SLO mode leaves client concurrency unbounded; it does not invent a maximum concurrency of 64.

The experiment intensities are `fast`, `balanced`, and `max`. `fast` performs narrow mechanism screening, `balanced` adds adaptive value refinement and combinations, and `max` permits up to 40 parameter candidates within a 48-trial default total budget and never uses the strong-gain early stop. Request count and steady-state validity remain tied to observed concurrency/capacity rather than a fixed 500-request rule.

For repeated work, leave trial history enabled. The default database is `<output-dir>/inferopt-history.sqlite3`. Historical results become weak priors only when checkpoint content, current SGLang argument contract, selected GPU architecture/topology, workload shape/data fingerprint, mode, objective, and SLOs all match exactly. History influences matched-parameter ordering and Bayesian confirmation; it never occupies a candidate slot.

Set `--cost-per-gpu-hour` and `--currency` at `init` to add a cost-per-token section. Use `--canonical-gpu-model NVIDIA H800` when a cluster exposes an internal alias instead of the actual GPU model; the runtime alias remains in the artifacts for audit.

Use **Concurrency points to measure** for an explicit online capacity/SLO curve such as `1,4,8,16,32` or `1 4 8 16 32`. Without explicit points, online mode measures the target first and adaptively backs off only when needed. Offline no-SLO mode discovers runtime capacity from the loaded server instead.

`doctor` and `plan` do not start a server. `run --yes` starts only SGLang process groups created by the current experiment and only after the task passes validation.

For non-interactive use, begin with [`assets/task.autopilot.example.json`](assets/task.autopilot.example.json):

```bash
cp assets/task.autopilot.example.json task.json
inferopt validate --task task.json
inferopt run --task task.json --yes --output final.json
```

## Task Inputs

A task describes the model path, SGLang repository, Python executable, output directory, target workload, SLOs, and budget. The key inputs are:

- `deployment_mode`: `online_latency` or `offline_throughput`.
- `workload.max_concurrency`: the online target or an SLO-constrained load. It is intentionally absent for an offline no-SLO task.
- `calibration`: explicit points are honored exactly; otherwise online calibration starts at the target and uses bounded adaptive fallback.
- `slo`: tail E2E, TTFT, TPOT/ITL, error-rate, and throughput constraints.
- `measurement`: warmup, minimum completed requests, and minimum steady-state duration.
- `budget`: trial, GPU-hour, and wall-clock limits; `measurement` controls repeat and variation requirements.
- `capability_overrides`: explicit feature constraints, such as disabling speculative decoding for a known model/version incompatibility.

See [`references/input-schema.md`](references/input-schema.md) and the example task for the full schema.

## Decision Model

The controller keeps the baseline as a valid candidate. It only recommends a changed launch configuration when all required gates pass:

1. The candidate completes and preserves correctness/error-rate requirements.
2. Every declared SLO passes.
3. The measured improvement clears the configured practical-improvement and noise thresholds.
4. Interleaved confirmation repetitions remain stable.

When no candidate clears those gates but all applicable mechanism classes were tested, the tool may retain the measured baseline. If model-native or workload-critical mechanisms were not completed, the status is `insufficient_optimization_evidence`; the report may retain a provisional configuration for analysis but emits no deployment command.

## Artifacts

Every run records inventory, the current SGLang parameter audit, exact launch commands, raw benchmark outputs, server logs, runtime observations, profiling outputs, trial results, and the final decision. Generated artifacts are ignored by Git by default because they can include private model paths and workload details.

## Safety

This tool is designed for explicitly authorized, single-host experiments. Review the generated plan before adding `--yes`. It never executes arbitrary shell snippets from the task, never installs packages, and never kills processes it did not create.

See [`SKILL.md`](SKILL.md) for the full operational workflow and [`references/safety-policy.md`](references/safety-policy.md) for the safety contract.

## License

Licensed under the [Apache License 2.0](LICENSE).
