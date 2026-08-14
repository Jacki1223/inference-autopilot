# Inference Autopilot

Evidence-driven, bounded optimization for [SGLang](https://github.com/sgl-project/sglang) inference deployments.

Given a model, local hardware, workload, SLOs, and an experiment budget, Inference Autopilot discovers the installed SGLang parameter surface, checks deployment feasibility, runs repeatable benchmark trials, captures a representative Nsight Systems trace, parses SGLang runtime logs, and produces a data-backed deployment recommendation.

It is a standalone Python command-line tool. It does not require Codex, an LLM agent, SSH access, package installation, source modifications, or kernel changes to run its core workflow.

See [ROADMAP.md](ROADMAP.md) for the planned multi-GPU, multi-node, multimodal, operator-optimization, and continuous-regression work.

## What It Does

- Inspects the local GPU topology, model configuration and weight footprint.
- Reads the checked-out SGLang `server_args.py` and `sglang.launch_server --help` on every run, so parameter discovery follows the installed version.
- Checks single-GPU feasibility and reports when a model needs quantization or multi-GPU parallelism. It never downloads or switches checkpoints automatically.
- Supports online latency and offline throughput objectives, with explicit E2E, TTFT, TPOT/ITL, error-rate, and throughput gates.
- Performs warmup, minimum-duration steady-state measurement, candidate screening, interleaved repeat confirmation, and noise/SLO gating.
- Builds workload-aware candidate families for scheduling, chunked prefill, KV/cache, CUDA Graph, memory pool, attention/MoE backend, and compatible speculative decoding features.
- Detects startup dependency and backend failures by capability family. After the first definitive MTP/EAGLE failure, it records the cause and skips remaining candidates in that family while continuing independent tuning work.
- Profiles a bounded serving window with Nsight Systems, then routes observed queueing, CPU/GPU overlap, cache, graph, communication, and kernel evidence into a second tuning stage.
- Writes structured artifacts, a reproducible launch command, rejected-trial evidence, and a Markdown report.

## What It Does Not Do

- It does not claim a global optimum. A result is the best configuration within the recorded SGLang version, tested parameter space, hardware, workload, budget, and acceptance gates.
- It does not modify drivers, CUDA packages, SGLang source, model weights, kernels, production services, or unowned processes.
- It does not make kernel changes automatically. Nsight Compute is used only after Nsight Systems has isolated a relevant kernel, and requires GPU performance-counter permission.
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

`init` asks both deployment modes for the same optional p99 latency limits: E2E (request start to final token), TTFT (request start to first token), and TPOT (average generated-token time), all in milliseconds. Leave a value blank or enter `0` to omit that limit; leaving all three blank creates an objective-only task with no SLO constraint. Online defaults to a target concurrency of `8`; offline defaults to `64`. It then asks for **Experiment intensity**. `fast` is a short compatibility and coarse-ranking pass; `balanced` is the default for routine tuning; `rigorous` uses longer steady-state windows, more candidates, and five confirmation repetitions for a final deployment decision.

Use **Concurrency points to measure** to provide an exact capacity/SLO curve such as `1,4,8,16,32` or `1 4 8 16 32`; the final point must equal the target concurrency. Leave it blank for the automatic geometric sweep. These points measure the baseline curve; final startup-parameter tuning remains targeted at the highest point.

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
- `workload.max_concurrency`: the highest concurrent-request load that must pass the final SLO and parameter-confirmation gates.
- `calibration.min_concurrency` / `calibration.max_concurrency`: an optional baseline capacity sweep. Tasks created by `inferopt init` sweep `1, 2, 4, ...` through the declared maximum before profiling; the final parameter search remains at that maximum.
- `slo`: tail E2E, TTFT, TPOT/ITL, error-rate, and throughput constraints.
- `measurement`: warmup, minimum completed requests, and minimum steady-state duration.
- `search`: trial, GPU-hour, wall-clock, repeat, and variation limits.
- `capability_overrides`: explicit feature constraints, such as disabling speculative decoding for a known model/version incompatibility.

See [`references/input-schema.md`](references/input-schema.md) and the example task for the full schema.

## Decision Model

The controller keeps the baseline as a valid candidate. It only recommends a changed launch configuration when all required gates pass:

1. The candidate completes and preserves correctness/error-rate requirements.
2. Every declared SLO passes.
3. The measured improvement clears the configured practical-improvement and noise thresholds.
4. Interleaved confirmation repetitions remain stable.

When no candidate clears those gates, `recommendation_status` is `retain_confirmed_baseline`. This is a successful, evidence-backed result rather than a failed run.

## Artifacts

Every run records inventory, the current SGLang parameter audit, exact launch commands, raw benchmark outputs, server logs, runtime observations, profiling outputs, trial results, and the final decision. Generated artifacts are ignored by Git by default because they can include private model paths and workload details.

## Safety

This tool is designed for explicitly authorized, single-host experiments. Review the generated plan before adding `--yes`. It never executes arbitrary shell snippets from the task, never installs packages, and never kills processes it did not create.

See [`SKILL.md`](SKILL.md) for the full operational workflow and [`references/safety-policy.md`](references/safety-policy.md) for the safety contract.

## License

Licensed under the [Apache License 2.0](LICENSE).
