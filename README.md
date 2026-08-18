# Inference Autopilot

Evidence-driven deployment optimization for [SGLang](https://github.com/sgl-project/sglang).

Inference Autopilot takes a model, a GPU host, a workload, optional latency SLOs, and an experiment budget. It then validates whether the model can be served, runs a bounded set of SGLang configurations, diagnoses the serving bottleneck, and emits a reproducible deployment command backed by measured data.

It is a standalone Python CLI. The normal workflow does not require Codex, an LLM agent, SSH, SGLang source changes, kernel changes, or automatic package installation.

## At A Glance

| You provide | The tool produces |
| --- | --- |
| Model path and SGLang checkout | Feasibility checks and a resolved launch command |
| GPU selection and topology | Valid parallelism candidates and capacity evidence |
| Input/output shape, concurrency, and dataset | Warm steady-state benchmark metrics |
| Online/offline objective and optional SLOs | SLO-safe or throughput-maximizing recommendation |
| Experiment intensity and budget | Trial artifacts, profiler evidence, and `report.md` |

The result is the best configuration found inside the recorded hardware, SGLang version, workload, search space, budget, and acceptance gates. It is not a claim of a global optimum.

## What It Does

The workflow has five stages. Each stage leaves evidence in the private run directory.

### 1. Discover and validate

- Inspects visible GPUs, topology, model config, dtype, and weight footprint.
- Reads the installed SGLang `server_args.py` and `sglang.launch_server --help` on every run. Launch commands therefore use flags and types from the actual checkout, not a hard-coded version.
- Checks single-host deployment feasibility, including GPU count, attention-head divisibility, KV-head sharding/replication, memory, and supported TP layouts.
- Reports when the checkpoint needs a different quantization or parallel layout. It never downloads or switches a checkpoint automatically.

### 2. Define and measure the workload

- Supports `online_latency` and `offline_throughput` objectives.
- Supports synthetic fixed-shape requests, SGLang-generated shared-prefix requests, local custom JSONL conversations, and ShareGPT JSON.
- Warms the service before measurement and enforces a minimum steady-state window. Request counts expand when a short run has not reached the requested stability criteria.
- Uses client-side `bench_serving --max-concurrency` for load control. It does not overwrite SGLang `--max-running-requests`; the resolved server value is recorded as diagnostic evidence.

### 3. Search deployment parameters

- Builds candidates from the installed flag catalog and the current workload: scheduling, chunked prefill, KV/prefix cache, memory pool, CUDA Graph, attention/MoE backend, parallelism, and compatible speculative decoding.
- Ranks mechanisms using workload shape, GPU-active profiler shares, SGLang queue/cache/graph observations, topology, and current defaults. It covers high-impact mechanisms before nearby value refinements.
- Screens individual candidates first, then measures promising combinations. A combination must win its own comparison; gains are not added together on paper.
- Skips only candidates with a recorded deterministic incompatibility. A startup failure, timeout, port conflict, or process kill is not treated as a reusable proof that a parameter is bad.
- Uses bounded early stopping only for `fast` mode after sufficient successful coverage and a strong measured gain. The final deployment gate remains the configured practical-improvement threshold (1% by default).

### 4. Profile and diagnose

- Captures a bounded serving-only Nsight Systems trace after warmup, not from model load.
- Separates GPU-active time from CPU and startup time. It extracts top kernels, queue/launch latency, synchronization, allocation, copy activity, communication, GPU gaps, and CUDA Graph coverage.
- Parses SGLang logs and runtime metrics for admission limits, prefix-cache behavior, graph capture, backend fallbacks, and missing fused MoE configs.
- Uses Nsys to route the next candidates. Nsys is not treated as a proof of occupancy, memory bandwidth, or instruction-level stalls; those require Nsight Compute or a focused microbenchmark.

### 5. Confirm and report

- Rechecks the selected candidate against a matched baseline window. Offline no-SLO runs preserve a longer cache-flushed baseline reference; SLO-constrained modes use repeated interleaved A/B measurements.
- Applies correctness, error-rate, SLO, improvement, noise, and secondary-regression gates.
- Emits `final.json`, a Markdown `report.md`, raw logs and traces, rejected-trial reasons, and an exact deployment command.
- Keeps the confirmed baseline when no candidate is supported by sufficient evidence. That is a valid result, not a failed experiment.

## Modes and SLOs

Both modes accept the same optional latency limits:

- E2E: request start to the final token.
- TTFT: request start to the first token.
- TPOT/ITL: generated-token time.
- Error rate and throughput can also be constrained in the task file.

Leave all limits blank or set them to `0` for an objective-only run.

- `online_latency`: optimize tail latency and SLO-safe concurrency. The adaptive calibration starts from the runtime-resolved admission capacity, then backs off and binary-searches only when needed.
- `offline_throughput`: maximize throughput. Without SLOs, the client does not impose a maximum concurrency and the server's resolved KV/admission policy determines sustainable throughput. With SLOs, the tool starts at the highest capacity probe and backs off to find the largest SLO-safe load.

Use explicit `calibration.concurrencies` or the `init` prompt when you need a fixed curve such as `1,4,8,16,32`. These points calibrate the baseline; parameter tuning is targeted at the selected highest load.

## Requirements

- Python 3.9 or newer.
- A local SGLang checkout or installation runnable by the selected Python interpreter.
- NVIDIA GPUs for automatic execution and profiling. AMD inventory/planning is available, but the automatic profiler executor is not.
- A locally available, SGLang-compatible model directory.
- A compatible SGLang benchmark entry point.
- `nsys` for the normal `inferopt run` workflow. Nsight Compute is optional and can require driver-level performance-counter permission.

Inference Autopilot has no mandatory third-party Python runtime dependency of its own. SGLang, CUDA, PyTorch, and the model runtime remain dependencies of the target environment.

## Install

Install or replace the package without changing the existing SGLang or CUDA environment:

```bash
python3 -m pip install \
  --no-deps \
  --no-build-isolation \
  --force-reinstall \
  "git+https://github.com/Jacki1223/inference-autopilot.git"
```

For a source checkout:

```bash
python3 -m pip install .
```

## Quick Start

```bash
inferopt init --output task.json
inferopt doctor --task task.json --output doctor.json
inferopt plan --task task.json --output plan.json
inferopt run --task task.json --yes --output final.json
inferopt report --result final.json --output report.md
```

`init` asks for the SGLang checkout, Python interpreter, model path, output directory, deployment mode, request shape, target concurrency, concurrency points, shared-prefix length, optional SLOs, dataset, GPU selection, and experiment intensity. Each prompt explains how the value is used. `online_latency` defaults to a low concurrency hint; `offline_throughput` defaults to a higher one.

`doctor` and `plan` do not start a server. They validate the host, model, installed SGLang flags, benchmark flags, profiler availability, and candidate plan. `run --yes` starts only process groups created by the current experiment. Progress output includes the current stage, candidate, trial count, throughput, p99 E2E, and SLO status.

For non-interactive use:

```bash
cp assets/task.autopilot.example.json task.json
inferopt validate --task task.json
inferopt run --task task.json --yes --output final.json
```

See [references/input-schema.md](references/input-schema.md) for the complete task schema.

## Using With An Agent

Inference Autopilot can run in two equivalent ways:

- **Standalone CLI:** the user creates the task and runs each command directly. No Agent or Codex is required.
- **Agent-assisted:** an Agent reads `SKILL.md`, gathers missing inputs conversationally, invokes the same CLI and scripts, and explains the resulting evidence.

To make the repository available as an Agent skill in an environment that supports skills, install the repository under that environment's skill directory so that `SKILL.md` is at the skill root. For example:

```bash
mkdir -p ~/.agents/skills
git clone https://github.com/Jacki1223/inference-autopilot.git \
  ~/.agents/skills/inference-autopilot
```

Then ask the Agent to use `inference-autopilot` for a deployment request. A typical Agent-assisted session is:

1. The user provides the model path, SGLang checkout, visible GPUs, workload shape or dataset, deployment mode, SLOs, and experiment budget.
2. The Agent reads `SKILL.md` and asks only for missing or ambiguous values, such as whether the workload is online latency or offline throughput.
3. The Agent runs `inferopt init`, `doctor`, and `plan`, then shows the generated plan and safety checks.
4. After explicit approval, the Agent runs `inferopt run --yes` and monitors the stage/trial progress.
5. The Agent opens `final.json` and `report.md`, summarizes the baseline, tested candidates, bottleneck evidence, rejected configurations, and recommended launch command.
6. If the report identifies a kernel-level follow-up, the Agent may suggest Nsight Compute, a microbenchmark, or the separate fused MoE tuning command. These high-cost or state-changing actions remain opt-in.

The Agent is an orchestration and explanation layer; it does not invent performance conclusions. Benchmark metrics, SGLang logs, Nsys evidence, confirmation repetitions, and the acceptance gates in the result files remain the source of truth. Without an Agent, the CLI executes the same validation, search, profiling, confirmation, and reporting logic, but the user enters the task values and interprets the report directly.

## Workload Data

Synthetic data gives exact token lengths and can request a generated shared-prefix workload. For real traffic, choose `custom` JSONL or `sharegpt` JSON. Dataset files remain local and are validated before any GPU launch.

Example custom row:

```json
{"conversations":[{"content":"Summarize this incident report..."},{"content":"The incident was caused by..."}]}
```

For real datasets, input/output token counts are planning hints; SGLang measures the actual distribution. Set shared-prefix tokens only for synthetic data. `doctor` checks that the installed benchmark supports the selected dataset, shared-prefix, and cache-flush flags.

## Search and Decision Rules

The controller keeps the baseline as a valid candidate. A changed configuration is deployable only when:

1. It completes with acceptable correctness and error rate.
2. Every declared SLO passes.
3. Improvement clears the configured practical-improvement and noise thresholds.
4. Mode-specific confirmation passes using a matched baseline and candidate workload.

Candidate counts depend on intensity and the total budget. `fast`, `balanced`, and `rigorous` plan progressively wider coverage before combinations. The controller does not assume that `chunked_prefill_size` is always the best knob: its direction is chosen from workload shape, online/offline objective, queue pressure, and shared-prefix behavior. Every decision and skipped candidate is recorded in `search-plan.json` and `final.json`.

## Profiling Evidence

Nsys is used to answer routing questions such as:

- Is GPU work busy, or is the request path stalled on CPU scheduling and synchronization?
- Which kernels dominate GPU-active time during serving?
- Are there large GPU gaps, excessive copies, communication costs, or low CUDA Graph coverage?
- Do SGLang logs show queue pressure, cache misses, backend fallback, or missing MoE configuration?

It does not replace the benchmark winner metric and it does not by itself prove an operator-level optimization. Use Nsight Compute or a microbenchmark only after a bounded Nsys trace identifies a relevant kernel.

## Optional Fused MoE Tuning

The normal workflow never runs the expensive fused MoE autotuner. When SGLang logs report missing configs, `report.md` records the filenames, observed shapes, and an explicit standalone command. Run it only when you choose to pay the compile and benchmark cost:

```bash
inferopt tune-moe \
  --task RUN_DIR/task.json \
  --profile RUN_DIR/profile/nsys-diagnosis.json \
  --result RUN_DIR/final.json \
  --output-dir RUN_DIR/optional-fused-moe-tuning \
  --yes \
  --output RUN_DIR/optional-fused-moe-tuning.json
```

The generated files follow the installed SGLang loader contract, including Triton-version directories and paired normal/`_down.json` files when the model requests them. A partial or unvalidated config is never emitted as deployable. For direct paired-file generation, see `inferopt generate-moe-config --help` and [references/sglang-adapter.md](references/sglang-adapter.md).

## Artifacts and Safety

Every run records inventory, the SGLang parameter audit, exact launch and benchmark commands, raw outputs, logs, runtime observations, profiler data, trial decisions, and the final recommendation under the configured private artifact directory. Artifacts are ignored by Git because they can contain model paths and workload details.

The tool never executes arbitrary shell snippets from a task, installs packages, modifies drivers/CUDA/SGLang/model weights, or kills processes it did not create. Review `plan.json` before using `--yes`.

See [SKILL.md](SKILL.md) for the full operational workflow, [references/execution-schema.md](references/execution-schema.md) for execution contracts, and [references/safety-policy.md](references/safety-policy.md) for the safety contract.

## Current Scope

Implemented: single-host inventory and feasibility checks, TP search, MoE EP candidates, SGLang DP-attention candidates, online/offline tuning, custom datasets, adaptive calibration, Nsys routing, and optional fused MoE workflows.

Not yet implemented: joint multi-replica DP placement, pipeline-stage placement, multi-node search, full multimodal workload handling, automatic operator changes, and production rollout orchestration. See [ROADMAP.md](ROADMAP.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
