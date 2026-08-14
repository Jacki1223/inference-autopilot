---
name: inference-autopilot
description: Analyze, benchmark, diagnose, and optimize large-model inference deployments from hardware inventory, model details, workload traces, and latency or throughput SLOs. Use when Codex needs to tune SGLang launch parameters, run bounded single-host GPU experiments, plan deployment topology, inspect GPU or CPU profiles, identify scheduler/KV/communication/kernel bottlenecks, propose operator optimizations, or validate that a candidate configuration improves performance without correctness or SLO regressions.
---

# Inference Autopilot

Run an evidence-driven optimization loop. Treat this skill as the control plane and the bundled scripts as deterministic utilities. Default to local, private, dry-run behavior.

## Start Here

For a new private single-host deployment, start with the one-shot interface. The
user supplies only local paths, workload, SLO, objective, budget, and optional
GPU visibility. The script discovers NVIDIA or AMD GPU memory and topology,
reads the local model config and weight sizes, regenerates the current SGLang
parameter contract from `server_args.py` and `sglang.launch_server --help`, captures a bounded Nsight Systems baseline, routes trace and workload
evidence to parameter families, runs bounded screening, automatically repeats
the decision, and emits a deployable command only after the confirmation gates
pass:

```bash
inferopt init --output /absolute/private/task.json
inferopt doctor --task /absolute/private/task.json --output /absolute/private/doctor.json
inferopt plan --task /absolute/private/task.json --output /absolute/private/plan.json
inferopt run --task /absolute/private/task.json --yes --output /absolute/private/final.json
inferopt report --result /absolute/private/final.json --output /absolute/private/report.md
```

The standalone `inferopt` command does not require Codex. `doctor` never starts
a server: it validates the local model, GPU/runtime, current SGLang CLI, profiler
availability, and a transparent single-GPU memory estimate. It may recommend a
quantized checkpoint class when the current model cannot fit, but never downloads,
switches, or approves a model variant without an explicit quality evaluation.

The underlying scripts remain available for controlled integration:

```bash
cp assets/task.autopilot.example.json /absolute/private/task.json
python3 scripts/autopilot.py validate --task /absolute/private/task.json
python3 scripts/autopilot.py plan --task /absolute/private/task.json --output /absolute/private/plan.json
python3 scripts/autopilot.py run --task /absolute/private/task.json --yes --output /absolute/private/final-pointer.json
```

Read [references/hardware-profiles.json](references/hardware-profiles.json) for
the official-source GPU capability catalog. Runtime discovery overrides catalog
values. Preserve the checked-out SGLang version's automatic defaults as the
baseline; never transplant a winning parameter from one GPU or workload to
another without measurement.

1. Locate the inference repository and read its local instructions.
2. Create a task specification using [references/input-schema.md](references/input-schema.md).
3. Run `scripts/inferopt.py validate --spec <task.json>`.
4. Run `scripts/inferopt.py inventory --output <run-dir>/inventory.json` on every target host when access is available.
5. Inspect the generated inventory and task constraints before launching anything.
6. Follow the optimization loop below.

For authorized single-host execution, copy `assets/task.execute.example.json`, read [references/execution-schema.md](references/execution-schema.md), validate it, render the exact command plan, obtain explicit approval, and only then run:

```bash
python3 scripts/autotune.py validate --spec task.json
python3 scripts/autotune.py plan --spec task.json --output plan.json
python3 scripts/autotune.py run --spec task.json --yes --output final-pointer.json
python3 scripts/autotune.py report --run-dir /absolute/completed-run --output decision.json
```

If the framework is SGLang, read [references/sglang-adapter.md](references/sglang-adapter.md). For another engine, discover its actual launch, benchmark, metrics, and profiling interfaces instead of assuming SGLang flags.

## Safety Contract

- Default to `dry_run`. Require the user or task specification to opt into execution.
- Run only on hosts, endpoints, repositories, models, datasets, and output directories in scope.
- Never modify a production deployment, autoscaler, traffic router, cloud resource, driver, system service, or shared cluster without explicit authorization.
- Never download a model or dependency, reserve paid hardware, publish results, or send traces externally without explicit authorization.
- Never kill processes not started by the current experiment. Record owned PIDs and terminate only those PIDs.
- Never use broad process-kill commands as cleanup.
- Redact prompts, API keys, tokens, user identifiers, and proprietary model paths from reports.
- Enforce trial, wall-time, GPU-hour, and failure budgets. Stop immediately on correctness failure, repeated OOM, thermal/power anomaly, or error-rate violation.
- Keep baseline artifacts immutable. Store every candidate in a distinct trial directory.
- Do not accept a faster candidate unless correctness, stability, and SLO gates pass.
- Treat `--yes` as confirmation that the generated command plan was reviewed; never add it before approval.

Read [references/safety-policy.md](references/safety-policy.md) before executing benchmarks, profilers, or code changes.

## Optimization Loop

### Deployment Modes

Set `deployment_mode` in the autopilot task. Use `online_latency` for an
interactive service: require at least one E2E, TTFT, TPOT, or ITL SLO and
maximize the selected objective only after all declared tail-latency gates
pass. Use `offline_throughput` for batch inference: maximize sustained
aggregate throughput at calibrated batch pressure; latency remains recorded
but is only a gate when the task explicitly declares it.

Before profiling, run the private geometric concurrency calibration generated
from the budget. Treat its selected concurrency as an analysis load only.
For a model with a verified official cookbook, first compare complete, locally
valid capability bundles: include the relevant model feature (for example MTP),
prefix/KV cache policy, scheduler/admission, memory pool, CUDA Graph, and MoE
backend variants when the workload and hardware make them applicable. Profile
the fastest SLO-valid initial configuration, even if its single screening
sample has not yet cleared the final improvement threshold. Then use Nsight
evidence to select and refine parameter families, and screen and confirm every
candidate against the original target workload. Never label a result as global
best: report it as the best configuration within the checked-out SGLang
parameter contract, tested search space, hardware, workload, and SLO gates.
Use `search_depth: thorough` (the default) to add a one-factor sensitivity
screen for every high-impact compatible family even when a short trace lacks a
single hotspot. Use `evidence_guided` only when experiment budget is tight.
Every run emits `search-plan.json.parameter_audit`, which accounts for each
CLI-visible, non-deprecated `ServerArgs` as selected, excluded, or
inapplicable with a concrete reason. It prevents a short candidate list from
being mistaken for the whole startup-parameter surface.
After the one-factor screen, the optimizer uses the remaining trial budget to
test explicit combinations of independent screened winners. It then confirms
the best combined configuration with interleaved baseline repetitions; it does
not perform an unbounded Cartesian product.

### 1. Normalize The Objective

Extract:

- hardware and interconnect topology;
- model architecture, dtype, quantization, context limit, and draft model;
- workload arrival process, input/output distributions, prefix reuse, modalities, structured outputs, and concurrency;
- TTFT, TPOT/ITL, end-to-end latency, throughput, goodput, cost, power, and correctness constraints;
- experiment budget and allowed mutation scope.

Do not optimize an unspecified scalar. Convert multiple SLOs into hard constraints plus one primary objective, for example: maximize request goodput subject to P99 TTFT and P99 TPOT.

### 2. Establish A Reproducible Baseline

- Pin repository commit, container/image, model revision, driver/runtime, environment variables, and exact launch command.
- Warm up before measurement.
- Warm up before every measured window. Require at least 30 seconds of completed-request measurement by default; when a run is shorter, automatically increase request count and repeat it, preserving the short raw sample as an audit artifact.
- Use at least five times the maximum concurrency in steady-state serving tests when affordable.
- Repeat noisy trials; preserve raw JSONL, logs, metrics, and profiler traces.
- Record idle and loaded GPU memory, utilization, power, clocks, host CPU, network, cache hit rate, request failures, accepted draft tokens, and queueing metrics when available.
- Validate output correctness before trusting performance numbers.

Use `scripts/inferopt.py analyze` for SGLang-compatible benchmark JSONL and `scripts/inferopt.py compare` to gate candidates. Use `scripts/autotune.py` only for an isolated, authorized, single-host SGLang experiment.

Treat a one-run `screening_winner` as a hypothesis. Require at least three interleaved repetitions, acceptable objective CV, and every repetition passing SLO before treating `winner_status=confirmed` as a deployable result.

Use `recommended_configuration` for the deployment decision. Preserve a confirmed baseline when `recommendation_status=retain_confirmed_baseline`; an empty `winner` can mean candidates were correctly rejected, not that the experiment failed.

### 3. Classify Before Tuning

Read [references/diagnosis-playbook.md](references/diagnosis-playbook.md). First classify the dominant bottleneck:

- admission or queueing;
- prefill compute;
- decode memory bandwidth;
- KV capacity, reuse, transfer, or host IO;
- CPU scheduler/tokenizer/frontend;
- TP/EP/PP collective communication;
- MoE imbalance or All-to-All;
- speculative draft/verify waste;
- multimodal preprocessing;
- one or more GPU operators.

Do not start kernel work while queueing, cache misses, bad topology, or an unsuitable backend dominates end-to-end time.

### 4. Search From Coarse To Fine

Search in this order unless evidence justifies a different order:

1. deployment topology and parallelism;
2. memory feasibility and KV allocation;
3. scheduler, batching, and chunked prefill;
4. attention, GEMM, MoE, and communication backends;
5. prefix cache, hierarchical cache, sparse attention, or PD transfer;
6. speculative decoding;
7. operator implementation and kernel parameters.

Change one conceptual factor per ablation. Use a coarse family screen before
fine tuning. Enumerate every current `server_args.py` parameter into a family,
and record every compatible parameter as selected, excluded with its reason,
or inapplicable to the current model/topology/deployment mode. Reuse prior trials
and reject infeasible configurations before launching a server.

### 5. Escalate Profiling By Evidence

Use the least expensive tool that can answer the current question. Read [references/profiler-routing.md](references/profiler-routing.md).

- Start with serving metrics and logs.
- Use framework/PyTorch traces for stage and operator attribution.
- Use Nsight Systems or equivalent for CPU/GPU overlap and communication timelines.
- Use Nsight Compute or a kernel benchmark only after isolating a specific hot operator and representative shapes.

Keep profiling windows short and representative. Do not compare profiled latency directly with unprofiled production latency.

### 6. Optimize Operators Only When Justified

Before editing a kernel, record:

- operator share of GPU-active execution time, and independently measured request-level attribution before claiming an end-to-end bound;
- exact shapes, dtypes, layouts, architecture, batch and sequence regimes;
- current backend and competing implementations;
- numerical tolerance and determinism requirements;
- expected upper-bound end-to-end gain from Amdahl's law.

Build an isolated correctness and performance benchmark. Cover boundary shapes and realistic distributions, not only one favorable shape. Integrate only after isolated and end-to-end gates pass.

### 7. Validate And Report

Require:

- correctness equivalence or an explicitly approved quality tradeoff;
- all hard SLOs pass at the target load;
- improvement exceeds the configured minimum and noise margin;
- no material error-rate, memory, stability, fairness, or cost regression;
- repeated or confidence-supported results.

Deliver:

- recommended launch/deployment configuration;
- baseline versus winner and confidence/noise statement;
- diagnosed bottleneck with supporting evidence;
- rejected trials and why they failed;
- raw artifact locations and reproduction commands;
- remaining risks and next experiments;
- code changes only when they produced validated gains.

## Bundled Commands

```bash
python3 scripts/inferopt.py validate --spec task.json
python3 scripts/inferopt.py inventory --output runs/inventory.json
python3 scripts/inferopt.py plan --spec task.json --output runs/plan.json
python3 scripts/inferopt.py analyze --input runs/baseline.jsonl --spec task.json --output runs/baseline-summary.json
python3 scripts/inferopt.py compare --baseline runs/baseline-summary.json --candidate runs/trial-001-summary.json --spec task.json
python3 scripts/autotune.py validate --spec task.json
python3 scripts/autotune.py plan --spec task.json --output plan.json
python3 scripts/autotune.py run --spec task.json --yes
python3 scripts/autopilot.py validate --task task.json
python3 scripts/autopilot.py plan --task task.json --output plan.json
python3 scripts/autopilot.py run --task task.json --yes --output final-pointer.json
```

`inferopt.py` never starts a server. `autopilot.py` starts only the local SGLang
process groups it creates, regenerates an installed-version parameter contract for every run, performs a bounded Nsight Systems CUDA-range capture,
then generates and executes two-stage `autotune.py` runs using structured
allowlisted parameters. It cannot execute shell snippets, remote SSH, Slurm,
Kubernetes, arbitrary modules, production endpoints, package installation,
source edits, or kernel changes. It records a shape-matched Nsight Compute
follow-up only when one kernel is trace-proven hot; it never edits a kernel.
