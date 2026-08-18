# Inference Autopilot

Evidence-driven, bounded optimization for [SGLang](https://github.com/sgl-project/sglang) inference deployments.

Given a model, local hardware, workload, SLOs, and an experiment budget, Inference Autopilot discovers the installed SGLang parameter surface, checks deployment feasibility, runs repeatable benchmark trials, captures a representative Nsight Systems trace, parses SGLang runtime logs, and produces a data-backed deployment recommendation.

It is a standalone Python command-line tool. It does not require Codex, an LLM agent, SSH access, package installation, source modifications, or kernel changes to run its core workflow.

See [ROADMAP.md](ROADMAP.md) for the planned full multi-GPU topology search, multi-node, multimodal, operator-optimization, and continuous-regression work.

## What It Does

- Inspects the local GPU topology, model configuration and weight footprint.
- Reads the checked-out SGLang `server_args.py` and `sglang.launch_server --help` on every run, then freezes those real flags and types into every emitted launch command.
- Checks the complete selected single-host GPU set, rejects TP layouts that violate GPU-count, attention-head, or SGLang KV-head sharding/replication rules, and reports when a model needs another parallel layout or quantized checkpoint. It never downloads or switches checkpoints automatically.
- Supports online latency and offline throughput objectives, with explicit E2E, TTFT, TPOT/ITL, error-rate, and throughput gates.
- Performs warmup, minimum-duration steady-state measurement, candidate screening, and mode-aware confirmation. Offline no-SLO runs preserve one unprofiled baseline instead of repeatedly restarting it; SLO-constrained runs retain repeated A/B confirmation.
- Supports fixed-shape synthetic token IDs, SGLang's generated shared-prefix workload, real custom JSONL conversations, and ShareGPT JSON. Real dataset files stay local and are validated before GPU launch.
- Builds workload-aware candidate families for scheduling, chunked prefill, KV/cache, CUDA Graph, memory pool, attention/MoE backend, and compatible speculative decoding features.
- Detects startup dependency and backend failures by capability family. After the first definitive MTP/EAGLE failure, it records the cause and skips remaining candidates in that family while continuing independent tuning work.
- Reserves trial budget for post-profile parameter candidates and confirmation before running exploratory Cookbook bundles. It never reports a baseline-only screen as an optimized deployment recommendation.
- Tests compatible combinations after one-factor screening. Every candidate that clears the configured improvement threshold is considered; strongest-candidate pairs run first, followed by other pairs and larger combinations as the remaining trial budget allows. Conflicting or duplicate configurations are excluded, and a combined configuration must win its own measured comparison before confirmation.
- Establishes a warm steady-state serving window before a bounded Nsight Systems capture, samples workload-time metrics during that capture, then routes queueing, CPU/GPU overlap, cache, graph, communication, and kernel evidence into a second tuning stage.
- Detects missing hardware/model/Triton-specific fused MoE configs in SGLang logs and records representative decode/prefill shapes plus an optional standalone tuning command.
- Writes structured artifacts, a reproducible launch command, rejected-trial evidence, and a Markdown report.
- Shows an elapsed-time and per-stage trial progress bar while experiments run. Interactive terminals and redirected logs use the same plain-text status, so `nohup` output remains readable.

## What It Does Not Do

- It does not claim a global optimum. A result is the best configuration within the recorded SGLang version, tested parameter space, hardware, workload, budget, and acceptance gates.
- It does not yet orchestrate multi-replica data parallelism, pipeline-parallel stage placement, or multi-node joint `(replica, TP, PP, EP)` search. Single-host TP feasibility/search, MoE EP candidates, and SGLang DP-attention candidates are supported and benchmark-gated; unsupported parallel dimensions are reported rather than implied to have been optimized.
- It does not modify drivers, CUDA packages, SGLang source, model weights, kernels, production services, or unowned processes.
- It does not make kernel changes automatically. Nsight Compute is used only after Nsight Systems has isolated a relevant kernel, and requires GPU performance-counter permission.
- It never runs the high-cost fused MoE autotuner as part of `inferopt run`. That operation has a separate command and requires explicit `--yes` approval.
- Single-host execution is implemented now. Multi-node and production rollout orchestration are intentionally out of scope for the first release.

## Requirements

- Python 3.9 or newer.
- A local SGLang checkout or installation runnable by the selected Python interpreter.
- An NVIDIA GPU host for the automatic execution workflow. AMD inventory/planning is available, but the automatic profiling executor is not yet implemented.
- A locally available model directory for execution.
- A SGLang-compatible benchmark entry point and an authorized output directory.
- `nsys` is required for the automatic `inferopt run` workflow. Nsight Compute is optional and may be blocked by driver-level performance-counter permissions.

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

During `init`, choose **Workload data**. `synthetic` preserves exact token lengths and can optionally use SGLang's generated shared-prefix dataset. `custom` reads a local JSONL file with at least two conversation turns per row; `sharegpt` reads a local ShareGPT JSON array. For real data, the entered input/output token counts are planning hints while SGLang measures the actual token distribution. Set **Shared prefix tokens** only for synthetic data; real datasets must contain their real shared prefixes.

Each `custom` JSONL row follows SGLang's native loader format. The first turn is the prompt and the second turn supplies the expected output length; either `content` or `value` is accepted:

```json
{"conversations":[{"content":"Summarize this incident report..."},{"content":"The incident was caused by..."}]}
```

`doctor` reads the installed benchmark's `--help` and blocks before any GPU launch when the selected SGLang version lacks the dataset, shared-prefix, or cache-flush flags required by the task.

The final **GPUs to use** prompt defaults to `all`, which preserves every GPU already visible to the `inferopt` process. Enter a comma-separated selection such as `0,1,2,3` to constrain the run. The chosen set controls feasibility checks, TP candidates, profiling, GPU-hour accounting, and the reproducible deployment environment.

`init` asks both deployment modes for the same optional p99 latency limits: E2E (request start to final token), TTFT (request start to first token), and TPOT (average generated-token time), all in milliseconds. Leave a value blank or enter `0` to omit that limit; leaving all three blank creates an objective-only task with no SLO constraint. Online defaults to a concurrency hint of `8`; offline defaults to `64`. In offline mode without SLOs, the benchmark omits `bench_serving --max-concurrency`, never searches SGLang `--max-running-requests`, and lets the server's resolved KV/admission policy determine sustainable throughput. It starts the baseline service exactly twice: one bounded Nsight Systems capture and one unprofiled parameter-screening service. Before stopping that unprofiled service, it flushes the cache and captures a second, longer confirmation-reference window. Candidate screens use five workload waves, at least 40 and at most 512 requests, while the final baseline/candidate window defaults to half of the selected experiment profile's former request count (`256` in balanced mode). The duration gate may expand a window that is still too short. The final candidate is compared only with the matched reference window, never with the shorter screening metric. With any SLO, client load is controlled through `bench_serving --max-concurrency`; the server's resolved `max_running_requests` is diagnostic evidence and is not overwritten by the tuner. Every resolved value and final benchmark command is saved in the trial artifacts. It then asks for **Experiment intensity**. `fast` is a short compatibility and coarse-ranking pass; `balanced` is the default evidence-driven search for routine tuning; `rigorous` adds coverage-style sensitivity sweeps and longer steady-state windows.

The profiler-directed screen is restart-aware. `fast`, `balanced`, and `rigorous` plan at most 6, 9, and 15 parameter candidates respectively before combinations, subject to the total trial budget. Candidates are scored from workload shape, GPU-active Nsys kernel shares, SGLang queue/cache/CUDA Graph observations, topology, and current resolved defaults. The controller covers one representative value from each high-impact mechanism before spending restarts on nearby values. Failed backend candidates do not satisfy successful coverage. After 3, 6, or 12 successful candidates respectively, a measured gain of at least 3% may stop the remaining fallback screen; otherwise the tool continues into lower-ranked KV, page-size, scheduler, and value-refinement candidates. The 3% early-stop threshold does not replace the 1% deployment acceptance threshold. This is a bounded best-in-tested-space search, not an exhaustive sweep.

Use **Concurrency points to measure** to provide an exact capacity/SLO curve such as `1,4,8,16,32` or `1 4 8 16 32`; the final point must equal the target concurrency. Leave it blank for the automatic target-first calibration. These points measure the baseline curve; final startup-parameter tuning remains targeted at the highest point.

`doctor` and `plan` do not start a server. `run --yes` starts only SGLang process groups created by the current experiment and only after the task passes validation. `doctor` exits nonzero and records `blocking_errors` when the checkpoint cannot fit a legal layout, the selected SGLang CLI is unavailable, the automatic executor is unsupported, or required `nsys` profiling is unavailable.

### Optional Fused MoE Autotuning

When SGLang logs show that the active fused MoE path fell back to a generic config, `report.md` records the missing filenames, observed shapes, and a standalone command similar to:

```bash
inferopt tune-moe \
  --task RUN_DIR/task.json \
  --profile RUN_DIR/profile/nsys-diagnosis.json \
  --result RUN_DIR/final.json \
  --output-dir RUN_DIR/optional-fused-moe-tuning \
  --yes \
  --output RUN_DIR/optional-fused-moe-tuning.json
```

This command can be expensive because it compiles and benchmarks many Triton kernel configurations. It is never called by the normal deployment workflow, does not install Ray or modify SGLang, and writes generated configs only under the specified output directory. After generation it automatically runs at least three interleaved baseline/candidate repetitions using the original workload, SLOs, 1% default improvement threshold, secondary-regression limits, and variation gate. Only `generated_config_deployable: true` authorizes the emitted `SGLANG_MOE_CONFIG_DIR` and deployment command; otherwise retain the original deployment. Use `--no-validate` only when you intentionally want a non-deployable config artifact without the end-to-end A/B stage.

The generated files follow the installed SGLang loader contract, not an arbitrary filename convention. They are placed under `candidate-config-root/configs/triton_<major>_<minor>_<patch>/`, use names of the form `E=<experts>,N=<per-TP-intermediate>,device_name=<torch.cuda.get_device_name() with spaces replaced by _>,dtype=<config dtype>[,block_shape=[n, k]][,per_channel_quant=True].json`, and contain a JSON object mapping token batch size `M` to a kernel configuration (`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, and `num_stages`, with optional `USE_TMA`). `N` is the loader's `w2.shape[2]` convention, not the unsharded model intermediate size. Triton version directories are mandatory because SGLang does not treat configs tuned for one Triton version as interchangeable.

If the log names a file ending in `_down.json`, the normal tuner is insufficient: SGLang is requesting paired up/down-projection configs. The tool will not mark a standard up-only result deployable. Use the official separate tuner workflow to capture `topk_ids`, then pass the directory with `--topk-ids-dir`; it must produce matching normal and `_down.json` files before end-to-end validation can authorize deployment. This workflow is version- and model-specific, and is intentionally explicit rather than silently generating a partial configuration.

For direct generation of the paired files without the full `tune-moe` A/B workflow, use:

```bash
inferopt generate-moe-config \
  --repository /sgl-workspace/sglang \
  --python /usr/bin/python3 \
  --model-path /workspace/models/MODEL \
  --tp-size 4 \
  --ep-size 1 \
  --dtype fp8_w8a8 \
  --topk-ids-dir /workspace/moe-topk-ids \
  --output-dir /workspace/moe-config-output \
  --yes
```

This defaults to `--mode paired`, invokes SGLang's `tuning_fused_moe_triton_sep.py`, verifies that both matching files were generated, and writes a ready-to-test `config-root/configs/triton_<version>/` tree. It deliberately refuses to fabricate `_down.json` by copying the normal file. `--mode standard` is available only for logs that do not request a down-kernel file.

For non-interactive use, begin with [`assets/task.autopilot.example.json`](assets/task.autopilot.example.json):

```bash
cp assets/task.autopilot.example.json task.json
inferopt validate --task task.json
inferopt run --task task.json --yes --output final.json
```

## Task Inputs

A task describes the model path, SGLang repository, Python executable, output directory, target workload, SLOs, and budget. The key inputs are:

- `deployment_mode`: `online_latency` or `offline_throughput`.
- `workload.max_concurrency`: a client-load hint for objective-only online experiments and explicit capacity curves. For adaptive SLO calibration, it is not a ceiling: the first probe uses SGLang's resolved `max_running_requests` after model/KV initialization.
- `calibration.strategy`: `adaptive` (the default) probes the runtime-resolved admission capacity when an SLO exists, then backs off geometrically and binary-searches only if it fails. This avoids assuming a small init-time concurrency is the server's capacity. Use `full_curve`, or supply `calibration.concurrencies`, when you explicitly need exact points such as `1, 2, 4, 8`.
- `calibration.min_concurrency` / `calibration.max_concurrency`: bounds the adaptive SLO fallback or a `full_curve` sweep. The final parameter search is revalidated at the selected SLO-safe load.
- `slo`: tail E2E, TTFT, TPOT/ITL, error-rate, and throughput constraints.
- `measurement`: warmup, minimum completed requests, and minimum steady-state duration.
- `workload.dataset`: optional real-data source. Use `{"name":"custom","path":"/absolute/requests.jsonl","apply_chat_template":true}` or the corresponding `sharegpt` form. Omit it for synthetic data.
- `search`: trial, GPU-hour, wall-clock, repeat, and variation limits.
- `capability_overrides`: explicit feature constraints, such as disabling speculative decoding for a known model/version incompatibility.
- `knowledge`: optional official model page and hardware references. With `allow_download: true`, `run` also creates a private, shallow snapshot of [`sgl-project/sgl-cookbook`](https://github.com/sgl-project/sgl-cookbook) under the run directory. The result records its commit and matching Markdown hashes; existing snapshot paths are reused without network access.

See [`references/input-schema.md`](references/input-schema.md) and the example task for the full schema.

## Decision Model

The controller keeps the baseline as a valid candidate. It only recommends a changed launch configuration when all required gates pass:

1. The candidate completes and preserves correctness/error-rate requirements.
2. Every declared SLO passes.
3. The measured improvement clears the configured practical-improvement and noise thresholds.
4. Mode-specific confirmation passes: offline no-SLO reruns only the selected candidate against the preserved baseline; SLO-constrained modes use repeated A/B confirmation.

Nsight Systems is routing evidence, not the final performance measurement. The tool extracts GPU-active kernel shares and top kernels, CUDA launch/queue latency, synchronization and allocation API shares, GPU activity gaps, memory-copy activity, and communication kernels. It combines these with SGLang logs and Prometheus samples to prioritize attention/MoE backends, CUDA Graph coverage, scheduler controls, prefix-cache policy, memory/KV settings, and parallelism. Nsys cannot determine memory-bandwidth utilization, occupancy, or instruction-level stalls; a trace-proven kernel hotspot requires a bounded Nsight Compute or kernel-microbenchmark follow-up. Profiled request latency is never used as the deployment winner metric.

For `online_latency`, undeclared latency metrics are also protected by the configured secondary-regression limit. For `offline_throughput`, latency is observational unless the user declares a latency SLO; an implicit latency-regression check does not veto a valid throughput gain.

`chunked_prefill_size` candidates are ordered rather than swept blindly. Online runs with a tail-latency SLO or measured prefill queue pressure test the nearest smaller, workload-derived values first, including an uncached shared-prefix suffix when applicable. Offline or objective-only runs test larger, throughput-amortizing values first. Every run records this direction and its evidence in `search-plan.json.chunked_prefill_strategy`.

When no candidate clears those gates, `recommendation_status` is `retain_confirmed_baseline`. This is a successful, evidence-backed result rather than a failed run.

## Artifacts

Every run records inventory, the current SGLang parameter audit, exact launch commands, raw benchmark outputs, server logs, runtime observations, profiling outputs, trial results, cookbook provenance/snapshot metadata, and the final decision. Generated artifacts are ignored by Git by default because they can include private model paths and workload details.

Prior candidate failures are reused only when model path and size, SGLang commit, selected hardware, visibility, and workload match exactly, and only for deterministic configuration/dependency/backend/memory failures. Transient timeouts, port conflicts, process kills, and GPU health events are never cached as reasons to skip future parameters.

## Safety

This tool is designed for explicitly authorized, single-host experiments. Review the generated plan before adding `--yes`. It never executes arbitrary shell snippets from the task, never installs packages, and never kills processes it did not create.

See [`SKILL.md`](SKILL.md) for the full operational workflow and [`references/safety-policy.md`](references/safety-policy.md) for the safety contract.

## License

Licensed under the [Apache License 2.0](LICENSE).
