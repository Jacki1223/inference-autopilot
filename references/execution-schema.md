# Single-Host Execution

Use `scripts/autotune.py` for bounded screening on one non-production host. It launches one SGLang server at a time, waits for `/v1/models`, runs `sglang.bench_serving`, analyzes the JSONL, stops the owned process group, and advances to the next trial.

## Required Additions

Add `execution`, `benchmark`, and `search` to the base task specification.

### Execution

- `python`: absolute Python executable from the SGLang environment.
- `host`: `127.0.0.1` or `localhost` only.
- `port`: dedicated unoccupied local port from 1024 through 65535.
- `offline`: default `true`; sets Hugging Face and Transformers offline modes. Setting it to `false` also requires `scope.allow_download=true`.
- `require_accelerator`: default `true`; refuse execution when NVIDIA/AMD hardware is not detected.
- `startup_timeout_sec`, `benchmark_timeout_sec`, `shutdown_timeout_sec`: positive bounds.
- `env`: optional scalar environment variables from the built-in allowlist only. Proxy variables (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, and lowercase forms) are supported; loopback is always added to the proxy bypass list so local benchmark traffic remains local.

### Benchmark

Supported fields:

- `dataset_name`: `random-ids`, `random`, `custom`, or `sharegpt`. Use `random-ids` for download-free synthetic input; the executor sends its generated token IDs directly so configured lengths are preserved. SGLang's `random` mode samples seed text from ShareGPT.
- `dataset_path`: required absolute existing path for `custom` and `sharegpt`; also required for `random` while offline. Online `random` may download ShareGPT only when `execution.offline=false` and `scope.allow_download=true`.
- `apply_chat_template`: optional boolean for `custom` and `sharegpt` prompts.
- `sharegpt_context_len`: optional positive context-length filter used by both real conversation loaders.
- `num_prompts`, `max_concurrency`, `request_rate`, `warmup_requests`, `seed`.
- `random_input_len`, `random_output_len`, `random_range_ratio` for `random-ids` and `random` data. The ratio is the minimum length divided by the configured maximum: use `1.0` for exact lengths; `0.2` samples uniformly from roughly 20% through 100% of each maximum.
- `output_details`; enabled automatically for goodput or error-rate gates and contains sensitive generated text.

Use `num_prompts >= 5 * max_concurrency` for final measurements when the budget permits.

The current SGLang `custom` loader expects JSONL. Each valid line contains `conversations` or `conversation` with at least a user and assistant turn; turn text may use `content` or `value`. `sharegpt` expects a JSON array with the corresponding conversation structure. The one-shot CLI checks this structure before starting a GPU service, but tokenizer-dependent length filtering still occurs inside the installed SGLang version.

### Search

`strategy` is currently `one_factor`. `baseline` is the fixed launch configuration. `space` maps one supported parameter to a list of values. Each candidate changes exactly one baseline parameter.

Measurement controls:

- `repetitions`: physical runs per configuration, from 1 through 9. Use 1 for screening and at least 3 for confirmation.
- `order`: currently `interleaved`. Odd rounds run baseline-to-candidate; even rounds reverse the order to reduce time-order bias.
- `max_cv_pct`: maximum population coefficient of variation for the objective metric. Default `10`.
- `min_confirm_repetitions`: repetitions required before a candidate can be confirmed. Default `3`.
- `require_all_slo_pass`: require every completed repetition to pass all SLO gates. Default `true`.

`budget.max_trials` counts physical trials, including repetitions. The executor uses the median of each metric across repetitions, emits all samples and objective CV, and distinguishes `screening_winner` from a statistically gated `winner`. A one-run search can identify a screening candidate but cannot produce a confirmed winner.

The final decision also contains `recommended_configuration` and `recommendation_status`. `confirmed_candidate` recommends a gated candidate; `retain_confirmed_baseline` explicitly recommends the stable baseline when every candidate is rejected. Inspect each candidate's `rejection_reasons` instead of treating an empty `winner` as an execution failure.

Re-evaluate a completed run after report or gate changes without launching a server:

```bash
python3 scripts/autotune.py report --run-dir /absolute/run/directory --output decision.json
```

The executor allowlists current SGLang parameters, including these families:

- parallelism: `tp_size`, `dp_size`, `pp_size`, `ep_size`, `moe_dp_size`;
- memory and admission: `mem_fraction_static`, `max_running_requests`, `max_total_tokens`, `page_size`, `kv_cache_dtype`;
- scheduling and graph coverage: `chunked_prefill_size`, `max_prefill_tokens`, `num_continuous_decode_steps`, `scheduler_recv_interval`, `cuda_graph_max_bs_decode`, `cuda_graph_max_bs_prefill`;
- backends: `attention_backend`, `prefill_attention_backend`, `decode_attention_backend`, `moe_runner_backend`, `bf16_gemm_backend`, `schedule_policy`.

Supported boolean parameters:

- `disable_cuda_graph`, `disable_radix_cache`, `disable_overlap_schedule`;
- `enable_mixed_chunk`, `enable_dp_attention`;
- `enable_two_batch_overlap`, `enable_single_batch_overlap`;
- `enable_mscclpp`, `enable_torch_symm_mem`, `disable_custom_all_reduce`;
- `enable_torch_compile`.

## Lifecycle And Artifacts

1. Validate all fields and reject unknown execution parameters.
2. Render the exact server and benchmark argument arrays without running them.
3. Require explicit approval and `--yes`.
4. Refuse a busy port and production scope.
5. Create a timestamped run directory and immutable spec, inventory, and manifest.
6. For each physical trial, store command arrays, PID/process group, logs, raw JSONL, summary, and status.
7. Stop only the process group created for that trial.
8. Stop on baseline failure, GPU health evidence, time/GPU-hour budget, or consecutive failure budget.
9. Aggregate repetitions into `aggregates.json`; write incremental `results.json` and final `final.json` for recovery and review.

## Current Limits

- One host and one SGLang server per trial.
- Sequential one-factor screening, not Bayesian optimization.
- No source or kernel modifications.
- `autotune.py` alone does not capture a profiler. `autopilot.py` requires a bounded baseline Nsight Systems capture (or a matching reusable capture) before it generates candidates.
- No correctness evaluator beyond benchmark request failures.
- No SSH, Slurm, Kubernetes, PD disaggregation, or multi-node orchestration.
