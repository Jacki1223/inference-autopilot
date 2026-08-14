# SGLang Adapter

## Discovery

Before running, locate and inspect:

- `python/sglang/bench_serving.py`
- `python/sglang/bench_offline_throughput.py`
- `python/sglang/profiler.py`
- `docs/developer_guide/benchmark_and_profiling.md`
- `docs/advanced_features/server_arguments.md`
- `benchmark/kernels/`

Use the checked-out version's help and documentation. Flags change quickly; never construct a launch command solely from this reference. Before every optimization run, build the allowed parameter set from `ServerArgs.add_cli_args` in the target repository's `python/sglang/srt/server_args.py`, then intersect it with the target runtime's `python -m sglang.launch_server --help`. Record both content hashes in the run artifact and reject a candidate absent from either surface.

## Baseline Command Shape

Use `python3 -m sglang.launch_server --help` and `python3 -m sglang.bench_serving --help` to resolve actual flags. A typical benchmark must specify the backend/URL, model, dataset or controlled lengths, request count, request rate or concurrency, and JSONL output.

For steady-state tests, prefer at least `num_prompts >= 5 * max_concurrency` when the budget permits, and require a completed-request window of at least 30 seconds by default. If the measured duration is shorter, increase request count, rerun the same warm-up/measurement procedure, and keep the short attempt only as an audit artifact. `request_goodput_rps` requires `--output-details`; store those files only in the private run directory because they include generated text and errors. Delete or redact generated text before sharing any report.

## Coarse-To-Fine Parameter Families

1. `tp/dp/ep/pp/cp`, aggregated versus PD, worker counts.
2. context and memory fraction, max tokens, max running requests, page size, KV dtype.
3. chunked prefill, max prefill tokens, mixed chunk, scheduling and overlap.
4. attention/GEMM/MoE/collective backend and CUDA Graph coverage.
5. radix/HiCache/HiSparse and routing policy.
6. speculative algorithm, depth, top-k, draft tokens and draft placement.

Reject incompatible combinations using server argument validation before consuming a full trial.

## Deployment Modes And Calibration

For `online_latency`, preserve declared E2E/TTFT/TPOT/ITL SLOs as hard gates.
Run a geometric closed-loop load calibration only to find a representative
analysis load, profile at that load, and validate every chosen launch parameter
again at the user's target workload. For `offline_throughput`, calibrate toward
the configured batch-pressure ceiling and prioritize KV allocation, admission,
continuous batching, chunk mixing, CUDA Graph coverage, and kernel backend
families. Do not claim a global optimum: identify the tested version-specific
search space and the evidence for every excluded parameter family.

## Metrics

Collect benchmark TTFT/TPOT/ITL and throughput plus server metrics for queue, running requests, token usage, cache hit, retractions, speculative acceptance, PD transfer, and expert distribution when enabled. Exact metric names are version-specific.

## Repository Integration

Keep orchestration outside SGLang initially. Upstream only narrow reusable changes such as metrics, profiler hooks, benchmark fields, config validation, or a demonstrated runtime optimization. Avoid embedding the whole optimizer into the scheduler.
