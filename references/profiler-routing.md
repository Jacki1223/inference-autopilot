# Profiler Routing

| Question | First tool | Escalation |
|---|---|---|
| Which SLO fails and at what load? | serving benchmark JSONL and metrics | repeat load sweep |
| Prefill or decode? | stage-level framework profiler | separate prefill/decode workers |
| CPU/GPU bubbles or communication overlap? | PyTorch trace or Nsight Systems | rank-merged timeline |
| GPU memory allocation/leak? | framework memory snapshot | allocator-specific analysis |
| Which operator dominates? | PyTorch operator table/system trace | kernel trace aggregation |
| Why is a CUDA kernel slow? | isolated benchmark | Nsight Compute counters |
| Why does EP scale poorly? | expert/token metrics and system trace | DeepEP microbenchmark/network counters |
| Why does host KV restore stall? | cache metrics and timeline | PCIe/NUMA/storage counters |

## SGLang Tools

- `python3 -m sglang.bench_serving`: online latency and throughput; write JSONL with details.
- `python3 -m sglang.bench_offline_throughput`: scheduler throughput without HTTP.
- `python3 -m sglang.bench_one_batch`: model runner and kernel-oriented fixed batches.
- `python3 -m sglang.profiler`: controlled PyTorch profiler capture; supports stage separation and merged ranks.
- `examples/profiler/nsys_profile_tools/gputrc2graph.py`: aggregate GPU kernel time from Nsight Systems reports.
- `benchmark/kernels/`: targeted communication, attention, quantization, GEMM, MoE, and scheduler microbenchmarks.

## Profiling Rules

- Capture after warmup.
- Use the same representative workload as the failing SLO.
- Start with tens of steps, not full benchmark duration.
- Profile prefill and decode separately for PD deployments.
- Save tool version, arguments, rank, and time window.
- Avoid Nsight Compute on the entire server. Isolate the operator and shapes first. An Nsight Systems kernel percentage is a share of GPU-active kernel time, not a share of model-load or wall-clock time; do not turn it into an end-to-end Amdahl bound without separate request-level attribution.
- The automatic SGLang path starts the server under `nsys`, uses the framework CUDA profiler range to exclude startup and warmup, then parses CUDA timeline, kernel, API, memory, and NVTX reports. It captures Prometheus metrics and resolved server arguments alongside the trace.
- A profiled benchmark is attribution evidence, not a production-latency measurement. The unprofiled screening and confirmation runs remain the source of performance decisions.
