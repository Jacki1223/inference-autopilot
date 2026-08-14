# Bottleneck Diagnosis Playbook

## Decision Table

| Evidence | Likely bottleneck | Next controlled test |
|---|---|---|
| TTFT rises with request rate, GPU not saturated | queue/admission/frontend | inspect queue, tokenizer, scheduler CPU, request batching |
| TTFT grows sharply with prompt length | prefill compute or KV allocation | prefill-only sweep, chunk size sweep, attention/GEMM trace |
| TPOT grows with context and batch, high HBM bandwidth | decode attention/KV bandwidth | KV dtype, attention backend, page size, sparse attention |
| High recomputed tokens, low prefix hit | cache/routing | validate prompt identity, radix cache, cache-aware routing |
| Restore/prefetch dominates TTFT | hierarchical KV IO | measure tier bandwidth, threshold, overlap, hit size distribution |
| GPU gaps aligned with CPU work | scheduler/tokenizer overhead | PyTorch/system timeline, overlap scheduling, CUDA Graph coverage |
| Collective time grows with scale | TP/EP communication | topology validation, message sizes, backend, quantized/fused collective |
| All-to-All tail and uneven expert tokens | MoE imbalance | expert histogram, EPLB placement, redundant experts, DeepEP mode |
| Draft work high, accepted tokens low | speculative waste | disable spec, depth sweep, per-workload acceptance analysis |
| Verify dominates with high acceptance | spec verify/kernel/batch shape | draft token tree and verify backend profiling |
| Decode GPU idle while prefill overloaded | PD rate mismatch | P/D capacity sweep, KV transfer and queue measurements |
| One kernel dominates trace | operator bottleneck | isolate representative shapes and use kernel profiler |

## Required Decomposition

Separate end-to-end latency into at least:

```text
frontend + queue + prefill + KV transfer/restore + decode + detokenize/network
```

For speculative decoding, further split decode into draft, verify, accepted extend, rejected work, and synchronization. For MoE, split attention, router, dispatch, expert compute, combine, and collective wait.

## Amdahl Gate

Do not optimize a hot operator until its maximum plausible end-to-end gain justifies the effort:

```text
max_speedup = 1 / ((1 - hot_fraction) + hot_fraction / operator_speedup)
```

Prefer a configuration or scheduling fix when it has a larger reachable gain and lower blast radius.

## Search Strategy

1. Run a low/mid/high load baseline.
2. Identify the first SLO to fail.
3. Change one subsystem at a time.
4. Use short screening trials to eliminate poor regions.
5. Re-run finalists for steady state and tail metrics.
6. Confirm the winner on at least one workload variation to prevent overfitting.
# Workload-Aware Chunked Prefill

For fixed-length closed-loop workloads, calculate a first-order batch boundary as `max_concurrency * input_tokens`. Include values below, at, and above that boundary when screening `chunked_prefill_size`, plus the framework default. Treat this only as a search-space prior.

Use scheduler logs to verify the mechanism. If a candidate turns one `N-request / T-token` prefill batch into several smaller prefill batches, and TTFT plus throughput regress while decode batch size stays unchanged, classify it as excessive prefill fragmentation. Do not generalize the result to variable-length or open-loop traffic; repeat against the production length and arrival distributions.
