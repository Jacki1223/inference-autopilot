# Inference Autopilot Roadmap

This roadmap prioritizes trustworthy, reproducible deployment decisions over a large but opaque parameter search. Dates are deliberately omitted: each phase is released when its acceptance criteria are met.

## Principles

- Measure the user's actual workload and SLOs before recommending a deployment command.
- Search only parameters that are valid for the installed SGLang version, model, topology, and observed bottleneck.
- Keep all recommendations reproducible with immutable task, environment, command, log, benchmark, and profiler artifacts.
- Treat a retained baseline as a valid result when no candidate clears the evidence gates.
- Never modify drivers, CUDA, model weights, or production processes without an explicit separate action.

## Phase 1: Strong Single-Host Foundation

### Task Experience

- Guided `init` for deployment mode, experiment intensity, workload sizes, shared-prefix behavior, target concurrency, and explicit concurrency points.
- Import and validate non-interactive JSON tasks for CI and repeatable experiments.
- Add workload distributions instead of fixed input/output lengths: percentiles, request-rate traces, prompt mixes, and configurable prefix locality.
- Make SLO setup explicit during `init`, including E2E, TTFT, TPOT/ITL, error-rate, and throughput gates.

### Efficient Evidence Collection

- Adaptive measurement budgets: short compatibility checks, bounded coarse screening, and longer confirmation only for close candidates.
- Sequential stopping based on SLO failure, confidence intervals, effect size, and measurement variance.
- Correct generated shared-prefix request accounting at every stage, including profiling ranges.
- Surface estimated remaining trials and wall-clock/GPU budget in live progress output.

### Single-GPU Parameter Search

- Expand workload-aware SGLang candidate routing for scheduling, chunked prefill, KV cache, CUDA Graph, attention, MoE, tokenizer, and CPU-front-end controls.
- Record every visible SGLang parameter as applied, inapplicable, unsupported, deferred, or rejected with evidence.
- Add a small Bayesian/surrogate refinement pass only after evidence-driven one-factor screening; never replace baseline confirmation with a model prediction.
- Add configuration minimization: remove one changed flag at a time from a winner to identify the smallest deployable command.

### Acceptance Criteria

- A one-command run produces a valid command, comparison table, raw evidence, and an explicit non-recommendation reason when appropriate.
- On a representative single-GPU workload, adaptive measurement reduces experiment time relative to fixed full-duration trials without changing the confirmed decision beyond its uncertainty bounds.

## Phase 2: Multi-GPU Single-Host

### Topology and Feasibility

- Discover GPU count, per-GPU memory, NVLink/NVSwitch versus PCIe topology, NUMA layout, CPU affinity, and local NIC placement.
- Enumerate deployable TP, PP, DP, DPA, EP, and MoE-DP layouts before starting expensive experiments.
- Reject invalid degree combinations early, including divisibility, model-architecture, sequence-length, and memory constraints.
- Recommend quantized local variants only after a declared quality gate clears them.

### Benchmark and Profiling

- Collect rank-aware SGLang logs and per-rank GPU metrics.
- Capture Nsight Systems traces on all ranks with synchronized workload ranges, then merge collective, compute, and idle-time evidence.
- Parse NCCL traces, collective sizes, overlap, rank imbalance, and stragglers rather than relying only on aggregate throughput.
- Add topology-aware bottleneck classes: PCIe saturation, NCCL serialization, PP bubble, EP imbalance, DPA overhead, and CPU/NIC affinity issues.

### Search Strategy

- Search parallel layouts in stages: feasible layouts, coarse throughput/SLO screening, then per-layout SGLang runtime tuning.
- Keep topology selection separate from fine-grained server flags so the search does not become a full Cartesian product.
- Compare normalized efficiency: throughput per GPU, p99 latency, memory headroom, communication fraction, and cost per generated token.

### Acceptance Criteria

- Given a local 2-8 GPU host, the tool returns ranked feasible parallel layouts and explains why rejected layouts do not fit or fail the SLO.
- A report identifies whether the limiting factor is compute, memory/KV capacity, communication, pipeline bubbles, or request scheduling.

## Phase 3: Multi-Node Serving

### Cluster Input and Safety

- Support an explicit inventory file describing nodes, GPU topology, interconnect, NICs, hostnames, and approved launch mechanism.
- Provide read-only cluster preflight: SSH reachability, version parity, CUDA/NCCL/SGLang compatibility, clock skew, routing, MTU, and fabric health.
- Integrate with explicit launch adapters for Slurm, Kubernetes, and static SSH hosts. No implicit cluster mutation.
- Use a per-run namespace, port plan, ownership manifest, and cleanup protocol so concurrent experiments cannot interfere.

### Distributed Deployment Options

- Evaluate TP/PP/DP/EP/DPA across nodes with topology constraints and communication-aware feasibility estimates.
- Evaluate prefill/decode disaggregation when workload shape and fabric latency justify it.
- Evaluate data-parallel replicas and request routing policies for online traffic.
- Add placement advice: which ranks should share an NVLink island, which traffic crosses nodes, and when a layout is not worthwhile.

### Distributed Observability

- Synchronized benchmark driver with request IDs and a common time base.
- Per-node Nsys capture plus NCCL and NIC telemetry; produce a unified critical-path timeline.
- Diagnose all-reduce/all-to-all contention, slow ranks, PCIe/NVLink/IB/RoCE saturation, packet retransmits, CPU starvation, and imbalanced experts.
- Report service-level metrics and rank-level causes together, rather than attributing distributed latency to one kernel.

### Acceptance Criteria

- A supplied cluster inventory produces a safe launch plan, explicit topology assumptions, and an auditable resource allocation.
- The system distinguishes a compute-bound layout from a communication-bound layout using synchronized evidence across ranks and nodes.

## Phase 4: Workload and Model Coverage

### Multimodal Serving

- Add `workload.type`: `text`, `image_text`, `video_text`, and `audio_text`.
- Describe image count, resolution, formats, video frames/FPS, audio duration, and encoder/decode request mixes.
- Benchmark realistic OpenAI-compatible multimodal requests instead of token-only synthetic prompts.
- Report encoder latency, vision/audio-token processing, prefill, decode, memory use, cache behavior, and end-to-end SLOs separately.

### Other Serving Modes

- Add first-class workload adapters for embedding, reranking, classification, and batch/offline jobs.
- Add LoRA and multi-adapter workloads, including load/eviction behavior and quality/correctness gates.
- Add model-specific Cookbook adapters with versioned evidence and compatibility tests.

### Acceptance Criteria

- A multimodal recommendation is based on measured multimodal traffic and separates encoder versus language-model bottlenecks.
- Text-only benchmarks are never presented as a full multimodal deployment conclusion.

## Phase 5: Operator and Runtime Optimization

### Escalation Path

- Use Nsys to prove a material serving-path bottleneck before requesting Nsight Compute counters.
- Automate NCU permission detection and collect counters only for selected kernels and final candidate configurations.
- Produce operator tickets with launch geometry, shapes, dtype, call frequency, GPU-active-time share, Amdahl bound, and reproduction command.
- Add optional benchmark-only kernel/plugin experiments behind explicit user approval; never patch an installed SGLang checkout automatically.

### Runtime Improvements

- Mine SGLang logs for dynamic batching, cache fragmentation, CUDA Graph misses, tokenizer pressure, and MoE expert skew.
- Support controlled A/B validation of custom kernels, attention backends, and MoE backends.
- Publish anonymized, versioned hardware-model-workload capability records only when users opt in.

### Acceptance Criteria

- Each low-level optimization proposal includes a measurable upper bound and an end-to-end serving validation plan.
- Kernel microbenchmarks alone cannot mark a deployment recommendation as improved.

## Phase 6: Continuous Optimization and Operations

- Scheduled regression runs against a versioned workload suite after SGLang, CUDA, driver, model, or hardware changes.
- Compare new results against a confirmed baseline with statistical gates and explain regressions.
- Export machine-readable results for dashboards, CI, experiment tracking, and issue templates.
- Add cost, energy, and capacity metrics: tokens per GPU-hour, tokens per watt, safe concurrency, and SLO-constrained cost per token.
- Add a deployment handoff bundle: final command, environment manifest, model revision, benchmark recipe, profiler evidence, rollback command, and known limitations.

## Explicit Non-Goals

- Claiming a global optimum from a bounded experiment.
- Automatically changing production configuration, drivers, packages, model weights, or kernels.
- Treating a single short benchmark as a deployment decision.
- Hiding unsupported model, topology, profiler-permission, or quality-gate limitations.
