# Inference Autopilot Roadmap

Inference Autopilot is evolving toward an automated optimization system for SGLang deployments: given hardware, a model, a representative workload, and optional SLOs, it should produce a measured deployment recommendation, explain the limiting factors, and identify the next optimization opportunity.

The roadmap is organized into three workstreams. They are not strictly sequential: single-host improvements remain continuous, while multi-node and kernel-optimization work can progress in parallel when the required hardware is available.

## 1. Strengthen the Current Single-Host Optimizer

The first priority is to make the existing tool more accurate, faster to run, easier to use, and useful across more models, GPUs, and workloads.

### Optimization quality

- Continue expanding the canonical Candidate Registry and bottleneck-driven mechanism rules.
- Derive candidate values from the installed SGLang version, effective defaults, hardware limits, model architecture, workload shape, SLOs, and measured runtime evidence.
- Improve adaptive search: broad mechanism coverage first, focused value refinement second, compatible composition third, and repeated confirmation last.
- Use controlled intervention results to update the initial bottleneck diagnosis instead of treating the pre-search profile as final truth.
- Keep precision-changing, model-switching, and quality-sensitive candidates behind explicit quality gates.
- Minimize the final command by rejecting parameters that do not add measurable value over their strongest parent configuration.

### Runtime and experiment cost

- Reduce unnecessary model restarts and reuse resident services where experimental isolation permits it.
- Improve sequential stopping so clear losses terminate quickly and ambiguous candidates receive additional evidence only when needed.
- Warm-start from strictly compatible historical trials without allowing history to consume discovery slots or override fresh measurements.
- Estimate remaining wall time, GPU-hours, requests, and trial stages before and during execution.
- Run independent single-GPU candidates concurrently on otherwise idle GPUs while preserving exclusive resource placement and comparable benchmark windows.

### Coverage and usability

- Improve one-command execution from validated task input through profiling, search, confirmation, and report generation.
- Expand realistic workload support: synthetic token shapes, shared prefixes, custom JSONL, ShareGPT, request-rate traces, mixed prompt lengths, and real production datasets.
- Add first-class multimodal, embedding, reranking, LoRA, and multi-adapter workload adapters without presenting text-only evidence as a complete conclusion.
- Expand model-specific Cookbook ingestion while always validating recipes against the current checkpoint, hardware, and installed SGLang parameter surface.
- Make every candidate auditable as applicable, scheduled, executed, measured, rejected, deferred, unsupported, or quality-sensitive.
- Improve reports, CI compatibility checks, resumability, trial history, cost per token, and deployment handoff artifacts.

### Exit criteria

- A new user can provide model, hardware, workload, mode, and optional SLOs once, then obtain a reproducible launch command and evidence report without an Agent.
- On representative workloads, adaptive search reaches the same deployment decision as a wider reference search while using materially less wall time and GPU budget.
- Unsupported or insufficiently tested mechanisms produce an explicit bounded-result warning instead of a false optimum claim.

## 2. Extend from Single-Host to Multi-Node Optimization

The second workstream extends the current single-host optimizer into a topology-aware distributed deployment optimizer.

### Cluster discovery and safety

- Accept an explicit cluster inventory describing nodes, GPUs, GPU interconnects, NICs, NUMA layout, hostnames, and the approved launch mechanism.
- Add read-only preflight checks for reachability, driver/CUDA/NCCL/SGLang parity, clock skew, MTU, RDMA/RoCE/InfiniBand health, ports, and model availability.
- Support controlled launch adapters for Slurm, Kubernetes, and static SSH hosts.
- Use per-run namespaces, port allocation, ownership manifests, and deterministic cleanup so concurrent experiments cannot interfere.

### Distributed topology search

- Enumerate feasible TP, PP, DP, DPA, EP, MoE-DP, and context-parallel layouts before running expensive benchmarks.
- Keep topology selection separate from fine-grained SGLang flag tuning to avoid a full Cartesian product.
- Model memory fit, head/KV-head divisibility, pipeline balance, expert placement, NVLink/NVSwitch islands, PCIe boundaries, and inter-node communication cost.
- Evaluate replica layouts, request routing, and prefill/decode disaggregation when the workload and network make them relevant.
- Compare absolute throughput and latency together with throughput per GPU, communication fraction, memory headroom, cost per token, and SLO-safe capacity.

### Distributed observability

- Run synchronized benchmark traffic with request IDs and a common time base across nodes.
- Collect per-rank SGLang logs, GPU telemetry, NCCL traces, NIC counters, and bounded Nsys captures.
- Diagnose collective serialization, all-to-all contention, PP bubbles, rank imbalance, expert skew, stragglers, CPU/NUMA affinity, retransmits, and fabric saturation.
- Produce a unified critical-path report that connects service-level latency or throughput loss to rank- and node-level causes.

### Exit criteria

- A supplied cluster inventory produces a safe, auditable launch plan and ranked feasible distributed layouts.
- The tool explains why rejected layouts do not fit, fail an SLO, or waste communication resources.
- Multi-node recommendations are confirmed with synchronized end-to-end measurements rather than extrapolated from single-host results.

## 3. Add Kernel-Level Analysis and Optimization

The third workstream continues beyond launch-parameter tuning. After Nsys identifies a material GPU hotspot, InferOpt should perform bounded NCU analysis and produce operator-level optimization directions.

### Automated escalation pipeline

1. Use Nsys to identify GPU-active operator families, call frequency, shapes, dtype, launch behavior, and their share of total GPU kernel time.
2. Estimate the operator's Amdahl upper bound and skip expensive NCU work when the maximum possible end-to-end gain is insignificant.
3. Detect NCU availability and GPU performance-counter permissions before scheduling any capture.
4. Select a small number of representative kernel launches or construct a shape-matched microbenchmark instead of profiling the entire serving process with NCU.
5. Collect compute throughput, Tensor Core utilization, DRAM/L2 traffic, arithmetic intensity, occupancy, warp stalls, launch geometry, instruction mix, and source correlation when available.
6. Classify the hotspot as compute-, memory-, latency-, launch-, synchronization-, communication-, or occupancy-bound.
7. Generate evidence-linked suggestions such as dtype/layout changes, fusion, tiling, vectorization, persistent kernels, CUDA Graph coverage, backend changes, MoE configuration tuning, or shape specialization.
8. Validate any implementation with both a kernel microbenchmark and the original end-to-end workload/SLO. A faster kernel alone must never become a deployment recommendation.

### Does this require an Agent?

An Agent should be optional, not required for the core workflow.

The standalone CLI must automatically handle:

- hotspot selection and Amdahl filtering;
- NCU permission and capability detection;
- bounded capture and metric collection;
- known counter-based bottleneck classification;
- source/configuration mapping where deterministic metadata is available;
- standard optimization suggestions with supporting evidence;
- reproducible commands, operator tickets, and end-to-end validation plans.

An optional Agent can add substantial value for open-ended work:

- reading the relevant SGLang, Triton, CUTLASS, FlashInfer, or CUDA source path;
- connecting unusual counter combinations to model architecture and workload behavior;
- searching upstream issues, papers, and implementation alternatives;
- proposing a custom kernel or source-level patch;
- generating and iterating benchmark code;
- explaining trade-offs when several optimizations are plausible.

The recommended architecture is therefore:

```text
InferOpt CLI
  -> deterministic Nsys/NCU collection
  -> structured operator_optimization.json
  -> rule-based diagnosis and standard recommendations
  -> optional Agent/Skill for deeper source-level optimization
  -> explicit user approval
  -> isolated implementation and A/B validation
```

The Agent must not be allowed to modify an installed production environment automatically. Source changes, compilation, custom kernels, and deployment require an explicit separate action, isolated workspace, recorded diff, rollback path, and end-to-end confirmation.

### Exit criteria

- Every kernel recommendation contains the measured hotspot share, expected upper bound, NCU evidence, reproduction command, suggested change, and validation plan.
- The no-Agent CLI produces useful, correct standard recommendations for known bottleneck patterns.
- Agent-assisted optimization can propose source-level changes, but only measured end-to-end gains that preserve correctness and SLOs are accepted.

## Guiding Principles

- Optimize the user's actual hardware, model, workload, and SLOs—not a universal synthetic score.
- Prefer a smaller evidence-backed search over a large opaque parameter sweep.
- Treat retained baselines and bounded non-recommendations as valid outcomes.
- Keep measurements, commands, logs, profiler evidence, decisions, and limitations reproducible.
- Never modify production services, drivers, packages, weights, or kernels without explicit authorization.
