# Safety And Experiment Integrity

## Authorization Gates

Require explicit authorization before:

- launching or stopping remote services;
- using production endpoints or traces;
- installing packages, drivers, profilers, or kernels;
- downloading models or datasets;
- allocating cloud or shared cluster resources;
- changing Kubernetes, Slurm, system, driver, power, or clock configuration;
- enabling profilers with material service overhead;
- applying generated source or kernel changes;
- publishing traces, reports, patches, or benchmark results.

## Process Ownership

For every launched process, record PID, command, working directory, start time, log paths, and parent experiment ID. Stop only recorded owned PIDs. If ownership is uncertain, leave the process running and report it.

## Data Privacy

Prefer token lengths, timestamps, hashes, and prefix relationships over raw prompts. Never place credentials or raw private prompts in task specifications, command lines, filenames, reports, or source control.

## Failure Gates

Stop a trial on:

- output mismatch or invalid structured output;
- request error rate above the task limit;
- OOM or memory corruption;
- server health failure;
- NaN/Inf in model output or kernel validation;
- repeated timeout, deadlock, or watchdog failure;
- GPU temperature, power, ECC, or Xid anomaly;
- budget exhaustion.

After three consecutive failures with the same cause, stop the optimization loop and report the blocking condition.

## Result Integrity

- Preserve raw output and exact commands.
- Never silently drop failed or slow requests.
- Separate warmup from measurement.
- Compare equal workloads and arrival schedules.
- Account for profiler overhead.
- Report regressions and inconclusive results.
- Do not cherry-pick a single run. Repeat the baseline when environment drift is possible.
