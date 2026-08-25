# CLI Progress

Long-running InferOpt commands write progress to stderr. Result JSON remains
on stdout or in the requested `--output` file.

Supported formats:

```bash
inferopt run --task task.json --yes --progress plain --output final.json
inferopt run --task task.json --yes --progress json --output final.json
inferopt run --task task.json --yes --progress none --output final.json
```

`plain` is the default. `json` emits one JSON object per event for CI, agents
and external UIs. `none` disables progress only; artifacts and results are
unchanged.

Known totals use a progress bar. Unknown-duration operations use elapsed-time
heartbeats every 30 seconds and never invent a percentage or ETA.

Trial progress tracks three independent counts:

```text
completed=<actually processed trials>
active=<currently running services/benchmarks>
queued=<not yet started>
```

Parallel trial indexes are labels, not completed counts. Outlier retries emit
an explicit plan-update event, including any displaced lower-priority trial.

Within a trial, progress reports:

- port availability;
- SGLang process launch;
- model/KV/CUDA Graph startup, including the latest bounded server-log line;
- server readiness;
- provisional smoke start/pass;
- benchmark attempt, request count and 30-second heartbeat;
- request-window expansion when measurement gates are not met;
- measurement-gate completion;
- process cleanup and accelerator-resource release.

Nsight progress reports server startup, steady-state preflight, bounded capture,
profile export, each `nsys stats` report and diagnosis completion. Fused-MoE
tuners report each shape group and a 30-second heartbeat with log locations.
