# SGLang Parameter Evolution

InferOpt reads the `ServerArgs` contract from the SGLang checkout selected by
the task. It does not assume that a packaged list of flags is current.

## Default: conservative

```json
{
  "parameter_evolution": {
    "mode": "conservative",
    "exploration_budget_pct": 10,
    "minimum_confidence": 0.8
  }
}
```

New, removed and changed flags appear in `doctor.json`, `plan.json` and the
final report. Every visible flag is classified in the Parameter Capability
Registry. Unknown or uncovered flags do not execute merely because they
exist; a safe bounded flag can execute only after semantic, context,
dependency/conflict, applicability and risk gates all pass.

This is not a per-parameter exception list. Mature parameters use versioned
trigger rules and workload-derived value functions. Other parameters use the
same generic evidence pipeline:

```text
current ServerArgs contract
  -> help/source/Cookbook semantics
  -> bounded value domain
  -> mechanism or catalog-family classification
  -> hardware/model/workload/SLO/bottleneck match
  -> dependency, conflict, applicability and risk checks
  -> ordinary measured Candidate Registry experiment
  -> refinement/composition/confirmation only after positive evidence
```

## Optional: bounded experimental discovery

Select `experimental` during `inferopt init`, or use:

```bash
inferopt init --non-interactive \
  ... \
  --parameter-evolution-mode experimental \
  --parameter-evolution-budget-pct 10 \
  --max-provisional-trials 2 \
  --output task.json
```

A new parameter is eligible only when all of the following are true:

- it was added relative to a previously saved SGLang contract;
- it is visible in the current `launch_server --help`;
- its type, action and candidate values are bounded;
- source paths, help text or current Cookbook content identify a serving-path
  performance mechanism with confidence at least `minimum_confidence`;
- it is not a path, secret, endpoint, debug, destructive or quality-sensitive
  control;
- the experiment mode is not `fast`;
- a reserved discovery slot is available.

The quota never reduces confirmation. Balanced uses at most two provisional
trials; max uses at most six, and both remain subject to the configured
percentage/absolute caps.

Every provisional configuration follows this lifecycle:

```text
current ServerArgs validation
  -> isolated server startup
  -> 4-16 request resident smoke
  -> full workload/SLO screen
  -> parameter-scoped failure circuit breaker
  -> normal refinement/composition (only with positive evidence)
  -> ordinary repeated confirmation before deployment
```

Failure evidence is fingerprinted by framework/model/hardware/workload and is
stored in the private history database. One local success remains provisional;
it does not modify the global rule catalog automatically.

## Contract tools

```bash
inferopt parameter-catalog \
  --repository /sgl-workspace/sglang \
  --output current-contract.json

inferopt parameter-diff \
  --baseline previous-contract.json \
  --current current-contract.json \
  --output parameter-diff.json
```

The scheduled GitHub workflow checks SGLang `main`, saves the current contract
for the next comparison, uploads the full audit artifact and creates or updates
a compatibility issue when the contract changes. CI uses a dependency-light
AST fallback only when the runtime `ServerArgs` import is unavailable; GPU
execution always uses the live argparse contract.

## Safety boundary

Parameter evolution never edits SGLang source, installs drivers, changes model
weights, enables lower precision, or executes arbitrary Cookbook commands.
Cookbook flags outside the validated rule list are retained as semantic/value
evidence only. The current ServerArgs binding, startup smoke, benchmark SLOs,
quality policy and repeated confirmation remain mandatory.
