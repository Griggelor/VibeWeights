# v0.8.0.dev21 candidate — bound task continuation

**CPU reference accepted on the tested code revision; release metadata CI pending.** Basis: dev20 `14bc45df97f5c273fe8cd36eea8e2f7ba83a5965`. Primary target Windows 11 / AMD Strix Halo; CPU reference and remote CPU CI. No GPU execution claimed.

## Hypothesis and contract

A single host-authorized typed task can continue through successive, locally initiated recruitment events without the host choosing the requester, provider or child event. `RecruitmentContinuation(GroupWork(...)).run(...)` sweeps the original and emitted child events in deterministic breadth-first order. Only the original target/group key receives task inputs; unrelated groups on the same global event never see the task values. The continuation does not rank or assign experts and does not change their weights. Group formation and learned request decisions remain local.

The task identity binds original event, target/group, numeric inputs, goal, validator topics, attested oracle identity, limits, and immutable session budgets. The actual validator callables are trusted host attestations; they cannot be serialized or verified as identical code. Local task receipts remain in the catalog; global event payloads still do not carry task values.

Before a group run, wait for the exact event to have responses from all currently resident, bound perception heads. Bounded delivery remains the existing round-robin transport; it may process unrelated events, charged to the same budget. Persist delivery reservations before tick (interruption may overcharge), and reserve immutable deterministic run IDs before work. The budget counts distinct run identities, not wall time or repeated computation if interrupted before the first receipt. No new group run after the session run budget. No manual choice of child event. Reuse an existing group receipt and replay its dispatch idempotently; successful historical receipts return HISTORICAL with no newly verified value. A fresh successful group execution returns OK. Stale ancestry, delivery exhaustion, and missing groups fail closed. No general autonomous planning or arbitrary group-to-group task handoff.

```python
from vibeweights.continuation import RecruitmentContinuation
# work: GroupWork with admitted PerceptionHeads and RequestHeads
result = RecruitmentContinuation(work).run(
    "session-1", root_event, "explore", "research",
    {"input": (4,)}, "answer", validators,
    oracle_id="independent-formula", max_runs=3, max_deliveries=64,
    recruitment_budget=2,
)
```

## Verification plan

Four new integration tests: three-run automatic learned chain with unrelated group isolation, immutable task/session budgets across restart, recovery after catalog receipt and source commit, and delivery budget refusing partial group execution. No new test outcome claimed.

Commits: executor `5831c694c59f33e749c737e250d2cc67c263a335`, tests `cf35ded59f434dccabe0533d5cff786e9fb66f35`, CI queue efficiency `e7ba15e921ce00465bdb5352bdbf9cfae7445f66`. CI runs on main pushes and pull requests; superseded revisions of the same PR are cancelled, no test matrix or verification stage removed. Local DNS remains unavailable. Windows dev18 suite passed both Python versions on run 46141672717, additional stages pending when written. dev19 Linux 245/245 before the LF/device-probe follow-up, dev20 CI pending.

Next: finish Windows dev18/19/20/21 acceptance in sequence. Then non-affine transfer with independent holdout gate, heterogeneous structural fragment assimilation, and explicit target-host GPU acceptance.

## Abnahme

[CI 46142418702](https://github.com/Griggelor/VibeWeights/actions/runs/46142418702) auf `bce96a02ff120610ea770160a607ec0f8fb6c33c`: 253/253 Tests je Windows/Linux/Python 3.12 und 3.13; Geräteprobe, Gewichtsreproduktion und sämtliche Smoke-Tests erfolgreich. [JSON-Evidenz](../evidence/v08-task-continuation-ci.json). Der tatsächliche Strix-Halo-GPU-Lauf bleibt offen.
