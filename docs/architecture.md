> Historische Referenz bis 0.2.0. Aktuelle Plattformverträge: [platform_v03.md](platform_v03.md).

# Architektur v0.1 — experimentelle Referenz

Ziel unverändert: ~512 unabhängig erzeugbare Experten à ~100M Parameter, nichtsprachlicher Austausch, adaptive rekurrente Verarbeitung, selbstverwalteter indexierter Kontext, separater Input-/Output-Experte. Kein LLM im Inference-Pfad dieser Referenz, kein Training. Quelle: PROJEKTAUFTRAG.md.

## Tatsächlich implementiert

```
structured input -> immutable State
                 -> [memory.action -> SEARCH -> memory.action -> READ -> READY] (optional)
                 -> arith.init -> arith.step (shared weights, repeated) -> arith.finish
                 -> logic.ge
                 -> [memory.action -> EVICT -> memory.action -> HALT] (optional)
                 -> symbolic verbalizer
```

All registered experts are PyTorch affine/ReLU networks with frozen parameters loaded from separate safetensors. Neither generator nor independent reference oracle is invoked during inference. State changes are validated before publication. Tensor exchange uses a deliberately explicit, structured numerical representation, not an emergent semantic embedding. The reference controller has an explicit fixed chain and hard step budget, and accepts only phase-permitted actions. The controller does not compute the product or comparison. The memory head determines action from validity masks, but not what semantic information is relevant to a free-form task. The supplied key comes from structured input. The output module formats values only.

## Architecture decisions

- ADR-001: FP32 CPU, bounded integer semantic slots, isolated tensor snapshots, immutable canonical state. No GPU promises. Safetensors + SHA-256 + manifests; integrity against accidental corruption, not authenticity.
- ADR-002: analytic construction first; no training, pretrained or search-based modification. A unit of functionality may be much smaller than 100M. No padding. The five current modules are proof modules, not five mature target-pool competencies. `arith.init` and `arith.finish` are support projections.
- ADR-003: deterministic single-threaded controller, phase-specific action permissions, decrement progress check, bounded total dispatches (1–128; default 48). No general adaptive router or independent semantic verifier. Fail statuses suppress verbalization of incomplete results. A wrong yet in-range result from corrupted in-memory weights can pass; mutation evidence demonstrates dependency, not built-in fault detection.
- ADR-004: SQLite exact-key index, immutable version rows, source label, positive stable ID, one active retrieved scalar. Search chooses latest version and read pins the ID. No semantic ranking or automatic fact discovery. External ingestion only. No external service.
- ADR-005: structured CLI input and deterministic verbalizer, classified SYMBOLIC. No general language encoding/decoding. Input/output outside planned 512 pool pending future research.

Target controller functions: select experts using numeric requests, enforce dependencies/write permissions and bounded compute, arbitrate conflicts, observe progress, permit verification before finalization. Native actions must remain numeric; backend side effects are deterministic and validated. Future ABI changes require versioning and compatibility tests. Current runtime is intentionally task-specific; parallel merges, adapter registry, generic plugin discovery and learned latent semantics are not implemented.
