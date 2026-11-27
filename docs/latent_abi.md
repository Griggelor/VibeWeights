> Historische Referenz bis 0.2.0. Aktuelle Plattformverträge: [platform_v03.md](platform_v03.md).

# Latent ABI 0.1

Canonical state: immutable tuples of nine bounded integers and nine booleans. Tensor views: CPU float32, shape [number of requested slots] (or batch for expert-forward tests), exact integer representation. Each expert reads ordered fields declared by manifest. `valid=false` means unavailable, canonical zero payload; valid zero is distinct. False for GE is valid zero. No separate unknown/not-applicable reason yet. IDs 1..16,777,215 (exact float32 range). All other slots nonnegative. No normalization beyond defined ranges.

| Slot/index | Meaning | Range | Writers |
|---|---|---|---|
| X/0 | factor | 0..31 | structured input |
| Y/1 | repetition count/factor | 0..31 | input OR READ; EVICT clears |
| THRESHOLD/2 | compare target | 0..961 | input |
| ACC/3 | accumulated sum | 0..961 | init, step |
| REMAIN/4 | remaining iterations | 0..31 | init, step |
| PRODUCT/5 | final product | 0..961 | finish |
| GE/6 | product >= threshold | 0..1 | logic.ge |
| KEY/7 | memory search key | 1..16777215 | input |
| RECORD/8 | pinned source record | 1..16777215 | SEARCH |

Expert contract: `Expert.forward(x: Tensor) -> Tensor`, `Expert.run(state: State) -> State | Tensor` (action head returns logits). `SPECS` defines the current registry/ordering, layer shapes and write permissions. State.apply checks output shape, dtype, finite values, integrality, allowed range and canonical mask; atomic update by immutable replacement. Action scores must be exactly one-hot across 5 actions. Core does not accept natural language messages from experts. Traces are human-readable debug metadata, not internal reasoning communication. No calibrated confidence or arbitrary routing requests yet.

For this single-writer sequential reference, updates occur in fixed phases with no merge. ACC/REMAIN may only be changed by declared projections; REMAIN must decrease exactly by 1. Expert input tensors are copied, not mutable views of canonical state. Output status and trace are separate runtime metadata. `steps` counts neural dispatches (not SQL calls or verbalization). On budget exhaustion, invalid state, stagnation or memory failure, status != OK. Intermediate state is diagnostic only. Failure cannot publish a successful verbalized result. No independent product correctness verifier yet.

## Analytical weight construction

All layers use ReLU after affine. For legal inputs:

- init: W=[[0],[1]], b=[0,0], input Y → [0,Y]. 4 parameters.
- step: W=[[1,1,0],[0,0,1]], b=[0,-1], input ACC,X,REMAIN → [ACC+X, REMAIN-1] on reachable positive-counter states. 8 parameters, reused on each iteration. Invariant after k steps: ACC=kX, REMAIN=Y-k; hence ACC=XY at halt. The step is not a standalone multiply expert.
- finish: W=[[1]], b=[0], ACC → PRODUCT. 2 parameters.
- GE: hidden affine [[1,-1],[1,-1]], b=[1,0]; output [1,-1], b=0. For integer d=PRODUCT-THRESHOLD, ReLU(d+1)-ReLU(d) equals indicator(d>=0). 9 parameters.
- memory.action: mask i=RECORD valid, v=Y valid, r=GE valid. Scores = ReLU([1-i-r, i-v-r, v-r, v+r-1, r-v]). On reachable masks: 000→SEARCH, 100→READ, 110→READY, 111→EVICT, 101→HALT. 20 parameters. Unsupported masks are not guaranteed meaningful; runtime validates one-hot and dependencies/phase. No examples used for weight design.

Total 43 scalar parameters including biases. FP32 payload 172 bytes excluding file headers. Actual artifacts in `artifacts/weights`; manifests state origin DIRECT. Experimental status remains even after scoped tests pass.
