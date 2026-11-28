# v0.8.0.dev20 candidate — trained local recruitment requests

**Not accepted yet.** Base dev19 candidate `a3e2d5b3d31d0075003d7532c08760c5a9051c59`. Primary target Windows 11 / AMD Strix Halo; CPU reference training and inference only in this slice.

## Hypothesis

Request initiation can be a learned local competence independent of the existing participation heads. On the same group-local missing-input situation, independently trained cores may choose different actions: request or abstain. A host authorizes an opt-in per-run budget and supplies the typed task and validators, but does not choose requesters or providers.

## Implementation

- `RequestHeads`: separate core-bound sigmoid head with four local binary features (source absent, output absent, RECRUIT parent, goal matches local output). SGD, zero initialization, max 512 steps, immutable SafeTensors payload, spec and weights content-bound, immutable independent holdout receipt. No automatic head replacement.
- `GroupWork(..., request_heads=heads).execute(..., recruitment_budget=2, recruitment_policy='learned')`: learned decisions only; an expert without an admitted head abstains. No symbolic fallback. Existing typed path remains opt-in and unchanged. Missing-input gate is a publication safety condition, not a request generator.
- All admitted decisions, including abstentions, are bound to the historical group receipt. Decisions are recomputed against current weights, report and local features before publication, replay and recursive event liveness. The complete authorized batch is required; manual publication cannot circumvent a learned abstention or exhausted budget. Requests contain topic metadata only.
- Existing receipt-first/source-DB-commit protocol remains; no claim of cross-DB atomicity or automatic continuation of the task. Replaying a committed intent is idempotent. Local feature contracts and acceptance labels remain host-designed; this is not open-ended learned problem discovery.

## Evidence and limits

Container-only synthetic SGD experiment (PyTorch 2.10.0+cpu, lr=0.5, 512 steps, OMP/MKL=1, eight training inputs and eight disjoint held-out binary inputs per rule). The last feature is zero for training and one for holdout, deliberately irrelevant to both artificial policies. Consumer rule: missing source AND missing output AND not RECRUIT. Middle rule: missing source AND missing output AND RECRUIT. Final observed loss 0.06857049465179443 and 0.10157248377799988 respectively; both 8/8 holdout. This is only an isolated torch experiment, not an import of the production module or a real-world task.

Four additional tests in `tests/test_neural_request_heads_v08.py` cover trained divergent decisions in the two-depth recruitment chain, no head/no fallback, independent rejected and immutable admission, and interruption/restart/replay. The first also corrupts an admitted request artifact and expects downstream events to become stale. No production-path or CI success claimed yet.

Commits: head storage/training `2b226867e3f7217c7e18e3228948b021a27ff636`, decision replay `cd7a776d0c6b87e285145f924bb541deb84e9c62`, group integration `0e669d4092d9582f33308c70401472729c343a83`, publication gate `c3b6ab57e6b65f8b84ab9165291a46559baa346b`, tests `4ee4deb144530179b961ca7d7d106a6de8e1c4e2`. Local Git/DNS reconfirmed unavailable; GitHub Actions required.

Next: full Windows/Linux acceptance; bound task continuation across child events (same independently attested task, no host hand-selected event), learned non-affine transfer and heterogeneous fragment ABI. GPU acceptance remains a separate target-host test.
