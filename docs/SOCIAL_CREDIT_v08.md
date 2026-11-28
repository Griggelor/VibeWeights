# dev30 candidate — distinguish individual and cooperative credit

**Implementation and prespecified experiment, results pending.** dev29 actual outcome: 45/48 joint tasks vs 0/48 without recall, but only 75/144 overall; one individual goal 0/48 with no activation. Both real fragment readouts passed original admission; form controls also 16/16, no pretrained-specific advantage. [Evidence](../evidence/v08-real-fragment-social-ci.json).

## Defect hypothesis

Current self-head SGD observes clipped full-group utility minus leave-one-out utility. A valid contribution can have zero marginal utility when a harmful shot or partner pushes both results into the saturation region. Example independent oracle U(y)=max(-1,1-|y-1|): useful y=1, harmful y=10, full y=11; U(11)=U(10)=-1, while U(1)-U(0)=1. Thus learning to initiate from the group-only signal is unreliable under interference. Not established as the sole cause of dev29 failures.

## Opt-in correction

`SocialPopulation(...,credit_policy='isolated-activation-v2')`: separately score the actually computed contribution alone against an empty group (same independently attested oracle). Self-head learns this isolated utility difference. Existing partner-head learning still requires verified joint output and actual positive leave-one-out marginal contributions; no positive peer reward for a failed group. All scores, thresholds, weights, data and shot schedules otherwise unchanged. The new policy is bound to the session identity; default v1 manifests/receipts unchanged. New receipt records empty and singleton utility, per-owner isolated credit and additional oracle-call cost (1+n calls, zero extra core forwards). Atomic transaction rolls back on extra assessment failure. No oracle input to local decisions, no new competence labels, no global selection.

Limitation: this favors individually helpful initiation; genuinely synergistic contributors with no isolated effect may still need historical invitations/shots. Requires a meaningful oracle for partial outputs, additive reference contract. Not a general credit assignment solution, not a change to the strict release/admission gates.

## Matched evaluation

Same real fragment readouts, train examples, goal format, seeds 7/19/41, 600 learning episodes/seed, 0/1/2 training shots, controls and thresholds as dev29. Both policies evaluated on 16 completely new authored sentence pairs (48 tasks/seed), checked disjoint from adapter train/admission and dev29 usage. No new readout selection or tuning on these cases. Report both policies and per-goal outcomes, including failed admissions or incomplete results. Exact same core handles, separate session/DB directories. All underlying assumptions and limitations of dev29 remain. No automatic claim of real model advantage from punctuation task. CPU reference only.

Three regression tests: known clipping counterexample, useful self-selection after restart and credit-policy identity, atomic rollback after additional oracle failure. Existing dev28 default tests must remain unchanged. Actual remote results pending. Next: independently evidenced adapter candidates learned from usage, productive shot-triggered invitations, latent communication beyond fixed ports; do not skip individual activation verification.
