# dev27 candidate — systematic competence localization

**User-directed research change; implementation, CI pending.** Do not arbitrarily assign one competence to a fragment. First map what existing regions contribute to a task. Base dev26 `5f76af4ca5e39051e78cd509f2e49aed6e039c39`: 16-block conditioned prefix passed 16/16 authored sentiment holdouts, 6-block improved variants 15/16, random and embedding controls 11/16; narrow experimental admission only. [Report](../evidence/v08-fragment-conditioning-a1.json).

## Method

The scanner runs the pinned full Qwen3-0.6B-Base original as an **offline research reference**, never as a production router or required runtime teacher. Dense decoder blocks normally execute for every input; activity alone cannot establish function. Record input RMS, residual update RMS, relative update RMS and last-token cosine separately for each decoder block. Compare baseline final next-token logits for a correct and foil continuation. Only cases the intact model prefers correctly are eligible for damage aggregation; all cases and baseline failures retained.

Reversible causal screening: substitute each selected block's output with its input (identity bypass), measuring loss of correct preference and signed drop in correct-minus-foil logit margin. No weight edits; hooks removed even on interruption, unchanged baseline recomputed after each case. Test all 28 single blocks and existing nonoverlapping partitions with sizes 6/8/16 including remainders; duplicate spans deduplicated. Rank regions only within each domain, not across incomparable domain scales. Both damage and improvement are possible. Bypass can disrupt generic computation, distributional compatibility and downstream processing; a large effect is not proof of isolated competence or necessity outside this intervention. Layer-level activity measurements for each case and all intervention scores preserved. No automatic extraction or admission from scores.

## Prespecified task families

36 host-authored forced-choice cases, six each in arithmetic, syntax, world knowledge, sentiment, logic, and code. Three contrasting pairs per domain. Verify exact tokenizer prefix and distinct single-token candidate continuations; log exclusions rather than silently substitute examples. Numerical prompts may be excluded if tokenization does not match; domain coverage must be checked before interpreting results. Scores are forced-choice, not unconstrained accuracy; small correlated cases and hand-written labels are not a benchmark. No independent fragment competence claim.

## Follow-up

1. Inspect both activity and causal maps. Hypotheses about sensitive regions must survive new cases, other templates, matched controls and different intervention sizes; a longer span has more capacity and may cause generic disruption.
2. Contrastive activation patching with matched clean/counterfactual inputs at aligned positions to test sufficiency, followed by matched-size submodule or attention/MLP interventions. See [Towards Best Practices of Activation Patching in Language Models](https://openreview.net/forum?id=Hf17y6u9BC) (general methodology; current implementation uses bypass, not activation patching).
3. Extract candidate spans at evidence-driven boundaries, preserve dependencies, learn appropriate input/output adapters, evaluate against random and adapter-only controls. Test on independent data families before assigning local competence labels/heads. A region can participate in multiple domains; a competence can be distributed across regions.
4. Only then use in decentralized group cooperation; the diagnostic map is not a permanent central routing table.

## Tests and execution

Three tests: known positive/negative/zero causal effects and exclusion of failed baseline; hook cleanup and bad input; actual tiny Qwen3 all spans and unchanged weights. Optional fragment test CI Windows/Linux; separate actual pretrained CPU scan workflow. No local full-model execution; no GPU measurement. Fixed model and original weights. Actual results pending.

```powershell
python -m pip install -e ".[fragments,test]"
python -m pytest -q experiments/fragment_tests
python scripts/probe_competence_mapping.py --download-pinned-qwen3 --report mapping/report.json --device cpu
```
