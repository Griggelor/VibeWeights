# dev26 candidate — controlled latent-interface follow-up

**Technical CPU CI passed; one narrowly qualified experimental candidate. Release metadata CI pending.** Base dev25 `348e4bd7d43f6911e73e9df997d1efbf74b755b7`. Prior experiment negative: pretrained 7/12, random 5/12, embedding-only 6/12, no admissions; all local heads 8/8. Full report [here](../evidence/v08-fragment-adapter-a1.json). Technical baseline and tiny tests passed (7/7 Linux in 3.67 s, Windows in 5.80 s). No population admission or assimilation.

## Hypothesis and implementation

The six-layer pretrained prefix produced holdout scores with magnitude below 0.0012 and loss near its initial value, unlike controls. Possible causes: dominant residual-stream directions, unequal token-vector norms and regularization relative to latent scale. Not an established causal diagnosis. Instead of weakening admission, change the latent interface and compare ablations.

- Token-L2-mean: normalize each token vector before pooling, then L2-normalize pooled vector. Preserve old mean-L2 default; optional last-token mode also implemented and tested but not used in the prespecified real experiment.
- Train-centered-radius: calculate mean and RMS Euclidean distance from the mean using training latents only. Gain = min(1000, 1/max(radius,1e-6)). Fit the same regularized linear readout on centered scaled features, then fold center/gain into the exported FP32 weights/bias. Evaluate exported readout itself. No extra learned parameters, no holdout normalization or labels used during fitting. Store gain, radius, mean norm, training latent digest and representation contract. Old artifact identity retained for default representation; new pooling is explicitly core-bound.
- All acceptance conditions unchanged, including full train and disjoint holdout accuracy. Failed variants remain non-executable. No model selection or threshold adjustment based on this round.

## Experiment

Fresh authored sentiment cases (16 train, 16 holdout) fixed in `scripts/probe_fragment_conditioning.py`; dev25 labels are not reused. Same model revision, same frozen embedding, same adapter width and optimizer budget, same labels across all arms. The prior negative experiment informs this hypothesis; this is not an external benchmark.

| Arm | Core | Blocks | Pooling | Conditioning |
|---|---|---:|---|---|
| p6-baseline | pretrained | 6 | mean-L2 | none |
| p6-pool | pretrained | 6 | token-L2-mean | none |
| p6-condition | pretrained | 6 | mean-L2 | centered-radius |
| p6-combined | pretrained | 6 | token-L2-mean | centered-radius |
| r6-combined | random | 6 | token-L2-mean | centered-radius |
| e-combined | embedding-only | 0 | token-L2-mean | centered-radius |
| p16-combined | pretrained | 16 | token-L2-mean | centered-radius |
| r16-combined | random | 16 | token-L2-mean | centered-radius |

Four-arm factorial at six blocks helps distinguish pooling and conditioning effects. Matched random and embedding-only controls, deeper-scale pair, all predictions and diagnostics reported. No teacher-model execution. Depths are host-selected, overlapping prefixes are not autonomous coalitions. Random control uses one seed, not a population estimate. All arms reported, even if none passes; no claim of broad competence. If a candidate passes, repeat on an untouched independent data family and check marginal benefit before A2.

Three new tests: normalization ablation on artificial vectors, mathematical conditioning/export equivalence, unchanged adapter capacity and identical exported weights under changed holdout labels, representation binding and rejected admission. Separate CI workflow for real follow-up; previous A1 and A0 tests remain. GPU/UMA and general latent ABI open.

## Actual results

CI 46163192152 on ad8b355544f208a6f2e4e30f3bef960c1a993160: all eight arms 16/16 training. Holdouts: p6-baseline 13/16; p6-pool, p6-condition and p6-combined each 15/16; r6-combined, e-combined and r16-combined each 11/16; p16-combined 16/16 and admitted. Six-block raw mean-L2 training radius 0.00896397, tokenwise pooled radius 0.374762 (about 42x), conditioning gain 111.56 and 2.67 respectively. These observations support the interface hypothesis, not unique causal attribution or general competence. All workflows green. [Raw report](../evidence/v08-fragment-conditioning-a1.json). New independent data family and localization required before general admission/coalition claims.
