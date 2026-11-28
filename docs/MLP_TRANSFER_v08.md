# dev23 candidate — trainable ReLU hidden layers

Base: dev22 code revision `aac7a2aa02c6980d0a4454cc4a30dd151e32da3d`, CPU CI accepted; inherited dev22 release metadata separately. **dev23 production CI pending.** Primary target Windows 11 / AMD Strix Halo.

## Hypothesis and contract

A two-layer MLP with a fixed ReLU activation can learn its hidden and output weights from independently verified coalition observations, without copying parent weights. The host chooses model width and task/validator schema; architecture discovery is not claimed. A new `mlp` core stores four FP32 tensors (`hw,hb,w,b`) and an immutable operation/dimension metadata tag. Logical input/output widths and actual tensor count are distinct. All four tensors count against resident payload budget. Existing affine and quadratic artifacts retain their serialization/semantics. No parent replacement or removal.

`GroupLearning.stage(..., method='TRAINED_MLP', hidden_width=2, steps=256)` requires the existing two-member verified group evidence, distinct input observations, full source and head provenance. Deterministic seed 7, CPU Float64 LBFGS, fixed ReLU, all four tensors optimized, FP32 export. Bounds: at most 64 input/output/hidden units, 64 samples, 256 optimizer steps, finite training values of absolute magnitude <=1000, <=65536 parameters. This is a bounded optimizer-step contract, not a wall-clock bound; line search may evaluate the closure more than once. Same stringent independent holdout/report gate as prior transfer, no relaxed tolerances. Staged weights not usable before admission.

## Experiments and negative results

Local standalone Torch 2.10.0+cpu, one thread, target 3*ReLU(2x-1)+2. First nine-point integer grid: all nine combinations of seeds 17/41/7 and hidden widths 2/4/8 failed the held-out check despite some nearly zero training loss. Example seed 7 width 2 final loss 4.10e-20, max test error 0.0267074. Revised experiment uses 17 inputs spaced by 0.5 from -4 to 4, seed 7, 256 steps. Fixed evaluation tolerances 1e-6 relative and absolute.

| Hidden units | Initial loss | Final loss | Train | Holdout |
|---|---:|---:|---:|---:|
| 1 | 97.23794 | 3.04e-21 | 17/17 | 6/6 |
| 2 | 95.90424 | 1.68e-20 | 17/17 | 6/6 |
| 4 | 125.48971 | 1.24e-20 | 17/17 | 4/6 |

All four parameter tensors changed for each revised model. Width 2 reproduced the six disjoint input values and 65/65 quarter-grid cases (grid overlaps training and holdout). The development holdouts were inspected during the model/training-data experiments; they are not an untouched post-selection generalization benchmark. The tests use separately supplied oracle IDs and disjoint inputs; this demonstrates only the controlled task. No production module was executed locally. Full raw values and unsuccessful trial: [JSON](../evidence/v08-mlp-transfer-isolated.json).

## Integration tests

Four tests cover real MLP+affine cooperation, trained transfer, separate holdout admission, persisted/new-group inference, weight-only subprocess, parent quarantine; undertraining rejection; interrupted staging recovery; all-tensor residency budget and operation metadata mismatch. CI pending. Next: acceptance, then broader nonlinear tasks and structural model-fragment assimilation. No arbitrary pretrained fragment import, learned semantic latent space or Strix Halo GPU measurement claimed.
