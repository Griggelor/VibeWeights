# dev25 candidate — A1 frozen fragment adapter and local participation head

**Technical CPU CI passed; competence experiment negative, no admission. Release metadata CI pending.** Base dev24 release metadata `abad0b78a124cd901e8d1e174afffda3bfddf2c2`. A0 real-model CPU parity verified at `b9d051921343914dc2f153e75268b0f4c7453c16`. This is the first restricted A1 experiment, not yet general fragment assimilation.

## Design

Frozen native Qwen3 embedding input adapter plus a contiguous prefix (first six of 28 decoder blocks for actual-model experiment). A mean-pooling/L2-normalized latent is mapped to a binary output by a learned linear readout. The decoder and embedding do not train. Native token/latent representations are specific to Qwen3; no universal cross-model input adapter or arbitrary middle-span adaptation. The host fixes the prefix, task and labels.

An independently trained four-input local head decides participation from topic match, input availability, recruitment and priority. The training label is a host-defined conjunction of the first two features. Its disjoint evaluation varies priority; this is a metadata-head mechanism test, not learned semantic routing. A head can abstain before large-fragment execution. No global selection among experts. Persisted experimental adapter and head are four FP32 tensors, source/embedding/span hashes, task, training/holdout digests and complete reports. Both competence and head must pass all observed and disjoint test examples for experimental admission, and only the imported-fragment path may admit. Immutable files, explicit reopening, source integrity checked on encoder reconstruction. No integration with general Population residency or global LearningEvents yet.

## Prespecified experiment and controls

Script `scripts/probe_fragment_adapters.py` fixes 12 training sentences and 12 distinct holdout sentences, balanced binary sentiment labels authored for this small task, no teacher logits or full-model inference, no external evaluation corpus. The pinned Qwen3 tokenizer encodes each sentence. Same frozen embedding, same training/holdout token sequences, identical output adapter dimensions, fixed LBFGS budget 64 optimizer steps, L2 regularization 0.001. Head uses separately fixed metadata examples (8 training + 8 held out). Three encoders:
- real imported six-block prefix;
- same six-block architecture, seeded random initialization (normal linear weights, seed 29), same pretrained embedding;
- pretrained embedding alone, no decoder blocks.

Random and imported cores have the same FP32 parameter payload; embedding-only has less compute, all train identical-size adapters and heads. No hidden control tuning or label-based architecture choice after seeing results. Report all predictions and scores including failed admissions. A workflow succeeds if the experiment executes and any admitted expert replays its decisions correctly; it does **not** assert the pretrained arm wins. Exact all-case admission is deliberately strict; the small authored holdout does not prove broad sentiment competence. Random and embedding-only controls are not eligible for experimental admission regardless of accuracy.

Three tiny-Qwen3 tests cover frozen core / restart without source / local abstention, comparable random controls and rejection, disjointness and artifact corruption. Existing four A0 tests run alongside them on Windows and Linux. Separate real-model CI on Ubuntu attempts the prespecified three-arm experiment, uploading reports and small adapter artifacts only. No GPU measurement or local full-model execution. Next: examine measured marginal contribution, improve latent input/output ports, then local head-driven multi-fragment cooperation and budgeted residency. No assimilation claim if controls are competitive.

## Local execution

```powershell
python -m pip install -e ".[fragments,test]"
python -m pytest -q experiments/fragment_tests
python scripts/probe_fragment_adapters.py --download-pinned-qwen3 --output fragment-a1 --device cpu
```

Actual experiment requires model download and several GB of memory/disk. No implicit GPU fallback. This is separate from target Windows 11 / Strix Halo hardware acceptance.

## Actual result

Run 46160839256, code 67075d59de22cc2199194e0af6f411c8e327f8e2: pretrained 11/12 training, 7/12 holdout; random 12/12, 5/12; embedding-only 12/12, 6/12. All metadata heads 8/8 held out. No candidate admitted. Pretrained final regularized loss 0.692910362651911 vs initial 0.6931471805599453; holdout logits within +/-0.0012. Hypothesis for next round: pooling may suppress discriminative directions and/or regularization scale is unsuitable; not yet a causal diagnosis. [Full report](../evidence/v08-fragment-adapter-a1.json). Technical baseline and tiny CI successful; no competence or GPU claim.
