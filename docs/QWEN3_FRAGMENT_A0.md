# dev24 candidate — Qwen3 structural fragments, A0

**A0 CPU reference passed on the tested revision; release metadata CI pending.** Base dev23 release-metadata commit `0d790de241a0ed769defa57c168f0be356068614`, functional code accepted in CI 46145571563. This implements the structure-oriented blind-cut variant B and stage A0 of [FRAGMENT_ASSIMILATION.md](FRAGMENT_ASSIMILATION.md). Reproduction is a necessary prerequisite, not evidence of assimilation or autonomous cooperation.

## Source and architecture

First supported model: [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd), pinned revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, Apache-2.0 upstream license. Configuration: 28 decoder layers, hidden width 1024, 16 attention heads, 8 KV heads, head dimension 128, intermediate width 3072, tied embedding/output weights. Reference implementation: [Transformers 4.57.1 Qwen3](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py). The optional extra pins this version; baseline runtime does not import it. Source tensors are never copied from untrusted executable manifests; only local single-file SafeTensors, config and license. No remote model code, pickle, quantization, arbitrary architecture, sliding attention, rope scaling, padding or KV cache. FP32 inference with original stored tensor dtype retained in each fragment.

- Extraction validates keys and shapes against the exact supported layer schema, preserves parameter tensors, requires complete coverage, and refuses unrecognized tensors. Tied output head is represented by the single embedding weight; if a duplicate is present, it must be identical.
- Immutable content-addressed block files plus separate embedding and final normalization. A hash-bound manifest records model identity, revision and origin, source/config/license hashes, each tensor's shape/dtype, logical parameters, unique payload and file bytes.
- Partitions contain consecutive complete blocks in sizes 6/8/16 for the 28-block model, retaining remainders. They are logical views sharing immutable block files, not three physical copies. This is a declared storage decision; report logical coverage and unique bytes separately. No semantic cut choice. Store validates configuration, license, manifest, complete partition coverage, per-file digest and tensors.
- DecoderSpan is independently instantiated from its own block files; latent interface FP32 B×T×H with zero-based contiguous positions, causal full attention, bounded sequence length (128), batch size (4), no cache. This is a narrow architecture-specific latent boundary, not the general VibeWeights latent ABI. No token embedding or LM head in a span.
- Original-order replay loads one span at a time; it does not call the full model. Reference evaluation uses upstream full Qwen3ForCausalLM with eager attention and no cache, compares full logits and per-position argmax across three token sequence lengths and two greedy extensions. Teacher is released before replay. Tolerances abs/rel 1e-5. Only original-order parity, no language-task benchmark or dynamic groups. No numerical pass asserted before CI.

## Run locally (Windows 11 or Linux)

Use a supported PyTorch environment (CPU, or explicit compatible CUDA/HIP build; no implicit fallback).

```powershell
python -m pip install -e ".[fragments,test]"
python -m pytest -q experiments/fragment_tests
python scripts/probe_qwen3_fragments.py --download-pinned-qwen3 --output fragment-store --report qwen3-parity.json --device cpu
```

For offline testing, use `--snapshot PATH` instead of download. That local snapshot identity is caller-attested, not independently established by the script; source bytes are hashed and must match between extraction and comparison. Output must not exist; artifacts are written in a staging directory and moved at completion. The experiment may use multiple GB of host RAM and disk. Source BF16 payload and FP32 execution memory are separate; no VRAM measurement yet. Use explicit `--device cuda:0` only with a working GPU backend.

## Verification

Separate workflow [fragment-reference.yml](../.github/workflows/fragment-reference.yml): tiny randomly initialized *real Qwen3 architecture* tests on Windows and Ubuntu; actual pinned pretrained model download and three-partition reference parity on Ubuntu once for this PR, or manual dispatch. Model weights stay on the runner/cache; only reports, source license/config and manifest are uploaded. CPU baseline CI remains separate. Four tiny integration tests: parity, source-free replay, latent boundary and frozen weights; corruption; complete partition/source binding; unsupported configurations and size budget. Synthetic architecture tests are not pretrained evidence.

Container check: Transformers absent, no model cache, local DNS fails for GitHub and Hugging Face. Source model config, license and revision verified from upstream repository listing, not downloaded locally. GitHub connector can write code and remote CI can attempt acquisition. Any network or parity failure must be recorded before acceptance.

Next A1: adapter contract and actual local heads on imported spans, negative controls (random weights, reversed order, adapter-only, original chain), independent task evidence, then multi-scale cooperation. No catalog admission, PRETRAINED expert admission, learning, autonomous routing, or assimilation claim in A0. No Strix Halo GPU claim.

## Measured acceptance

CI 46148469004, code b9d051921343914dc2f153e75268b0f4c7453c16: 15/15 full-logit comparisons exactly equal, all argmax equal. 596,049,920 parameters, 1,192,099,840 bytes unique tensor payload; 27.436 s combined experiment elapsed on remote CPU (not a benchmark). Tiny architecture Windows 4/4 (4.99 s), Linux 4/4 (2.74 s). Baseline CI 46148468885 successful. [Raw report](../evidence/v08-qwen3-fragment-a0.json). No assimilation claim.
