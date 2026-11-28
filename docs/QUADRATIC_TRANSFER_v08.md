# v0.8.0.dev22 candidate — nonlinear behavioral transfer

**CPU reference accepted on the tested code revision; release metadata CI pending.** Base dev21 `bce96a02ff120610ea770160a607ec0f8fb6c33c`. Primary target Windows 11 / AMD Strix Halo. CPU reference only in this slice.

## Hypothesis and implementation

A coalition containing a nonlinear expert can transfer verified input/output behavior into an independently trained additive expert without copying parents' tensors. The first bounded model class is a fixed quadratic feature map (linear terms followed by upper-triangle products) with a learned FP32 affine readout. It is nonlinear in the original inputs, not an arbitrary deep neural network or learned latent representation. Logical shape and stored parameter count are distinct. Direct quadratic seed experts and TRAINED_QUADRATIC artifacts use the numeric-events ABI and immutable SafeTensors artifacts; explicit affine composition rejects nonlinear cores. Existing affine contracts are preserved.

GroupLearning requires two current participating cores, a single input, independently verified complete group runs, exact source/head/run provenance, and distinct training inputs. Replay validates evidence. Trainer receives only observations, with full-rank expanded design, bounded dimensions (input width <=16, <=65536 parameters, <=64 observed runs), bounded input values, CPU Float64 SGD with zero initialization and FP32 export. Promotion uses disjoint independently attested holdout inputs and observed samples; parents remain intact. Existing population resident tensor accounting counts actual stored readout tensors. Group work and request-head core compatibility accept the new operation.

## Measured isolated experiments

- Local container Torch 2.10.0+cpu, OMP/MKL threads 1, seven training inputs -3..3, 512 SGD steps, standardized quadratic features; loss 654.5 -> 2.6145104287324965e-29. FP32 readout [3,6], bias 5. Five unseen inputs [-8,-5,4,6,9] give [365,140,113,239,518], all exact.
- Two-dimensional nine-input grid {-1,0,1}², rank six, 1024 SGD steps; loss 38.611111111111114 -> 6.89157651922245e-30. FP32 readout [4,-2,2,3,-1], bias 7; unseen (2,-3) gives 2 as expected.
- Both experiments are standalone local reproductions of the mathematical training procedure; not production-module imports or GPU benchmarks.

## Tests

Four new integration tests: transfer from genuine quadratic+affine cooperation, independent holdout and immutable parents, weight-only subprocess and restart, rejoining through a group event, quarantine; rejected/undertrained admission; multivariate full-rank/cross terms and rejection of incompatible operations; interruption after training and before staging. Full CI is pending. No new test pass claimed.

## Boundaries

The feature basis is developer-chosen, not discovered; general nonlinear MLP training, architecture discovery, heterogeneous pretrained fragment assimilation, learned latent interface and GPU execution remain open. No autonomous group learning authority is added. Full-rank sample requirement constrains practical dimensionality more than the declared input limit. No evidence of general reasoning.

## Artifact operation binding

The nonlinear SafeTensors file includes the exact fixed feature contract and logical input width as one metadata field. Equal coefficient tensors in affine and quadratic experts are not treated as identical artifacts. Catalog loading validates that operation metadata matches the manifest and rejects mismatches. Legacy affine artifacts still use no metadata. A fifth integration test covers hash distinction and manifest operation substitution.

## CPU CI acceptance

[Run 46143869357](https://github.com/Griggelor/VibeWeights/actions/runs/46143869357): four green jobs, 258/258 each, all device probes, rebuilds and smokes. [Machine-readable evidence](../evidence/v08-quadratic-transfer-ci.json). Target GPU acceptance remains open.
