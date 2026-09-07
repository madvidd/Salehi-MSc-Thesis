# SEAM Context and Attention Ablation: Results

Independent uncertainty-aware context, relative geometry and QKNorm interventions.

## Comparison

| Variant | Selected minADE6 | Change | MR | b-minFDE6 | minADE1 | Vector minADE6 | minFDE1 | minFDE6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SEAM reference | 0.727668 | +0.00% | 0.194958 | 2.025440 | 1.751487 | 0.727668 | 4.372325 | 1.418612 |
| Uncertainty-aware context | 0.708755 | -2.60% | 0.187595 | 1.995327 | 1.723248 | 0.708755 | 4.325522 | 1.389651 |
| Relative geometry | 0.717261 | -1.43% | 0.187955 | 1.997381 | 1.713196 | 0.718735 | 4.282954 | 1.394853 |
| QKNorm | 0.703845 | -3.27% | 0.181393 | 1.992296 | 1.687949 | 0.703845 | 4.244209 | 1.375525 |

All errors are lower-is-better. Change is relative to the first row's selected minADE6. A dash means that a comparable numerical record is not supplied. Selection values and complete metric vectors are explicitly distinguished below.

## Interpretation

All three independent interventions improve selected minADE6 against the matched control: uncertainty -2.60%, geometry -1.43% and QKNorm -3.27%. QKNorm achieves the lowest selected error, 0.703845. These independent gains do not establish that their composition will be additive.

## Protocol and Evidence

20 epochs; global batch 48; three GPUs; seed 2333; AdamW; LR 1e-3 to 1e-5.

For relative geometry, selected minADE6 is 0.717261 while the final vector contains 0.718735. These are not the same checkpoint. The earlier batch-32 attempt is retained separately and is not pooled with this batch-48 study.

- [Recorded source](Results/SEAM_AV2_20EPOCH_4TEST_MAXRES_20260903-114512/Comparison.csv)
- [Original run records](Results/SEAM_AV2_20EPOCH_4TEST_MAXRES_20260903-114512/)
- [Implementation and reproduction](README.md)

## Training Time

Timing is reported separately from forecasting error. See the source records for the retained duration definition.

| Variant | Duration (h:mm:ss) |
|---|---:|
| SEAM reference | 15:36:46 |
| Uncertainty-aware context | 15:40:25 |
| Relative geometry | 15:45:02 |
| QKNorm | 17:55:47 |

[Recorded timing table](Results/SEAM_AV2_20EPOCH_4TEST_MAXRES_20260903-114512/Timing.csv)
