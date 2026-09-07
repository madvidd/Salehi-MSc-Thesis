# SHARP Architectural Ablation: Results

Nine independent memory, context, pooling, geometry and decoder interventions against a shared control.

## Comparison

| Variant | Selected minADE6 | Change | MR | b-minFDE6 | minADE1 | Vector minADE6 | minFDE1 | minFDE6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline SHARP | 0.752339 | +0.00% | 0.192652 | 2.064007 | 1.886442 | 0.752339 | 4.672195 | 1.434819 |
| Confidence-gated memory | 0.755349 | +0.40% | 0.194413 | 2.067328 | 1.876097 | 0.755349 | 4.694828 | 1.443534 |
| Cross-window consistency | 0.776207 | +3.17% | 0.202497 | 2.096022 | 1.872110 | 0.776207 | 4.785388 | 1.476766 |
| Learned temporal pooling | 0.757637 | +0.70% | 0.201897 | 2.080306 | 1.891240 | 0.757637 | 4.739688 | 1.463850 |
| Uncertainty-aware target context | 0.749961 | -0.32% | 0.198575 | 2.074243 | 1.863236 | 0.749961 | 4.676969 | 1.447852 |
| Relative geometry attention bias | 0.749679 | -0.35% | 0.190852 | 2.051891 | 1.875754 | 0.749679 | 4.660576 | 1.432463 |
| Kinematic motion stem | 0.785164 | +4.36% | 0.198175 | 2.068197 | 1.942671 | 0.785164 | 4.701560 | 1.448078 |
| Endpoint refinement decoder | 0.755898 | +0.47% | 0.196254 | 2.058200 | 1.867779 | 0.755898 | 4.641530 | 1.441044 |
| Lane topology graph | 0.754238 | +0.25% | 0.200616 | 2.065051 | 1.903412 | 0.754238 | 4.696756 | 1.443290 |
| Agent temporal Mamba | 0.780190 | +3.70% | 0.212822 | 2.141835 | 1.993448 | 0.780190 | 4.973433 | 1.514158 |

All errors are lower-is-better. Change is relative to the first row's selected minADE6. A dash means that a comparable numerical record is not supplied. Selection values and complete metric vectors are explicitly distinguished below.

## Interpretation

Relative geometry gives the lowest selected minADE6 (0.749679, -0.35%), followed by uncertainty-aware context (0.749961, -0.32%). The remaining interventions do not improve this primary error. Endpoint refinement improves minFDE1 despite its higher minADE6, illustrating that trajectory-average and endpoint criteria need not move together.

## Protocol and Evidence

20 epochs; global batch 32; four GPUs; seed 2333; AdamW; LR 1e-4 to 1e-5; 13 warm-up epochs.

The captured summary derives selected minADE6 from checkpoint filenames and uses retained or nearest validation records for other metrics. Treat the complete vector as reported evidence, not a newly re-evaluated selected checkpoint.

- [Recorded source](Runs/SHARP_AV2_20EPOCH_10TEST_20260805-031733/Summary.md)
- [Original run records](Runs/SHARP_AV2_20EPOCH_10TEST_20260805-031733/)
- [Implementation and reproduction](README.md)
