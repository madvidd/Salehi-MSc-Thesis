# SEAM 80-Epoch Combined Experiment

SEAM + uncertainty-aware target context + relative-geometry bias + QKNorm + future-head Mamba.

Captured: 2026-09-10T14:23:34.270943+01:00
Training: 80 epochs completed.
Final selected-checkpoint evaluation: complete.

Protocol: 80 epochs; seed 2333; global batch 32 (2 GPUs x 8 x accumulation 2); FP32; AdamW; LR 0.001 to 0.00001; warm-up ratio 0.167; weight decay 0.01; gradient clipping norm 5.

| Measurement | MR | minADE1 | minADE6 | minFDE1 | minFDE6 | b-minFDE6 |
|---|---:|---:|---:|---:|---:|---:|
| Best validation epoch 77 | 0.146110 | 1.575834 | 0.647280 | 3.902214 | 1.238342 | 1.859444 |
| Latest validation epoch 79 | 0.147551 | 1.576387 | 0.647462 | 3.902743 | 1.238793 | 1.864359 |
| Selected-checkpoint evaluation | 0.146110 | 1.575836 | 0.647280 | 3.902212 | 1.238343 | 1.859444 |

Last recorded progress: epoch 79, batch 12450/12495, step 499817.
This is a saved observation, not confirmation of live progress.

## Training Time

| Measurement | Hours |
|---|---:|
| Recorded training attempts | 95.639 |

Attempt duration includes training and epoch validation; failed attempts are retained separately in ATTEMPTS.json. Offline gaps are not added.

Full logs and model checkpoints remain on the PC in `/home/server01/M/Results/SEAM_AV2_80EPOCH_COMBINED_20260906-144118`. Checkpoints are not uploaded to GitHub.
The combined model is an experiment, not a guaranteed improvement over its individual components.
