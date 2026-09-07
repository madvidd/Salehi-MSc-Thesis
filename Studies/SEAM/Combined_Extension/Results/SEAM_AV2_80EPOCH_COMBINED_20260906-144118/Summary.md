# SEAM 80-Epoch Combined Experiment

SEAM + uncertainty-aware target context + relative-geometry bias + QKNorm + future-head Mamba.

Captured: 2026-09-06T14:45:23.838126+01:00
Training: not yet confirmed complete.
Final selected-checkpoint evaluation: not yet complete.

Protocol: 80 epochs; seed 2333; global batch 32 (2 GPUs x 8 x accumulation 2); FP32; AdamW; LR 0.001 to 0.00001; warm-up ratio 0.167; weight decay 0.01; gradient clipping norm 5.

| Measurement | MR | minADE1 | minADE6 | minFDE1 | minFDE6 | b-minFDE6 |
|---|---:|---:|---:|---:|---:|---:|
| No validation result yet | - | - | - | - | - | - |

Last recorded progress: epoch 0, batch 550/12495, step 275.
This is a saved observation, not confirmation of live progress.

## Training Time

| Measurement | Hours |
|---|---:|
| Recorded training attempts | 0.000 |

Attempt duration includes training and epoch validation; failed attempts are retained separately in ATTEMPTS.json. Offline gaps are not added.

Full logs and model checkpoints remain on the PC in `/home/server01/M/Results/SEAM_AV2_80EPOCH_COMBINED_20260906-144118`. Checkpoints are not uploaded to GitHub.
The combined model is an experiment, not a guaranteed improvement over its individual components.
