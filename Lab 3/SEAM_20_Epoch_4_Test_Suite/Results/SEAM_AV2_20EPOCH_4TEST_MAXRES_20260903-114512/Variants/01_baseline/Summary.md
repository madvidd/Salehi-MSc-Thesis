# SEAM baseline

The verified SEAM AV2 architecture is retained unchanged and trained for the common 20-epoch budget.

## Status

- State: **complete**
- Progress: not started
- Saved checkpoints: 21
- Active training time: 15:36:46

## Metrics

| MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---:|---:|---:|---:|---:|---:|
| 0.195 | 2.025 | 1.751 | 0.728 | 4.372 | 1.419 |

## Best Saved Checkpoint

- Epoch: 19
- minADE6: 0.727668
- Local path: `/home/server01/M/Results/SEAM_AV2_20EPOCH_4TEST_MAXRES_20260903-114512/01_baseline/checkpoints/epoch_19-minADE6_0.7276679277420044.ckpt`

## Controlled Setup

All four variants use AV2, 20 epochs, seed 2333, AdamW, peak/minimum learning rates 1e-3/1e-5, warm-up ratio 0.167, weight decay 1e-2, gradient clipping 5, 3 GPUs, microbatch 8 per GPU, 2-step gradient accumulation, and effective global batch 48.
