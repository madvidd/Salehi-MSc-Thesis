# Relative-geometry attention bias

Relative displacement, distance, and heading difference provide a learned per-head bias in the current-window scene self-attention blocks.

## Status

- State: **complete**
- Progress: not started
- Saved checkpoints: 21
- Active training time: 15:45:02

## Metrics

| MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---:|---:|---:|---:|---:|---:|
| 0.188 | 1.997 | 1.713 | 0.719 | 4.283 | 1.395 |

## Best Saved Checkpoint

- Epoch: 18
- minADE6: 0.717261
- Local path: `/home/server01/M/Results/SEAM_AV2_20EPOCH_4TEST_MAXRES_20260903-114512/03_relative_geometry_bias/checkpoints/epoch_18-minADE6_0.7172605991363525.ckpt`

## Controlled Setup

All four variants use AV2, 20 epochs, seed 2333, AdamW, peak/minimum learning rates 1e-3/1e-5, warm-up ratio 0.167, weight decay 1e-2, gradient clipping 5, 3 GPUs, microbatch 8 per GPU, 2-step gradient accumulation, and effective global batch 48.
