# Relative-geometry attention bias

Relative displacement, distance, and heading difference provide a learned per-head bias in the current-window scene self-attention blocks.

## Status

- State: **pending**
- Progress: not started
- Saved checkpoints: 0
- Active training time: -

## Metrics

| MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---:|---:|---:|---:|---:|---:|
| - | - | - | - | - | - |

## Best Saved Checkpoint

No metric-named checkpoint is available yet.

## Controlled Setup

All four variants use AV2, 20 epochs, seed 2333, AdamW, peak/minimum learning rates 1e-3/1e-5, warm-up ratio 0.167, weight decay 1e-2, gradient clipping 5, 3 GPUs, microbatch 8 per GPU, 2-step gradient accumulation, and effective global batch 48.
