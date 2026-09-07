# SEAM baseline

The verified SEAM AV2 architecture is retained unchanged and trained for the common 20-epoch budget.

## Status

- State: **running/resumable**
- Progress: not started
- Saved checkpoints: 0
- Active training time: 00:00:12

## Metrics

| MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---:|---:|---:|---:|---:|---:|
| - | - | - | - | - | - |

## Best Saved Checkpoint

No metric-named checkpoint is available yet.

## Controlled Setup

All four variants use AV2, 20 epochs, seed 2333, AdamW, peak/minimum learning rates 1e-3/1e-5, warm-up ratio 0.167, weight decay 1e-2, gradient clipping 5, and effective global batch 32.
