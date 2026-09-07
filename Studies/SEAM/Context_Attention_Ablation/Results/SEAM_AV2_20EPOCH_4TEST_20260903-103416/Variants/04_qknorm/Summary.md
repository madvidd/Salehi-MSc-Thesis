# QKNorm attention

Per-head L2-normalised queries and keys with learned logit scales replace the attention kernels while preserving SEAM's topology and residual paths.

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

All four variants use AV2, 20 epochs, seed 2333, AdamW, peak/minimum learning rates 1e-3/1e-5, warm-up ratio 0.167, weight decay 1e-2, gradient clipping 5, and effective global batch 32.
