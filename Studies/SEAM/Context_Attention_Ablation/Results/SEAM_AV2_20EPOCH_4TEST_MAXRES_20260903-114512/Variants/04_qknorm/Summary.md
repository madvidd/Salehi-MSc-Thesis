# QKNorm attention

Per-head L2-normalised queries and keys with learned logit scales replace the attention kernels while preserving SEAM's topology and residual paths.

## Status

- State: **complete**
- Progress: not started
- Saved checkpoints: 21
- Active training time: 17:55:47

## Metrics

| MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---:|---:|---:|---:|---:|---:|
| 0.181 | 1.992 | 1.688 | 0.704 | 4.244 | 1.376 |

## Best Saved Checkpoint

- Epoch: 19
- minADE6: 0.703845
- Local path: `/home/server01/M/Results/SEAM_AV2_20EPOCH_4TEST_MAXRES_20260903-114512/04_qknorm/checkpoints/epoch_19-minADE6_0.7038450837135315.ckpt`

## Controlled Setup

All four variants use AV2, 20 epochs, seed 2333, AdamW, peak/minimum learning rates 1e-3/1e-5, warm-up ratio 0.167, weight decay 1e-2, gradient clipping 5, 3 GPUs, microbatch 8 per GPU, 2-step gradient accumulation, and effective global batch 48.
