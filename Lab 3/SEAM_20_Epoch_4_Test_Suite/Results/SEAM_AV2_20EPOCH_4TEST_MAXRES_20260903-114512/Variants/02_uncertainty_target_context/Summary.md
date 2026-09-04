# Uncertainty-aware target context

Previous-window modal probabilities control the radius and feature gain of SEAM's existing endpoint-centred target-context branch.

## Status

- State: **complete**
- Progress: not started
- Saved checkpoints: 21
- Active training time: 15:40:25

## Metrics

| MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---:|---:|---:|---:|---:|---:|
| 0.188 | 1.995 | 1.723 | 0.709 | 4.326 | 1.390 |

## Best Saved Checkpoint

- Epoch: 19
- minADE6: 0.708755
- Local path: `/home/server01/M/Results/SEAM_AV2_20EPOCH_4TEST_MAXRES_20260903-114512/02_uncertainty_target_context/checkpoints/epoch_19-minADE6_0.7087548971176147.ckpt`

## Controlled Setup

All four variants use AV2, 20 epochs, seed 2333, AdamW, peak/minimum learning rates 1e-3/1e-5, warm-up ratio 0.167, weight decay 1e-2, gradient clipping 5, 3 GPUs, microbatch 8 per GPU, 2-step gradient accumulation, and effective global batch 48.
