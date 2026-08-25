# SEAM AV2 Three-Run Progress

- Generated: `2026-08-25T10:22:15.565385+01:00`
- Results root: `/home/server01/M/Results/SEAM_AV2_MAMBA_3RUN_20260824-202936`
- Experiment root: `/home/server01/M/Codes/SEAM_AV2_MAMBA_3RUN_20260824-202936`
- Active training ranks detected: **18**
- Completed variants: **0/3**

## Progress

| Run | Status | Current progress | Checkpoints | Best minADE6 |
|---|---|---:|---:|---:|
| SEAM baseline | running/resumable | not started | 11 | 0.870494 (epoch 14) |
| SEAM + agent-history Mamba | pending | not started | 0 | not available |
| SEAM with future-head Mamba replacement | pending | not started | 0 | not available |

## Latest Completed Validation

| Run | Validation epoch | MR | minADE1 | minADE6 | minFDE1 | minFDE6 | b-minFDE6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SEAM baseline | 15 | 0.279 | 2.159 | 0.870 | 5.533 | 1.750 | 2.376 |
| SEAM + agent-history Mamba | - | - | - | - | - | - | - |
| SEAM with future-head Mamba replacement | - | - | - | - | - | - | - |

## Controlled Setup

All three runs use AV2, 80 epochs, seed 2333, AdamW, peak/minimum LR 1e-3/1e-5, 13 warm-up epochs, weight decay 1e-2, gradient clipping 5, and effective global batch 32. The Lab 3 hardware adaptation is two GPUs, microbatch 8 per GPU, and two-step gradient accumulation.

## Error Scan

No fatal error pattern was found in the retained log tails.
