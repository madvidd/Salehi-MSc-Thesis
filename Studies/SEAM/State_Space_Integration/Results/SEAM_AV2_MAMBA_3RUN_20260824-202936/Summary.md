# SEAM AV2 Three-Run Progress

- Generated: `2026-09-03T04:17:55.889638+01:00`
- Results root: `/home/server01/M/Results/SEAM_AV2_MAMBA_3RUN_20260824-202936`
- Experiment root: `/home/server01/M/Codes/SEAM_AV2_MAMBA_3RUN_20260824-202936`
- Matching active suite processes detected: **0**
- Completed variants: **3/3**
- Append-only local Terminal.txt: `/home/server01/M/Terminal/SEAM_AV2_Mamba_3_Run/SEAM_AV2_MAMBA_3RUN_20260824-202936/Terminal.txt` (146905 bytes); all previously saved lines were retained.

## Progress

| Run | Status | Current progress | Checkpoints | Best minADE6 |
|---|---|---:|---:|---:|
| SEAM baseline | complete | validated epoch 81 | 11 | 0.662859 (epoch 77) |
| SEAM + agent-history Mamba | complete | validated epoch 81 | 11 | 0.664814 (epoch 77) |
| SEAM with future-head Mamba replacement | complete | validated epoch 81 | 11 | 0.648480 (epoch 79) |

## Latest Completed Validation

| Run | Validation epoch | MR | minADE1 | minADE6 | minFDE1 | minFDE6 | b-minFDE6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SEAM baseline | 81 | 0.158 | 1.614 | 0.662 | 3.996 | 1.262 | 1.872 |
| SEAM + agent-history Mamba | 81 | 0.156 | 1.622 | 0.664 | 3.999 | 1.257 | 1.869 |
| SEAM with future-head Mamba replacement | 81 | 0.142 | 1.588 | 0.644 | 3.945 | 1.230 | 1.855 |

## Controlled Setup

All three runs use AV2, 80 epochs, seed 2333, AdamW, peak/minimum LR 1e-3/1e-5, 13 warm-up epochs, weight decay 1e-2, gradient clipping 5, and effective global batch 32. The Lab 3 hardware adaptation is two GPUs, microbatch 8 per GPU, and two-step gradient accumulation.

## Error Scan

No fatal error pattern was found in the retained log tails.
