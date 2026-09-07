# SEAM AV2: mamba_future_replace

Replaces only the decoder trajectory-coordinate MLP with a two-block Mamba head over the ordered 80-step future sequence.

## Run Status

- Status: **complete**
- Current progress: epoch unknown, unknown%
- Checkpoints saved: 11
- Run directory: `/home/server01/M/Results/SEAM_AV2_MAMBA_3RUN_20260824-202936/mamba_future_replace`

## Controlled Training Setup

| Setting | Value |
|---|---:|
| Dataset | Argoverse 2 motion forecasting |
| Observation / prediction | 3 s / 6 s |
| Streaming passes | t = 3, 4, 5 s |
| Epochs | 80 |
| Effective global training batch | 32 |
| Optimizer | AdamW |
| Peak / minimum LR | 1e-3 / 1e-5 |
| Warm-up | 13 epochs |
| Weight decay | 1e-2 |
| Gradient clipping | norm 5 |
| Seed | 2333 |
| GPUs | 2 x RTX 2080 Ti |
| Microbatch / accumulation | 8 per GPU / 2 steps |
| SyncBatchNorm activation batch | 16 (hardware adaptation) |

## Latest Validation

| Epoch | MR | minADE1 | minADE6 | minFDE1 | minFDE6 | b-minFDE6 |
|---:|---:|---:|---:|---:|---:|---:|
| 81 | 0.142 | 1.588 | 0.644 | 3.945 | 1.230 | 1.855 |

## Best Saved Checkpoint

- Epoch: 79
- minADE6: 0.648480
- Local path: `/home/server01/M/Results/SEAM_AV2_MAMBA_3RUN_20260824-202936/mamba_future_replace/checkpoints/epoch_79-minADE6_0.6484796404838562.ckpt`

## Reference

Released SEAM AV2 checkpoint metrics: MR 0.153, minADE1 1.598, minADE6 0.663, minFDE1 3.962, minFDE6 1.249, b-minFDE6 1.848.

The paper prints peak LR 1e-2, but the released configuration and checkpoint scheduler state both use 1e-3. These runs use 1e-3 for reproducibility with the released implementation.
