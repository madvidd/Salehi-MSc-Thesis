# SHARP Paper And Public-Code Alignment

## Pinned Sources

- Paper: *SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting*, CVPR 2026.
- Official repository: `https://github.com/a-pru/sharp`
- Audited commit: `f6bf2fc0109f9838cdc24bfb763b5c3e6847c2ae`

## AV2 Optimization Target

The paper supplement states one NVIDIA Quadro RTX 8000, batch size 32, 80 AV2 epochs, 13 warm-up epochs, LR increasing to `1e-4` and then cosine-decaying to `1e-5`, AdamW, weight decay, and gradient clipping. It states that no data augmentation or cross-dataset training is used.

The pinned public Hydra config currently says 60 epochs and LR `1e-3`. Therefore, this suite preserves the pinned official architecture and dataset/model configuration but deliberately replaces those two drifted config values with the paper schedule. The warm-up ratio is set to exactly `13/80 = 0.1625`.

## Four-GPU Execution

The paper used one GPU. Lab 2 uses four GPUs to reduce wall time while preserving global batch 32: batch 8 per rank, no LR scaling, FP32, seed 2333, DDP, and SyncBatchNorm. Loader workers and persistent prefetching affect throughput, not model mathematics or sample content.

## Baseline Stability Patches

Run 1 makes no learned architecture change. It applies only these behavior-preserving compatibility fixes:

- uses `R^T` instead of a general matrix inverse for an orthonormal rotation matrix;
- creates advanced-index tensors on the same CUDA device as their source;
- gives attention and padding masks compatible dtypes;
- supplies the real validation batch size to distributed metric logging;
- uses the current `timm.layers.DropPath` import;
- asserts that every trainable parameter belongs to exactly one AdamW group;
- records validation history and checkpoints without changing loss or forward outputs.

## Published AV2 Reference

| MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---:|---:|---:|---:|---:|---:|
| 0.140 | 1.822 | 1.569 | 0.639 | 3.850 | 1.197 |

All listed metrics are lower-is-better. Generated summaries report `run - article`, so a negative delta is an improvement.
