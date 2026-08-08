# Lab 2 SHARP 20-Epoch Ten-Test Summary

Last checked: 2026-08-08 07:18 BST  
Run: `SHARP_AV2_20EPOCH_10TEST_20260805-031733`

## Run configuration

- Dataset: Argoverse 2
- Epochs per variant: 20
- Seed: 2333
- GPUs: four RTX 2080 Ti GPUs using DDP
- Batch size: 8 per GPU, 32 global
- Learning rate: `1e-4` to `1e-5`
- Warm-up ratio: `0.65` (13 of 20 epochs)
- Weight decay: `0.01`
- SyncBatchNorm: enabled
- Checkpointing: every epoch plus `last.ckpt`

## Results so far

| Test | Status | Best epoch | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline SHARP | Complete | 19 | **0.192652** | **2.064007** | 1.886442 | **0.752339** | **4.672195** | **1.434819** |
| Confidence-gated memory | Complete | 19 | 0.194413 | 2.067328 | **1.876097** | 0.755349 | 4.694828 | 1.443534 |
| Cross-window consistency | Running, epoch 5/20 at 96% | 4 so far | - | - | - | 1.139935 | - | - |
| Tests 4-10 | Not started | - | - | - | - | - | - | - |

## Current comparison

Relative to the completed 20-epoch baseline, confidence-gated memory produced:

- 0.40% worse minADE6
- 0.91% worse MR
- 0.61% worse minFDE6
- 0.48% worse minFDE1
- 0.55% better minADE1

Confidence-gated memory therefore did not provide an overall accuracy improvement in this controlled 20-epoch run.

## Health and persistence

- `VARIANT_COMPLETE=baseline` and `VARIANT_COMPLETE=confidence_gated_memory` are present.
- The suite automatically advanced to `cross_window_consistency`.
- Each completed variant has checkpoints for epochs 0-19 plus `last.ckpt`.
- No new traceback, CUDA illegal-memory-access failure, NCCL failure, out-of-memory error, or variant failure was found.
- The separate `nvidia-smi` status query still reports the known NVML driver/library mismatch; the isolated CUDA runtime continues to train.
- `VARIANT_STATUS.txt` incorrectly labels completed tests as `PENDING` because the publication check looks for `COMPLETE` while the runner writes `COMPLETED`. The completion markers, final validation, and checkpoint sets confirm both tests completed.

This file reports an in-progress suite. Results for the active and remaining variants must not be treated as final until their best-checkpoint validation completes.
