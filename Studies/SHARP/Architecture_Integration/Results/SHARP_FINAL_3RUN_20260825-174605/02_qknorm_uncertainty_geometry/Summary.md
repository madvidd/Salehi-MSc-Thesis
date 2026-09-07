# SHARP + QKNorm + uncertainty context + geometry bias

All SHARP MHA constructors use QKNorm; streamed target context adapts to mode uncertainty; scene attention receives a learned relative-pose bias.

## Status

- Completed: True
- Official source commit: `f6bf2fc0109f9838cdc24bfb763b5c3e6847c2ae`
- Best checkpoint: `/home/server00/M/Results/SHARP_FINAL_3RUN_20260825-174605/02_qknorm_uncertainty_geometry/run/checkpoints/epoch_79-minADE6_0.6805892586708069.ckpt`
- Training duration seconds: 546800
- Final evaluation: one GPU, batch 32, full AV2 validation split
- Best recorded training epoch by minADE6: 80

## Controlled Training Setup

- AV2 processed train/validation data; 10 historical steps; 100 future steps; split points 10/20/30/40/50; radius 150 m.
- 80 epochs; 13 warm-up epochs; AdamW; LR 1e-4 to 1e-5 cosine decay; weight decay 1e-2.
- Seed 2333; global batch 32 (8 per rank); four GPUs; FP32; gradient clipping norm 5; SyncBatchNorm.
- No data augmentation and no cross-dataset training.

## Final Metrics

| Metric | Final validation | SHARP article | Delta (run - article) |
|---|---:|---:|---:|
| MR | 0.155515 | 0.140000 | 0.015515 |
| b-minFDE6 | 1.918870 | 1.822000 | 0.096870 |
| minADE1 | 1.677496 | 1.569000 | 0.108496 |
| minADE6 | 0.680589 | 0.639000 | 0.041589 |
| minFDE1 | 4.130601 | 3.850000 | 0.280601 |
| minFDE6 | 1.285462 | 1.197000 | 0.088462 |

Delta means run metric minus article metric; negative is better for every listed error metric.

Full checkpoints and the unfiltered log remain on Lab 2 and are intentionally excluded from GitHub.
