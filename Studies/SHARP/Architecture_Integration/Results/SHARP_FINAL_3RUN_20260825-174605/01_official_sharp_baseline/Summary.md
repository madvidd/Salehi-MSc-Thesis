# Official SHARP baseline

Pinned official architecture with the paper's AV2 optimization schedule.

## Status

- Completed: True
- Official source commit: `f6bf2fc0109f9838cdc24bfb763b5c3e6847c2ae`
- Best checkpoint: `/home/server00/M/Results/SHARP_FINAL_3RUN_20260825-174605/01_official_sharp_baseline/run/checkpoints/epoch_78-minADE6_0.6792823672294617.ckpt`
- Training duration seconds: 452534
- Final evaluation: one GPU, batch 32, full AV2 validation split
- Best recorded training epoch by minADE6: 79

## Controlled Training Setup

- AV2 processed train/validation data; 10 historical steps; 100 future steps; split points 10/20/30/40/50; radius 150 m.
- 80 epochs; 13 warm-up epochs; AdamW; LR 1e-4 to 1e-5 cosine decay; weight decay 1e-2.
- Seed 2333; global batch 32 (8 per rank); four GPUs; FP32; gradient clipping norm 5; SyncBatchNorm.
- No data augmentation and no cross-dataset training.

## Final Metrics

| Metric | Final validation | SHARP article | Delta (run - article) |
|---|---:|---:|---:|
| MR | 0.155955 | 0.140000 | 0.015955 |
| b-minFDE6 | 1.924560 | 1.822000 | 0.102560 |
| minADE1 | 1.676867 | 1.569000 | 0.107867 |
| minADE6 | 0.679282 | 0.639000 | 0.040282 |
| minFDE1 | 4.138639 | 3.850000 | 0.288639 |
| minFDE6 | 1.287697 | 1.197000 | 0.090697 |

Delta means run metric minus article metric; negative is better for every listed error metric.

Full checkpoints and the unfiltered log remain on Lab 2 and are intentionally excluded from GitHub.
