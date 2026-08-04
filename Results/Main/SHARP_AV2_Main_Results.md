# SHARP AV2 Main Results

All trajectory metrics are **lower is better**. Values are reported from the best comparable validation record available for each run. The published SHARP validation result is included as the primary article reference because it uses the same AV2 validation split and reports the same six metrics as the local experiments.

## Metrics

| Experiment | Machine | Evaluation record | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **SHARP article** | Quadro RTX 8000 | AV2 validation, standard 5 s context | **0.14** | **1.82** | **1.57** | **0.64** | **3.85** | **1.20** |
| Normal SHARP | Lab 1, RTX A4000 | Best checkpoint, epoch 59 | 0.149 | 1.841 | 1.578 | 0.650 | 3.893 | 1.223 |
| Normal SHARP | Lab 2, 4 x RTX 2080 Ti | Best checkpoint, epoch 56 | 0.150608 | 1.864411 | 1.599569 | 0.661463 | 3.962160 | 1.242391 |
| Normal SHARP (baseline MHA) | Lab 3, 3 x RTX 2080 Ti | Final validation | 0.153231 | 1.927635 | 1.671288 | 0.686853 | 4.063601 | 1.287932 |
| SHARP + scene-encoder Mamba | Lab 2, 4 x RTX 2080 Ti | Best checkpoint, epoch 78 | 0.157353 | 1.926419 | 1.680364 | 0.680362 | 4.130003 | 1.290422 |
| SHARP + temporal-agent Mamba | Lab 2, 4 x RTX 2080 Ti | Final validation, epoch 79 | 0.157 | 1.920 | 1.640 | 0.676 | 4.050 | 1.290 |

## SHARP Article Official Test-Set Result

The article's official AV2 single-agent test-set table reports four metrics. It does not report minADE1 or minFDE1 in that table.

| Experiment | Dataset split | MR | b-minFDE6 | minADE6 | minFDE6 |
|---|---|---:|---:|---:|---:|
| SHARP article | AV2 test | 0.14 | 1.83 | 0.64 | 1.19 |

## Training Setup and Time

| Experiment | GPUs | Epochs | Global batch | Approximate training time |
|---|---:|---:|---:|---:|
| SHARP article | 1 x Quadro RTX 8000, 48 GB | 80 | 32 | Not reported |
| Normal SHARP, Lab 1 | 1 x RTX A4000 | 60 | 16 | 88 h 45 min |
| Normal SHARP, Lab 2 | 4 x RTX 2080 Ti | 60 | 16 | 109 h |
| Normal SHARP baseline MHA, Lab 3 | 3 x RTX 2080 Ti | 80 | 24 | 99 h 39 min active training |
| SHARP + scene-encoder Mamba, Lab 2 | 4 x RTX 2080 Ti | 80 | 32 | 129 h 17 min |
| SHARP + temporal-agent Mamba, Lab 2 | 4 x RTX 2080 Ti | 80 | 32 | 160 h 20 min |

## Best Checkpoint minADE6

| Experiment | Best epoch | Best saved minADE6 |
|---|---:|---:|
| SHARP article, published validation result | Not reported | **0.640000** |
| Normal SHARP, Lab 1 | 59 | **0.650232** |
| Normal SHARP, Lab 2 | 56 | 0.661463 |
| Normal SHARP baseline MHA, Lab 3 | Not retained in the available summary | 0.673460 |
| SHARP + scene-encoder Mamba, Lab 2 | 78 | 0.680362 |
| SHARP + temporal-agent Mamba, Lab 2 | 77 | 0.673736 |

## Interpretation

- The SHARP article's standard AV2 validation result is the strongest row across all six reported metrics.
- Normal SHARP on the Lab 1 A4000 is the closest local reproduction: its best minADE6 is 0.010232 higher than the article result, a 1.60% relative gap.
- Temporal-agent Mamba improved best-checkpoint minADE6 over scene-encoder Mamba by 0.006626, approximately 0.97%.
- Neither Mamba placement improved on the normal SHARP Lab 1 or Lab 2 result.
- The Lab 3 normal-SHARP best checkpoint and temporal-agent Mamba best checkpoint were effectively tied: 0.673460 versus 0.673736 minADE6.
- Temporal-agent Mamba was the slowest run, taking approximately 51 hours longer than normal SHARP on Lab 2 and 31 hours longer than scene-encoder Mamba.

## Comparison Caveat

This is not a perfectly controlled architecture ablation. The runs differ in GPU count, epoch count, global batch size, SyncBatchNorm use, and whether the complete metric vector came from final validation or a best-checkpoint record. For the Lab 3 baseline and temporal-agent Mamba runs, the best checkpoint filename provides the best minADE6, while the complete six-metric vector comes from a separate final-validation record. The article test-set row is shown separately because test and validation results should not be treated as identical evaluation records.

## Source

- Alexander Prutsch, Christian Fruhwirth-Reisinger, David Schinagl, and Horst Possegger, [SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting](https://arxiv.org/abs/2603.28091), CVPR 2026. The validation reference comes from the standard AV2 setting in Table 1; the official test result comes from Table 3; the hardware and optimization settings come from supplementary Section B.6.
