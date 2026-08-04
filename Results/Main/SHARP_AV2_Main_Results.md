# SHARP Main Results Across AV2, AV1, and nuScenes

All trajectory metrics are **lower is better**. Values are reported from the best comparable validation record available for each run. The published SHARP validation result is included as the primary article reference because it uses the same AV2 validation split and reports the same six metrics as the local experiments.

## AV2 Metrics

| Experiment | Machine | Evaluation record | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **SHARP article** | Quadro RTX 8000 | AV2 validation, standard 5 s context | **0.14** | **1.82** | **1.57** | **0.64** | **3.85** | **1.20** |
| Normal SHARP | Lab 1, RTX A4000 | Best checkpoint, epoch 59 | 0.149 | 1.841 | 1.578 | 0.650 | 3.893 | 1.223 |
| Normal SHARP | Lab 2, 4 x RTX 2080 Ti | Best checkpoint, epoch 56 | 0.150608 | 1.864411 | 1.599569 | 0.661463 | 3.962160 | 1.242391 |
| Normal SHARP (baseline MHA) | Lab 3, 3 x RTX 2080 Ti | Final validation | 0.153231 | 1.927635 | 1.671288 | 0.686853 | 4.063601 | 1.287932 |
| SHARP + QKNorm attention | Lab 3, 3 x RTX 2080 Ti | Best checkpoint, epoch 65 | 0.154471 | 1.899642 | 1.671904 | 0.669777 | 4.119903 | 1.266240 |
| SHARP + scene-encoder Mamba | Lab 2, 4 x RTX 2080 Ti | Best checkpoint, epoch 78 | 0.157353 | 1.926419 | 1.680364 | 0.680362 | 4.130003 | 1.290422 |
| SHARP + temporal-agent Mamba | Lab 2, 4 x RTX 2080 Ti | Final validation, epoch 79 | 0.157 | 1.920 | 1.640 | 0.676 | 4.050 | 1.290 |

## SHARP Article Official Test-Set Result

The article's official AV2 single-agent test-set table reports four metrics. It does not report minADE1 or minFDE1 in that table.

| Experiment | Dataset split | MR | b-minFDE6 | minADE6 | minFDE6 |
|---|---|---:|---:|---:|---:|
| SHARP article | AV2 test | 0.14 | 1.83 | 0.64 | 1.19 |

## AV2 Training Setup and Time

| Experiment | GPUs | Epochs | Global batch | Approximate training time |
|---|---:|---:|---:|---:|
| SHARP article | 1 x Quadro RTX 8000, 48 GB | 80 | 32 | Not reported |
| Normal SHARP, Lab 1 | 1 x RTX A4000 | 60 | 16 | 88 h 45 min |
| Normal SHARP, Lab 2 | 4 x RTX 2080 Ti | 60 | 16 | 109 h |
| Normal SHARP baseline MHA, Lab 3 | 3 x RTX 2080 Ti | 80 | 24 | 99 h 39 min active training |
| SHARP + QKNorm attention, Lab 3 | 3 x RTX 2080 Ti | 80 | 24 | Not retained in lightweight archive |
| SHARP + scene-encoder Mamba, Lab 2 | 4 x RTX 2080 Ti | 80 | 32 | 129 h 17 min |
| SHARP + temporal-agent Mamba, Lab 2 | 4 x RTX 2080 Ti | 80 | 32 | 160 h 20 min |

## AV2 Best Checkpoint minADE6

| Experiment | Best epoch | Best saved minADE6 |
|---|---:|---:|
| SHARP article, published validation result | Not reported | **0.640000** |
| Normal SHARP, Lab 1 | 59 | **0.650232** |
| Normal SHARP, Lab 2 | 56 | 0.661463 |
| Normal SHARP baseline MHA, Lab 3 | Not retained in the available summary | 0.673460 |
| SHARP + QKNorm attention, Lab 3 | 65 | 0.669777 |
| SHARP + scene-encoder Mamba, Lab 2 | 78 | 0.680362 |
| SHARP + temporal-agent Mamba, Lab 2 | 77 | 0.673736 |

## Argoverse 1 Results

The SHARP article's supplementary Table 11 and the Lab 2 run both report AV1 validation metrics at `K=6`, so MR6, minADE6, and minFDE6 are directly comparable. The article does not report the other three values in that table.

| Source | Machine | Evaluation record | MR6 | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **SHARP article** | Quadro RTX 8000 | AV1 validation | **0.07** | Not reported | Not reported | **0.58** | Not reported | **0.90** |
| Normal SHARP, Lab 2 | RTX 2080 Ti independent run | AV1 validation | 0.088670 | 1.603313 | 1.289943 | 0.635403 | 2.765118 | 1.027457 |

Compared with the article validation result, the Lab 2 run is higher by 0.018670 MR6, 0.055403 minADE6, and 0.127457 minFDE6. These correspond to relative gaps of approximately 26.67%, 9.55%, and 14.16%, respectively. The article trained AV1 for 60 epochs with a global batch size of 32 on one Quadro RTX 8000.

## nuScenes Results

The article's main nuScenes challenge table reports test-set metrics at `K=5`, whereas the Lab 2 validation output is labeled at `K=6`. The values are therefore included for reference but are not a strict like-for-like comparison.

| Source | Evaluation record | MR | b-minFDE | minADE1 | minADEK | minFDE1 | minFDEK |
|---|---|---:|---:|---:|---:|---:|---:|
| **SHARP article** | nuScenes challenge test, K=5 | **0.28** | Not reported | Not reported | **1.13** | **6.02** | Not reported |
| SHARP article, full 2 s context | Varying-length evaluation, K=5 | Not reported | Not reported | Not reported | 1.13 | Not reported | 1.92 |
| Normal SHARP, Lab 2 | nuScenes validation, logged at K=6 | 0.302300 | 2.601139 | 2.998736 | 1.150025 | 6.344474 | 1.965479 |

Against the article's challenge-test values, the Lab 2 output is higher by 0.022300 MR, 0.020025 minADE, and 0.324474 minFDE1. These differences are only indicative because both the split and number of modes differ. The article trained nuScenes for 25 epochs with a global batch size of 32 on one Quadro RTX 8000.

## Interpretation

- The SHARP article's standard AV2 validation result is the strongest row across all six reported metrics.
- Normal SHARP on the Lab 1 A4000 is the closest local reproduction: its best minADE6 is 0.010232 higher than the article result, a 1.60% relative gap.
- QKNorm improved the Lab 3 best-checkpoint minADE6 from 0.673460 for baseline MHA to 0.669777, an absolute improvement of 0.003683 or approximately 0.55%.
- QKNorm also reduced the reported b-minFDE6 and minFDE6, while MR, minADE1, and minFDE1 were slightly higher; the complete baseline vector came from final validation whereas QKNorm was evaluated at its best checkpoint.
- Temporal-agent Mamba improved best-checkpoint minADE6 over scene-encoder Mamba by 0.006626, approximately 0.97%.
- Neither Mamba placement improved on the normal SHARP Lab 1 or Lab 2 result.
- The Lab 3 normal-SHARP best checkpoint and temporal-agent Mamba best checkpoint were effectively tied: 0.673460 versus 0.673736 minADE6.
- Temporal-agent Mamba was the slowest run, taking approximately 51 hours longer than normal SHARP on Lab 2 and 31 hours longer than scene-encoder Mamba.

## Comparison Caveat

This is not a perfectly controlled architecture ablation. The runs differ in dataset split, number of predicted modes, GPU count, epoch count, global batch size, SyncBatchNorm use, and whether the complete metric vector came from final validation or a best-checkpoint record. For the Lab 3 baseline and temporal-agent Mamba runs, the best checkpoint filename provides the best minADE6, while the complete six-metric vector comes from a separate final-validation record. The article test-set row is shown separately because test and validation results should not be treated as identical evaluation records.

## Source

- Alexander Prutsch, Christian Fruhwirth-Reisinger, David Schinagl, and Horst Possegger, [SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting](https://arxiv.org/abs/2603.28091), CVPR 2026. The AV2 validation reference comes from Table 1, the AV2 official test result from Table 3, the nuScenes challenge result from Table 5, the varying-length references from Table 6, the AV1 validation result from supplementary Table 11, and the hardware and optimization settings from supplementary Section B.6.
