# SHARP Attention Experiments Compared with the Published SHARP Result

All trajectory metrics are **lower is better**. The SHARP article row is its
standard Argoverse 2 validation result. Completed Lab 3 rows are evaluations of
locally saved checkpoints on the AV2 validation split.

## Full Metric Comparison

| Method | Evaluation record | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---|---|---:|---:|---:|---:|---:|---:|
| **SHARP article** | Published AV2 validation result | **0.140000** | **1.820000** | **1.570000** | **0.640000** | **3.850000** | **1.200000** |
| SHARP baseline MHA | Lab 3 final validation | 0.153231 | 1.927635 | 1.671288 | 0.686853 | 4.063601 | 1.287932 |
| SHARP + QKNorm | Lab 3 best checkpoint, epoch 65 | 0.154471 | 1.899642 | 1.671904 | **0.669777** | 4.119903 | **1.266240** |
| SHARP + Talking Heads | Lab 3 best checkpoint, epoch 71 | 0.156510 | 1.927414 | 1.675181 | 0.674991 | 4.130968 | 1.285642 |
| SHARP + QKNorm + Talking Heads | Lab 3, currently training | Not final | Not final | Not final | Not final | Not final | Not final |

Among the completed Lab 3 alternatives, QKNorm has the best minADE6,
b-minFDE6, and minFDE6. Baseline MHA has the best MR, minADE1, and minFDE1.
Talking Heads does not lead any of the six completed-run metrics.

## Gap from the SHARP Article

Positive values mean the Lab 3 metric is higher, and therefore worse, than the
published SHARP result.

| Method | MR delta | b-minFDE6 delta | minADE1 delta | minADE6 delta | minFDE1 delta | minFDE6 delta |
|---|---:|---:|---:|---:|---:|---:|
| Baseline MHA | +0.013231 | +0.107635 | +0.101288 | +0.046853 | +0.213601 | +0.087932 |
| QKNorm | +0.014471 | **+0.079642** | +0.101904 | **+0.029777** | +0.269903 | **+0.066240** |
| Talking Heads | +0.016510 | +0.107414 | +0.105181 | +0.034991 | +0.280968 | +0.085642 |

QKNorm is closest to the paper for b-minFDE6, minADE6, and minFDE6. Baseline
MHA is closest for MR, minADE1, and minFDE1. None of the completed Lab 3
attention configurations matches or improves the published SHARP result.

## Controlled Best-Checkpoint minADE6

SHARP checkpoint selection monitors minADE6. Comparing the saved minimum for
each completed Lab 3 run is therefore the clearest local attention ablation.

| Method | Best epoch | Best saved minADE6 | Delta from Lab 3 baseline | Relative change |
|---|---:|---:|---:|---:|
| Baseline MHA | 66 | 0.673460 | Reference | Reference |
| **QKNorm** | **65** | **0.669777** | **-0.003683** | **0.55% better** |
| Talking Heads | 71 | 0.674991 | +0.001531 | 0.23% worse |
| QKNorm + Talking Heads | In progress | Not final | Not comparable | Not comparable |

This checkpoint comparison shows a small but real QKNorm improvement over the
local baseline. Talking Heads alone is slightly worse than the local baseline.
The combined method must finish before it can be ranked.

## What Changed

- **Baseline MHA:** the original SHARP multi-head dot-product attention.
- **QKNorm:** L2-normalizes every query and key head before their dot product,
  then applies one learned, bounded logit scale per head. This controls logit
  magnitude while preserving SHARP's attention topology and streaming design.
- **Talking Heads:** learns head-mixing matrices before softmax and after
  softmax. This lets attention heads exchange information while leaving the
  surrounding SHARP blocks unchanged.
- **QKNorm + Talking Heads:** applies both changes in the same replacement
  attention module. Its result is still provisional.

The variants replace only the `MultiheadAttention` constructor used in SHARP's
standard and custom transformer blocks. The short-window processing,
instance-aware context streaming, decoder, losses, optimizer schedule, and data
pipeline are retained.

## Training-Control Caveat

The Lab 3 completed runs share the same local control:

- AV2 processed training and validation data;
- 80 epochs, seed 2333, LR 1e-4 to 1e-5, and 13 warm-up epochs;
- three RTX 2080 Ti GPUs, per-GPU batch 8, global batch 24;
- DDP with synchronized BatchNorm.

The SHARP article used global batch 32 on one Quadro RTX 8000. Consequently,
the Lab 3 experiments are controlled against each other, but they are not exact
hardware-and-batch reproductions of the article. The complete baseline metric
vector is from final validation, whereas the QKNorm and Talking Heads vectors
are evaluations of their best saved checkpoints. The separate minADE6 table
uses best-checkpoint values for all three local methods.

## Sources

- SHARP article: Alexander Prutsch, Christian Fruhwirth-Reisinger, David
  Schinagl, and Horst Possegger, *SHARP: Short-Window Streaming for Accurate
  and Robust Prediction in Motion Forecasting*, CVPR 2026, Table 1.
- Baseline record:
  `Lab3_SHARP_Baseline_MHA_final_20260726-023208/FINAL_VALIDATION.txt`.
- QKNorm record:
  `Lab3_SHARP_Attention_qknorm_completed_20260731-200639/FINAL_METRICS.json`.
- Talking Heads record:
  `Lab3_SHARP_Attention_talking_heads_completed_20260805-181428/FINAL_METRICS.json`.
- Combined-run status:
  `Snapshots/20260805-213248/RUN_STATUS.txt`.
