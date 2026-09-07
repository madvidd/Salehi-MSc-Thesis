# SHARP and Related AV2 Results

Updated: 2026-08-25

Lower is better for every forecasting metric. The `metric-vector minADE6` column belongs to the displayed metric vector. `Best checkpoint minADE6` is separately reported when the retained best checkpoint differs from the displayed final-validation vector.

## Main AV2 Validation Results

| Model and experiment | Evaluation record | MR | b-minFDE6 | minADE1 | Metric-vector minADE6 | minFDE1 | minFDE6 | Best checkpoint minADE6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SHARP article | Article validation table | 0.140000 | 1.820000 | 1.570000 | 0.640000 | 3.850000 | 1.200000 | 0.640000 |
| SHARP, Lab 1 A4000 | Validated epoch 59 of 60 | 0.149248 | 1.840676 | 1.577801 | 0.650232 | 3.892962 | 1.223235 | 0.650232 |
| SEAM baseline, Lab 1 A4000 | Final validation after 80 epochs | 0.157916 | 1.881538 | 1.600458 | 0.664724 | 3.965043 | 1.272579 | Not independently retained |
| Normal SHARP, Lab 2 | Validated best checkpoint, epoch 56 | 0.150608 | 1.864411 | 1.599569 | 0.661463 | 3.962160 | 1.242391 | 0.661463 |
| SHARP + scene-token Mamba, Lab 2 | Validated best checkpoint, epoch 78 of 80 | 0.157353 | 1.926419 | 1.680364 | 0.680362 | 4.130003 | 1.290422 | 0.680362 |
| SHARP + residual temporal-agent Mamba, Lab 2 | Final epoch 79 progress vector | 0.157000 | 1.920000 | 1.640000 | 0.676000 | 4.050000 | 1.290000 | 0.673736 at epoch 77 |
| SHARP baseline MHA, Lab 3 | Final validation after 80 epochs | 0.153231 | 1.927635 | 1.671288 | 0.686853 | 4.063601 | 1.287932 | 0.673460 at epoch 66 |
| SHARP + QKNorm, Lab 3 | Validated retained metrics | 0.154471 | 1.899642 | 1.671904 | 0.669777 | 4.119903 | 1.266240 | 0.669777 at epoch 65 |
| SHARP + Talking-Heads, Lab 3 | Validated retained metrics | 0.156510 | 1.927414 | 1.675181 | 0.674991 | 4.130968 | 1.285642 | 0.674991 at epoch 71 |
| SHARP + QKNorm + Talking-Heads, Lab 3 | Validated retained metrics | 0.152064 | 1.931623 | 1.663308 | 0.674492 | 4.090390 | 1.291059 | 0.674492 at epoch 71 |

The Lab 1 SHARP result is the closest recorded local SHARP reproduction to the article on `minADE6`: `0.650232` versus `0.64`, an absolute gap of `0.010232` or approximately `1.60%`.

The Lab 3 attention comparison is controlled internally. QKNorm improved best `minADE6` by `0.003683` (`0.55%`) versus its local MHA control. It also improved b-minFDE6 and minFDE6 in the retained vector, although MR was slightly worse. Talking-Heads and the combined operator did not improve best `minADE6`; the combined operator did produce the best MR.

## SHARP Article AV2 Test Set

The article's AV2 test-set values are included for reference and are not directly comparable with local validation rows.

| Source | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---|---:|---:|---:|---:|---:|---:|
| SHARP article, AV2 test | 0.140000 | 1.830000 | Not reported | 0.640000 | Not reported | 1.190000 |

## Lab 2 20-Epoch Controlled Screening

All rows use AV2, seed 2333, 20 epochs, global batch 32 across four RTX 2080 Ti GPUs, AdamW, LR `1e-4` to `1e-5`, 13 warm-up epochs, and SyncBatchNorm. The baseline and modifications were trained independently. This table is suitable for screening conclusions, not for claiming the article's 80-epoch accuracy.

| Test | Best epoch | MR | b-minFDE6 | minADE1 | minADE6 | Delta minADE6 | minFDE1 | minFDE6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline SHARP | 19 | 0.192652 | 2.064007 | 1.886442 | 0.752339 | +0.000000 | 4.672195 | 1.434819 |
| Confidence-gated memory | 19 | 0.194413 | 2.067328 | 1.876097 | 0.755349 | +0.003010 | 4.694828 | 1.443534 |
| Cross-window consistency | 18 | 0.202497 | 2.096022 | 1.872110 | 0.776207 | +0.023868 | 4.785388 | 1.476766 |
| Learned temporal pooling | 19 | 0.201897 | 2.080306 | 1.891240 | 0.757637 | +0.005298 | 4.739688 | 1.463850 |
| Uncertainty-aware target context | 19 | 0.198575 | 2.074243 | 1.863236 | 0.749961 | -0.002378 | 4.676969 | 1.447852 |
| Relative-geometry attention bias | 19 | 0.190852 | 2.051891 | 1.875754 | 0.749679 | -0.002660 | 4.660576 | 1.432463 |
| Kinematic motion stem | 19 | 0.198175 | 2.068197 | 1.942671 | 0.785164 | +0.032825 | 4.701560 | 1.448078 |
| Endpoint refinement decoder | 19 | 0.196254 | 2.058200 | 1.867779 | 0.755898 | +0.003559 | 4.641530 | 1.441044 |
| Lane topology graph | 19 | 0.200616 | 2.065051 | 1.903412 | 0.754238 | +0.001899 | 4.696756 | 1.443290 |
| Agent temporal Mamba replacement | 19 | 0.212822 | 2.141835 | 1.993448 | 0.780190 | +0.027851 | 4.973433 | 1.514158 |

`Delta minADE6` is variant minus baseline, so a negative value is better. Relative geometry was the strongest 20-epoch screen, followed closely by uncertainty-aware target context. Endpoint refinement did not improve minADE6 but did improve minFDE1 from `4.672195` to `4.641530`.

Evidence: [completed suite summary](../../Lab%202/SHARP_20_Epoch_10_Test_Suite/Runs/SHARP_AV2_20EPOCH_10TEST_20260805-031733/Summary.md).

## Training Setup and Time Context

| Run | Epochs | Global batch | Hardware | Approximate elapsed time | Status |
|---|---:|---:|---|---:|---|
| SHARP article AV2 | 80 | 32 | 1 x RTX 8000 | Not stated | Article reference |
| Normal SHARP, Lab 1 | 60 | 16 | 1 x A4000 | 88 h 45 min | Completed |
| Normal SHARP, Lab 2 | 60 | 16 | 4 x RTX 2080 Ti | About 109 h | Completed |
| SHARP baseline MHA, Lab 3 | 80 | 24 | 3 x RTX 2080 Ti | About 99 h 39 min | Completed |
| SHARP + scene-token Mamba, Lab 2 | 80 | 32 | 4 x RTX 2080 Ti | About 129 h 17 min | Completed |
| SHARP + temporal-agent Mamba, Lab 2 | 80 | 32 | 4 x RTX 2080 Ti | About 160 h 20 min | Completed |

More GPUs did not guarantee lower wall-clock time. DDP communication, validation, data loading, different global batches, synchronized BatchNorm, architecture cost, and shared-machine load all affect throughput.

## AV1 Validation

| Source | Evaluation | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---|---|---:|---:|---:|---:|---:|---:|
| SHARP article | AV1 validation | 0.070000 | Not reported | Not reported | 0.580000 | Not reported | 0.900000 |
| Normal SHARP, Lab 2 | Local AV1 validation | 0.088670 | 1.603313 | 1.289943 | 0.635403 | 2.765118 | 1.027457 |

The local Lab 2 run was worse by `0.018670` MR, `0.055403` minADE6, and `0.127457` minFDE6.

## nuScenes Validation

| Source | Evaluation | MR | b-minFDE | minADE1 | minADEK | minFDE1 | minFDEK |
|---|---|---:|---:|---:|---:|---:|---:|
| SHARP article | nuScenes challenge test, K=5 | 0.280000 | Not reported | Not reported | 1.130000 | 6.020000 | Not reported |
| SHARP article | Full 2 s context, varying-length evaluation, K=5 | Not reported | Not reported | Not reported | 1.130000 | Not reported | 1.920000 |
| Normal SHARP, Lab 2 | Local nuScenes validation, logged at K=6 | 0.302300 | 2.601139 | 2.998736 | 1.150025 | 6.344474 | 1.965479 |

Against the article challenge-test row, the local validation row was higher by `0.022300` MR, `0.020025` minADE, and `0.324474` single-trajectory minFDE. The split and number of modes differ, so this comparison is descriptive rather than controlled.

## Partial and Prepared Work

- The Lab 3 SEAM baseline is partial. Its latest committed snapshot reports best checkpoint `minADE6=0.870494` at epoch 14 and a latest validation vector at epoch 15. The SEAM Mamba-addition and Mamba-replacement runs remain pending.
- The Lab 2 final three-run SHARP package is prepared but has no committed metrics. It must not be included in final rankings until its result files exist.

See [the complete run registry](All_Run_Registry.md) for status and evidence links, and [the modification glossary](../Info.md) for architectural explanations and hypotheses.

## References

- SHARP paper: [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Prutsch_SHARP_Short-Window_Streaming_for_Accurate_and_Robust_Prediction_in_Motion_Forecasting_CVPR_2026_paper.html)
- SHARP repository: [cpr-uzh/SHARP](https://github.com/cpr-uzh/SHARP)
- SHARP local attention setup: [Lab 3 controlled-ablation documentation](../../Lab%203/README.md)
