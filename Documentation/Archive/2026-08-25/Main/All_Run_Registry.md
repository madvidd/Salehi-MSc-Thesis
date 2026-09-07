# Complete Cross-Lab Experiment Registry

Updated: 2026-08-25

This registry covers substantive model configurations found in the Thesis repository and the retained Lab 1 A4000 backup. Restarts, checkpoint recoveries, terminal snapshots, and publication retries are operational records rather than separate scientific runs and are not counted as new model variants.

## Lab 1: A4000

| Run or configuration | Dataset and setup | Status | Recorded outcome | Evidence |
|---|---|---|---|---|
| DeMo/SEAM launcher development, nine timestamped attempts | AV2 subset, 3 epochs, 3,000 train and 400 validation scenarios | Grouped setup attempts | The attempts converged on the final `20260505-013724` quick comparison; earlier retries are not independent ablations. | [A4000 snapshot](https://github.com/madvidd/Thesis_A4000/tree/main/Otter49/Snapshots/20260731-053841/files/scratch/.V/work/thesis_results) |
| DeMo + RealMotion quick screening | AV2 subset, 3 epochs, limited batches | Completed | MR `0.685000`, minADE6 `3.455753`, minFDE6 `7.508242`, b-minFDE6 `8.206079` | [comparison summary](https://github.com/madvidd/Thesis_A4000/blob/main/Otter49/Snapshots/20260731-053841/files/scratch/.V/work/thesis_results/20260505-013724/metrics/comparison_summary.json) |
| SEAM streaming quick screening | Same AV2 subset and 3-epoch screen | Completed | MR `0.816000`, minADE6 `4.048000`, minFDE6 `8.341000`, b-minFDE6 `9.013000` | [comparison summary](https://github.com/madvidd/Thesis_A4000/blob/main/Otter49/Snapshots/20260731-053841/files/scratch/.V/work/thesis_results/20260505-013724/metrics/comparison_summary.json) |
| Full-data DeMo + RealMotion | Full AV2, requested 60 epochs, one A4000 | Failed | Repeated CUDA allocator/runtime failures at progressively smaller batches did not produce a comparable completed result. | [full-run logs](https://github.com/madvidd/Thesis_A4000/tree/main/Otter49/Snapshots/20260731-053841/files/scratch/.V/work/thesis_full_results/20260506-092858/logs) |
| Full-data SEAM baseline | Full AV2, 80 epochs, one A4000, requested batch 32 | Completed | MR `0.157916`, b-minFDE6 `1.881538`, minADE1 `1.600458`, minADE6 `0.664724`, minFDE1 `3.965043`, minFDE6 `1.272579` | [SEAM training log](https://github.com/madvidd/Thesis_A4000/blob/main/Otter49/Snapshots/20260731-053841/files/scratch/.V/work/thesis_full_results/20260506-092858/logs/seam_train.log) |
| SHARP preprocessing/setup `20260603-173447` | Full AV2 | Grouped setup attempt | Dataset preprocessing evidence exists; no comparable final metric vector was retained for this setup attempt. | [A4000 snapshot](https://github.com/madvidd/Thesis_A4000/tree/main/Otter49/Snapshots/20260731-053841/files/scratch/.V/work/thesis_sharp_results/20260603-173447) |
| Full-data SHARP baseline `20260603-173704` | Full AV2, 60 epochs, seed 2333, one A4000, batch 16 | Completed | MR `0.149248`, b-minFDE6 `1.840676`, minADE1 `1.577801`, minADE6 `0.650232`, minFDE1 `3.892962`, minFDE6 `1.223235` | [captured terminal evidence](https://github.com/madvidd/Thesis_A4000/blob/main/Otter49/Snapshots/20260731-053841/metadata/tmux_panes/Thesis_0.0.txt) |

The quick subset rows are engineering screens and must not be compared directly with the full-dataset rows.

## Lab 2: SHARP Baselines and Mamba

| Run or configuration | Dataset and setup | Status | Recorded outcome | Evidence |
|---|---|---|---|---|
| Normal SHARP AV2 | AV2, 60 epochs, four RTX 2080 Ti GPUs, recorded global batch 16 | Completed | Best epoch 56; minADE6 `0.661463` | [consolidated metrics](SHARP_AV2_Main_Results.md) |
| Normal SHARP AV1 | AV1 validation | Completed | MR `0.088670`, minADE6 `0.635403`, minFDE6 `1.027457` | [consolidated metrics](SHARP_AV2_Main_Results.md#av1-validation) |
| Normal SHARP nuScenes | nuScenes validation, locally logged at K=6 | Completed | MR `0.302300`, minADE6 `1.150025`, minFDE6 `1.965479` | [consolidated metrics](SHARP_AV2_Main_Results.md#nuscenes-validation) |
| SHARP + scene-token Mamba | AV2, 80 epochs, global batch 32, four RTX 2080 Ti GPUs | Completed | Best epoch 78; minADE6 `0.680362` | [final metrics](../../Lab%202/Main_Results/Mamba_Encoder_Addition_Results/FINAL_METRICS.txt) |
| SHARP + residual temporal-agent Mamba | AV2, 80 epochs, global batch 32, four RTX 2080 Ti GPUs | Completed | Best epoch 77; minADE6 `0.673736` | [best checkpoint](../../Lab%202/Main_Results/Mamba_Encoder_Addition_Results/Temporal_Agent_Mamba/Snapshots/20260803-045226/BEST_CHECKPOINT.txt) |

## Lab 2: Completed 20-Epoch Screening Suite

All ten rows use AV2, seed 2333, 20 epochs, four RTX 2080 Ti GPUs, global batch 32, and the same optimization controls. This is a short-horizon screen, not an 80-epoch article reproduction.

| Test | Status | Best epoch | Best minADE6 | Delta vs 20-epoch baseline | Result |
|---|---|---:|---:|---:|---|
| Baseline SHARP | Completed | 19 | 0.752339 | +0.000000 | Control |
| Confidence-gated memory | Completed | 19 | 0.755349 | +0.003010 | Worse |
| Cross-window consistency | Completed | 18 | 0.776207 | +0.023868 | Worse |
| Learned temporal pooling | Completed | 19 | 0.757637 | +0.005298 | Worse |
| Uncertainty-aware target context | Completed | 19 | 0.749961 | -0.002378 | Improved |
| Relative-geometry attention bias | Completed | 19 | 0.749679 | -0.002660 | Improved; best screen |
| Kinematic motion stem | Completed | 19 | 0.785164 | +0.032825 | Worse |
| Endpoint refinement decoder | Completed | 19 | 0.755898 | +0.003559 | Worse minADE6; better minFDE1 |
| Lane topology graph | Completed | 19 | 0.754238 | +0.001899 | Slightly worse |
| Agent temporal Mamba replacement | Completed | 19 | 0.780190 | +0.027851 | Worse |

Evidence: [completed suite summary](../../Lab%202/SHARP_20_Epoch_10_Test_Suite/Runs/SHARP_AV2_20EPOCH_10TEST_20260805-031733/Summary.md).

## Lab 2: Final Three-Run Package

| Planned run | Configuration | Status | Current result |
|---|---|---|---|
| Official SHARP baseline | Pinned official SHARP source and article-aligned 80-epoch AV2 setup | Prepared | No committed metric yet |
| QKNorm + uncertainty + geometry | Baseline plus QKNorm, uncertainty-aware target context, and relative-geometry bias | Prepared | No committed metric yet |
| QKNorm + uncertainty + geometry + residual Mamba | Run 2 plus a small residual bidirectional Mamba between temporal-agent blocks 2 and 3 | Prepared | No committed metric yet |

Evidence: [final-suite specification](../../Lab%202/SHARP_Final_3_Run_Suite/README.md).

## Lab 3: Completed SHARP Attention Suite

All four rows use AV2, 80 epochs, seed 2333, three RTX 2080 Ti GPUs, global batch 24, and identical non-attention architecture and optimization settings.

| Attention configuration | Status | Best epoch | Best minADE6 | Delta vs local MHA | Result |
|---|---|---:|---:|---:|---|
| Baseline MHA | Completed | 66 | 0.673460 | +0.000000 | Control |
| QKNorm | Completed | 65 | 0.669777 | -0.003683 | Improved by 0.55% |
| Talking-Heads | Completed | 71 | 0.674991 | +0.001531 | Worse by 0.23% |
| QKNorm + Talking-Heads | Completed | 71 | 0.674492 | +0.001032 | Worse by 0.15%; best MR of the four |

Evidence: [completed attention summary](../../Lab%203/Main_Results/Attention_Experiments/Runs/SHARP_ATTENTION_ABLATION_20260717-123916/Summary.md).

## Lab 3: SEAM and Mamba Suite

| Run | Setup | Status at latest committed snapshot | Recorded outcome |
|---|---|---|---|
| SEAM baseline | AV2, configured for 80 epochs, effective global batch 32 on two GPUs | Partial | Best checkpoint minADE6 `0.870494` at epoch 14; latest recorded validation at epoch 15: MR `0.279`, minADE6 `0.870`, minFDE6 `1.750` |
| SEAM + agent-history Mamba | Baseline plus residual Mamba over each agent's observed history | Prepared/pending | No committed metric yet |
| SEAM with future-head Mamba replacement | Baseline with the trajectory-coordinate MLP replaced by a future-sequence Mamba head | Prepared/pending | No committed metric yet |

Evidence: [latest SEAM suite summary](../../Lab%203/SEAM_AV2_Mamba_3_Run/Results/SEAM_AV2_MAMBA_3RUN_20260824-202936/Summary.md).

## Interpretation Rules

- Negative deltas are improvements because lower forecasting errors are better.
- A checkpoint-selected `minADE6` and a full metric vector are only treated as the same evaluation when the retained evidence says that checkpoint was explicitly validated.
- The Lab 2 20-epoch screen, Lab 3 80-epoch attention suite, and long-run Lab 1/Lab 2 results have different hardware and batch construction. Their absolute values are informative but not a strict controlled ablation across laboratories.
- Partial and prepared rows must not be included in final ranking tables until completion and validation records are committed.
