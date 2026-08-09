# Lab 3 Attention Experiment Summary

Lower values are better for every listed metric.

| Attention | Status | Best epoch | Best minADE6 | Latest epoch | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline MHA | Completed | 66 | 0.673460 | 79 | 0.153231 | 1.927635 | 1.671288 | 0.686853 | 4.063601 | 1.287932 |
| QK-Norm | Completed | 65 | 0.669777 | 79 | 0.154471 | 1.899642 | 1.671904 | 0.669777 | 4.119903 | 1.266240 |
| Talking-Heads | Completed | 71 | 0.674991 | 79 | 0.156510 | 1.927414 | 1.675181 | 0.674991 | 4.130968 | 1.285642 |
| QK-Norm + Talking-Heads | Running (resumed after OOM) | 33 | 0.716198 | 36 | 0.177 | 2.020 | 1.800 | 0.722 | 4.490 | 1.370 |

Best minADE6 comes from checkpoint filenames. The remaining metrics are the latest or final values visible in the logs and are not necessarily from the best-minADE6 checkpoint. Baseline MHA includes its resumed run; the interrupted pre-resume segment ended at epoch 5. The combined experiment resumed from its epoch-35 checkpoint after a CUDA out-of-memory failure during epoch 36.
