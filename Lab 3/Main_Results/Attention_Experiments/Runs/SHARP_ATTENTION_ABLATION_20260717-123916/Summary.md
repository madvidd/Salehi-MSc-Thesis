# Lab 3 Attention Experiment Summary

Generated: 2026-08-15T16:49:49.419017+01:00

Lower values are better for every listed metric.

| Attention | Status | Best epoch | Best minADE6 | Latest epoch | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline MHA | Completed | 66 | 0.673460 | 79 | 0.153231 | 1.927635 | 1.671288 | 0.686853 | 4.063601 | 1.287932 |
| QK-Norm | Completed | 65 | 0.669777 | 79 | 0.154471 | 1.899642 | 1.671904 | 0.669777 | 4.119903 | 1.266240 |
| Talking-Heads | Completed | 71 | 0.674991 | 79 | 0.156510 | 1.927414 | 1.675181 | 0.674991 | 4.130968 | 1.285642 |
| QK-Norm + Talking-Heads | Running (batch 4, accumulation 2) | 63 | 0.675038 | 64 | 0.152000 | 1.920000 | 1.660000 | 0.675000 | 4.090000 | 1.280000 |

Best minADE6 is read from checkpoint filenames. Remaining metrics use validated metrics.json values when available, otherwise the latest values recoverable from logs; they may not belong to the best-minADE6 checkpoint.
