# Lab 3 Attention Experiment Summary

Lower values are better for every listed metric.

| Attention | Status | Best epoch | Best minADE6 | Latest epoch | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline MHA | Saved/inactive | 4 | 1.102881 | 5 | 0.356 | 2.790 | 2.700 | 1.100 | 6.550 | 2.790 |
| QK-Norm | Running | 65 | 0.669777 | 79 | 0.15447057783603668 | 1.8996416330337524 | 1.6719043254852295 | 0.6697766780853271 | 4.119902610778809 | 1.266239881515503 |
| Talking-Heads | Saved/inactive | 71 | 0.674991 | 79 | 0.1565099060535431 | 1.927414059638977 | 1.6751811504364014 | 0.674991250038147 | 4.130967617034912 | 1.2856419086456299 |
| QK-Norm + Talking-Heads | Running (recovered after OOM) | 33 | 0.716198 | 36 | 0.177 | 2.020 | 1.800 | 0.722 | 4.490 | 2.020 |

Best minADE6 comes from checkpoint filenames. Other metrics are the latest or final values visible in the logs.
