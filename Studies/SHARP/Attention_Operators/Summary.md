# SHARP Attention Operators: Results

Multi-head attention, query-key normalisation, head mixing and their combination.

## Comparison

| Variant | Selected minADE6 | Change | MR | b-minFDE6 | minADE1 | Vector minADE6 | minFDE1 | minFDE6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline MHA | 0.673460 | +0.00% | 0.153231 | 1.927635 | 1.671288 | 0.686853 | 4.063601 | 1.287932 |
| QK-Norm | 0.669777 | -0.55% | 0.154471 | 1.899642 | 1.671904 | 0.669777 | 4.119903 | 1.266240 |
| Talking-Heads | 0.674991 | +0.23% | 0.156510 | 1.927414 | 1.675181 | 0.674991 | 4.130968 | 1.285642 |
| QK-Norm + Talking-Heads | 0.674492 | +0.15% | 0.152064 | 1.931623 | 1.663308 | 0.674492 | 4.090390 | 1.291059 |

All errors are lower-is-better. Change is relative to the first row's selected minADE6. A dash means that a comparable numerical record is not supplied. Selection values and complete metric vectors are explicitly distinguished below.

## Interpretation

QKNorm improves selected minADE6 from 0.673460 to 0.669777 (-0.55%). Talking-Heads alone gives +0.23%, and QKNorm with head mixing gives +0.15%. The combined operator has the lowest MR in the retained vectors, but not the lowest trajectory-average error.

## Protocol and Evidence

80 epochs; global batch 24; three GPUs; seed 2333; AdamW; LR 1e-4 to 1e-5.

The MHA vector is final-epoch validation, whereas the alternatives have retained selected-checkpoint evaluations. The primary comparison therefore uses the saved selected minADE6 for every operator; do not attribute all secondary-metric changes to a fully matched checkpoint evaluation.

- [Recorded source](Main_Results/Attention_Experiments/Runs/SHARP_ATTENTION_ABLATION_20260717-123916/Summary.md)
- [Original run records](Main_Results/Attention_Experiments/)
- [Implementation and reproduction](README.md)
