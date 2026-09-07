# SEAM State-Space Integration: Results

Residual agent-history refinement and future-sequence decoder replacement.

## Comparison

| Variant | Selected minADE6 | Change | MR | b-minFDE6 | minADE1 | Vector minADE6 | minFDE1 | minFDE6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SEAM baseline | 0.662859 | +0.00% | 0.158000 | 1.872000 | 1.614000 | 0.662000 | 3.996000 | 1.262000 |
| SEAM + agent-history Mamba | 0.664814 | +0.29% | 0.156000 | 1.869000 | 1.622000 | 0.664000 | 3.999000 | 1.257000 |
| SEAM with future-head Mamba replacement | 0.648480 | -2.17% | 0.142000 | 1.855000 | 1.588000 | 0.644000 | 3.945000 | 1.230000 |

All errors are lower-is-better. Change is relative to the first row's selected minADE6. A dash means that a comparable numerical record is not supplied. Selection values and complete metric vectors are explicitly distinguished below.

## Interpretation

Future-head replacement reduces selected minADE6 from 0.662859 to 0.648480 (-2.17%). Agent-history addition gives 0.664814 (+0.29%). The result supports sequence modelling at the future-coordinate head in this study, rather than a universal benefit from adding recurrence.

## Protocol and Evidence

80 epochs; global batch 32; two GPUs; seed 2333; AdamW; LR 1e-3 to 1e-5.

The best-checkpoint metric and rounded final log vector are separate observations. The captured summary labels the final validation as epoch 81; that label is retained, not interpreted as an 82-epoch training schedule.

- [Recorded source](Results/SEAM_AV2_MAMBA_3RUN_20260824-202936/Summary.md)
- [Original run records](Results/SEAM_AV2_MAMBA_3RUN_20260824-202936/)
- [Implementation and reproduction](README.md)
