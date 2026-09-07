# SHARP Architecture Integration: Results

A reference reproduction followed by composed QKNorm, uncertainty and relative-geometry modifications.

## Comparison

| Variant | Selected minADE6 | Change | MR | b-minFDE6 | minADE1 | Vector minADE6 | minFDE1 | minFDE6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SHARP reference | 0.679282 | +0.00% | 0.155955 | 1.924560 | 1.676867 | 0.679282 | 4.138639 | 1.287697 |
| SHARP + QKNorm + uncertainty + geometry | 0.680589 | +0.19% | 0.155515 | 1.918870 | 1.677496 | 0.680589 | 4.130601 | 1.285462 |

All errors are lower-is-better. Change is relative to the first row's selected minADE6. A dash means that a comparable numerical record is not supplied. Selection values and complete metric vectors are explicitly distinguished below.

## Interpretation

The combined model improves MR, b-minFDE6, minFDE1 and minFDE6. Selected minADE6 changes from 0.679282 to 0.680589 (+0.19%), so the composition does not improve every accuracy criterion. MR changes from 0.155955 to 0.155515 (-0.28%). Independent screening gains are therefore not assumed to sum when mechanisms are composed.

## Protocol and Evidence

80 epochs; global batch 32; four GPUs; seed 2333; AdamW; LR 1e-4 to 1e-5; 13 warm-up epochs.

Two final selected-checkpoint evaluations are available. The residual-Mamba extension has implementation code but no committed final metrics in this evidence set. The older Current_Progress snapshot predates the second completed evaluation and is not the current comparison.

- [Recorded source](Results/SHARP_FINAL_3RUN_20260825-174605/01_official_sharp_baseline/metrics.json)
- [Recorded source](Results/SHARP_FINAL_3RUN_20260825-174605/02_qknorm_uncertainty_geometry/metrics.json)
- [Original run records](Results/SHARP_FINAL_3RUN_20260825-174605/)
- [Implementation and reproduction](README.md)

## Training Time

Timing is reported separately from forecasting error. See the source records for the retained duration definition.

| Variant | Duration (h:mm:ss) |
|---|---:|
| SHARP reference | 125:42:14 |
| SHARP + QKNorm + uncertainty + geometry | 151:53:20 |
