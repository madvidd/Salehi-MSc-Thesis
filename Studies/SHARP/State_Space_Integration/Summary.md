# SHARP State-Space Integration: Results

Scene-token recurrence and residual temporal-agent recurrence, retained as placement experiments.

## Comparison

| Variant | Selected minADE6 | Change | MR | b-minFDE6 | minADE1 | Vector minADE6 | minFDE1 | minFDE6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scene-token Mamba | 0.680362 | +0.00% | 0.157353 | 1.926419 | 1.680364 | 0.680362 | 4.130003 | 1.290422 |
| Residual temporal-agent Mamba | 0.673736 | -0.97% | - | - | - | - | - | - |

All errors are lower-is-better. Change is relative to the first row's selected minADE6. A dash means that a comparable numerical record is not supplied. Selection values and complete metric vectors are explicitly distinguished below.

## Interpretation

Moving recurrence to the ordered agent history reduces selected minADE6 from 0.680362 to 0.673736 (-0.97%) relative to the scene-token placement. The result motivates preserving temporal order and a small residual pathway; it is not evidence that arbitrary recurrence improves every SHARP configuration.

## Protocol and Evidence

The reported placement runs use 80 epochs and global batch 32 on four GPUs. Earlier setup versions also remain available.

Only the scene-token placement has a full-precision final vector in this comparison. No secondary metrics are invented for the temporal-agent checkpoint. These historical placement runs are not merged with the independent architecture-integration control.

- [Recorded source](Main_Results/Mamba_Encoder_Addition_Results/FINAL_METRICS.txt)
- [Recorded source](Main_Results/Mamba_Encoder_Addition_Results/Temporal_Agent_Mamba/Snapshots/20260803-045226/BEST_CHECKPOINT.txt)
- [Original run records](Main_Results/Mamba_Encoder_Addition_Results/)
- [Implementation and reproduction](README.md)
