# SHARP State-Space Integration

Scene-token recurrence and residual temporal-agent recurrence, retained as placement experiments.

## Recorded Finding

Moving recurrence to the ordered agent history reduces selected minADE6 from 0.680362 to 0.673736 (-0.97%) relative to the scene-token placement. The result motivates preserving temporal order and a small residual pathway; it is not evidence that arbitrary recurrence improves every SHARP configuration.

Only the scene-token placement has a full-precision final vector in this comparison. No secondary metrics are invented for the temporal-agent checkpoint. These historical placement runs are not merged with the independent architecture-integration control.

## Navigation

- [Results and interpretation](Summary.md)
- [Original result records](Main_Results/Mamba_Encoder_Addition_Results/)
- [Setup and operational details](SETUP.md)
- [Preparation or launch entry point](setup_lab2_sharp_mamba_article80.py)
- [Modification definitions](../../../Results/Info.md)
- [Repository reproduction guide](../../../Documentation/Reproduction.md)

## Shared Controls

The reported placement runs use 80 epochs and global batch 32 on four GPUs. Earlier setup versions also remain available.

Numerical run records and timestamped snapshots are preserved as captured. Historical filenames identify the actual run; the study title identifies its scientific purpose. Earlier launch versions remain available for tracing source evolution, not as additional experiments.
