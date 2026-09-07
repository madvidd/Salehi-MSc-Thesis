# SEAM Combined Extension

Uncertainty, geometry and QKNorm composed with a future-head Mamba module.

## Recorded Finding

This implementation combines all three context/attention mechanisms with future-head Mamba. Accuracy claims require its own completed evaluation; they cannot be inferred from independent ablations.

The latest committed snapshot records early training and no final selected-checkpoint evaluation. The code is available for reproducibility, but this extension is not included in completed-result rankings. No separate SEAM + QKNorm + uncertainty + geometry run without Mamba is recorded.

## Navigation

- [Results and interpretation](Summary.md)
- [Original result records](Results/SEAM_AV2_80EPOCH_COMBINED_20260906-144118/)
- [Setup and operational details](SETUP.md)
- [Preparation or launch entry point](launch_combined.sh)
- [Artifact publisher](publish_combined.py)
- [Modification definitions](../../../Results/Info.md)
- [Repository reproduction guide](../../../Documentation/Reproduction.md)

## Shared Controls

80 epochs configured; global batch 32; two GPUs; seed 2333; AdamW; LR 1e-3 to 1e-5.

Numerical run records and timestamped snapshots are preserved as captured. Historical filenames identify the actual run; the study title identifies its scientific purpose. Earlier launch versions remain available for tracing source evolution, not as additional experiments.
