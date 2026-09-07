# SHARP Architecture Integration

A reference reproduction followed by composed QKNorm, uncertainty and relative-geometry modifications.

## Recorded Finding

The combined model improves MR, b-minFDE6, minFDE1 and minFDE6. Selected minADE6 changes from 0.679282 to 0.680589 (+0.19%), so the composition does not improve every accuracy criterion. MR changes from 0.155955 to 0.155515 (-0.28%). Independent screening gains are therefore not assumed to sum when mechanisms are composed.

Two final selected-checkpoint evaluations are available. The residual-Mamba extension has implementation code but no committed final metrics in this evidence set. The older Current_Progress snapshot predates the second completed evaluation and is not the current comparison.

## Navigation

- [Results and interpretation](Summary.md)
- [Original result records](Results/SHARP_FINAL_3RUN_20260825-174605/)
- [Setup and operational details](SETUP.md)
- [Preparation or launch entry point](launch_final_3run_suite.sh)
- [Artifact publisher](publish_final_artifacts.sh)
- [Modification definitions](../../../Results/Info.md)
- [Repository reproduction guide](../../../Documentation/Reproduction.md)

## Shared Controls

80 epochs; global batch 32; four GPUs; seed 2333; AdamW; LR 1e-4 to 1e-5; 13 warm-up epochs.

Numerical run records and timestamped snapshots are preserved as captured. Historical filenames identify the actual run; the study title identifies its scientific purpose. Earlier launch versions remain available for tracing source evolution, not as additional experiments.
