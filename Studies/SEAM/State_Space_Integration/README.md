# SEAM State-Space Integration

Residual agent-history refinement and future-sequence decoder replacement.

## Recorded Finding

Future-head replacement reduces selected minADE6 from 0.662859 to 0.648480 (-2.17%). Agent-history addition gives 0.664814 (+0.29%). The result supports sequence modelling at the future-coordinate head in this study, rather than a universal benefit from adding recurrence.

The best-checkpoint metric and rounded final log vector are separate observations. The captured summary labels the final validation as epoch 81; that label is retained, not interpreted as an 82-epoch training schedule.

## Navigation

- [Results and interpretation](Summary.md)
- [Original result records](Results/SEAM_AV2_MAMBA_3RUN_20260824-202936/)
- [Setup and operational details](SETUP.md)
- [Preparation or launch entry point](launch_lab3_seam_3run.sh)
- [Artifact publisher](publish_seam_suite_snapshot.sh)
- [Modification definitions](../../../Results/Info.md)
- [Repository reproduction guide](../../../Documentation/Reproduction.md)

## Shared Controls

80 epochs; global batch 32; two GPUs; seed 2333; AdamW; LR 1e-3 to 1e-5.

Numerical run records and timestamped snapshots are preserved as captured. Historical filenames identify the actual run; the study title identifies its scientific purpose. Earlier launch versions remain available for tracing source evolution, not as additional experiments.
