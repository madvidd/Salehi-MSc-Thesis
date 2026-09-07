# SHARP Architectural Ablation

Nine independent memory, context, pooling, geometry and decoder interventions against a shared control.

## Recorded Finding

Relative geometry gives the lowest selected minADE6 (0.749679, -0.35%), followed by uncertainty-aware context (0.749961, -0.32%). The remaining interventions do not improve this primary error. Endpoint refinement improves minFDE1 despite its higher minADE6, illustrating that trajectory-average and endpoint criteria need not move together.

The captured summary derives selected minADE6 from checkpoint filenames and uses retained or nearest validation records for other metrics. Treat the complete vector as reported evidence, not a newly re-evaluated selected checkpoint.

## Navigation

- [Results and interpretation](Summary.md)
- [Original result records](Runs/SHARP_AV2_20EPOCH_10TEST_20260805-031733/)
- [Setup and operational details](SETUP.md)
- [Preparation or launch entry point](launch_lab2_10test_suite_v5.sh)
- [Artifact publisher](publish_lab2_10test_snapshot.sh)
- [Modification definitions](../../../Results/Info.md)
- [Repository reproduction guide](../../../Documentation/Reproduction.md)

## Shared Controls

20 epochs; global batch 32; four GPUs; seed 2333; AdamW; LR 1e-4 to 1e-5; 13 warm-up epochs.

Numerical run records and timestamped snapshots are preserved as captured. Historical filenames identify the actual run; the study title identifies its scientific purpose. Earlier launch versions remain available for tracing source evolution, not as additional experiments.
