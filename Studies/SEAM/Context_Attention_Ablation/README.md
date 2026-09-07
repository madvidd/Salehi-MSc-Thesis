# SEAM Context and Attention Ablation

Independent uncertainty-aware context, relative geometry and QKNorm interventions.

## Recorded Finding

All three independent interventions improve selected minADE6 against the matched control: uncertainty -2.60%, geometry -1.43% and QKNorm -3.27%. QKNorm achieves the lowest selected error, 0.703845. These independent gains do not establish that their composition will be additive.

For relative geometry, selected minADE6 is 0.717261 while the final vector contains 0.718735. These are not the same checkpoint. The earlier batch-32 attempt is retained separately and is not pooled with this batch-48 study.

## Navigation

- [Results and interpretation](Summary.md)
- [Original result records](Results/SEAM_AV2_20EPOCH_4TEST_MAXRES_20260903-114512/)
- [Setup and operational details](SETUP.md)
- [Preparation or launch entry point](launch_lab3_seam_20epoch_4test_max_resources.sh)
- [Artifact publisher](publish_seam_20epoch_snapshot.sh)
- [Modification definitions](../../../Results/Info.md)
- [Repository reproduction guide](../../../Documentation/Reproduction.md)

## Shared Controls

20 epochs; global batch 48; three GPUs; seed 2333; AdamW; LR 1e-3 to 1e-5.

Numerical run records and timestamped snapshots are preserved as captured. Historical filenames identify the actual run; the study title identifies its scientific purpose. Earlier launch versions remain available for tracing source evolution, not as additional experiments.
