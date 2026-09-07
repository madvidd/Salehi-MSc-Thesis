# SHARP Attention Operators

Multi-head attention, query-key normalisation, head mixing and their combination.

## Recorded Finding

QKNorm improves selected minADE6 from 0.673460 to 0.669777 (-0.55%). Talking-Heads alone gives +0.23%, and QKNorm with head mixing gives +0.15%. The combined operator has the lowest MR in the retained vectors, but not the lowest trajectory-average error.

The MHA vector is final-epoch validation, whereas the alternatives have retained selected-checkpoint evaluations. The primary comparison therefore uses the saved selected minADE6 for every operator; do not attribute all secondary-metric changes to a fully matched checkpoint evaluation.

## Navigation

- [Results and interpretation](Summary.md)
- [Original result records](Main_Results/Attention_Experiments/)
- [Setup and operational details](SETUP.md)
- [Preparation or launch entry point](setup_lab3_attention_ablation.py)
- [Artifact publisher](Codes/publish_lab3_attention_snapshot.sh)
- [Modification definitions](../../../Results/Info.md)
- [Repository reproduction guide](../../../Documentation/Reproduction.md)

## Shared Controls

80 epochs; global batch 24; three GPUs; seed 2333; AdamW; LR 1e-4 to 1e-5.

Numerical run records and timestamped snapshots are preserved as captured. Historical filenames identify the actual run; the study title identifies its scientific purpose. Earlier launch versions remain available for tracing source evolution, not as additional experiments.
