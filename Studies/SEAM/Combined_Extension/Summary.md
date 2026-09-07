# SEAM Combined Extension: Results

Uncertainty, geometry and QKNorm composed with a future-head Mamba module.

## Comparison

No final metric vector is available in the committed evidence.

## Interpretation

This implementation combines all three context/attention mechanisms with future-head Mamba. Accuracy claims require its own completed evaluation; they cannot be inferred from independent ablations.

## Protocol and Evidence

80 epochs configured; global batch 32; two GPUs; seed 2333; AdamW; LR 1e-3 to 1e-5.

The latest committed snapshot records early training and no final selected-checkpoint evaluation. The code is available for reproducibility, but this extension is not included in completed-result rankings. No separate SEAM + QKNorm + uncertainty + geometry run without Mamba is recorded.

- [Original run records](Results/SEAM_AV2_80EPOCH_COMBINED_20260906-144118/)
- [Implementation and reproduction](README.md)
