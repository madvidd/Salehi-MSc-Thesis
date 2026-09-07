# SEAM Combined Extension

[Study summary and source evidence](../../../../Studies/SEAM/Combined_Extension/Summary.md).

This implementation combines all three context/attention mechanisms with future-head Mamba. Accuracy claims require its own completed evaluation; they cannot be inferred from independent ablations.

The latest committed snapshot records early training and no final selected-checkpoint evaluation. The code is available for reproducibility, but this extension is not included in completed-result rankings. No separate SEAM + QKNorm + uncertainty + geometry run without Mamba is recorded.
