# SEAM AV2 20-Epoch Controlled Four-Test Suite

This package runs four controlled SEAM experiments on the Argoverse 2 motion-forecasting dataset:

1. `baseline`
2. `uncertainty_target_context`
3. `relative_geometry_bias`
4. `qknorm`

Every run uses the same SEAM data pipeline, optimiser, learning-rate schedule, seed, effective batch size, model width, and 20-epoch budget. Only the named architectural intervention changes. Previous SEAM code, checkpoints, and result directories are never modified.

The Lab 3 launcher performs source/configuration audits, CUDA and real-data smoke tests, then executes the variants in order. Each epoch writes a resumable checkpoint. A completed variant is summarised and published before the next variant begins; large logs and checkpoints remain only on Lab 3.

Suite version 2 classifies optimizer parameters at their owning modules. This preserves the baseline decay groups and prevents a module name containing `bias` from assigning geometry-network weights to both optimizer groups. A failed preflight-only legacy directory remains preserved; the launcher automatically creates a corrected directory when that failed attempt has no training checkpoint.

See `EXPERIMENT_MANIFEST.md` for the controlled design, `SOURCE_PROVENANCE.md` for the source-isolation audit, and `TERMINAL_COMMANDS.md` for the single terminal block to launch or resume the suite.
