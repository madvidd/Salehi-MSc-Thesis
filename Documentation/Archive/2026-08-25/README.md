# Thesis Results Index

Updated: 2026-08-25

This directory is the consolidated index for the experiments performed on Lab 1, Lab 2, and Lab 3. It distinguishes completed results from partial, failed, and prepared runs so that provisional values are not presented as final results.

## Documents

- [All experiment runs](Main/All_Run_Registry.md): cross-lab registry of every substantive model configuration and its current status.
- [SHARP metrics](Main/SHARP_AV2_Main_Results.md): consolidated AV2, AV1, and nuScenes metric tables, the completed 20-epoch screening study, and training-time context.
- [Modification glossary](Info.md): what each architectural modification changes, why it was expected to help, and whether the recorded experiment supported that hypothesis.

## Status Definitions

| Status | Meaning |
|---|---|
| Completed | Training reached its configured endpoint and a retained validation result is available. |
| Partial | Checkpoints or validation metrics exist, but the configured run has not completed. |
| Failed | The attempted configuration stopped without a comparable final result. |
| Prepared | Reproducible code exists, but no result has been committed yet. |
| Grouped setup attempts | Multiple operational retries of the same intended experiment; these are recorded without treating each retry as a scientific ablation. |

## Current High-Level Findings

- The closest local SHARP AV2 reproduction is the Lab 1 A4000 run with `minADE6=0.650232`, compared with `0.64` reported by the SHARP article.
- In the controlled Lab 3 attention study, QKNorm improved best `minADE6` from `0.673460` to `0.669777`. Talking-Heads and the combined variant did not improve best `minADE6`.
- In the Lab 2 20-epoch screening study, relative-geometry attention bias and uncertainty-aware target context produced small improvements over the 20-epoch baseline. The other seven modifications were neutral or worse on `minADE6` at that training horizon.
- The temporal-agent residual Mamba placement was better than the earlier scene-token Mamba placement, but neither beat unmodified SHARP in the completed long runs.
- The Lab 3 SEAM baseline has provisional metrics only. Its two Mamba variants have not produced committed results yet.
- The final three-run SHARP package is prepared but has no committed run metrics yet.

All forecasting metrics in these documents are lower-is-better. Cross-lab comparisons are descriptive unless the dataset, model, epoch schedule, batch construction, seed, and hardware adaptation are all controlled.
