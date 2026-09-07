# Experiment Registry

This catalogue is organised by scientific study. Counts refer to evaluated configurations represented in the linked records, not setup attempts, snapshots or the number of scripts. No result is inferred from an implementation alone.

| Study | Evaluated configurations | Evidence |
|---|---:|---|
| SEAM State-Space Integration | 3 | [Summary](../../Studies/SEAM/State_Space_Integration/Summary.md) |
| SEAM Context and Attention Ablation | 4 | [Summary](../../Studies/SEAM/Context_Attention_Ablation/Summary.md) |
| SEAM Combined Extension | 0 | [Summary](../../Studies/SEAM/Combined_Extension/Summary.md) |
| SHARP Attention Operators | 4 | [Summary](../../Studies/SHARP/Attention_Operators/Summary.md) |
| SHARP Architectural Ablation | 10 | [Summary](../../Studies/SHARP/Architecture_Ablation/Summary.md) |
| SHARP State-Space Integration | 2 | [Summary](../../Studies/SHARP/State_Space_Integration/Summary.md) |
| SHARP Architecture Integration | 2 | [Summary](../../Studies/SHARP/Architecture_Integration/Summary.md) |

## Separate Historical Records

Earlier SHARP AV1, nuScenes and shorter baseline summaries are retained in the [historical consolidated report](../../Documentation/Archive/2026-08-25/Main/SHARP_AV2_Main_Results.md). Its capture date and split/mode differences remain part of the record. They are not inserted as controls into the current matched AV2 studies.

The earlier single-GPU work is held separately in [Thesis_A4000](https://github.com/madvidd/Thesis_A4000); this repository does not contain all of that external archive. No cross-dataset or cross-budget ranking is constructed.

## Snapshot Precedence

SEAM state-space integration has three completed variants; SEAM context/attention ablation has four; SHARP attention has four; the SHARP architectural ablation has ten. The SHARP integration suite has two committed final evaluations, superseding its older progress snapshot. Its residual-Mamba extension and the SEAM combined extension have no committed final evaluation in this catalogue.

The [file inventory](../../Documentation/Repository_Audit/Original_File_Inventory.json) and [migration map](../../Documentation/Repository_Audit/Path_Migration.csv) retain the original path and content identity of every pre-organisation file.
