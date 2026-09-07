# Repository Guide

## Reading Order

Start with the [results overview](../Results/README.md), select a study, then read
its `Summary.md` and protocol. Follow the source links beside each comparison
to inspect the original evaluation. Implementation details are in the package's
Python modules and `runtime/` or bundled source tree.

## File Conventions

| File or directory | Purpose |
|---|---|
| Study `README.md` | Scientific scope, main finding and navigation |
| Study `Summary.md` | Current evidence-linked comparison, separate from dated snapshots |
| `SETUP.md` | Retained operational setup documentation |
| `EXPERIMENT_MANIFEST*.md`, `protocol.json` | Dataset, architecture and training controls |
| `TERMINAL_COMMANDS*.md` | Host-specific launch and resume instructions |
| `setup_*.py`, `launch_*.sh`, `run_*.sh` | Source preparation and experiment orchestration |
| `preflight_*.py`, `test_*.py` | Runtime checks and regression tests |
| `publish_*.sh`, `publish_*.py` | Curated artifact publication |
| `runtime/` | Study-specific architectural and runtime modules |
| `upstream/`, `sharp_original/` | Bundled model source; consult provenance before modification |
| `Results/`, `Runs/`, `Main_Results/` | Original run records with stable run identifiers |
| `Snapshots/`, `Current_Progress/` | Dated observations, not necessarily the latest final evaluation |
| `metrics.json`, `FINAL_METRICS.json` | Recorded numerical evaluations |
| `CHECKPOINTS.txt`, checkpoint inventories | Checkpoint names and storage locations, not binary weights |
| `Terminal.txt`, `LOG_TAIL.txt` | Recorded or compacted output, with its original capture scope |
| `Documentation/Archive/` | Superseded consolidated documentation retained for provenance |

Descriptive study folders replace machine labels in navigation. Original
timestamped run names, script version suffixes and captured filenames remain
unchanged because manifests, checkpoint records and resume scripts refer to them.
These identifiers are provenance, not additional model variants.

## Evidence Hierarchy

1. Explicit selected-checkpoint evaluation files support a full metric vector.
2. A checkpoint filename supports its recorded selection metric, not all metrics.
3. Validation log vectors describe their own evaluation epoch and precision.
4. A progress snapshot describes only its capture time; a later final evaluation
   takes precedence for the same run.

Curated summaries distinguish these sources. Rounded progress values are not
silently promoted to full-precision measurements. The presence of code or a
successful upload does not establish that training and evaluation finished.

## Historical Paths

[Path_Migration.csv](Repository_Audit/Path_Migration.csv) lists old and new
locations. [Original_File_Inventory.json](Repository_Audit/Original_File_Inventory.json)
records the original SHA-256 digest of every tracked file and identifies records
that must remain byte-identical. Original absolute paths inside logs and
configurations are intentionally retained; they refer to the execution host.

The archived consolidated summaries describe an earlier capture date. Their
superseded status text is not the current result registry. Upstream README links
may refer to assets in the original project rather than assets distributed here.

Run `python Tools/Repository/validate_repository.py` to check the layout,
preserved evidence, maintained Markdown links and Python syntax without training.
