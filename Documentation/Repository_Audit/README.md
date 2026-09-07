# Repository Organisation Audit

The model-based layout replaces the earlier machine-based navigation. No
training was started, resumed, stopped or re-evaluated during organisation.

## Preservation

- All 661 original tracked files have a retained destination.
- 503 recorded outputs, captured configurations, bundled source files and the
  licence are checked byte-for-byte against their original SHA-256 digests.
- Maintained launchers and publishers have updated repository paths. Recorded
  machine paths and timestamps remain unchanged.
- Superseded consolidated documentation is retained under `Documentation/Archive`.
- Trained weights and large raw logs were not newly uploaded; checkpoint
  inventories continue to identify their original storage locations.

## Checks

The [validator](../../Tools/Repository/validate_repository.py) checks original
file preservation, Python syntax, maintained Markdown links, Markdown table
shape, generated result summaries and the combined SEAM source-hash contract.

At organisation, 49 shell files passed `bash -n`. The existing combined-SEAM
tests passed ten storage/setup tests, with three PyTorch-dependent tests skipped
because the inspection environment did not contain PyTorch. Four existing SHARP
recovery-patch regression tests passed. These are repository checks, not a new
Linux/CUDA training validation.

## Files

- [Original file inventory](Original_File_Inventory.json): original revision,
  content hash, size, destination and preservation rule.
- [Path migration](Path_Migration.csv): searchable old-to-new filename mapping.

Historical snapshots remain historical. The maintained study summaries are the
current navigation layer and distinguish completed evaluations from available
code without final metrics. Licences and upstream attribution are retained.
