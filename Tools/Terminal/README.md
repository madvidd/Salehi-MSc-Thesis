# Log Capture and Checkpoint Inspection

[Capture_Training_Log.txt](Capture_Training_Log.txt) and
[Inspect_Best_Checkpoint.txt](Inspect_Best_Checkpoint.txt) preserve the original
host-specific command examples. They are reference snippets, not universal
commands for every study. The capture example writes a one-time destination
snapshot and can overwrite that destination; use a new destination to preserve
an earlier copy.

For current studies, use the corresponding `publish_*.sh` or `publish_*.py`
script. Read its arguments and size limits first. Logs can contain local paths
and configuration values, so publication requires both a credential scan and a
file-size check. A checkpoint inventory is not a backup of checkpoint weights.
