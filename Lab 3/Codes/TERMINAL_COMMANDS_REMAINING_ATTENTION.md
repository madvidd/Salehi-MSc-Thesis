# Lab 3 remaining attention experiments

Use
[`TERMINAL_COMMANDS_LAB3_RECOVERY.md`](TERMINAL_COMMANDS_LAB3_RECOVERY.md)
for the validated restart sequence.

The active sequence explicitly excludes `baseline_mha` and runs:

1. `qknorm`
2. `talking_heads`
3. `qknorm_talking_heads`

A variant advances only after training, evaluation, finite-metric validation,
local archival, branch push, and merge complete successfully. A native failure
is retried once from `last.ckpt` when available. A persistent failure is
archived locally and stops the sequence; failed results are never published as
completed experiments.
