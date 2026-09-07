# Reproduction and Artifact Publication

## Scope

Each study keeps its own protocol, preprocessing assumptions and recorded
environment. It is not a single interchangeable environment: SHARP and SEAM,
and their older state-space variants, use different pinned dependencies. Read
the study protocol before installing packages or comparing measurements.

The retained host paths are `/home/server00/M` for four-GPU SHARP experiments
and `/home/server01/M` for SEAM and the three-GPU attention experiment. Paths
can be adapted for another machine, but changes to batch construction,
normalisation or data processing must be reported with the reproduced results.

## Entry Points

Run these scripts from their study directory, after checking the linked
instructions. They are not commands to start all experiments simultaneously.

| Study | Launcher or instructions |
|---|---|
| SEAM state-space integration | [launch_lab3_seam_3run.sh](../Studies/SEAM/State_Space_Integration/launch_lab3_seam_3run.sh) |
| SEAM context and attention ablation | [TERMINAL_COMMANDS_MAX_RESOURCES.md](../Studies/SEAM/Context_Attention_Ablation/TERMINAL_COMMANDS_MAX_RESOURCES.md) |
| SEAM combined extension | [TERMINAL_COMMANDS.md](../Studies/SEAM/Combined_Extension/TERMINAL_COMMANDS.md) |
| SHARP attention operators | [SETUP.md](../Studies/SHARP/Attention_Operators/SETUP.md) |
| SHARP architectural ablation | [TERMINAL_COMMANDS_DRIVER_SAFE.md](../Studies/SHARP/Architecture_Ablation/TERMINAL_COMMANDS_DRIVER_SAFE.md) |
| SHARP state-space integration | [SETUP.md](../Studies/SHARP/State_Space_Integration/SETUP.md) |
| SHARP architecture integration | [launch_final_3run_suite.sh](../Studies/SHARP/Architecture_Integration/launch_final_3run_suite.sh) |

Launchers create or resume timestamped experiment directories. Do not delete
those directories, completion markers or checkpoint files to restart a
publisher. Epoch checkpoints retain optimiser and scheduler state; work after
the most recent valid checkpoint may need to be repeated following interruption.
No preflight can guarantee that a long GPU job will never fail.

## Updating an Existing Host Checkout

The repository was reorganised without moving data on either execution host.
Pulling the new layout does not update orchestration scripts already copied into
an active experiment. Before later publication, update the repository-destination
paths in that experiment's copied publisher using the migration map. Some
publishers require a generated `suite.env` beside the script and cannot be run
directly from a fresh source checkout. Preserve that environment file and follow
the package's argument contract. Do not replace trained model code, checkpoint
paths or run configuration merely to update repository navigation.

Older copied publishers can still target the former machine-based directories.
Use the [migration map](Repository_Audit/Path_Migration.csv) when updating those
paths. A fresh checkout is preferable to moving directories inside a running
training checkout. Existing historical snapshots remain readable in Git history.

## Data and Storage

Obtain Argoverse data through its official distribution and follow the selected
model's preprocessing instructions. No dataset licence is superseded by the
repository licence. Full checkpoints, processed tensors and complete large logs
remain on the execution hosts; GitHub contains bounded transcripts, numerical
results, configuration snapshots, manifests and checkpoint inventories.

Never commit `Token.txt`, passwords, access tokens or credential-bearing URLs.
Use a local token file or the host credential manager for authenticated Git
operations. The repository contains no login credentials or portable guarantee
of access to the original machines.

## Review Checks

The repository validator and CPU regression tests check organisation, source
contracts and storage logic. CUDA, NCCL, real-data forward/backward tests and
long-run behaviour require the documented Linux GPU environment. A local syntax
check is not a replacement for those runtime checks.
