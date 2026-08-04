# Lab 2 SHARP 20-Epoch Ten-Test Suite

This package prepares and runs one baseline plus nine isolated architecture/loss ablations on AV2. See `EXPERIMENT_MANIFEST.md` for the exact controls and variant definitions.

## Files

- `setup_lab2_sharp_20epoch_10test.py`: copies clean SHARP source into a new timestamped experiment and applies controlled patches.
- `run_10_test_suite.sh`: generated into the experiment directory with absolute Lab 2 paths.
- `preflight_suite.py`: verifies four GPUs, all ten model variants, optimizer coverage, and fused CUDA Mamba before training.
- `summarize_variant.py`: records each completed variant's best checkpoint and final validation metrics.
- `suite_status.py`: reports completed, resumable, started, and pending variants.
- `TERMINAL_COMMANDS.md`: commands to preserve the current run, update from GitHub, set up, run, resume, and inspect status.

The setup refuses to modify a SHARP source tree that already contains Mamba. It creates new code and result directories and leaves all earlier experiments intact.
