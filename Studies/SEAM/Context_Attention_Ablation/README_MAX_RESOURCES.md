# SEAM AV2 20-Epoch Four-Test Max-Resource Suite

This profile repeats the same four SEAM interventions and 20-epoch schedule as the controlled suite while using all three Lab 3 GPUs. It writes to independent experiment and result directories and never deletes or overwrites the original batch-32 run.

The resource settings are three DDP ranks, microbatch 8 per GPU, two-step gradient accumulation, and a derived data-worker count based on the available logical CPUs. This produces an effective global batch of 48. The batch-48 results are controlled within this four-run family, but they must not be presented as a direct batch-matched continuation of the earlier batch-32 study.

The launcher performs the same structural, optimizer, CUDA, warning, and real-data smoke tests before training. Every epoch is checkpointed, interrupted variants resume from `last.ckpt`, and each completed variant is summarized and published before the next begins.

Run `stop_current_seam_20epoch_suite.sh` once to preserve and stop the superseded active batch-32 attempt, then run `launch_lab3_seam_20epoch_4test_max_resources.sh` from a new terminal.
