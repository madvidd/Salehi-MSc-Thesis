# Final SHARP AV2 Three-Run Suite

This package creates a new, isolated Lab 2 experiment and leaves every previous code and result directory unchanged.

## Sequence

1. `01_official_sharp_baseline`: official SHARP architecture at commit `f6bf2fc0109f9838cdc24bfb763b5c3e6847c2ae`, trained with the AV2 schedule stated in the paper supplement.
2. `02_qknorm_uncertainty_geometry`: run 1 plus QKNorm in every SHARP attention constructor, uncertainty-aware streamed target context, and learned relative-geometry scene-attention bias.
3. `03_qknorm_uncertainty_geometry_temporal_mamba`: run 2 plus a small residual bidirectional Mamba between temporal-agent encoder blocks 2 and 3.

All three runs use seed 2333, 80 epochs, 13 warm-up epochs, global batch 32, AdamW, LR `1e-4` to `1e-5`, weight decay `1e-2`, FP32, gradient clipping norm 5, and the same AV2 preprocessing/model dimensions. Four GPUs use batch 8 per rank without LR scaling.

## Reproducibility And Recovery

- Reviewed official files are SHA-256 checked before patching.
- Each variant receives an independent copy of the pinned source.
- A bounded NCCL probe validates broadcast, all-reduce, and synchronization on all four GPUs with `NCCL_P2P_DISABLE=1`. Native P2P passed the short collective probe but failed during the real Lab 2 workload, so the recorded shared-memory transport is used without changing GPU count or training hyperparameters.
- A real AV2 batch of 8 samples runs through forward, loss, backward, and optimizer steps for each variant on one GPU. Separating model/data validation from the four-rank collective probe avoids `fast_dev_run` rank skew while preserving the production DDP configuration.
- Before full training, every variant must also complete 256 real AV2 optimizer steps with four-GPU DDP, SyncBatchNorm, global batch 32, synchronous CUDA error reporting, and the same optimizer schedule. Checkpoints and validation are disabled only for this bounded diagnostic.
- Every preflight subprocess has a timeout and isolated process group, so a failed check cannot remain for 30 minutes or leave stale ranks behind.
- TQDM provides terminal-safe progress reporting for both redirected preflight logs and foreground training; the Rich live-console callback is intentionally excluded because it can corrupt its internal stack under redirected `fast_dev_run` output.
- Streamed validation losses and metrics declare the actual per-rank scenario count explicitly, preventing Lightning from ambiguously inferring batch size from SHARP's nested window collection.
- AdamW grouping inspects each module's direct parameters once. This preserves SHARP's decay/no-decay rules while preventing composed module names such as `relative_geometry_bias` from placing a weight in both groups; preflight requires complete optimizer coverage.
- `last.ckpt` is written every epoch. A failed run retries up to three times from the newest loadable checkpoint.
- Rerunning the launcher reuses the active suite, skips completed variants, and retries evaluation or publication without retraining.
- A completed run is evaluated once on one GPU with batch 32, avoiding distributed-validation sample padding.

## Outputs

Each run produces exact JSON/CSV/Markdown metrics, checkpoint inventory, warning report, compact `Terminal.txt`, training curves in PNG and SVG, and a dissertation summary. The suite produces comparison tables and plots. Full logs and checkpoints stay on Lab 2; compact files below 10 MiB are automatically merged and pushed under this package's `Results` directory.

Exact reproduction of a published floating-point result cannot be guaranteed. The paper trained on one RTX 8000, whereas this suite uses four RTX 2080 Ti GPUs. DDP sample order, CUDA kernels, dependency versions, and dataset preprocessing can cause small differences even with identical optimization hyperparameters.
