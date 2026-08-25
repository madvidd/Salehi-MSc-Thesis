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
- Static audits and real AV2 batches run on four-rank DDP before full training.
- `last.ckpt` is written every epoch. A failed run retries up to three times from the newest loadable checkpoint.
- Rerunning the launcher reuses the active suite, skips completed variants, and retries evaluation or publication without retraining.
- A completed run is evaluated once on one GPU with batch 32, avoiding distributed-validation sample padding.

## Outputs

Each run produces exact JSON/CSV/Markdown metrics, checkpoint inventory, warning report, compact `Terminal.txt`, training curves in PNG and SVG, and a dissertation summary. The suite produces comparison tables and plots. Full logs and checkpoints stay on Lab 2; compact files below 10 MiB are automatically merged and pushed under this package's `Results` directory.

Exact reproduction of a published floating-point result cannot be guaranteed. The paper trained on one RTX 8000, whereas this suite uses four RTX 2080 Ti GPUs. DDP sample order, CUDA kernels, dependency versions, and dataset preprocessing can cause small differences even with identical optimization hyperparameters.
