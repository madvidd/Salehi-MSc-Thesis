# SEAM AV2 20-Epoch Max-Resource Experiment Manifest

## Variants

1. Unmodified SEAM baseline.
2. Baseline with uncertainty-aware target context.
3. Baseline with relative-geometry attention bias.
4. Baseline with QKNorm attention.

## Shared Training Protocol

- Dataset: the same read-only processed Argoverse 2 train and validation tensors.
- Epochs: 20.
- Seed: 2333.
- Optimizer: AdamW.
- Peak and minimum learning rates: 1e-3 and 1e-5.
- Warm-up ratio: 0.167.
- Weight decay: 1e-2.
- Gradient clipping: norm 5.
- Precision: 32-bit.
- Hardware: three RTX 2080 Ti GPUs under DDP with SyncBatchNorm.
- Microbatch: 8 examples per GPU.
- Gradient accumulation: 2.
- Effective global batch: 48.
- Data workers: derived from `nproc`, with a stable range of 4 to 12 workers per rank.

Only the named architectural intervention changes among the four runs. This max-resource family is internally controlled. Its batch size differs from the earlier batch-32 family and therefore its metrics and runtime must be labelled separately.
