# SEAM AV2 Article and Released-Code Audit

This package uses the official single-agent streaming SEAM implementation supplied
in `seam-main.zip`. It does not use the multi-agent extension.

## Controlled Baseline

| Setting | Value |
|---|---:|
| Dataset | Argoverse 2 motion forecasting |
| Observation / prediction | 3 s / 6 s |
| Streaming passes | t = 3, 4, 5 s |
| Agent-history samples | 30 |
| Auxiliary output horizon | 80 steps total |
| Embedding dimension | 128 |
| Agent temporal MHA blocks | 4 |
| Scene MHA blocks | 4 |
| Target-context blocks | 2 |
| Decoder cross-attention stages | 3 |
| Attention heads | 8 |
| Drop path | 0.2 |
| Epochs | 80 |
| Effective global batch | 32 |
| Optimizer | AdamW |
| Minimum LR | 1e-5 |
| Warm-up | 13 epochs |
| Weight decay | 1e-2 |
| Gradient clipping | norm 5 |
| Seed | 2333 |

The paper prints a peak learning rate of `1e-2`. The released `Seam.yaml` and the
released checkpoint scheduler state both use `1e-3`. This package uses `1e-3` so
the baseline reproduces the released implementation rather than the apparent
paper typo.

The article trained with batch 32 on one Quadro RTX 8000. Lab 3 has three 11 GB
RTX 2080 Ti cards, and 32 cannot be divided evenly over three DDP ranks. The
runner therefore uses two GPUs, microbatch 8 per GPU, and two-step gradient
accumulation. The effective optimizer batch remains exactly 32. Precision remains
32-bit. Because SEAM contains lane-embedding BatchNorm, SyncBatchNorm sees 16
samples per forward pass rather than the RTX 8000 run's 32; gradient accumulation
cannot reproduce BatchNorm activation statistics. This unavoidable 11 GB hardware
adaptation is disclosed rather than being presented as bitwise reproduction.

## Three Variants

1. `baseline`: unchanged official SEAM architecture.
2. `mamba_agent_add`: adds one gated residual Mamba block after all four temporal
   MHA blocks and before max pooling across the ordered 30-step agent history.
3. `mamba_future_replace`: leaves the encoder untouched and replaces only the
   decoder localization MLP with a two-block Mamba head over the ordered 80-step
   future sequence.

Both Mamba configurations use `d_model=128`, `d_state=16`, `d_conv=4`,
`expand=2`, and the fused CUDA fast path. The future replacement uses two blocks.

These are controlled hypotheses, not guaranteed improvements. Mamba is applied
only where token order is meaningful. SEAM's context streaming, target-centric
context, scene attention, decoder cross-attention, and trajectory relay are
preserved.

## Released AV2 Checkpoint Reference

| MR | minADE1 | minADE6 | minFDE1 | minFDE6 | b-minFDE6 |
|---:|---:|---:|---:|---:|---:|
| 0.153 | 1.598 | 0.663 | 3.962 | 1.249 | 1.848 |
