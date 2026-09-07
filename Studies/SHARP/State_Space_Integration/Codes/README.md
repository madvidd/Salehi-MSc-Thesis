# Lab 2: SHARP Temporal-Agent Mamba

This package creates one new, isolated AV2 experiment:

- Original SHARP architecture and training process from the completed Lab 2
  scene-Mamba run.
- The previous scene-token Mamba module is absent.
- One bidirectional Mamba block is added to the temporal agent-history encoder.

The setup copies `/home/server00/M/Codes/SHARP/Code` into a new timestamped
experiment directory. It does not edit or delete the original SHARP source,
previous Mamba code, checkpoints, logs, or results.

## Controlled training settings

The new run uses the same controllable settings as the completed Lab 2
scene-Mamba run:

- AV2 processed train and validation sets already on Lab 2
- seed `2333`
- 80 epochs
- four RTX 2080 Ti GPUs using DDP
- batch size 8 per GPU, global batch size 32
- six DataLoader workers per process
- SyncBatchNorm enabled
- the existing SHARP AdamW optimizer and gradient clipping of 5
- learning rate `1e-4`, minimum learning rate `1e-5`
- warm-up ratio `0.167`
- top three checkpoints monitored by `minADE6`, plus `last.ckpt`
- final validation from Lightning's best checkpoint

These settings provide a direct comparison with the completed Lab 2
scene-Mamba experiment. They align the key AV2 schedule and global batch with
the SHARP experiment, while the hardware execution remains four-GPU DDP rather
than the paper's single-GPU hardware.

## Only architecture difference

The original SHARP agent-history encoder contains four temporal self-attention
blocks. All four remain unchanged. The new Mamba module is inserted after
block 2 and before block 3:

`temporal attention 1 -> temporal attention 2 -> temporal-agent Mamba -> temporal attention 3 -> temporal attention 4`

For every agent, Mamba scans the ten chronological observation embeddings.
Valid observations are compacted before the scan so padding is not treated as
history. The module uses:

- separate forward and backward Mamba directions
- `d_state=8`
- `d_conv=3`
- `expand=1`
- dropout `0.1`
- learned forward/backward fusion
- per-channel LayerScale initialized to `0.01`
- a residual connection back to the unmodified SHARP features

The stable-CUDA runner uses PyTorch/cuDNN `Conv1d` on the GPU and keeps the
official fused CUDA selective-scan kernel. It verifies that the saved best
checkpoint contains temporal-agent Mamba parameters and no other Mamba module.

## Running

Use `TERMINAL_COMMANDS_TEMPORAL_AGENT_STABLECUDA80.md`. The generated runner reserves all four GPUs and
writes to a new timestamped results directory. Run the launcher with `bash`,
not `source`, so it executes in the foreground as a child process and always
returns control to the same terminal.

## Stable-CUDA correction after the batch-122 failure

The fully custom fused temporal-agent variants are retained for diagnosis, but
must not be resumed. Both eventually produced an illegal CUDA memory access.
The current `stablecuda80` experiment leaves the architecture, parameters and
80-epoch training settings unchanged. It disables only Mamba's optional custom
`causal_conv1d_cuda` function, so the short convolution uses CUDA/cuDNN through
`nn.Conv1d`; the fused CUDA selective scan remains enabled.

Before the full run, the current launcher performs 256 real AV2 training
batches with the exact four-GPU DDP and SyncBatchNorm path under synchronous
CUDA error reporting. This deliberately runs beyond the earlier batch-122
failure. Use `TERMINAL_COMMANDS_TEMPORAL_AGENT_STABLECUDA80.md` for the current
Lab 2 command and one-time terminal snapshot.
