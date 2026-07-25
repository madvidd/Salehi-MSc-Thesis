# Lab 2: Controlled SHARP Temporal-Agent Mamba Ablation

This package creates two new, isolated AV2 experiments on Lab 2:

1. `baseline_control`: original SHARP with no Mamba parameters.
2. `temporal_agent_mamba`: the same SHARP model with one gated bidirectional
   Mamba block placed on each agent's chronological history between temporal
   attention blocks 2 and 3.

The setup copies `/home/server00/M/Codes/SHARP/Code` into two new timestamped
directories. It never edits or deletes the original source, completed Mamba
experiment, checkpoints, logs, or previous results.

## Controlled settings

Both experiments use:

- AV2 processed train and validation sets already on Lab 2
- seed `2333`
- 80 epochs
- four RTX 2080 Ti GPUs with DDP
- batch size 8 per GPU, global batch size 32
- six DataLoader workers per process
- SyncBatchNorm enabled
- AdamW and gradient clipping 5
- learning rate `1e-4`, minimum learning rate `1e-5`
- warm-up ratio `0.167`
- top three checkpoints monitored by `minADE6`, plus `last.ckpt`
- validation from Lightning's best checkpoint

## Temporal-agent Mamba design

The temporal module scans a real sequence: the ten chronological observations
of one agent. It does not scan the unordered scene-token set.

- Original four temporal self-attention blocks remain present.
- Mamba is inserted after block 2 and before block 3.
- Forward and backward directions use separate Mamba modules.
- Valid observations are compacted chronologically before each scan.
- Padding never precedes valid observations in either scan direction.
- Forward/backward features use a learned per-channel gate.
- A per-channel LayerScale initialized to `0.01` keeps the model close to
  baseline at initialization.
- Initial configuration: `d_state=8`, `d_conv=3`, `expand=1`, dropout `0.1`.

The setup reuses the verified fused CUDA package directory from the completed
Lab 2 scene-Mamba run through a symbolic link. The dependency files are not
modified.

## Running

Use the exact commands in `TERMINAL_COMMANDS.md`. Run only one experiment at a
time because each runner reserves all four GPUs.

For the strongest comparison, run the baseline first and then the temporal
variant. The sequential runner does this automatically, but it will occupy the
terminal until both 80-epoch experiments finish.
