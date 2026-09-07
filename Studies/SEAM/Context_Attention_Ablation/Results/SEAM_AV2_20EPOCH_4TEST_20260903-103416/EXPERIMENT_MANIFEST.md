# Experiment Manifest

## Constant training setup

| Setting | Value |
|---|---:|
| Dataset | Argoverse 2 motion forecasting, processed SEAM data |
| Epochs | 20 |
| GPUs | 2 x RTX 2080 Ti |
| Batch size per GPU | 8 |
| Gradient accumulation | 2 |
| Effective global batch | 32 |
| Optimiser | AdamW |
| Initial learning rate | 0.001 |
| Minimum learning rate | 0.00001 |
| Warm-up ratio | 0.167 |
| Weight decay | 0.01 |
| Gradient clipping | 5.0 |
| Random seed | 2333 |
| Forecast modes | 6 |
| Validation selection metric | minimum minADE6 |

This is the verified previous SEAM baseline recipe with one controlled budget change: all variants stop after 20 epochs instead of 80.

## Variant-specific changes

### 01 Baseline

The previous SEAM baseline architecture is retained unchanged. It provides the within-suite 20-epoch control.

### 02 Uncertainty-aware target context

SEAM already builds target-centric context around the previous observation window's six predicted endpoints. This variant uses the previous window's detached modal probabilities at that existing branch. A learned uncertainty controller adjusts each mode's target-context radius and feature contribution. Ambiguous modes receive broader context, while confident modes retain a tighter context. All controller outputs are initialised to the baseline radius and unit feature gain.

### 03 Relative-geometry attention bias

A learned per-head additive bias is applied to the four scene self-attention blocks after SEAM context streaming. The bias uses relative displacement, distance, and heading difference between current agent and lane tokens. This is the location where tokens share one current-window coordinate frame, so the geometric relations are well-defined and directly condition agent-lane interaction.

### 04 QKNorm

Every SEAM attention kernel is replaced by a topology-compatible QKNorm kernel: temporal agent self-attention, current scene self-attention, context-stream cross-attention, trajectory-relay cross-attention, target-context self-attention, and decoder cross-attention. Queries and keys are normalised per head and use a learnable per-head logit scale. Residual paths, feed-forward networks, depth, masks, and data flow are unchanged.

## Isolation and recovery

- Generated code and results use timestamped directories.
- Existing SEAM and SHARP directories are never removed or overwritten.
- `last.ckpt` is resumed automatically for an incomplete variant.
- `TRAINING_COMPLETE` skips a finished variant on relaunch.
- Checkpoints are saved every epoch.
- Publication excludes checkpoints and full raw logs from GitHub.
- A failed publication preserves local results and stops sequencing before the next variant.
