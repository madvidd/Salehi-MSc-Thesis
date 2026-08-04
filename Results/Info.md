# SHARP Mamba Experiment and Related-Work Notes

This document records two architecture questions:

1. Where and how Mamba was added to SHARP's temporal agent encoder.
2. Which of DeMo, SEAM, and SHARP originally used Mamba.

## 1. Temporal-Agent Mamba Added to SHARP

### Previous scene-token experiment

The earlier Lab 2 experiment applied Mamba after the temporal histories had already been encoded and pooled. Mamba therefore processed the combined scene-token sequence containing agent and lane tokens. The order of those tokens is not inherently chronological.

### Current temporal-agent experiment

The scene-token Mamba was removed completely. The replacement operates inside each agent's temporal-history encoder, before temporal pooling:

```text
Per-agent observations [agent, 10 timesteps, 128 features]
  -> original SHARP temporal self-attention block 1
  -> original SHARP temporal self-attention block 2
  -> bidirectional temporal-agent Mamba
  -> original SHARP temporal self-attention block 3
  -> original SHARP temporal self-attention block 4
  -> temporal pooling
  -> one encoded feature per agent
```

All four original SHARP temporal-attention blocks remain present. The Mamba module is inserted after block 2 by `temporal_mamba_after_block = 2`. It is not applied to pooled agent tokens, map/lane tokens, the scene encoder, or the trajectory decoder.

Implementation: [setup_lab2_temporal_agent_mamba_rotationfix80.py](../Lab%202/Codes/setup_lab2_temporal_agent_mamba_rotationfix80.py)

### Mamba configuration

| Property | Configuration |
|---|---|
| Input sequence | One agent's valid chronological observations |
| Sequence length | Up to 10 observations |
| Model/feature dimension | 128, matching SHARP's embedding dimension |
| Direction | Separate forward and backward Mamba branches |
| `d_state` | 8 |
| `d_conv` | 3 |
| `expand` | 1 |
| Normalization | Pre-Mamba `LayerNorm(128)` |
| Direction fusion | Learned per-channel sigmoid gate |
| Initial direction weights | 0.5 forward and 0.5 backward |
| Dropout | 0.1 |
| LayerScale initialization | 0.01 per channel |
| Residual form | `x + layer_scale * dropout(fused_update)` |
| Agent chunk size | 128 agents; time is never divided into chunks |
| Padding treatment | Valid observations are compacted before recurrence |
| CUDA path | Official fused selective-scan CUDA kernel |
| Convolution path | PyTorch/cuDNN `Conv1d` for runtime stability |
| Mamba fast path | Disabled for the stable fused-selective-scan setup |

The backward branch reverses only the valid observation prefix, scans it, and restores chronological order. Padded observations therefore do not contaminate the recurrent state. Agent chunking limits memory usage without changing the temporal sequence seen by Mamba.

The optimizer construction verifies that every temporal-agent Mamba parameter belongs to the existing SHARP AdamW parameter groups. The setup also rejects a source tree that already contains another Mamba integration, preventing the old and new experiments from being stacked accidentally.

### Controlled training configuration

The temporal-agent run retained the previous controlled SHARP-plus-Mamba training setup:

| Setting | Value |
|---|---:|
| Dataset | Argoverse 2 |
| Seed | 2333 |
| Epochs | 80 |
| GPUs | 4 x RTX 2080 Ti with DDP |
| Batch size | 8 per GPU, 32 global |
| DataLoader workers | 6 per process, 24 total |
| Batch normalization | Synchronized BatchNorm |
| Optimizer | Existing SHARP AdamW configuration |
| Gradient clipping | 5 |
| Learning rate | `1e-4` to `1e-5` |
| Warm-up ratio | 0.167 |
| Checkpoint selection | Three best `minADE6` checkpoints plus `last.ckpt` |

### Why this placement could help

1. An individual agent's observation history is a genuinely ordered sequence, making it a better semantic match for a selective state-space scan than an arbitrarily ordered set of scene tokens.
2. Mamba processes motion before temporal pooling removes timestep-level detail, allowing it to model velocity changes, acceleration, braking, turning, and stop/start behaviour.
3. The middle insertion lets two original attention blocks construct temporal features, Mamba integrate them across time, and the remaining two attention blocks refine the updated representation.
4. Bidirectional processing is permissible because the complete observed history is available before forecasting; it does not inspect the future ground-truth trajectory.
5. The residual connection and small `0.01` LayerScale make the initial network close to original SHARP, reducing the risk of destabilizing training.
6. `d_state=8` and `expand=1` deliberately limit the added capacity and overfitting risk.

This was a testable hypothesis, not a guarantee. The temporal-agent placement improved best `minADE6` from **0.680362** for scene-token Mamba to **0.673736**, a reduction of **0.006626** (approximately **0.97%**). It nevertheless remained worse than the recorded unmodified SHARP results, including Lab 2 `0.661463` and Lab 1 `0.650232`. A likely explanation is that only ten observations are present and SHARP's four original temporal-attention blocks already model this short sequence effectively; the extra module may require dedicated hyperparameter tuning.

## 2. Mamba in DeMo, SEAM, and SHARP

Among the three original articles, **only DeMo uses Mamba as an architectural component**.

| Article | Original Mamba use | Principal mechanism |
|---|---|---|
| DeMo | Yes | Unidirectional and bidirectional Mamba combined with attention |
| SEAM | No | Transformer attention with endpoint-aware streaming |
| SHARP | No | Transformer attention with instance-aware short-window streaming |

### DeMo

DeMo is an Attention-Mamba hybrid and uses Mamba in three places.

#### Historical agent encoder

```text
Agent history -> MLP embedding -> unidirectional Mamba stack -> agent feature
```

The released Argoverse 2 code uses four unidirectional Mamba blocks at feature dimension 128, with RMSNorm, FP32 residual accumulation, fused add-normalization, and drop-path 0.2. The map is encoded separately with PointNet. Agent and map features are concatenated and processed by a Transformer scene encoder.

#### State Consistency Module

```text
Future-time state queries
  -> cross-attention with scene context
  -> two bidirectional Mamba blocks across future states
```

For Argoverse 2, DeMo represents the future with 60 state queries. Bidirectional scanning is used because these are simultaneously available latent future-state queries, rather than causal observed-history samples.

#### Hybrid Coupling Module

```text
Mode queries + state queries
  -> scene/mode/hybrid attention
  -> two bidirectional Mamba blocks
  -> multimodal trajectory outputs
```

Attention models global scene interactions and interactions among motion modes. Mamba models ordered state sequences and promotes consistency across future timestamps. DeMo's ablation reports that bidirectional Mamba performed better than attention, unidirectional Mamba, Conv1D, and GRU for future-state sequence modelling.

DeMo does not explicitly override the lower-level `d_state`, `d_conv`, or `expand` values in its block factory; those values follow the installed `mamba_ssm` implementation defaults. They should therefore not be assumed to equal the custom Lab 2 values above.

Paper: [DeMo: Decoupling Motion Forecasting into Directional Intentions and Dynamic States](https://papers.nips.cc/paper_files/paper/2024/file/c0ff9e52e94ae331bc0f2d28be06a9ca-Paper-Conference.pdf)

### SEAM

SEAM contains no Mamba layer. Its central contribution is endpoint-aware streaming:

- previous predicted trajectory endpoints become anchors;
- target-centric context is encoded around those anchors;
- agent-centric and target-centric scene features are processed by a dual-context attention decoder;
- information from the preceding prediction is propagated to the current streaming step.

Mamba appears in SEAM's bibliography as related work, not as part of the model.

Paper: [Streaming Real-Time Trajectory Prediction Using Endpoint-Aware Modeling](https://openaccess.thecvf.com/content/WACV2026/papers/Prutsch_Streaming_Real-Time_Trajectory_Prediction_Using_Endpoint-Aware_Modeling_WACV_2026_paper.pdf)

### SHARP

Original SHARP also contains no Mamba layer. It uses a Transformer-based architecture with:

- incrementally processed short observation windows;
- instance-aware context streaming between windows;
- latent feature propagation for matched agents;
- streaming and non-streaming training passes;
- a dual objective intended to maintain accuracy across different observation lengths.

Mamba-related approaches are cited by SHARP, but the original paper and released model do not integrate Mamba.

Paper: [SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting](https://openaccess.thecvf.com/content/CVPR2026/html/Prutsch_SHARP_Short-Window_Streaming_for_Accurate_and_Robust_Prediction_in_Motion_CVPR_2026_paper.html)

### Relationship between DeMo and the Lab 2 modification

The Lab 2 experiment is conceptually related to DeMo's historical-agent encoder because both apply Mamba to a meaningful per-agent temporal sequence. They are not the same architecture:

```text
DeMo:
MLP -> four unidirectional Mamba blocks -> agent feature

Modified SHARP:
attention 1 -> attention 2 -> one custom bidirectional Mamba module
-> attention 3 -> attention 4 -> temporal pooling
```

DeMo additionally uses bidirectional Mamba in two decoder modules. The modified SHARP decoder remains unchanged. Therefore, the Lab 2 experiment is a conservative Mamba augmentation of SHARP, not a reproduction of DeMo.
