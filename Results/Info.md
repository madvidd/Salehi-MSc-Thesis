# Architecture Modification Glossary and Outcomes

Updated: 2026-08-25

This document explains what each tested modification changes, why it was a plausible accuracy improvement, and what the retained experiment showed. Lower metric values are better. A plausible mechanism is a hypothesis, not evidence of improvement; the outcome column records the evidence.

## SHARP Features Preserved by the Ablations

Original SHARP uses Transformer attention and does not contain Mamba. Its central design is preserved unless a row explicitly says otherwise:

- short observation windows are processed incrementally;
- matched agent instances propagate latent context between windows;
- streaming and non-streaming passes share the forecasting model;
- previous predictions and target-centric context are used by later windows;
- multimodal trajectories are trained with SHARP's original losses.

The controlled ablations change a local operator or auxiliary pathway rather than removing SHARP's short-window, instance-aware streaming mechanism.

## Lab 3 Attention Operators

The Lab 3 suite replaced all 18 `MultiheadAttention` constructors in the single-agent SHARP model while preserving embedding dimension, block counts, eight heads, MLPs, positional encodings, streaming memory, decoder, losses, optimizer, data, and seed.

| Operator | What changes | Why it might improve accuracy | Recorded outcome |
|---|---|---|---|
| Baseline MHA | Standard scaled dot-product multi-head attention. Query-key dot products produce logits, softmax produces weights, and each head aggregates values. | It is the original SHARP control and provides unrestricted content-based interaction. | Best minADE6 `0.673460`. |
| QKNorm | L2-normalizes each projected query and key head before their dot product and learns a scale per head. | It prevents query/key magnitude from arbitrarily sharpening attention, controls logit scale, and can make optimization and head specialization more stable. | Best minADE6 `0.669777`, an improvement of `0.003683` or `0.55%`. b-minFDE6 and minFDE6 also improved in the retained vector; MR was slightly worse. |
| Talking-Heads | Applies identity-initialized learned mixing across attention heads before softmax and again after softmax. | Heads can exchange information instead of remaining independent, potentially coordinating complementary spatial and temporal patterns. | Best minADE6 `0.674991`, `0.001531` or `0.23%` worse than local MHA. It did not lead any listed metric. |
| QKNorm + Talking-Heads | Combines normalized/scaled query-key logits with learned head mixing. | QKNorm could stabilize logits while head mixing increases interaction capacity. | Best minADE6 `0.674492`, `0.001032` or `0.15%` worse than local MHA. It produced the best MR (`0.152064`) but did not improve displacement error. |

The QKNorm result supports carrying QKNorm into a longer controlled SHARP combination run. The Talking-Heads results do not support adding its extra mixing solely to improve minADE6.

Evidence: [Lab 3 completed attention summary](../Lab%203/Main_Results/Attention_Experiments/Runs/SHARP_ATTENTION_ABLATION_20260717-123916/Summary.md).

## Lab 2 20-Epoch Screening Modifications

The ten-test suite independently compared one baseline with nine modifications. All used the same 20-epoch AV2 training controls. Deltas below are variant minADE6 minus baseline `0.752339`; negative is better.

| Modification | What changes | Why it might help | Outcome |
|---|---|---|---|
| Confidence-gated memory | Learns a reliability gate between current tokens and instance-aware streamed updates. | Noisy or stale memory could be suppressed while reliable history is retained. | `0.755349`, delta `+0.003010`: worse. |
| Cross-window consistency | Adds a `0.05`-weight Smooth L1 term between confidence-weighted current predictions and transformed previous predictions. | Adjacent streaming windows should describe a physically consistent future and fluctuate less. | `0.776207`, delta `+0.023868`: worse. The fixed auxiliary weight likely constrained early learning too strongly. |
| Learned temporal pooling | Replaces temporal max pooling with masked learned attention pooling. | The model can weight informative timesteps instead of selecting each feature dimension independently by a hard maximum. | `0.757637`, delta `+0.005298`: worse. |
| Uncertainty-aware target context | Uses previous mode probabilities to adapt endpoint-centric context radius and gate target features. | Ambiguous prior predictions should use broader or more cautious context, while confident predictions can focus on a smaller relevant region. | `0.749961`, delta `-0.002378`: improved. |
| Relative-geometry attention bias | Adds learned per-head relative position and heading biases to the four scene-attention blocks. | Motion interactions depend strongly on relative distance, bearing, and orientation; an explicit geometric prior reduces how much must be inferred from content embeddings alone. | `0.749679`, delta `-0.002660`: best screen. It also improved MR, b-minFDE6, minFDE1, and minFDE6. |
| Kinematic motion stem | Adds masked velocity, acceleration, and speed-change embeddings before temporal encoding. | Explicit derivatives can expose braking, turning, and acceleration patterns that raw positions make the network learn implicitly. | `0.785164`, delta `+0.032825`: worst screen. Derivative noise or redundant features likely hurt this short schedule. |
| Endpoint refinement decoder | Adds endpoint-conditioned trajectory/logit refinement and a `0.2`-weight auxiliary coarse-prediction loss. | Correcting endpoints directly could reduce final displacement and improve mode separation. | `0.755898`, delta `+0.003559`: worse minADE6, but minFDE1 improved from `4.672195` to `4.641530`. |
| Lane topology graph | Adds sparse, geometry-derived lane-neighbor message passing before scene encoding. | Explicit lane connectivity can improve route continuity and legal maneuver reasoning. | `0.754238`, delta `+0.001899`: slightly worse. |
| Agent temporal Mamba replacement | Replaces all four temporal agent-history attention blocks with four unidirectional Mamba blocks (`d_state=16`, `d_conv=4`, `expand=2`) and keeps max pooling. | A selective state-space model could efficiently accumulate chronological motion state and regularize the short history. | `0.780190`, delta `+0.027851`: worse. Complete replacement removed useful attention behavior and was not supported by the screen. |

These 20-epoch results identify candidates, not final 80-epoch conclusions. Relative geometry and uncertainty-aware context were selected for the prepared final suite because they improved under identical screening controls.

## Mamba Placements Tested in SHARP

### Scene-Token Mamba Addition

The first Lab 2 Mamba experiment ran after temporal histories had already been encoded and pooled. Mamba processed the combined scene-token sequence containing agent and lane tokens. That token order is not inherently chronological, so the state-space scan had a weak sequence interpretation.

Result: best minADE6 `0.680362` at epoch 78 of 80. This was worse than unmodified Lab 2 SHARP (`0.661463`).

### Residual Temporal-Agent Mamba Addition

The scene-token Mamba was removed. A bidirectional residual Mamba module was inserted inside each agent's chronological history encoder:

```text
10 observed timesteps
  -> SHARP temporal attention 1
  -> SHARP temporal attention 2
  -> bidirectional residual Mamba
  -> SHARP temporal attention 3
  -> SHARP temporal attention 4
  -> temporal pooling
```

All four original attention blocks remain. Mamba is not applied to lane tokens, scene encoding, streaming memory, or the decoder.

| Property | Configuration |
|---|---|
| Feature dimension | 128 |
| Direction | Separate forward and backward scans |
| `d_state`, `d_conv`, `expand` | `8`, `3`, `1` |
| Normalization | Pre-Mamba `LayerNorm(128)` |
| Fusion | Learned per-channel sigmoid gate, initialized to equal directions |
| Residual stabilization | Dropout `0.1`, LayerScale `0.01` |
| Padding | Valid observations compacted before recurrence |
| Agent chunking | 128 agents; time is never chunked |
| CUDA path | Fused selective scan; cuDNN `Conv1d` |

This placement was plausible because an agent history is genuinely ordered, it acts before pooling discards timestep detail, and the small residual update starts close to original SHARP. It improved over scene-token Mamba: best minADE6 fell from `0.680362` to `0.673736` (`0.97%`). It still did not beat unmodified SHARP. Ten observations may be too short for Mamba to add value beyond SHARP's four temporal-attention blocks.

Implementation: [temporal-agent Mamba setup](../Lab%202/Codes/setup_lab2_temporal_agent_mamba_rotationfix80.py).

### Full Temporal-Attention Replacement

The 20-epoch screen replaced all four temporal attention blocks with four unidirectional Mamba blocks. Unlike the residual addition, no temporal attention remained.

Result: minADE6 `0.780190` versus the screen baseline `0.752339`. This is evidence against wholesale temporal-attention replacement in the tested configuration.

## SEAM Mamba Hypotheses

The Lab 3 SEAM suite contains one partial baseline and two not-yet-run Mamba variants. No accuracy conclusion should be drawn for the Mamba variants yet.

| Variant | Change | Rationale | Evidence status |
|---|---|---|---|
| Agent-history Mamba addition | Adds a residual Mamba refinement over chronological observed-agent features while preserving SEAM's endpoint-aware streaming and decoder. | The observed history is a semantically ordered sequence and may benefit from selective recurrent state without removing attention. | Pending; no metric. |
| Future-head Mamba replacement | Replaces the trajectory-coordinate MLP with a Mamba sequence head over ordered future steps. | Future coordinates form an ordered sequence; recurrence could promote smooth, dynamically consistent trajectories. | Pending; no metric. |

The latest committed SEAM baseline snapshot is partial, so even the baseline is not yet a final comparison row.

## Prepared Final SHARP Combination

The final three-run Lab 2 package is prepared but has no committed metrics:

1. pinned official SHARP baseline;
2. baseline plus QKNorm, uncertainty-aware target context, and relative-geometry bias;
3. run 2 plus the small residual bidirectional temporal-agent Mamba.

This design combines only modifications that either improved a controlled screen or have a conservative residual formulation. The third run tests whether Mamba can add value after the empirically supported attention/context changes. Until those runs finish, this remains a hypothesis.

Specification: [final three-run suite](../Lab%202/SHARP_Final_3_Run_Suite/README.md).

## Mamba in DeMo, SEAM, and Original SHARP

| Original model | Uses Mamba? | Main mechanism |
|---|---|---|
| DeMo | Yes | Unidirectional Mamba for observed agent histories and bidirectional Mamba in future-state and hybrid coupling modules, combined with attention |
| SEAM | No | Transformer attention with endpoint-aware streaming |
| SHARP | No | Transformer attention with instance-aware short-window streaming |

The Lab 2 temporal-agent addition is conceptually closest to DeMo's historical-agent encoder, but it is not a DeMo reproduction. DeMo uses a deeper Mamba design and additional decoder Mamba modules; modified SHARP retains its own decoder and streaming architecture.

- [DeMo paper](https://papers.nips.cc/paper_files/paper/2024/file/c0ff9e52e94ae331bc0f2d28be06a9ca-Paper-Conference.pdf)
- [SEAM paper](https://openaccess.thecvf.com/content/WACV2026/papers/Prutsch_Streaming_Real-Time_Trajectory_Prediction_Using_Endpoint-Aware_Modeling_WACV_2026_paper.pdf)
- [SHARP paper](https://openaccess.thecvf.com/content/CVPR2026/html/Prutsch_SHARP_Short-Window_Streaming_for_Accurate_and_Robust_Prediction_in_Motion_Forecasting_CVPR_2026_paper.html)

See [the complete run registry](Main/All_Run_Registry.md) for statuses and [the consolidated metric tables](Main/SHARP_AV2_Main_Results.md) for exact values.
