# Architectural Modifications: Mechanisms and Evidence

SEAM and SHARP forecast multiple possible agent trajectories while transferring
context between observations. SEAM emphasises endpoint-aware streaming; SHARP
processes short observation windows with instance-aware context transfer. Each
study changes defined operators or pathways while retaining the corresponding
streaming structure. An architectural rationale is a hypothesis; the measured
outcome is reported separately.

## Context and Attention Mechanisms

| Mechanism | Implementation and location | Motivation | Recorded outcome |
|---|---|---|---|
| Multi-head attention (MHA) | Scaled query-key dot products produce softmax weights that aggregate values within each head. | Content-dependent interaction provides the reference operator. | Unchanged control within each attention study. |
| Query-key normalisation (QKNorm) | Projected queries and keys are L2-normalised within each head, with a learned logit scale. Surrounding attention topology is preserved. | Decouple vector magnitude from attention sharpness and control logit scale. | Selected minADE6 improves by 3.27% in the SEAM context/attention study and 0.55% in the SHARP attention study. |
| Uncertainty-aware target context | Previous mode probabilities adjust the radius and feature gain of endpoint-centred target context. | Ambiguous predictions may benefit from broader context than confident predictions. | Selected minADE6 improves by 2.60% in the SEAM ablation and 0.32% in the SHARP architectural ablation. |
| Relative-geometry attention bias | Learned per-head biases encode relative displacement, distance and heading in current-window scene attention. | Spatial interaction depends on geometry, not content features alone. | Selected minADE6 improves by 1.43% in the SEAM ablation and 0.35% in the SHARP architectural ablation. |
| Talking-Heads | Identity-initialised learned mixing across heads before and after softmax. | Exchange complementary information between heads. | SHARP selected minADE6 is 0.23% higher than MHA; combining it with QKNorm gives 0.15% higher error. |

Percentages use each study's own control. Implementation references:
[SEAM operators](../Studies/SEAM/Context_Attention_Ablation/upstream/seam-main/src/model/layers/controlled_ablation.py),
[SHARP operators](../Studies/SHARP/Attention_Operators/attention_variants.py).

## SEAM State-Space Integration

Mamba uses an input-dependent state-space recurrence to select which sequence
information is retained or suppressed. Sequence order is therefore part of the
architectural choice, rather than simply another implementation of attention.

| Placement | Modification | Rationale | Selected minADE6 |
|---|---|---|---:|
| Reference | SEAM without added Mamba | Matched endpoint-aware control | 0.662859 |
| Agent history | Residual refinement of observed agent-history features | Accumulate chronological evidence without removing the reference pathway | 0.664814 |
| Future head | Replace the coordinate MLP with a two-block Mamba head over future steps | Model dependencies between ordered future coordinates | 0.648480 |

Future-head replacement improves selected minADE6 by 2.17%; history addition
does not improve that metric. These are different insertion locations, not
equivalent increases in capacity. See the
[architecture audit](../Studies/SEAM/State_Space_Integration/ARTICLE_AND_CODE_AUDIT.md)
and [results](../Studies/SEAM/State_Space_Integration/Summary.md).

## SHARP Architectural Ablation

Each intervention was trained independently against the same control. The
shortened training horizon is specified in the protocol, not treated as a
replacement for the separate full-length integration experiment.

| Intervention | What changes | Why it was tested | Selected minADE6 |
|---|---|---|---:|
| Baseline | Original architecture | Shared control | 0.752339 |
| Confidence-gated memory | Gate between current tokens and streamed updates | Attenuate stale or noisy history | 0.755349 |
| Cross-window consistency | Smooth L1 agreement with weight 0.05 between aligned adjacent-window predictions | Reduce inconsistent forecasts across observations | 0.776207 |
| Learned temporal pooling | Masked learned pooling replaces max pooling | Weight informative observations before compression | 0.757637 |
| Uncertainty-aware context | Probability-conditioned context radius and gain | Adapt selection to multimodal ambiguity | 0.749961 |
| Relative geometry | Per-head scene-attention bias | Introduce a geometric interaction prior | 0.749679 |
| Kinematic stem | Masked velocity, acceleration and speed-change embeddings | Expose motion derivatives before temporal encoding | 0.785164 |
| Endpoint refinement | Endpoint-conditioned trajectory/logit correction and 0.2-weight coarse-prediction loss | Improve endpoint localisation and mode separation | 0.755898 |
| Lane topology graph | Sparse geometry-derived lane-neighbour message passing | Represent lane connectivity before scene interaction | 0.754238 |
| Temporal Mamba replacement | Four unidirectional Mamba blocks replace history attention; max pooling remains | Test recurrence as a complete history encoder | 0.780190 |

Only uncertainty and geometry improve selected minADE6 in this screen. Endpoint
refinement improves minFDE1 but not minADE6. The measurements identify candidates;
they do not establish causal explanations for every unsuccessful intervention.
See the [manifest](../Studies/SHARP/Architecture_Ablation/EXPERIMENT_MANIFEST.md)
and [complete comparison](../Studies/SHARP/Architecture_Ablation/Summary.md).

## SHARP State-Space Placements

The scene-token experiment applies recurrence after agent-history pooling over
combined scene tokens. The temporal-agent addition instead places a small
bidirectional residual module between history-attention blocks two and three,
before pooling. All four attention blocks remain in the addition experiment.

The temporal addition uses feature dimension 128, state size 8, convolution
width 3, expansion 1, pre-normalisation, dropout 0.1 and residual LayerScale
0.01. A learned gate combines scan directions. It differs from the architectural
ablation's complete replacement: four unidirectional blocks with state size 16,
convolution width 4 and expansion 2.

Chronological agent histories provide a clearer sequence interpretation than
scene-token ordering. Selected errors are 0.680362 for scene recurrence and
0.673736 for temporal recurrence. See
[placement evidence](../Studies/SHARP/State_Space_Integration/Summary.md).

## Composed Mechanisms

The [SHARP integration study](../Studies/SHARP/Architecture_Integration/Summary.md)
combines QKNorm, uncertainty and geometry. It improves four secondary metrics,
including MR, while selected minADE6 is 0.19% higher than the matched reference.
Complementary motivations do not imply additive empirical gains.

The residual-Mamba SHARP extension and the
[combined SEAM extension](../Studies/SEAM/Combined_Extension/README.md) retain
their implementations and observations, but no committed final evaluation
supports an accuracy claim for either. A SEAM composition without Mamba is not
a recorded completed experiment.

## Metric Definitions

- **minADE6:** minimum average displacement error among six forecast modes,
  following the selection rule in the corresponding model's metric code.
- **minFDE6:** minimum final displacement error among six modes.
- **minADE1 / minFDE1:** corresponding single-mode criteria.
- **MR:** miss rate, the fraction exceeding the evaluation's endpoint threshold.
- **b-minFDE6:** Brier-adjusted final displacement error, incorporating a
  probability penalty alongside endpoint error.

Displacement errors are measured in metres; MR is a fraction. The probability
penalty in b-minFDE6 is dimensionless, making it a composite score. All are
lower-is-better. Selection values and full vectors are distinguished wherever
their evaluation records differ. [Source attribution](../Documentation/Source_Attribution.md)
identifies the baseline implementations and audits.
