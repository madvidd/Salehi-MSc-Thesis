# Streaming Motion Forecasting: Code and Experimental Results

Research software and recorded Argoverse 2 experiments on uncertainty-aware
context, relative geometry, attention operators and selective state-space
modelling in SEAM and SHARP. The repository contains implementations,
configurations, evaluation records and analysis.

## Start Here

- [Results overview](Results/README.md): current evidence and within-study comparisons.
- [Research figures](Results/Figures/README.md): architecture diagrams, study plots and graphical overview, indexed by figure number in SVG, PDF and PNG formats.
- [Experiment registry](Results/Main/All_Run_Registry.md): every study and its recorded outputs.
- [Modification guide](Results/Info.md): implementation locations, hypotheses and outcomes.
- [Repository guide](Documentation/Repository_Guide.md): file conventions and evidence hierarchy.
- [Reproduction guide](Documentation/Reproduction.md): prerequisites, launchers and publication.

## Studies

| Model | Study | Research question |
|---|---|---|
| SEAM | [State-space integration](Studies/SEAM/State_Space_Integration/README.md) | Where can selective recurrence complement endpoint-aware forecasting? |
| SEAM | [Context and attention ablation](Studies/SEAM/Context_Attention_Ablation/README.md) | How do uncertainty, geometry and query-key normalisation behave independently? |
| SEAM | [Combined extension](Studies/SEAM/Combined_Extension/README.md) | Can the mechanisms be composed with a future-sequence head? |
| SHARP | [Attention operators](Studies/SHARP/Attention_Operators/README.md) | Which attention operator supports short-window streaming? |
| SHARP | [Architectural ablation](Studies/SHARP/Architecture_Ablation/README.md) | Which memory, context, pooling and decoder changes improve the matched control? |
| SHARP | [State-space integration](Studies/SHARP/State_Space_Integration/README.md) | How do scene-token and agent-history recurrence differ? |
| SHARP | [Architecture integration](Studies/SHARP/Architecture_Integration/README.md) | What is the effect of combining QKNorm, uncertainty and geometry? |

Study names describe the scientific intervention; epoch counts, batch sizes and
hardware are specified in each protocol. Results are compared within their own
study, not ranked across different training or evaluation settings.

## Layout

```text
Studies/        Model-specific source, protocols, tests and original run records
Results/        Curated summaries, comparison tables and indexed research figures
Documentation/ Repository guide, reproduction instructions and migration audit
Tools/          Log-inspection utilities and repository validation
LICENSE         Repository licence; upstream attribution is retained separately
README.md       Main entry point
```

Datasets, trained checkpoint binaries, credentials and oversized raw logs are
not distributed here. Checkpoint inventories and recorded machine paths identify
the corresponding local artifacts. A compact `Terminal.txt` is not necessarily
the complete original terminal history.

The [path migration index](Documentation/Repository_Audit/Path_Migration.csv)
maps the former machine-based layout to the study folders. Numerical records,
captured configurations and bundled model sources were preserved during this
organisation. [Source attribution](Documentation/Source_Attribution.md) identifies
the upstream research implementations.
