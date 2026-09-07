# SEAM AV2 20-Epoch Controlled Four-Test Study

- Updated: `2026-09-03T11:15:26.422872+01:00`
- Completed variants: **0/4**
- Active/resumable variant: **SEAM baseline**

## Results

| Run | State | Progress | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 | Best saved minADE6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SEAM baseline | running/resumable | - | - | - | - | - | - | - | - |
| Uncertainty-aware target context | pending | - | - | - | - | - | - | - | - |
| Relative-geometry attention bias | pending | - | - | - | - | - | - | - | - |
| QKNorm attention | pending | - | - | - | - | - | - | - | - |

All reported displacement and miss metrics are lower-is-better. Only within-study comparisons are valid because the four runs share one data pipeline, schedule, seed, and 20-epoch budget.

## Training Time

| Run | Active training time |
|---|---:|
| SEAM baseline | 00:00:12 |
| Uncertainty-aware target context | - |
| Relative-geometry attention bias | - |
| QKNorm attention | - |

## Interventions

### 01. SEAM baseline

The verified SEAM AV2 architecture is retained unchanged and trained for the common 20-epoch budget.

### 02. Uncertainty-aware target context

Previous-window modal probabilities control the radius and feature gain of SEAM's existing endpoint-centred target-context branch.

### 03. Relative-geometry attention bias

Relative displacement, distance, and heading difference provide a learned per-head bias in the current-window scene self-attention blocks.

### 04. QKNorm attention

Per-head L2-normalised queries and keys with learned logit scales replace the attention kernels while preserving SEAM's topology and residual paths.

## Diagnostics

No warning or fatal-error signature is present in the retained logs.
