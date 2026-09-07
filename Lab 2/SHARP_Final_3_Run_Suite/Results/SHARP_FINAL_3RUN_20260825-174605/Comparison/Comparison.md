# Final SHARP AV2 Three-Run Comparison

All three training/evaluation runs completed: False

| Run | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 |
|---|---:|---:|---:|---:|---:|---:|
| SHARP article | 0.140000 | 1.822000 | 1.569000 | 0.639000 | 3.850000 | 1.197000 |
| Official SHARP baseline | 0.155955 | 1.924560 | 1.676867 | 0.679282 | 4.138639 | 1.287697 |
| SHARP + QKNorm + uncertainty context + geometry bias | 0.155515 | 1.918870 | 1.677496 | 0.680589 | 4.130601 | 1.285462 |

All values are lower-is-better. Final run values use single-GPU full validation to avoid DDP sampler padding.
