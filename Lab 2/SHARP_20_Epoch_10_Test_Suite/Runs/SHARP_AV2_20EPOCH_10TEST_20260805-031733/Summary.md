# Lab 2 SHARP 20-Epoch Ten-Test Summary

Generated: 2026-08-13T19:41:20.853489+01:00

All variants use AV2, seed 2333, 20 epochs, global batch 32 (8 per GPU across four RTX 2080 Ti GPUs), AdamW, LR 1e-4 to 1e-5, 13 warm-up epochs, and SyncBatchNorm. Lower is better for all metrics.

| Test | Status | Live progress | Best epoch | Checkpoints | MR | b-minFDE6 | minADE1 | minADE6 | Delta minADE6 vs baseline | minFDE1 | minFDE6 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline SHARP | Completed | epoch index 19/19, 100% (6248/6248) | 19 | 21 | 0.192652 | 2.064007 | 1.886442 | 0.752339 | +0.000000 | 4.672195 | 1.434819 |
| Confidence-gated memory | Completed | epoch index 19/19, 100% (6248/6248) | 19 | 21 | 0.194413 | 2.067328 | 1.876097 | 0.755349 | +0.003010 | 4.694828 | 1.443534 |
| Cross-window consistency | Completed | epoch index 19/19, 100% (6248/6248) | 18 | 21 | 0.202497 | 2.096022 | 1.872110 | 0.776207 | +0.023868 | 4.785388 | 1.476766 |
| Learned temporal pooling | Completed | epoch index 19/19, 100% (6248/6248) | 19 | 21 | 0.201897 | 2.080306 | 1.891240 | 0.757637 | +0.005298 | 4.739688 | 1.463850 |
| Uncertainty-aware target context | Completed | epoch index 19/19, 100% (6248/6248) | 19 | 21 | 0.198575 | 2.074243 | 1.863236 | 0.749961 | -0.002378 | 4.676969 | 1.447852 |
| Relative geometry attention bias | Running (epoch index 6/19, 23%) | epoch index 6/19, 23% (1424/6248) | 5 | 7 | 0.338000 | 2.720000 | 2.520000 | 1.057035 | +0.304695 | 6.440000 | 2.090000 |
| Kinematic motion stem | Pending | - | - | 0 | - | - | - | - | - | - | - |
| Endpoint refinement decoder | Pending | - | - | 0 | - | - | - | - | - | - | - |
| Lane topology graph | Pending | - | - | 0 | - | - | - | - | - | - | - |
| Agent temporal Mamba | Pending | - | - | 0 | - | - | - | - | - | - | - |

Completed variants: 5/10. Currently active: Relative geometry attention bias.

`Delta minADE6 vs baseline` is variant minADE6 minus baseline minADE6; a negative value is better. Best minADE6 is read from checkpoint filenames. The other metrics use a saved variant summary when available, otherwise the closest matching or latest validation record in the logs. Lightning labels the 20 epochs from 0 through 19. Running results are provisional. This is a 20-epoch screening suite and should not be presented as equivalent to SHARP's full 80-epoch result.
