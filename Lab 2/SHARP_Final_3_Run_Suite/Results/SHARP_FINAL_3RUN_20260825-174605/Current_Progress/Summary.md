# Lab 2 Final SHARP Three-Run Progress

- Captured: 2026-08-31T20:04:52+01:00
- Results root: `/home/server00/M/Results/SHARP_FINAL_3RUN_20260825-174605`
- Completed variants: 1/3
- Active variant: `none detected`
- Suite complete: False
- Complete local transcript: `/home/server00/M/Terminal/SHARP_Final_3_Run_Suite/SHARP_FINAL_3RUN_20260825-174605/Snapshots/Progress_20260831-200431/Terminal.txt` (174241486 bytes)
- GitHub Terminal.txt mode: compact transcript; complete transcript retained locally
- Append-only local Terminal.txt: `/home/server00/M/Terminal/SHARP_Final_3_Run_Suite/SHARP_FINAL_3RUN_20260825-174605/Terminal.txt` (174241486 bytes); previously saved lines were retained.

## Progress And Best Validation Metrics

All metrics are lower-is-better. Best values are selected by the lowest recorded minADE6 epoch.

| Run | Status | Current progress | Latest validated epoch | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 | Checkpoints |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official SHARP baseline | completed | epoch 79, 100% | 79 | 0.1560 | 1.9246 | 1.6769 | 0.6793 | 4.1386 | 1.2877 | 11 |
| SHARP + QKNorm + uncertainty + geometry | pending | not started | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |
| Run 2 + residual temporal-agent Mamba | pending | not started | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |

## Runtime Diagnostics

- Matched warning/error lines in the complete transcript: 0 retained (latest 200 maximum).
- Snapshot generation only read logs and `/proc`; it did not send signals to training.

No matched runtime warnings or errors were found.
