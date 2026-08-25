# Lab 2 Final SHARP Three-Run Progress

- Captured: 2026-08-25T16:44:02+01:00
- Results root: `/home/server00/M/Results/SHARP_FINAL_3RUN_20260825-144225`
- Completed variants: 0/3
- Active variant: `none detected`
- Suite complete: False
- Complete local transcript: `/home/server00/M/Terminal/SHARP_Final_3_Run_Suite/SHARP_FINAL_3RUN_20260825-144225/Progress_20260825-164402/Terminal.txt` (22564 bytes)
- GitHub Terminal.txt mode: complete transcript

## Progress And Best Validation Metrics

All metrics are lower-is-better. Best values are selected by the lowest recorded minADE6 epoch.

| Run | Status | Current progress | Latest validated epoch | MR | b-minFDE6 | minADE1 | minADE6 | minFDE1 | minFDE6 | Checkpoints |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official SHARP baseline | pending | not started | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |
| SHARP + QKNorm + uncertainty + geometry | pending | not started | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |
| Run 2 + residual temporal-agent Mamba | pending | not started | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0 |

## Runtime Diagnostics

- Matched warning/error lines in the complete transcript: 3 retained (latest 200 maximum).
- Snapshot generation only read logs and `/proc`; it did not send signals to training.

```text
FINAL_SUITE_PREFLIGHT_FAILED: Preflight failed (1) in /home/server00/M/Results/SHARP_FINAL_3RUN_20260825-144225/preflight/01_official_sharp_baseline/preflight.log; forbidden=['Traceback (most recent call last)', 'Error executing job with overrides']
Traceback (most recent call last):
RuntimeError: Preflight failed (1) in /home/server00/M/Results/SHARP_FINAL_3RUN_20260825-144225/preflight/01_official_sharp_baseline/preflight.log; forbidden=['Traceback (most recent call last)', 'Error executing job with overrides']
```
