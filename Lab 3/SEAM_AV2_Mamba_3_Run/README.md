# Lab 3: SEAM AV2 Baseline and Two Mamba Ablations

This folder is a self-contained, resumable three-run package built from the
official SEAM source and the article's Argoverse 2 setup.

Run order:

1. Official SEAM baseline.
2. SEAM plus agent-history Mamba refinement.
3. SEAM with the trajectory-coordinate MLP replaced by a future-sequence Mamba
   head.

The launcher creates new timestamped code and result directories under
`/home/server01/M`. It never removes or overwrites prior Lab 3 runs. Every epoch
saves `last.ckpt`; rerunning the launcher resumes the incomplete variant and skips
completed variants. A failure stops before the next variant so comparisons cannot
silently continue from a broken run.

The existing processed AV2 data is reused read-only. Checkpoints and full logs
remain local. Summaries, configurations, checkpoint inventories, and bounded log
tails are published to GitHub after each completed or interrupted attempt.

See `ARTICLE_AND_CODE_AUDIT.md` for the exact control settings, the released-code
learning-rate discrepancy, and the two Mamba insertion decisions.
