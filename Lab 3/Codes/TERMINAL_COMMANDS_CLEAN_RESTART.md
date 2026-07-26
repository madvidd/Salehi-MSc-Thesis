# Lab 3 clean restart

The previous restart procedure is obsolete because it used the old GitHub
account, the CUDA 12.8 PyTorch environment, and continued after failed runs.

Use
[`TERMINAL_COMMANDS_LAB3_RECOVERY.md`](TERMINAL_COMMANDS_LAB3_RECOVERY.md).
That procedure:

- uses no `sudo`;
- removes the `madvidd` GitHub CLI credential;
- verifies `madviddd` with a real pull/integrate/push test;
- preserves all partial results;
- validates PyTorch 2.8 with CUDA 12.6 on all three GPUs;
- excludes `baseline_mha`;
- runs `qknorm`, `talking_heads`, and `qknorm_talking_heads` in order;
- retries one native failure and stops rather than publishing a failed run.
