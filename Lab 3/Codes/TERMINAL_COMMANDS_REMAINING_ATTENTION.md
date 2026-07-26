# Lab 3 remaining attention experiments

This continuation never runs `baseline_mha`. It runs, in order:

1. `qknorm`
2. `talking_heads`
3. `qknorm_talking_heads`

Each experiment uses the existing generated `run_variant.sh`, preserving its
80 epochs, seed 2333, three-GPU DDP, batch size 8 per GPU, global batch 24,
SyncBatchNorm, optimizer, learning-rate schedule, workers, checkpoints and AV2
dataset configuration.

After every attempted variant, the wrapper:

- creates a complete local archive, including checkpoints, logs and code;
- publishes only files below 10 MB and excludes credentials and large files;
- pushes the Lab 3 branch;
- automatically merges it into `main` and pushes `main`;
- continues to the next variant even if training, publication or merging fails.

## Run in the current terminal

```bash
BASE=/home/server01/M
REPO=$(cat "$BASE/Codes/LATEST_THESIS_LAB3_CLONE.txt")

cd "$REPO"
chmod +x "Lab 3/Codes/run_remaining_attention_suite.sh"

bash "Lab 3/Codes/run_remaining_attention_suite.sh"

STATUS=$?
echo "Final suite status: $STATUS"
echo "Terminal remains open."
```

The foreground command keeps progress visible. Closing that terminal while a
variant is training will terminate the foreground suite.

