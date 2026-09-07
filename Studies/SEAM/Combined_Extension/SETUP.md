# SEAM: Combined 80-Epoch Experiment

One new run: SEAM + uncertainty-aware target context + relative-geometry
attention bias + QKNorm + future-head Mamba. It starts from newly initialised
weights, not from any previous run's trained checkpoint. The individual
improvements motivate this experiment but do not establish that their
combination will improve accuracy.

## Protocol

The reference is the existing `SEAM_AV2_MAMBA_3RUN_20260824-202936` baseline,
using the released SEAM configuration: 80 epochs, seed 2333, AdamW,
LR 0.001 to 0.00001, warm-up ratio 0.167 (13 epochs), weight decay 0.01,
gradient clipping norm 5, FP32, SyncBatchNorm, and global batch 32.
Processed AV2 tensors are taken only from that run's DATA_ROOT.txt.
Three observation windows end at 3, 4 and 5 seconds; all three contribute
training gradients. Model width is 128, with eight attention heads, six modes,
four encoder blocks and 80 internally generated future points. Scoring uses
the dataset's six-second future horizon, unchanged from the reference.

Two RTX 2080 Ti GPUs each process a microbatch of 8, accumulating two steps.
This preserves the previous update batch (32) and SyncBatchNorm activation
batch (16). Three equal per-GPU batches cannot total 32. Therefore, the third
GPU performs the final selected-checkpoint evaluation rather than silently
changing the update batch to 48. CPU workers are bounded by available CPU
affinity, with pinned host memory and bounded prefetch. RAM and VRAM are not
deliberately filled: headroom is required for variable-size AV2 scenes.
No mixed precision, learning-rate scaling, smaller batches, dataset subsampling
or altered optimisation parameters are introduced as automatic fallbacks.

## Combined Architecture

- **Uncertainty:** the previous window's six modal probabilities and normalised
  entropy control the target-context feature gain and retrieval radius. This
  is the same intervention as the 20-epoch control. The radius uses discrete
  neighbourhood selection; no claim is made that this selection is differentiable.
- **Geometry:** the same zero-initialised five-feature relative-position,
  distance and heading network biases the four current scene-attention blocks.
- **QKNorm:** all 20 attention modules use headwise L2-normalised queries and
  keys, with a learned per-head logit scale. Geometry is added to these logits,
  and padding masks are preserved. Fully masked rows have zero attention mass
  and finite backward gradients.
- **Future-head Mamba:** only the decoder's coordinate MLP is replaced by the
  previous two-block ordered-future Mamba head: width 128, state 16, convolution
  width 4, expansion 2, time embeddings, pre-normalisation and residual gates
  initialised at sigmoid(-2). It does not mix unordered agents as a sequence.

Agent encoding, map encoding, modal probabilities, context streaming,
trajectory relay, loss terms and the AV2 data protocol otherwise remain intact.
The setup assembles a fresh code copy from the two existing packages. Pinned
input hashes and a complete generated-source inventory detect unintended drift.
No previous experiment's code, results, checkpoints or environment is modified.

## Checks and Recovery

The launcher verifies the existing pinned torch 2.1.1/cu121 environment,
GitHub token identity/access, two GPUs, free disk/memory, model composition,
Mamba initialisation and complete/non-overlapping optimizer groups.
An actual two-GPU smoke training runs four real AV2 batches per GPU, validation,
and a full checkpoint save. A second smoke phase restores this checkpoint and
must advance its optimiser step. These disposable preflight weights never
initialise the 80-epoch experiment.

Every completed epoch retains a full Lightning checkpoint including optimizer,
scheduler and loop state. Saves use Lightning's atomic local checkpoint I/O.
An unreadable newest epoch can fall back to an older readable epoch without
deleting the damaged file. An interrupted epoch may be replayed: this is not
an exact mid-batch or bitwise-identical recovery guarantee. Three bounded
fresh-process attempts are allowed per invocation; persistent faults stop with
evidence intact. Reusing the same launch command resumes the same run and
skips completed training. Publication failures never retrain an 80-epoch model.
An advisory lock prevents a second launcher for the same experiment.

The preflight cannot prove that every later variable-size scene fits in VRAM,
nor prevent hardware, power or network failures. Warnings are recorded, not
globally hidden. Existing known optimizer grouping, deprecated import and
ambiguous validation batch-size issues are already corrected in the source.
Unseen failures are surfaced with their logs rather than declared impossible.
Lightning can report that the checkpoint directory is already non-empty on
resume; this describes deliberately retained checkpoints, not lost progress.

Progress is printed every 50 batches and written to HEARTBEAT.json; validation
metrics are appended to epoch_metrics.csv. Final evaluation reports all six
metrics from the checkpoint selected by minimum validation minADE6.

## Publication

Token-only authentication reads `/home/server01/M/Token/Token.txt` and requires
the writable `madviddd` account. The remote remains `madvidd/Thesis`.
Completion automatically creates a compact snapshot, merges current `main`,
pushes and reads back every published file to verify exact contents.
The destination is `Studies/SEAM/Combined_Extension/Results/<new-run-name>`.
The snapshot contains Terminal.txt, Summary.md, all six metrics, epoch CSV,
attempt durations, configuration/source evidence and checkpoint hashes.
Large model files, full raw logs and TensorBoard events remain on the lab PC.
The complete local Terminal.txt is append-preserved; GitHub receives a bounded,
credential-sanitised excerpt if needed. Older runs' result folders are untouched.

For an on-demand snapshot in another terminal:

```bash
BASE=/home/server01/M
EXPERIMENT=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SEAM_AV2_80EPOCH_COMBINED_CODE.txt")
"$BASE/Codes/envs/seam_av2_mamba_torch211/bin/python" \
  "$EXPERIMENT/publish_combined.py" --experiment "$EXPERIMENT"
echo "Terminal remains open."
```

Checkpoint and distributed-execution semantics follow the pinned implementation:
[Lightning 2.4 ModelCheckpoint](https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/pytorch/callbacks/model_checkpoint.py),
[Lightning atomic checkpoint I/O](https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/plugins/io/torch_io.py).
