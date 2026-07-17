# Lab 3 - SHARP attention ablation

This package creates a new, timestamped AV2 experiment on Lab 3 without
deleting or modifying any previous code, checkpoints, logs, datasets, or
results.

## What is tested

The untouched control is the original SHARP model with standard PyTorch
multi-head dot-product attention. Three controlled variants replace only the
attention operator:

1. `qknorm`: L2-normalizes projected queries and keys and uses a learnable
   per-head logit scale.
2. `talking_heads`: adds identity-initialized head mixing before and after the
   attention softmax.
3. `qknorm_talking_heads`: combines both changes.

The embedding dimension, four-block encoders, eight heads, MLPs, positional
embeddings, streaming memory, target-context mechanism, decoder, losses,
optimizer, data and random seed remain the same across the four experiments.

## Where SHARP uses attention

The replacement covers every `MultiheadAttention` constructor in the original
single-agent SHARP model:

- Agent-history temporal self-attention: `Sharp.h_embed`, four blocks.
- Agent/lane scene self-attention: `Sharp.blocks`, four blocks.
- Instance-aware scene-memory cross-attention: `Sharp.scene_interact`, two
  blocks.
- Previous-trajectory relay cross-attention: `Sharp.traj_interact`, two blocks.
- Target-centric context self-attention: `Sharp.target_blocks`, two blocks.
- Mode-query decoder cross-attention: three scene blocks plus three
  target-context blocks.

That is 18 attention modules in total. The baseline remains unmodified.

## Training protocol

- AV2, 80 epochs.
- AdamW, peak learning rate `1e-4`, minimum learning rate `1e-5`.
- 13 warmup epochs followed by cosine decay.
- Original SHARP seed `2333`, model dimension 128, eight heads, four encoder
  blocks, DropPath 0.2, gradient clipping 5 and weight decay `1e-2`.
- Three RTX 2080 Ti GPUs through DDP.
- Default batch size 8 per GPU, global batch size 24.
- Default worker count is selected from the available CPUs and capped at eight
  workers per rank.

The SHARP paper used one RTX 8000 and global batch size 32. Three equal DDP
ranks cannot produce global batch 32 exactly. Global batch 24 is the stable
three-GPU setting already demonstrated on Lab 3. Therefore the baseline is
required for the controlled comparison, and exact article metrics cannot be
guaranteed.

## Preserve and stop the previous Lab 3 run

In the terminal currently displaying the old run, press `Ctrl+C` once and wait
for the shell prompt. Then verify that only processes whose working directory
is the old code directory are stopped:

```bash
OLD=/home/server01/M/Codes/AV2/SHARP_context_encoder_code

for pid in $(pgrep -u "$USER" -f 'python.*train.py' || true); do
  cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
  printf 'pid=%s cwd=%s\n' "$pid" "$cwd"
  if [[ "$cwd" == "$OLD"* ]]; then
    kill -INT "$pid"
  fi
done

sleep 10
nvidia-smi
```

This sends an interrupt only. It does not remove any files.

## Clone into `/home/server01/M`

```bash
mkdir -p /home/server01/M/Codes
cd /home/server01/M/Codes

if [ -d /home/server01/M/Codes/Thesis/.git ]; then
  cd /home/server01/M/Codes/Thesis
  git fetch origin
  git switch lab3-sharp-attention-ablation
  git pull --ff-only origin lab3-sharp-attention-ablation
else
  git clone \
    --branch lab3-sharp-attention-ablation \
    --single-branch \
    https://github.com/madviddd/Thesis.git \
    /home/server01/M/Codes/Thesis
fi

cd "/home/server01/M/Codes/Thesis/Lab 3"
```

The repository package belongs at:

```text
/home/server01/M/Codes/Thesis/Lab 3
```

The setup creates new experiment copies under:

```text
/home/server01/M/Codes/SHARP_ATTENTION_ABLATION_<timestamp>
/home/server01/M/Results/SHARP_ATTENTION_ABLATION_<timestamp>
```

It reuses, without modifying:

```text
/home/server01/M/Codes/AV2/envs/sharp_av2
/home/server01/M/Datasets/AV2/sharp_processed
```

## Set up and start

```bash
cd "/home/server01/M/Codes/Thesis/Lab 3"

/home/server01/M/Codes/AV2/envs/sharp_av2/bin/python \
  setup_lab3_attention_ablation.py

ROOT=$(cat /home/server01/M/Codes/LATEST_SHARP_ATTENTION_ABLATION.txt)
RESULTS=$(cat /home/server01/M/Results/LATEST_SHARP_ATTENTION_ABLATION.txt)

nohup "$ROOT/run_all_attention_experiments.sh" \
  > "$RESULTS/launcher.log" 2>&1 < /dev/null &

echo $! > "$RESULTS/launcher.pid"
disown
echo "ROOT=$ROOT"
echo "RESULTS=$RESULTS"
```

The baseline and three variants run sequentially, each using all three GPUs.
Closing the terminal after `disown` does not stop the suite.

## Monitor and resume

```bash
ROOT=$(cat /home/server01/M/Codes/LATEST_SHARP_ATTENTION_ABLATION.txt)
RESULTS=$(cat /home/server01/M/Results/LATEST_SHARP_ATTENTION_ABLATION.txt)

tail -f "$RESULTS/launcher.log"
nvidia-smi
```

If a run stops, launch the same `run_all_attention_experiments.sh` command
again. Completed variants are skipped and an incomplete variant resumes from
its `last.ckpt`.

Run or resume one variant only:

```bash
ROOT=$(cat /home/server01/M/Codes/LATEST_SHARP_ATTENTION_ABLATION.txt)
"$ROOT/run_variant.sh" qknorm
```

Accepted names are `baseline_mha`, `qknorm`, `talking_heads`, and
`qknorm_talking_heads`.

## Compare results

```bash
ROOT=$(cat /home/server01/M/Codes/LATEST_SHARP_ATTENTION_ABLATION.txt)
"$ROOT/compare_now.sh"
```

The final files are:

```text
attention_comparison.csv
attention_comparison.md
```

Each variant also contains `metrics.json`, `best_checkpoint.txt`, `train.log`,
`eval.log`, checkpoints and a `COMPLETE` marker.
