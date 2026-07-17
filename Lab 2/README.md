# Lab 2: SHARP with Mamba Encoder on AV2

This folder contains the Lab 2 setup for a new, isolated SHARP experiment on
the AV2 motion-forecasting dataset. It copies the existing SHARP source before
patching it, uses all four RTX 2080 Ti GPUs with DDP, trains for 60 epochs, and
stores new outputs separately. Existing SHARP code and results are not deleted
or overwritten.

## Article-aligned 80-epoch Mamba run

`setup_lab2_sharp_mamba_article80.py` preserves the verified Mamba architecture
and placement from the 60-epoch experiment while matching the SHARP paper's
AV2 optimization protocol: 80 epochs, global batch size 32, 13 warm-up epochs,
linear warm-up to `1e-4`, cosine decay to `1e-5`, AdamW, gradient clipping and
weight decay. It creates new `SHARP_AV2_MAMBA_ARTICLE80_<timestamp>` code and
result directories and does not overwrite the earlier 60-epoch experiment.

```bash
cd "/home/server00/M/Codes/Thesis/Lab 2"

/home/server00/M/Codes/envs/sharp/bin/python \
  setup_lab2_sharp_mamba_article80.py

EXPERIMENT_ROOT=$(cat \
  /home/server00/M/Codes/LATEST_SHARP_AV2_MAMBA_ARTICLE80.txt)

bash "$EXPERIMENT_ROOT/run_av2_mamba_4gpu.sh"
```

The launcher refuses to start while another `train.py` process is using Lab 2.
It uses all four GPUs, batch size 8 per GPU and six DataLoader workers per DDP
rank. The three best `minADE6` checkpoints and `last.ckpt` are retained.

## Required Lab 2 layout

The scripts expect these existing paths:

```text
/home/server00/M/
|-- Codes/
|   |-- SHARP/Code/
|   |-- envs/sharp/bin/python
|   `-- miniforge3/etc/profile.d/conda.sh
|-- Datasets/
|   `-- AV2/sharp_processed/
|       |-- train/
|       `-- val/
`-- Results/
```

The AV2 processed counts should be:

```text
train: 199908 .pt files
val:    24988 .pt files
```

## Download on Lab 2

Clone this repository under `/home/server00/M/Codes`:

```bash
cd /home/server00/M/Codes
git clone https://github.com/madviddd/Thesis.git Thesis
cd "/home/server00/M/Codes/Thesis/Lab 2"
```

If the repository is already cloned:

```bash
cd /home/server00/M/Codes/Thesis
git pull
cd "/home/server00/M/Codes/Thesis/Lab 2"
```

## Verify prerequisites

```bash
test -d /home/server00/M/Codes/SHARP/Code
test -x /home/server00/M/Codes/envs/sharp/bin/python

echo "AV2 train:"
find /home/server00/M/Datasets/AV2/sharp_processed/train -name "*.pt" | wc -l
echo "AV2 val:"
find /home/server00/M/Datasets/AV2/sharp_processed/val -name "*.pt" | wc -l

/home/server00/M/Codes/envs/sharp/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
raise SystemExit(
    0 if torch.cuda.is_available() and torch.cuda.device_count() == 4 else 1
)
PY
```

## Build the isolated experiment

```bash
cd "/home/server00/M/Codes/Thesis/Lab 2"
source /home/server00/M/Codes/miniforge3/etc/profile.d/conda.sh
conda activate /home/server00/M/Codes/envs/sharp

/home/server00/M/Codes/envs/sharp/bin/python \
  setup_lab2_sharp_mamba_v2.py

/home/server00/M/Codes/envs/sharp/bin/python \
  finalize_mamba_experiment.py
```

The setup creates new timestamped paths:

```text
/home/server00/M/Codes/SHARP_AV2_MAMBA_ENCODER_V2_<timestamp>/
/home/server00/M/Results/SHARP_AV2_MAMBA_ENCODER_V2_<timestamp>/
```

## Train on all four GPUs

```bash
EXPERIMENT_ROOT=$(cat /home/server00/M/Codes/LATEST_SHARP_AV2_MAMBA_ENCODER_V2.txt)
bash "$EXPERIMENT_ROOT/run_av2_mamba_4gpu.sh"
```

The default configuration uses:

- four GPUs through DDP
- batch size 8 per GPU, global batch size 32
- six DataLoader workers per process, 24 workers total
- 60 epochs
- three best checkpoints selected by `minADE6`
- automatic validation of the best checkpoint

If batch size 8 raises a CUDA out-of-memory error, start a new isolated attempt
with batch size 6:

```bash
EXPERIMENT_ROOT=$(cat /home/server00/M/Codes/LATEST_SHARP_AV2_MAMBA_ENCODER_V2.txt)
BATCH_SIZE=6 bash "$EXPERIMENT_ROOT/run_av2_mamba_4gpu.sh"
```

## Verification markers

The output must include:

```text
MAMBA_EXPERIMENT_FINALIZED
MAMBA_SMOKE_TEST_OK
MAMBA_ACTIVE=True
MAMBA_CHECKPOINT_KEYS=<positive number>
MAMBA_CHECKPOINT_PARAMETERS=<positive number>
SHARP_MAMBA_RUN_COMPLETE
```

A positive Mamba checkpoint key count proves that Mamba parameters were part of
the trained checkpoint. The implementation is a trainable native-PyTorch
selective state-space encoder placed after SHARP scene-token positional
embedding and before dual training and instance-aware context streaming.

## Monitor

Run these in another terminal:

```bash
watch -n 2 nvidia-smi
```

```bash
ps -u "$USER" -o pid,ppid,etime,%cpu,%mem,cmd \
  | grep -E "train.py|run_av2_mamba_4gpu" \
  | grep -v grep
```

Do not delete previous SHARP result folders. Each attempt writes to a new
timestamped result directory.

## Fused CUDA Mamba, 80 epochs

`setup_lab2_sharp_mamba_fused80.py` creates another isolated experiment. It
keeps the same encoder placement and Mamba dimensions, but replaces the slow
Python selective-scan loop with the official fused CUDA Mamba implementation.
It does not modify the shared SHARP environment or any previous code/results.

Stop any active `train.py` process before installing. Then build the new
experiment and install its pinned Python 3.11, PyTorch 2.8, CUDA 12 wheels:

```bash
cd "/home/server00/M/Codes/Thesis/Lab 2"

/home/server00/M/Codes/envs/sharp/bin/python \
  setup_lab2_sharp_mamba_fused80.py

EXPERIMENT_ROOT=$(cat \
  /home/server00/M/Codes/LATEST_SHARP_AV2_MAMBA_FUSED80.txt)

bash "$EXPERIMENT_ROOT/install_fused_mamba.sh"
```

Run the 80-epoch experiment in the foreground:

```bash
cd "$EXPERIMENT_ROOT/Code"
bash "$EXPERIMENT_ROOT/run_av2_mamba_fused_4gpu.sh"
```

The run uses four GPUs through DDP, batch size 8 per GPU (global batch 32),
a CPU-aware DataLoader worker count, 80 epochs, 13 warm-up epochs, and
the article-aligned `1e-4` to `1e-5` learning-rate schedule. It saves the top
three `minADE6` checkpoints plus the last checkpoint in a fresh result folder.

Before epoch 0, output must contain all of these markers:

```text
FUSED_CUDA_MAMBA_INSTALL_OK
FUSED_MAMBA_CUDA_SMOKE_TEST_OK
FUSED_MAMBA_CUDA_ACTIVE=True
MAMBA_ACTIVE=True FUSED_MAMBA_CUDA_ACTIVE=True
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
```

The smoke test performs a real CUDA forward/backward pass and verifies that
the official fused `mamba_inner_fn` fast path was called. The installer places
all Mamba packages under the timestamped experiment, so PyTorch and the shared
environment remain unchanged.
