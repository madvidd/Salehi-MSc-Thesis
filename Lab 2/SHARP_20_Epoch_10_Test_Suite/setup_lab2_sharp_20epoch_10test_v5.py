#!/usr/bin/env python3
"""Create the suite with strict YAML and CUDA-local loss-index compatibility."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
V4_SETUP = HERE / "setup_lab2_sharp_20epoch_10test_v4.py"

subprocess.run([sys.executable, str(V4_SETUP)], check=True)

base = Path("/home/server00/M")
v4_pointer = base / "Codes/LATEST_SHARP_AV2_20EPOCH_10TEST_V4.txt"
experiment_root = Path(v4_pointer.read_text(encoding="utf-8").strip())
pl_path = experiment_root / "Code/src/model/pl_modules.py"
source = pl_path.read_text(encoding="utf-8")

# The pinned CUDA/PyTorch stack failed reproducibly when a CPU row-index tensor
# and CUDA best-mode tensor were combined in advanced indexing. Put both index
# tensors on the prediction device. This is mathematically identical and does
# not change any model, loss, optimizer, batch, seed, or schedule setting.
replacements = (
    (
        "torch.arange(y_hat.shape[0]), best_mode",
        "torch.arange(y_hat.shape[0], device=y_hat.device), best_mode",
        2,
        "primary trajectory loss indices",
    ),
    (
        "torch.arange(new_y_hat.shape[0]), best_mode",
        "torch.arange(new_y_hat.shape[0], device=new_y_hat.device), best_mode",
        1,
        "single-trajectory loss index",
    ),
)

for old, new, expected, label in replacements:
    count = source.count(old)
    if count != expected:
        raise RuntimeError(f"Expected {expected} {label}, found {count}")
    source = source.replace(old, new)

compile(source, str(pl_path), "exec")
pl_path.write_text(source, encoding="utf-8", newline="\n")

v5_pointer = base / "Codes/LATEST_SHARP_AV2_20EPOCH_10TEST_V5.txt"
v5_pointer.write_text(str(experiment_root) + "\n", encoding="utf-8", newline="\n")

print("CUDA_LOCAL_LOSS_INDICES_APPLIED=True")
print(f"V5_EXPERIMENT_ROOT={experiment_root}")
