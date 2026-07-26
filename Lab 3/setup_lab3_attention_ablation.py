#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import py_compile
import shutil
import sys
from pathlib import Path


BASE = Path("/home/server01/M")
PACKAGE = Path(__file__).resolve().parent
SOURCE = PACKAGE / "sharp_original"
ENV_PYTHON = BASE / "Codes/AV2/envs/sharp_av2/bin/python"
CONDA_SH = BASE / "Codes/AV2/miniforge3/etc/profile.d/conda.sh"
DATA = BASE / "Datasets/AV2/sharp_processed"
VARIANTS = ("baseline_mha", "qknorm", "talking_heads", "qknorm_talking_heads")


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_checkpoint_config(code: Path) -> None:
    config = code / "conf/config.yaml"
    text = config.read_text()
    if "    save_last: True" not in text:
        text = text.replace("    save_top_k: 10\n", "    save_top_k: 10\n    save_last: True\n", 1)
    config.write_text(text)


def patch_attention(code: Path, variant: str) -> None:
    adapter = (PACKAGE / "attention_variants.py").read_text().replace("__VARIANT__", variant)
    adapter_path = code / "src/model/layers/attention_variants.py"
    adapter_path.write_text(adapter)

    expected = {
        "custom_transformer_blocks.py": 1,
        "transformer_blocks.py": 2,
    }
    for filename, expected_count in expected.items():
        path = code / "src/model/layers" / filename
        text = path.read_text()
        marker = "from .attention_variants import VariantMultiheadAttention\n"
        if marker not in text:
            text = text.replace("from torch import Tensor\n", "from torch import Tensor\n" + marker, 1)
        count = text.count("torch.nn.MultiheadAttention(")
        if count != expected_count:
            raise RuntimeError(f"Expected {expected_count} MHA constructors in {path}, found {count}")
        text = text.replace("torch.nn.MultiheadAttention(", "VariantMultiheadAttention(")
        path.write_text(text)


def write_runtime_patch(root: Path) -> Path:
    patch = root / "runtime_patch"
    patch.mkdir(parents=True)
    (patch / "sitecustomize.py").write_text(
        "import warnings\n"
        "\n"
        "import numpy as np\n"
        "for _name, _value in ((\"bool\", bool), (\"int\", int), (\"float\", float)):\n"
        "    if _name not in np.__dict__:\n"
        "        setattr(np, _name, _value)\n"
        "warnings.filterwarnings(\n"
        "    \"ignore\",\n"
        "    message=r\"The 'repr' attribute with value False was provided.*\",\n"
        ")\n"
        "warnings.filterwarnings(\n"
        "    \"ignore\",\n"
        "    message=r\"The 'frozen' attribute with value True was provided.*\",\n"
        ")\n"
        "warnings.filterwarnings(\n"
        "    \"ignore\",\n"
        "    message=r\"No device id is provided via .*\",\n"
        ")\n"
    )
    return patch


def write_run_scripts(code_root: Path, results_root: Path, runtime_patch: Path) -> None:
    runner = code_root / "run_variant.sh"
    runner.write_text(f'''#!/usr/bin/env bash
set -euo pipefail

VARIANT="${{1:?usage: run_variant.sh VARIANT}}"
case "$VARIANT" in
  baseline_mha|qknorm|talking_heads|qknorm_talking_heads) ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

BASE={BASE}
ROOT={code_root}
RESULTS_ROOT={results_root}
CODE="$ROOT/variants/$VARIANT/Code"
OUT="$RESULTS_ROOT/$VARIANT"
ENV="$BASE/Codes/AV2/envs/sharp_av2"
DATA="$BASE/Datasets/AV2/sharp_processed"
PATCH={runtime_patch}

mkdir -p "$OUT/run" "$OUT/eval"
source "$BASE/Codes/AV2/miniforge3/etc/profile.d/conda.sh"
conda activate "$ENV"

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONPATH="$PATCH:$CODE:${{PYTHONPATH:-}}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export NO_COLOR=1
export RICH_NO_COLOR=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CPU_COUNT=$(nproc)
DEFAULT_WORKERS=$(( (CPU_COUNT - 3) / 3 ))
(( DEFAULT_WORKERS < 4 )) && DEFAULT_WORKERS=4
(( DEFAULT_WORKERS > 8 )) && DEFAULT_WORKERS=8
WORKERS_PER_RANK=${{WORKERS_PER_RANK:-$DEFAULT_WORKERS}}
BATCH_PER_GPU=${{BATCH_PER_GPU:-8}}
EPOCHS=${{EPOCHS:-80}}
GLOBAL_BATCH=$(( BATCH_PER_GPU * 3 ))

echo "VARIANT=$VARIANT"
echo "CODE=$CODE"
echo "RESULTS=$OUT"
echo "GPUS=3 BATCH_PER_GPU=$BATCH_PER_GPU GLOBAL_BATCH=$GLOBAL_BATCH"
echo "WORKERS_PER_RANK=$WORKERS_PER_RANK TOTAL_WORKERS=$((WORKERS_PER_RANK * 3))"
nvidia-smi

if [[ -f "$OUT/COMPLETE" && -f "$OUT/metrics.json" ]]; then
  echo "$VARIANT is already complete; skipping training."
  exit 0
fi

cd "$CODE"
ARGS=(
  seed=2333
  gpus=3
  epochs="$EPOCHS"
  batch_size="$BATCH_PER_GPU"
  output_dir="$OUT/run"
  datamodule.pl_module.data_root="$DATA"
  datamodule.pl_module.num_workers="$WORKERS_PER_RANK"
  model.pl_module.optim.lr=0.0001
  model.pl_module.optim.min_lr=0.00001
  model.pl_module.optim.warmup_ratio=0.1625
  trainer.devices=3
  trainer.strategy=ddp_find_unused_parameters_false
  trainer.sync_batchnorm=true
  +trainer.num_sanity_val_steps=0
  callbacks.0.save_top_k=3
  callbacks.0.monitor=minADE6
)

LAST="$OUT/run/checkpoints/last.ckpt"
if [[ -f "$LAST" ]]; then
  echo "Resuming $VARIANT from $LAST"
  ARGS+=(checkpoint="$LAST")
fi

set +e
python -u train.py "${{ARGS[@]}}" 2>&1 | tee -a "$OUT/train.log"
TRAIN_RC=${{PIPESTATUS[0]}}
set -e
if (( TRAIN_RC != 0 )); then
  echo "Training failed for $VARIANT with exit code $TRAIN_RC" >&2
  exit "$TRAIN_RC"
fi

BEST=$(python "$ROOT/find_best_checkpoint.py" "$OUT/run/checkpoints")
echo "$BEST" > "$OUT/best_checkpoint.txt"
export SHARP_METRICS_JSON="$OUT/metrics.json"

python -u eval_to_json.py \
  seed=2333 \
  gpus=3 \
  batch_size="$BATCH_PER_GPU" \
  output_dir="$OUT/eval" \
  checkpoint="$BEST" \
  datamodule.pl_module.data_root="$DATA" \
  datamodule.pl_module.num_workers="$WORKERS_PER_RANK" \
  2>&1 | tee "$OUT/eval.log"

date --iso-8601=seconds > "$OUT/COMPLETE"
echo "Completed $VARIANT"
''')
    runner.chmod(0o755)

    run_all = code_root / "run_all_attention_experiments.sh"
    run_all.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
ROOT={code_root}
RESULTS_ROOT={results_root}
for variant in baseline_mha qknorm talking_heads qknorm_talking_heads; do
  "$ROOT/run_variant.sh" "$variant"
done
"{ENV_PYTHON}" "$ROOT/compare_results.py" "$RESULTS_ROOT"
''')
    run_all.chmod(0o755)

    compare = code_root / "compare_now.sh"
    compare.write_text(f'''#!/usr/bin/env bash
set -euo pipefail
"{ENV_PYTHON}" "{code_root / 'compare_results.py'}" "{results_root}"
''')
    compare.chmod(0o755)


def main() -> int:
    for path, description in (
        (SOURCE / "src/model/sharp.py", "bundled pristine SHARP source"),
        (ENV_PYTHON, "Lab 3 SHARP Python environment"),
        (CONDA_SH, "Lab 3 conda initialization"),
        (DATA / "train", "preprocessed AV2 training data"),
        (DATA / "val", "preprocessed AV2 validation data"),
    ):
        require(path, description)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    code_root = BASE / "Codes" / f"SHARP_ATTENTION_ABLATION_{stamp}"
    results_root = BASE / "Results" / f"SHARP_ATTENTION_ABLATION_{stamp}"
    code_root.mkdir(parents=True)
    results_root.mkdir(parents=True)
    runtime_patch = write_runtime_patch(code_root)

    manifest = {
        "created": stamp,
        "source": "Bundled pristine SHARP source",
        "source_sharp_sha256": sha256(SOURCE / "src/model/sharp.py"),
        "variants": list(VARIANTS),
        "dataset": str(DATA),
        "environment": str(ENV_PYTHON.parent.parent),
        "epochs": 80,
        "paper_learning_rate": 1e-4,
        "paper_min_learning_rate": 1e-5,
        "paper_warmup_epochs": 13,
        "gpus": 3,
        "batch_per_gpu": 8,
        "global_batch": 24,
        "note": "Paper global batch is 32; three equal DDP ranks cannot reproduce 32 exactly.",
    }

    for variant in VARIANTS:
        code = code_root / "variants" / variant / "Code"
        shutil.copytree(SOURCE, code)
        patch_checkpoint_config(code)
        shutil.copy2(PACKAGE / "eval_to_json.py", code / "eval_to_json.py")
        if variant != "baseline_mha":
            patch_attention(code, variant)

        model_config = (code / "conf/model/Sharp_av2.yaml").read_text()
        if "encoder_depth: 4" not in model_config or "drop_path: 0.2" not in model_config:
            raise RuntimeError(f"{variant} does not preserve the original SHARP encoder configuration")

        for path in code.rglob("*.py"):
            py_compile.compile(str(path), doraise=True)

    shutil.copy2(PACKAGE / "compare_results.py", code_root / "compare_results.py")
    shutil.copy2(PACKAGE / "find_best_checkpoint.py", code_root / "find_best_checkpoint.py")
    (code_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (code_root / "RESULTS_ROOT.txt").write_text(str(results_root) + "\n")
    write_run_scripts(code_root, results_root, runtime_patch)

    (BASE / "Codes/LATEST_SHARP_ATTENTION_ABLATION.txt").write_text(str(code_root) + "\n")
    (BASE / "Results/LATEST_SHARP_ATTENTION_ABLATION.txt").write_text(str(results_root) + "\n")

    print("SETUP_COMPLETE")
    print(f"CODE_ROOT={code_root}")
    print(f"RESULTS_ROOT={results_root}")
    print(f"RUN_ALL={code_root / 'run_all_attention_experiments.sh'}")
    print(f"RUN_ONE={code_root / 'run_variant.sh'} VARIANT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
