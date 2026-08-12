#!/usr/bin/env bash

# Recover test 5 after its Boolean target-context write triggered a CUDA fault.
# Existing variants, logs, checkpoints, and failed-attempt evidence are retained.

set -uo pipefail

BASE=/home/server00/M
PACKAGE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_POINTER="$BASE/Codes/LATEST_SHARP_AV2_20EPOCH_10TEST_V5.txt"
RESULTS_POINTER="$BASE/Results/LATEST_SHARP_AV2_20EPOCH_10TEST.txt"
PYTHON_BIN="$BASE/Codes/envs/sharp/bin/python"
PREPARE="$PACKAGE_DIR/prepare_lab2_nvidia_58015903_userspace.sh"
STATUS=1

if [[ ! -s "$EXPERIMENT_POINTER" || ! -s "$RESULTS_POINTER" ]]; then
  echo "FATAL: current experiment or results pointer is missing."
  exit 1
fi

EXPERIMENT=$(tr -d '\r\n' < "$EXPERIMENT_POINTER")
RESULTS=$(tr -d '\r\n' < "$RESULTS_POINTER")
RUNNER="$EXPERIMENT/run_10_test_suite.sh"
STAMP=$(date +%Y%m%d-%H%M%S)
RECOVERY="$RESULTS/recovery/checked_target_remap_$STAMP"

if [[ ! -x "$PYTHON_BIN" || ! -f "$RUNNER" || ! -d "$RESULTS" ]]; then
  echo "FATAL: current Python, runner, or results directory is unavailable."
  exit 1
fi

RUNNER_RESULTS=$(grep -F 'RESULTS_ROOT=' "$RUNNER" | head -1 | cut -d= -f2- | tr -d '"')
if [[ "$RUNNER_RESULTS" != "$RESULTS" ]]; then
  echo "FATAL: experiment and result pointers do not describe the same run."
  exit 1
fi

ACTIVE=$(
  python3 - "$EXPERIMENT" "$RESULTS" <<'PY'
import pathlib
import sys

needles = tuple(sys.argv[1:])
matches = []
for process in pathlib.Path("/proc").glob("[0-9]*"):
    try:
        command = process.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
    except OSError:
        continue
    if any(needle in command for needle in needles) and (
        "train.py" in command or "run_10_test_suite.sh" in command
    ):
        matches.append(f"{process.name} {command.strip()}")
print("\n".join(matches))
PY
)
if [[ -n "$ACTIVE" ]]; then
  echo "FATAL: this suite still has active processes; no files were changed."
  echo "$ACTIVE"
  exit 1
fi

for variant in \
  01_baseline \
  02_confidence_gated_memory \
  03_cross_window_consistency \
  04_learned_temporal_pool
do
  if [[ ! -s "$RESULTS/$variant/COMPLETED" ]]; then
    echo "FATAL: completed marker is missing for $variant."
    exit 1
  fi
done

mkdir -p "$RECOVERY"
cp -p "$RUNNER" "$RECOVERY/run_10_test_suite.sh.before_recovery"
[[ -f "$RESULTS/suite.log" ]] && cp --reflink=auto \
  "$RESULTS/suite.log" "$RECOVERY/suite.log.before_recovery"
[[ -f "$RESULTS/05_uncertainty_target_context/full_run.log" ]] && \
  cp --reflink=auto "$RESULTS/05_uncertainty_target_context/full_run.log" \
  "$RECOVERY/test5_failed_attempts.log"
find "$RESULTS" -name '*.ckpt' \
  -printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %p\n' | sort \
  > "$RECOVERY/CHECKPOINTS_BEFORE_RECOVERY.txt"

"$PYTHON_BIN" "$PACKAGE_DIR/patch_uncertainty_context_ddp.py" \
  "$EXPERIMENT" "$RECOVERY"
STATUS=$?
if (( STATUS != 0 )); then
  echo "FATAL: uncertainty DDP patch validation failed."
  exit "$STATUS"
fi

"$PYTHON_BIN" "$PACKAGE_DIR/patch_uncertainty_target_context_cuda.py" \
  "$EXPERIMENT" "$RECOVERY"
STATUS=$?
if (( STATUS != 0 )); then
  echo "FATAL: checked target-remap patch failed."
  exit "$STATUS"
fi

CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" \
  "$PACKAGE_DIR/validate_uncertainty_ddp_bridge.py" "$EXPERIMENT" \
  | tee "$RECOVERY/DDP_BRIDGE_AUDIT.txt"
STATUS=${PIPESTATUS[0]}
if (( STATUS != 0 )); then
  echo "FATAL: DDP bridge audit failed."
  exit "$STATUS"
fi

CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" \
  "$PACKAGE_DIR/validate_uncertainty_target_context_cuda.py" "$EXPERIMENT" \
  | tee "$RECOVERY/TARGET_REMAP_AUDIT.txt"
STATUS=${PIPESTATUS[0]}
if (( STATUS != 0 )); then
  echo "FATAL: target-remap equivalence audit failed."
  exit "$STATUS"
fi

LOADED_NVIDIA=$(cat /sys/module/nvidia/version 2>/dev/null || true)
if [[ "$LOADED_NVIDIA" == "580.159.03" ]]; then
  bash "$PREPARE"
  STATUS=$?
  if (( STATUS == 0 )); then
    export LD_LIBRARY_PATH="$BASE/Codes/NVIDIA_USERSPACE_580.159.03/runtime/lib:${LD_LIBRARY_PATH:-}"
    export PATH="$BASE/Codes/NVIDIA_USERSPACE_580.159.03/runtime/bin:$PATH"
  fi
else
  nvidia-smi -L
  STATUS=$?
fi
if (( STATUS != 0 )); then
  echo "FATAL: compatible GPU runtime validation failed."
  exit "$STATUS"
fi

SMOKE_ROOT="$RECOVERY/real_data_cuda_smoke"
mkdir -p "$SMOKE_ROOT/run"
SMOKE_LOG="$RECOVERY/REAL_DATA_CUDA_SMOKE.log"

smoke_log_is_valid() {
  local log=$1
  [[ -s "$log" ]] || return 1
  grep -aFq 'Trainer.fit` stopped: `max_steps=256` reached.' "$log" || return 1
  grep -aFq '256/256' "$log" || return 1
  ! grep -aEqi \
    'illegal memory access|CUDA error|ZeroDivisionError|Target-context remap count mismatch|Expected to have finished reduction|out of memory|AssertionError|NCCL WARN' \
    "$log"
}

PREVIOUS_SMOKE=$(
  find "$RESULTS/recovery" -type f -name REAL_DATA_CUDA_SMOKE.log \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
)
SMOKE_STATUS=1
SMOKE_REUSED=false

if [[ -n "$PREVIOUS_SMOKE" ]] && smoke_log_is_valid "$PREVIOUS_SMOKE"; then
  cp --reflink=auto "$PREVIOUS_SMOKE" "$SMOKE_LOG"
  printf 'source=%s\nreused=%s\n' "$PREVIOUS_SMOKE" \
    "$(date --iso-8601=seconds)" > "$RECOVERY/REUSED_CUDA_SMOKE.txt"
  SMOKE_STATUS=0
  SMOKE_REUSED=true
  echo "REUSING_VALID_256_BATCH_CUDA_SMOKE=$PREVIOUS_SMOKE"
fi

cd "$EXPERIMENT/Code" || exit 1
export SHARP_EXPERIMENT_VARIANT=uncertainty_target_context
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

if [[ "$SMOKE_REUSED" != true ]]; then
  echo "Running 256 real-data batches past the previous batch-148 failure point..."
  export CUDA_LAUNCH_BLOCKING=1

  set +e
  timeout --signal=INT --kill-after=60s 30m \
    "$PYTHON_BIN" train.py \
    seed=2333 \
    gpus=4 \
    epochs=20 \
    batch_size=8 \
    "output_dir=$SMOKE_ROOT/run" \
    "datamodule.pl_module.data_root=$BASE/Datasets/AV2/sharp_processed" \
    datamodule.pl_module.num_workers=6 \
    model.pl_module.optim.lr=0.0001 \
    model.pl_module.optim.min_lr=0.00001 \
    model.pl_module.optim.weight_decay=0.01 \
    model.pl_module.optim.warmup_ratio=0.65 \
    trainer.devices=4 \
    trainer.strategy=ddp_find_unused_parameters_false \
    trainer.gradient_clip_val=5 \
    trainer.gradient_clip_algorithm=norm \
    trainer.sync_batchnorm=true \
    trainer.num_sanity_val_steps=0 \
    +trainer.limit_train_batches=256 \
    +trainer.limit_val_batches=1 \
    +trainer.max_steps=256 \
    callbacks.0.save_top_k=0 \
    callbacks.0.save_last=false \
    callbacks.0.every_n_epochs=1 \
    checkpoint=null \
    2>&1 | tee "$SMOKE_LOG"
  SMOKE_STATUS=${PIPESTATUS[0]}
  set +e

  unset CUDA_LAUNCH_BLOCKING
fi

if smoke_log_is_valid "$SMOKE_LOG"; then
  if (( SMOKE_STATUS != 0 )); then
    if grep -aFq 'RuntimeError: No best checkpoint was recorded' "$SMOKE_LOG"; then
      echo "SMOKE_POSTFIT_CHECKPOINT_ERROR_ACCEPTED=True"
      SMOKE_STATUS=0
    fi
  fi
fi

if (( SMOKE_STATUS != 0 )) || ! smoke_log_is_valid "$SMOKE_LOG"; then
  echo "FATAL: the 256-batch CUDA smoke test was not valid."
  echo "The ten-test suite was not restarted."
  exit 1
fi

{
  echo "recovery_validated=$(date --iso-8601=seconds)"
  echo "experiment=$EXPERIMENT"
  echo "results=$RESULTS"
  echo "completed_variants_preserved=01,02,03,04"
  echo "restart_variant=05_uncertainty_target_context"
  echo "continue_variants=06,07,08,09,10"
  echo "target_remap_forward_equivalent=true"
  echo "target_remap_gradient_equivalent=true"
  echo "smoke_training_batches=256"
  echo "smoke_reused=$SMOKE_REUSED"
  echo "training_hyperparameters_changed=false"
} > "$RECOVERY/RECOVERY_VALIDATED.txt"

echo "CUDA_TARGET_REMAP_RECOVERY_VALIDATED=True"
echo "Tests 1-4 will be skipped; tests 5-10 will run sequentially."
echo "Recovery archive: $RECOVERY"

bash "$RUNNER"
STATUS=$?

echo
echo "Recovered suite foreground status: $STATUS"
echo "Previous results were not deleted."
echo "Terminal remains open."
exit "$STATUS"
