#!/usr/bin/env bash

# Recover the existing Lab 2 suite after the uncertainty-context DDP failure.
# Completed variants and all existing logs/checkpoints are preserved.

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
RECOVERY="$RESULTS/recovery/uncertainty_ddp_bridge_$STAMP"

if [[ ! -x "$PYTHON_BIN" || ! -f "$RUNNER" || ! -d "$RESULTS" ]]; then
  echo "FATAL: current Python, runner, or results directory is unavailable."
  echo "Experiment: $EXPERIMENT"
  echo "Results:    $RESULTS"
  exit 1
fi

RUNNER_RESULTS=$(grep -F 'RESULTS_ROOT=' "$RUNNER" | head -1 | cut -d= -f2- | tr -d '"')
if [[ "$RUNNER_RESULTS" != "$RESULTS" ]]; then
  echo "FATAL: experiment and result pointers do not describe the same run."
  echo "Runner results:  $RUNNER_RESULTS"
  echo "Pointer results: $RESULTS"
  exit 1
fi

ACTIVE=$(
  python3 - "$EXPERIMENT" "$RESULTS" <<'PY'
import pathlib
import sys

needles = tuple(str(pathlib.Path(value).resolve()) for value in sys.argv[1:])
matches = []
for process in pathlib.Path("/proc").glob("[0-9]*"):
    try:
        command = process.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
    except OSError:
        continue
    if ("train.py" in command or "run_10_test_suite.sh" in command) and any(
        needle in command for needle in needles
    ):
        matches.append(f"{process.name} {command.strip()}")
print("\n".join(matches))
PY
)
if [[ -n "$ACTIVE" ]]; then
  echo "FATAL: the suite is already active; no recovery action was taken."
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
    echo "No model or result file was changed."
    exit 1
  fi
done

mkdir -p "$RECOVERY"
cp -p "$RUNNER" "$RECOVERY/run_10_test_suite.sh.before_recovery"
[[ -f "$RESULTS/suite.log" ]] && cp --reflink=auto \
  "$RESULTS/suite.log" "$RECOVERY/suite.log.before_recovery"
[[ -f "$RESULTS/05_uncertainty_target_context/full_run.log" ]] && \
  cp --reflink=auto "$RESULTS/05_uncertainty_target_context/full_run.log" \
  "$RECOVERY/test5_failed_attempt.log"
find "$RESULTS" -name '*.ckpt' \
  -printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %p\n' | sort \
  > "$RECOVERY/CHECKPOINTS_BEFORE_RECOVERY.txt"

"$PYTHON_BIN" "$PACKAGE_DIR/patch_uncertainty_context_ddp.py" \
  "$EXPERIMENT" "$RECOVERY"
STATUS=$?
if (( STATUS != 0 )); then
  echo "FATAL: model recovery patch failed."
  echo "Previous results remain unchanged."
  exit "$STATUS"
fi

"$PYTHON_BIN" "$PACKAGE_DIR/validate_uncertainty_ddp_bridge.py" \
  "$EXPERIMENT" | tee "$RECOVERY/BRIDGE_AUDIT.txt"
STATUS=${PIPESTATUS[0]}
if (( STATUS != 0 )); then
  echo "FATAL: DDP bridge audit failed; training was not started."
  exit "$STATUS"
fi

{
  echo "recovery_started=$(date --iso-8601=seconds)"
  echo "experiment=$EXPERIMENT"
  echo "results=$RESULTS"
  echo "completed_variants_preserved=01,02,03,04"
  echo "restart_variant=05_uncertainty_target_context"
  echo "continue_variants=06,07,08,09,10"
  echo "training_configuration_changed=false"
  echo "forward_values_changed=false"
  echo "ddp_strategy=ddp_find_unused_parameters_false"
} > "$RECOVERY/RECOVERY_MANIFEST.txt"

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

if (( STATUS == 0 )); then
  echo "RECOVERY_VALIDATED=True"
  echo "Completed tests 1-4 will be skipped."
  echo "Test 5 will restart from epoch 0; tests 6-10 will follow automatically."
  echo "Recovery archive: $RECOVERY"
  cd "$EXPERIMENT/Code" || STATUS=1
fi
if (( STATUS == 0 )); then
  bash "$RUNNER"
  STATUS=$?
fi

echo
echo "Recovered suite foreground status: $STATUS"
echo "Previous results were not deleted."
echo "Terminal remains open."
exit "$STATUS"
