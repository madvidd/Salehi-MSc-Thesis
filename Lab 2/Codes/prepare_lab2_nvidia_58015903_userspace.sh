#!/usr/bin/env bash
set -euo pipefail

BASE=/home/server00/M
DRIVER_VERSION=580.159.03
ARCHIVE_BASE=https://download.nvidia.com/XFree86/Linux-x86_64/580.159.03
COMPAT_ROOT="$BASE/Codes/NVIDIA_USERSPACE_$DRIVER_VERSION"
DOWNLOAD_DIR="$COMPAT_ROOT/packages"
RUNTIME_DIR="$COMPAT_ROOT/runtime"
LIB_DIR="$RUNTIME_DIR/lib"
BIN_DIR="$RUNTIME_DIR/bin"
MARKER="$COMPAT_ROOT/READY.txt"

RUNFILE_NAME="NVIDIA-Linux-x86_64-$DRIVER_VERSION-no-compat32.run"
CHECKSUM_NAME="$RUNFILE_NAME.sha256sum"
RUNFILE="$DOWNLOAD_DIR/$RUNFILE_NAME"
CHECKSUM_FILE="$DOWNLOAD_DIR/$CHECKSUM_NAME"

LOADED_DRIVER=$(cat /sys/module/nvidia/version 2>/dev/null || true)
if [ "$LOADED_DRIVER" != "$DRIVER_VERSION" ]; then
  echo "ERROR: loaded NVIDIA module is '$LOADED_DRIVER', expected '$DRIVER_VERSION'."
  echo "This compatibility package is restricted to $DRIVER_VERSION."
  exit 1
fi

command -v curl >/dev/null 2>&1 || {
  echo "ERROR: curl is required."
  exit 1
}
command -v sha256sum >/dev/null 2>&1 || {
  echo "ERROR: sha256sum is required."
  exit 1
}
test -x "$BASE/Codes/envs/sharp/bin/python"

mkdir -p "$DOWNLOAD_DIR" "$RUNTIME_DIR" "$LIB_DIR" "$BIN_DIR"

if [ ! -s "$RUNFILE" ]; then
  echo "Downloading the official NVIDIA $DRIVER_VERSION user-space archive."
  curl --fail --location --retry 3 --retry-delay 3 \
    "$ARCHIVE_BASE/$RUNFILE_NAME" \
    --output "$RUNFILE"
else
  echo "Reusing $RUNFILE"
fi

curl --fail --location --retry 3 --retry-delay 3 \
  "$ARCHIVE_BASE/$CHECKSUM_NAME" \
  --output "$CHECKSUM_FILE"

EXPECTED_SHA256=$(awk 'NR == 1 {print $1}' "$CHECKSUM_FILE")
ACTUAL_SHA256=$(sha256sum "$RUNFILE" | awk '{print $1}')
if [ -z "$EXPECTED_SHA256" ] || [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  echo "ERROR: NVIDIA archive checksum verification failed."
  exit 1
fi

EXTRACT_DIR=$(find "$COMPAT_ROOT" -maxdepth 1 -type d \
  -name "NVIDIA-Linux-x86_64-$DRIVER_VERSION*" -print -quit)
if [ -z "$EXTRACT_DIR" ]; then
  echo "Extracting the NVIDIA archive without installing it."
  (
    cd "$COMPAT_ROOT"
    sh "$RUNFILE" --extract-only
  )
  EXTRACT_DIR=$(find "$COMPAT_ROOT" -maxdepth 1 -type d \
    -name "NVIDIA-Linux-x86_64-$DRIVER_VERSION*" -print -quit)
fi

test -n "$EXTRACT_DIR"
CUDA_LIBRARY=$(find "$EXTRACT_DIR" -type f \
  -name "libcuda.so.$DRIVER_VERSION" -print -quit)
NVML_LIBRARY=$(find "$EXTRACT_DIR" -type f \
  -name "libnvidia-ml.so.$DRIVER_VERSION" -print -quit)
NVIDIA_SMI=$(find "$EXTRACT_DIR" -type f -name nvidia-smi -print -quit)

test -s "$CUDA_LIBRARY"
test -s "$NVML_LIBRARY"
test -x "$NVIDIA_SMI"

ln -sfn "$CUDA_LIBRARY" "$LIB_DIR/libcuda.so.$DRIVER_VERSION"
ln -sfn "libcuda.so.$DRIVER_VERSION" "$LIB_DIR/libcuda.so.1"
ln -sfn "libcuda.so.1" "$LIB_DIR/libcuda.so"
ln -sfn "$NVML_LIBRARY" "$LIB_DIR/libnvidia-ml.so.$DRIVER_VERSION"
ln -sfn "libnvidia-ml.so.$DRIVER_VERSION" "$LIB_DIR/libnvidia-ml.so.1"
ln -sfn "libnvidia-ml.so.1" "$LIB_DIR/libnvidia-ml.so"
ln -sfn "$NVIDIA_SMI" "$BIN_DIR/nvidia-smi"

export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"
export PATH="$BIN_DIR:$PATH"

echo "Checking isolated NVIDIA $DRIVER_VERSION user-space libraries."
nvidia-smi -L

"$BASE/Codes/envs/sharp/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable with the isolated driver libraries")
if torch.cuda.device_count() != 4:
    raise SystemExit(f"Expected four GPUs, found {torch.cuda.device_count()}")

for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        value = torch.ones(1024, device=f"cuda:{index}")
        if value.sum().item() != 1024:
            raise SystemExit(f"CUDA allocation check failed on GPU {index}")
        print(index, torch.cuda.get_device_name(index), "CUDA_OK")
PY

cat > "$MARKER" <<EOF
driver_version=$DRIVER_VERSION
library_directory=$LIB_DIR
binary_directory=$BIN_DIR
source=$ARCHIVE_BASE
archive_sha256=$ACTUAL_SHA256
EOF

echo "NVIDIA_USERSPACE_COMPAT_READY=$COMPAT_ROOT"
