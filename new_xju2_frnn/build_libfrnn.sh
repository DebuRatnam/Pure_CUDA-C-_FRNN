#!/bin/bash
# Build xju2/libFRNN (_frnn.so) for use in benchmark_master.py.
# Run this from a GPU interactive node:
#   srun -C gpu -q interactive -N 1 -G 1 -c 32 -t 01:00:00 -A m3443 --pty /bin/bash -l
#   module load pytorch/2.8.0
#   bash new_xju2_frnn/build_libfrnn.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/libFRNN/build"

echo "=== Building xju2/libFRNN ==="
echo "Source: $SCRIPT_DIR/libFRNN"
echo "Build:  $BUILD_DIR"
echo "Output: $SCRIPT_DIR/_frnn*.so"

# pybind11 cmake config dir
PYBIND11_DIR=$(python3 -m pybind11 --cmakedir 2>/dev/null) || {
    pip install --user pybind11
    PYBIND11_DIR=$(python3 -m pybind11 --cmakedir)
}
echo "pybind11 cmake dir: $PYBIND11_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DFRNN_BUILD_PYTHON=ON \
  -DFRNN_BUILD_TESTS=OFF \
  -DFRNN_BUILD_BENCHMARKS=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DCMAKE_CUDA_ARCHITECTURES=80

make _frnn -j"$(nproc)"

# Copy the .so next to this script so benchmark_master.py can find it
# via sys.path.insert(0, new_xju2_frnn_dir).
SO=$(find "$BUILD_DIR" -name "_frnn*.so" | head -1)
if [ -z "$SO" ]; then
    echo "ERROR: _frnn*.so not found in $BUILD_DIR" >&2
    exit 1
fi
cp "$SO" "$SCRIPT_DIR/"
echo ""
echo "=== Build complete ==="
echo "Installed: $SCRIPT_DIR/$(basename "$SO")"
echo ""
echo "Verify with:"
echo "  python3 -c \"import sys; sys.path.insert(0, '$SCRIPT_DIR'); import _frnn; print('OK', _frnn.__version__)\""
