# FRNN-master

A pure-CUDA **fixed-radius nearest-neighbor (FRNN)** engine for *N*-dimensional point
clouds, built to beat **FAISS**, **FlashLib (FlashML)**, and the original
**xju2 / lxxue FRNN** in wall-clock latency across `D ∈ {2,3,4,8,16}` × `N ∈ {1K,10K,100K}`.

The engine auto-dispatches between two GPU paths:

- **Uniform grid** (spatial hash) — used when `3^D < total_cells`. Sub-quadratic; dominates at low/mid `D`.
- **Tiled brute-force** — used when `3^D ≥ total_cells` or `res ≤ 1` (i.e. high `D`). `float4`-vectorized distance kernel.

All point data is stored **Structure-of-Arrays** (`p[d*P + i]`) for coalesced warp access.
Engine hard limits: `K ≤ 128`, `D ≤ 128`, `ceil(1/R)^D ≤ 1,000,000`.

---

## Environment

| | |
|---|---|
| Machine | Perlmutter (NERSC) |
| GPU | NVIDIA A100 (`sm_80`) |
| Module | `pytorch/2.8.0` → Python 3.12, CUDA 12.9 (torch used only for FAISS and xju2 baselines) |
| Runtime | `cupy-cuda12x`, `faiss-gpu`, `nvidia-ml-py` — **no PyTorch required for the FRNN engine** |

> The `frnn_cuda` extension (pure pybind11, no torch) is the primary engine interface.
> PyTorch is only needed at runtime for the FAISS and xju2 baseline blocks in `benchmark_master.py`.

---

## 1. Get a GPU node

```bash
srun -C gpu -q interactive -N 1 -G 1 -c 32 -t 02:00:00 -A m3443 --pty /bin/bash -l
```

Do everything below **inside** this shell — `nvcc` and the A100 are required for builds and
benchmarks. (Change `-A m3443` to your own allocation if different.)

> ⚠️ The GPU may be shared if you are not in an exclusive allocation. Check with
> `nvidia-smi`; if another process is at high utilization, your latency numbers will be
> inflated. Run `nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv`
> to see who's on the device.

## 2. Load the environment

```bash
module load pytorch/2.8.0
cd /global/u1/d/dratnam/FRNN-master
export LD_LIBRARY_PATH=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
```

Install runtime packages once (login node has outbound network; compute node does not):

```bash
pip install --user cupy-cuda12x faiss-gpu nvidia-ml-py
```

## 3. Verify the engine loads

`frnn_cuda` (the primary engine) is already compiled and present in the repo root as
`frnn_cuda.cpython-312-*.so`. It is a pure pybind11 extension with no PyTorch dependency —
**no rebuild needed unless `python_interface/frnn_engine.cu` or `frnn_engine.h` changes.**

```bash
python3 -c "import frnn_cuda; from frnn_cupy import FRNNCuPy; import cupy as cp; print('OK')"
```

## 4. (Optional) Build the xju2 baseline

The benchmark compares against the original lxxue/FRNN in `xju2_frnn/`. It needs two
extensions built for **this** Python, with a **GCC ≥ 9 host compiler** (the system
`/usr/bin/c++` is GCC 7.5 and will fail libtorch's version check):

```bash
export TORCH_CUDA_ARCH_LIST=8.0
export CC=/opt/cray/pe/gcc-native/13/bin/gcc
export CXX=/opt/cray/pe/gcc-native/13/bin/g++

cd xju2_frnn/prefix_sum && python3 setup.py build_ext --inplace && cd -
cd xju2_frnn/FRNN       && python3 setup.py build_ext --inplace && cd -
```

The benchmark adds `xju2_frnn/FRNN` and `xju2_frnn/prefix_sum` to `sys.path` ahead of cwd, so
`import frnn` resolves to this package rather than the repo's local `./frnn/` source dir.
xju2 requires PyTorch tensors internally; if torch is absent the `xju2` column is `None`.

If you skip this step, the `xfrnn_ms` column is simply `None` everywhere — the rest of the
benchmark still runs.

## 4b. Set up FlashLib (`flash_lib_knn`)

`benchmark_master.py` times **FlashLib** (FlashML's fused brute-force exact top-K KNN,
`flash_knn`) as a baseline. It lives in `flash_lib_knn/` and the benchmark passes it a CuPy
array via DLPack interop. Install on a **login node** (compute nodes have no outbound
internet); the Triton / CuteDSL kernels compile JIT on first GPU call:

```bash
# 1. Clone the source into the existing flash_lib_knn/ folder (must be empty):
git clone https://github.com/FlashML-org/flashlib.git flash_lib_knn

# 2. Install FlashLib and its runtime deps:
pip install --user --no-deps -e flash_lib_knn
pip install --user "triton>=3.6" nvidia-cutlass-dsl
```

If FlashLib is not installed the `flash_ms` column is reported as `None` (skip-logged
`[FlashLib] ...`) and the rest of the benchmark still runs.

## 5. Run the benchmark

Run **from the repo root** with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python3 Tests/benchmark_master.py 2>&1 | tee benchmark_run.log
grep REGRESSION benchmark_run.log          # any cell where FRNN lost a baseline
```

Sweeps `D ∈ {3,16}` × `N ∈ {100K,200K,300K,400K,500K}` (10 cells by default), timing
FRNN, FAISS, FlashLib, and xju2 in-process on GPU-resident CuPy arrays. A 3-second GPU
warm-up runs first. Override the sweep at runtime:

```bash
D_SWEEP=3,4,8,16 PYTHONPATH=. python3 Tests/benchmark_master.py
```

FRNN uses `frnn_cupy.FRNNCuPy.search_projected()`: a CuPy wrapper around `frnn_cuda` with
zero host↔device copies. AoS→SoA transpose is done on the GPU; results are returned as
`(N, K)` CuPy arrays. The projection two-stage path (PCA→3D candidate search→full-D verify)
engages automatically when the top-3 principal components capture ≥ 90% of variance.

All baselines are timed on GPU-resident data (no H2D copies in the timed loop): FRNN and
FlashLib use CuPy arrays; FAISS and xju2 use pre-transferred CUDA tensors.

## 5b. Run the benchmark on low-rank data (projection path)

Set `LOWRANK=<intrinsic_dim>` to generate points near a low-dimensional manifold embedded in
the ambient D-space. The projection two-stage path (PCA→3D candidate search→full-D verify)
engages automatically when the top-3 principal components capture ≥ 90% of variance — exactly
the condition that holds for low-rank data. Baselines (FAISS, FlashLib, xju2) still run on
the original D-dimensional points; only FRNN uses projection.

```bash
LOWRANK=3 PYTHONPATH=. python3 Tests/benchmark_master.py 2>&1 | tee benchmark_run_lowrank.log
grep REGRESSION benchmark_run_lowrank.log
```

This overwrites `benchmark_results.json` and `benchmark_comparison.png` with the low-rank
results. On low-rank data the projection two-stage delivers a **4.7–7.1× speedup** over the
plain brute-force path at D=16 with exact recall (see `projection_comparison.json`).

To compare projection vs no-projection directly (rather than vs other libraries), run the
dedicated head-to-head script:

```bash
PYTHONPATH=. python3 compare_projection.py 2>&1 | tee projection_run.log
```

This writes `projection_comparison.json` with per-stage breakdowns (`project_ms`,
`stage1_ms`, `verify_ms`), recall, and speedup for both uniform and low-rank regimes.

## 6. Validate correctness

Checks FRNN returns the *right* neighbors, against xju2 and a float64 brute-force truth
(exits 0 = all pass, for CI):

```bash
PYTHONPATH=. python3 Tests/validate_correctness.py
```

Confirms FRNN returns the exact **K-nearest** points within the radius (matches the
brute-force oracle, and matches xju2 wherever the answer is unambiguous). On dense queries
(>K points in radius) FRNN returns the nearest K while xju2 returns any K — a semantic
difference the check accounts for, not a bug.

---

## Outputs

| File | Contents |
|---|---|
| `benchmark_results.json` | per-cell `{R, latency_ms (FRNN), peak_mb, faiss_ms, flash_ms, xfrnn_ms}` |
| `benchmark_run.log` | full console log; `!! REGRESSION` lines mark FRNN losses |
| `benchmark_comparison.png` | latency-vs-N plot per D dimension (log scale) |

---

## Repository layout

```
python_interface/
  frnn_engine.cu/.h    # pure pybind11 engine (CPU + raw-device-ptr GPU search paths)
frnn/csrc/
  grid/                # insert_points.cu, find_nbrs.cu — uniform-grid kernels (SoA)
  no_grid_frnn/        # no_grid_frnn.cu — float4-vectorized tiled brute-force (SoA)
  projection/          # project.cu (PCA->3D), verify.cu (fused full-D verify)
frnn_cupy.py              # CuPy interface to frnn_cuda: zero-copy search() + search_projected()
projection_frnn.py        # two-stage projection dispatcher (pure CuPy, no torch)
Tests/
  benchmark_master.py     # the sweep (FRNN vs FAISS vs FlashLib vs xju2)
  validate_correctness.py # FRNN vs xju2 vs float64 brute-force oracle
xju2_frnn/             # original lxxue/FRNN baseline (FRNN/ + prefix_sum/)
flash_lib_knn/         # FlashLib (FlashML) baseline — git clone + pip install -e (step 4b)
```

---

## Rebuild triggers

Rebuild the xju2 extensions (step 4) when you switch the `pytorch` module (different torch/Python ABI).

Rebuild `frnn_cuda` (the primary engine) when you edit:

- `python_interface/frnn_engine.cu` or `python_interface/frnn_engine.h`, or
- any `.cu` / `.h` under `frnn/csrc/`.

`frnn_cupy.py` and `projection_frnn.py` are pure Python — no rebuild needed after edits.
