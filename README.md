# FRNN-master

A pure-CUDA **fixed-radius nearest-neighbor (FRNN)** engine for *N*-dimensional point
clouds, built to beat **FAISS**, **FlashLib (FlashML)**, and the original
**xju2 / lxxue FRNN** in wall-clock latency across `D ∈ {2,3,4,8,16}` × `N ∈ {1K,10K,100K}`.

The engine auto-dispatches between three GPU paths:

- **Full-D uniform grid** (spatial hash) — used when the complete `D`-dimensional grid and its `3^D` neighbor-cell shell are feasible. Sub-quadratic; dominates at low/mid `D`.
- **First-coordinate grid with full-D ranking** — used for larger high-dimensional searches when the full-D grid is infeasible. The grid is built and scanned on `grid_dim = (D > 4) ? 4 : min(D, 3)` coordinates, but every encountered point is compared using its exact full-D squared distance and inserted directly into the K-nearest heap.
- **Tiled brute-force** — used for small workloads and as the fallback when a useful grid cannot be built. Uses a `float4`-vectorized distance kernel.

The high-D grid is an exact acceleration structure, not an approximate projection. Because
distance in the first `grid_dim` coordinates is a lower bound on full-D distance, its
radius-sized cell scan cannot omit a true in-radius neighbor. There is no PCA and no
"find O candidates, then verify" stage: all grid candidates are ranked in full D inline.

All point data is stored **Structure-of-Arrays** (`p[d*P + i]`) for coalesced warp access.
Engine hard limits: `K ≤ 128`, `D ≤ 128`. Grid paths use at most 1,000,000 cells;
the engine falls back to tiled brute force when the applicable grid is degenerate.

---

## Environment

| | |
|---|---|
| Machine | Perlmutter (NERSC) |
| GPU | NVIDIA A100 (`sm_80`) |
| Module | `pytorch/2.8.0` → Python 3.12, CUDA 12.9 (torch used only for FAISS, FlashLib, and xju2 baselines) |
| Runtime | `pynvml`, `faiss-gpu` — **no PyTorch or CuPy required for the FRNN engine** |

> The `frnn_cuda` extension (pure nanobind, no torch, no cupy) is the primary engine interface.
> PyTorch is only needed at runtime for the FAISS, FlashLib, and xju2 baseline blocks in `benchmark_master.py`.

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
pip install --user pynvml faiss-gpu
```

## 3. Build the engine

`frnn_cuda` is a **nanobind** extension (no torch, no cupy). Build it from the repo root:

```bash
cd /global/u1/d/dratnam/FRNN-master
pip install --user nanobind scikit-build-core
pip install --user --no-build-isolation -e .
```

Verify it loads:

```bash
python3 -c "import frnn_cuda; e = frnn_cuda.FRNNEngine(100); print('OK')"
```

**Rebuild** (after editing any `.cu` / `.h` under `python_interface/` or `frnn/csrc/`):

```bash
pip install --user --no-build-isolation -e .   # re-runs CMake + nvcc
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
`flash_knn`) as a baseline. Install on a **login node** (compute nodes have no outbound
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
PYTHONPATH=. python3 Tests/scripts/benchmark_master.py 2>&1 | tee benchmark_run.log
grep REGRESSION benchmark_run.log          # any cell where FRNN lost a baseline
```

Sweeps `D ∈ {3,16}` × `N ∈ {100K,200K,300K,400K,500K}` plus a singular `D=12, N=200K`
probe cell (11 cells by default), timing FRNN, FAISS, FlashLib, and xju2 in-process on
GPU-resident data. A 3-second GPU warm-up runs first.

### Data distributions

Control the point cloud generator with the `DIST` environment variable:

| `DIST` | Description | Radius |
|---|---|---|
| `lowrank` (default) | `INTRINSIC`-dim structure linearly embedded in D + noise; realistic non-uniform high-D data | calibrated bisection |
| `uniform` | iid uniform in `[0,1]^D` | analytic `radius_for(D,N)` |

```bash
# Low-rank data (default; exercises the first-coordinate/full-D grid path at D=16)
DIST=lowrank  PYTHONPATH=. python3 Tests/scripts/benchmark_master.py 2>&1 | tee benchmark_run_lowrank.log

# Uniform data
DIST=uniform  PYTHONPATH=. python3 Tests/scripts/benchmark_master.py 2>&1 | tee benchmark_run_uniform.log
```

Tunable env vars: `N_SWEEP` (comma-separated, default `100000,...,500000`), `D_SWEEP`
(default `3,16`), `EXTRA_CELLS` (singular `D:N` probe cells appended to the grid,
default `12:200000`), `INTRINSIC` (intrinsic dim for lowrank, default `3`),
`LOWRANK_NOISE` (default `0.02`).

FRNN is timed via `frnn_cuda.FRNNEngine.search_gpu()` called with a PyTorch tensor's raw
device pointer (AoS→SoA transpose done on the GPU with `.T.contiguous()`). No H2D/D2H
copies occur inside the timed loop — same footing as FAISS and xju2.

All baselines are timed on GPU-resident PyTorch tensors (no H2D copies in the timed loop).

## 6. Validate correctness

Checks that FRNN returns the right neighbors against an exact float64 brute-force truth
(exits 0 = all pass, for CI). The default sweep covers the D=3 full-grid path and the
D=12/D=16 first-coordinate grids:

```bash
# Default validation (LOWRANK=4)
PYTHONPATH=. python3 Tests/scripts/validate_correctness.py

# Change the intrinsic dimension of generated high-D data
LOWRANK=3 PYTHONPATH=. python3 Tests/scripts/validate_correctness.py
```

Confirms FRNN returns the exact **K-nearest** points within the radius (matches the
brute-force oracle) and reports any returned out-of-radius neighbors. At high D this
validates the inline full-D rankings produced during the first-coordinate grid scan.

---

## Outputs

| File | Contents |
|---|---|
| `Tests/json_results/benchmark_results.json` | per-cell `{R, dist, radius_mode, latency_ms (FRNN), peak_mb, faiss_ms, flash_ms, xfrnn_ms}` |
| `benchmark_run.log` | full console log; `!! REGRESSION` lines mark FRNN losses |
| `Tests/png_results/benchmark_comparison.png` | latency-vs-N plot per D dimension (log scale) |

Save results under a named file after each distribution run so they are not overwritten:
```bash
cp Tests/json_results/benchmark_results.json Tests/json_results/benchmark_results_lowrank.json
```

---

## Repository layout

```
python_interface/
  frnn_engine.h          # FRNNEngine class declaration
  frnn_engine.cu         # full-grid/first-coordinate-grid/BF auto-dispatch
  nanobind_module.cu     # frnn_cuda nanobind extension (NB_MODULE)
frnn/csrc/
  grid/
    grid.h               # GridParams struct
    insert_points.cu     # uniform-grid insertion kernel (SoA)
    find_nbrs.cu         # full-D neighbor ranking (SoA, D=3 AoS fast path)
  no_grid_frnn/
    no_grid_frnn.cu      # float4-vectorized tiled brute-force kernel
    no_grid_frnn.h
Tests/
  scripts/
    benchmark_master.py          # FRNN vs all available baselines
    benchmark_vs_libfrnn.py      # reusable FRNN vs stored/live libFRNN sweep
    validate_correctness.py      # FRNN vs exact brute-force oracle
    _run_frnn_isolated.py        # subprocess benchmark worker
  json_results/                  # structured benchmark outputs
  png_results/                   # generated benchmark plots
xju2_frnn/             # original lxxue/FRNN baseline (FRNN/ + prefix_sum/)
flash_lib_knn/         # FlashLib (FlashML) baseline — git clone + pip install -e (step 4b)
CMakeLists.txt         # scikit-build-core + nanobind build (replaces setup_frnn_torch.py)
pyproject.toml         # build-system declaration
```

---

## Rebuild triggers

Rebuild the xju2 extensions (step 4) when you switch the `pytorch` module (different torch/Python ABI).

Rebuild `frnn_cuda` (step 3) when you edit any `.cu` or `.h` under `python_interface/` or `frnn/csrc/`.

Pure-Python files (`Tests/scripts/benchmark_master.py`, etc.) need no rebuild after edits.
