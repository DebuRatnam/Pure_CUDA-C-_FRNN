# FRNN-master

A pure-CUDA **fixed-radius nearest-neighbor (FRNN)** engine for *N*-dimensional point
clouds, built to beat **FAISS**, **PyTorch Geometric (PyG)**, and the original
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
| Module | `pytorch/2.8.0` → torch 2.8 (cu129), Python 3.12, plus `faiss`, `torch_cluster`, `pynvml` |

> The compiled extensions are tied to the exact PyTorch version + Python version + GPU arch
> they were built against. **Rebuild whenever you switch the `pytorch` module.**

---

## 1. Get a GPU node

```bash
srun -C gpu -q interactive -N 1 -G 1 -c 32 -t 01:00:00 -A m3443 --pty /bin/bash -l
```

Do everything below **inside** this shell — the build needs `nvcc` and the A100, and the
benchmark needs the GPU. (Change `-A m3443` to your own allocation if different.)

> ⚠️ The GPU may be shared if you are not in an exclusive allocation. Check with
> `nvidia-smi`; if another process is at high utilization, your latency numbers will be
> inflated. Run `nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv`
> to see who's on the device.

## 2. Load the environment

```bash
module load pytorch/2.8.0
cd /global/u1/d/dratnam/FRNN-master
```

## 3. Build the FRNN engine (`frnn_torch`)

This is the zero-copy PyTorch/CUDA extension that wraps the engine kernels. It replaces the
old `make` build (the Makefile is stale — its `.cu` sources moved into `Tests/`).

```bash
rm -rf build frnn_torch*.so                 # clean any stale build
python3 setup_frnn_torch.py build_ext --inplace
```

Produces `frnn_torch.cpython-312-*.so` in the repo root. Verify:

```bash
python3 -c "import torch, frnn_torch; print('OK', hasattr(frnn_torch,'FRNNTorch'))"
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
lxxue/FRNN only supports `D ∈ {2,3}`; higher dims are reported as `None` (skip-logged).

If you skip this step, the `xju2` column is simply `None` everywhere — the rest of the
benchmark still runs.

## 5. Run the benchmark

Run **from the repo root** (so `import frnn_torch` resolves) with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python3 Tests/benchmark_master.py 2>&1 | tee benchmark_run.log
grep REGRESSION benchmark_run.log          # any cell where FRNN lost a baseline
```

Sweeps `D ∈ {3,16}` × `N ∈ {10K, 25K, 50K, 75K, 100K, 150K, 200K}` (14 cells), timing
FRNN, FAISS, PyG, and xju2 in-process on GPU-resident tensors. A 3-second GPU warm-up runs
first so the first cell isn't measured at idle clocks. (Edit `D_SWEEP` / `N_SWEEP` at the
top of the script to cover more of the engine's range — the engine itself handles `D` up
to 128.)

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
| `benchmark_results.json` | per-cell `{R, latency_ms (FRNN), peak_mb, faiss_ms, pyg_ms, xfrnn_ms}` |
| `benchmark_run.log` | full console log; `!! REGRESSION` lines mark FRNN losses |

---

## Repository layout

```
python_interface/
  frnn_torch.cu        # PyTorch/CUDA extension: torch.Tensor in/out, zero host copies
  frnn_engine.cu/.h    # original pybind engine (CPU + raw-device-ptr search paths)
frnn/csrc/
  grid/                # insert_points.cu, find_nbrs.cu — uniform-grid kernels (SoA)
  bruteforce/          # bruteforce.cu — float4-vectorized tiled brute-force (SoA)
setup_frnn_torch.py    # builds frnn_torch
Tests/
  benchmark_master.py  # the sweep (FRNN vs FAISS vs PyG vs xju2)
  test_frnn.cu         # C++ grid-vs-bruteforce correctness check (built via Makefile)
  benchmark_frnn.cu    # C++ grid timing benchmark
xju2_frnn/             # original lxxue/FRNN baseline (FRNN/ + prefix_sum/)
```

---

## Rebuild triggers

Rebuild `frnn_torch` (step 3) **and** the xju2 extensions (step 4) when you:

- switch the `pytorch` module (different torch/Python ABI), or
- edit any `.cu` / `.h` under `python_interface/` or `frnn/csrc/`.

For repeated runs in the same module with no source changes, only step 5 is needed.
