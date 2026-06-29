# FRNN-master — Performance Engineering Handbook

**Goal:** make the pure-CUDA engine beat FAISS, PyG, and xju2/FRNN in wall-clock latency
across D ∈ {2,3,4,8,16} × N ∈ {1K,10K,100K}. Auto-dispatch: grid when `3^D < total_cells`,
tiled brute-force when `3^D ≥ total_cells` or `res ≤ 1`.

**Engine hard limits** (`frnn_engine.cu`): K ≤ 128, D ≤ 128, `ceil(1/R)^D ≤ 1,000,000`.
Dimension auto-detected: `dim = points_raw.size() / max_p` — flat array must be exactly N×D.

**Known bug:** `test_frnn_specific.py:79` has `import fais` — fix only if running that script.
`frnn/csrc/backward/backward.cu` is dead legacy code (xju2, uses `at::Tensor`); not built.

---

## 1. Think Before Coding

**Why subprocess isolation is mandatory for FRNN timing:**
PyTorch's caching allocator pre-warms the CUDA context, cuBLAS/cuDNN init steals 100–300 ms,
and `torch.cuda.synchronize()` syncs all streams including FRNN's. Timing FRNN in the same
process as PyTorch produces contaminated numbers. The fix: run `_run_frnn_isolated.py` as a
subprocess using only `frnn_cuda`, `numpy`, `ctypes`, `pynvml`. IPC overhead (~5–50 ms) is
per-invocation, not per-trial — the 20 warm-up + 10 timed trials run uncontaminated inside.
Use temp `.npy` files for data transfer (not JSON) to avoid float rounding at scale.

**High-D dispatch rules:**
- xju2/FRNN: works at arbitrary D. Call it for every D; wrap in `try/except` so a missing build logs and yields `None` (never raise).
- PyG: dimension-agnostic but kd-tree degrades 5–50× at D ≥ 8. Wrap in `try/except`.
- Our engine: BF path engages automatically for all high-D cases. No engine changes needed.

---

## 2. Simplicity First

Two new files only. No existing files modified.

### 2.1 `_run_frnn_isolated.py` — Pure CUDA Worker

Never imports `torch`, `faiss`, or `torch_cluster`. Reads JSON from stdin, writes one JSON line to stdout.

```python
#!/usr/bin/env python3
import sys, json, time, ctypes, numpy as np, frnn_cuda, pynvml

data = json.load(sys.stdin)
pts_flat = np.load(data["pts_path"])          # float32, shape (N*D,)
N, D, K, R = data["N"], data["D"], data["K"], data["R"]
_cu = ctypes.CDLL("libcudart.so"); sync = _cu.cudaDeviceSynchronize

engine = frnn_cuda.FRNNEngine(max_points=N)
for _ in range(data.get("warmup", 20)):       # warm up engine + driver
    engine.search(pts_flat, K=K, radius=R)
sync()

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
times = []
for _ in range(data.get("trials", 10)):
    sync(); t0 = time.perf_counter()
    engine.search(pts_flat, K=K, radius=R)
    sync(); times.append(time.perf_counter() - t0)
peak_mb = (pynvml.nvmlDeviceGetMemoryInfo(handle).used - mem_before) / 1024**2
pynvml.nvmlShutdown()
print(json.dumps({"latency_ms": float(np.median(times))*1000, "peak_mb": float(max(peak_mb,0))}))
```

### 2.2 `benchmark_master.py` — Orchestrator

Sweeps N ∈ {1K,10K,100K} × D ∈ {2,3,4,8,16}. Writes `benchmark_results.json`. Prints `!! REGRESSION` when FRNN loses.

```python
#!/usr/bin/env python3
import subprocess, sys, json, time, os, tempfile, numpy as np, torch, pynvml
from math import pi, gamma, ceil

N_SWEEP = [1_000, 10_000, 100_000]
D_SWEEP = [2, 3, 4, 8, 16]
K, SEED, WARMUP, TRIALS = 16, 1234, 20, 10

def radius_for(D, N):
    v = pi**(D/2) / gamma(D/2+1)
    r = min((K/(N*v))**(1/D), 2.0)
    if r < 1.0 and ceil(1/r)**D > 900_000: r = 2.0
    return round(r, 5)

def timed_gpu(fn):
    # [Standard warm-up + synchronize timing loop]
    ...

def run_frnn_isolated(pts_flat, N, D, R):
    # [Write .npy to temp file, subprocess _run_frnn_isolated.py, parse JSON stdout]
    ...

def run_baselines(pts_np, D, R):
    # [FAISS GPU range search, PyG radius, xju2 D==3 only — all wrapped in try/except]
    # [torch.cuda.empty_cache() + del pts_t between each framework]
    ...

pynvml.nvmlInit()
all_results = {}
for D in D_SWEEP:
    for N in N_SWEEP:
        # [seed data, run_frnn_isolated, run_baselines, print REGRESSION if FRNN loses]
        ...
with open("benchmark_results.json", "w") as f: json.dump(all_results, f, indent=2)
pynvml.nvmlShutdown()
```

---

## 3. Surgical Changes

**File rules:** Only `_run_frnn_isolated.py` and `benchmark_master.py` may be created/modified.
Do not touch `frnn/csrc/` or `python_interface/` unless a kernel-level
regression fix is confirmed. Every kernel edit requires a rebuild:
`python3 setup_frnn_torch.py build_ext --inplace`.

### 3.2 Terminal Commands

There is **no Makefile** — the legacy `make` / `./frnn_test` / `./frnn_bench` C++ harness
was removed (commit `52410b5`). The only build is the PyTorch/CUDA extension
(`setup_frnn_torch.py`); correctness and latency are validated from Python under `Tests/`.
See `README.md` for the authoritative build/run flow.

```bash
# New interactive session (every login):
srun -C gpu -q interactive -N 1 -G 1 -c 32 -t 01:00:00 -A m3443 --pty /bin/bash -l
module load pytorch/2.8.0          # exact version — extension is ABI-pinned to torch 2.8.0

# Install missing packages (once):
pip install --user faiss-gpu pynvml frnn torch-cluster

# Build / rebuild after any .cu/.h change under python_interface/ or frnn/csrc/:
rm -rf build frnn_torch*.so
python3 setup_frnn_torch.py build_ext --inplace
export LD_LIBRARY_PATH=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH
python3 -c "import torch, frnn_torch; print('OK', hasattr(frnn_torch,'FRNNTorch'))"

# Validate correctness before benchmarks (replaces ./frnn_test):
PYTHONPATH=. python3 Tests/validate_correctness.py    # exit 0 = matches float64 oracle + xju2

# Run primary benchmark (replaces ./frnn_bench):
PYTHONPATH=. python3 Tests/benchmark_master.py 2>&1 | tee benchmark_run.log
grep "REGRESSION" benchmark_run.log
```

**Memory cleanup** (already in `run_baselines()`; do not remove):
`torch.cuda.empty_cache()` → `del pts_t` → `torch.cuda.empty_cache()` after each framework block.
FRNN subprocess exits after each (N,D) run; `~FRNNEngine()` calls `cudaFree` automatically.

---

## 4. Goal-Driven Execution

**Pass criteria:** FRNN latency < every baseline for all N ≥ 10K. Zero `!! REGRESSION` lines.
25 rows in `benchmark_results.json` (5D × 5N). xju2 skip-logs present for D ≠ 3. No subprocess crashes.

### 4.2 Verification Phases

| Phase | Command | Check |
|---|---|---|
| 0 — env | `python3 -c "import torch, frnn_torch, faiss, pynvml; print('OK')"` | No ImportError |
| 1 — build | `python3 setup_frnn_torch.py build_ext --inplace && ls frnn_torch*.so` | `.so` non-zero |
| 2 — correctness | `PYTHONPATH=. python3 Tests/validate_correctness.py` | exit 0 (matches oracle) |
| 3 — xju2 skip | Call `run_baselines(pts_np, D=8, R=2.0)` | `xfrnn_ms=None`, no exception |
| 4 — single trial | D=3, N=10K full run | `latency_ms` finite, no REGRESSION |
| 5 — full sweep | `Tests/benchmark_master.py` | all cells, 0 REGRESSION for N≥10K |

### 4.3 Regression Diagnostic (in priority order)

If `!! REGRESSION` appears for N ≥ 10K, diagnose in this order. Each fix needs a rebuild:
`python3 setup_frnn_torch.py build_ext --inplace`.

**A — Non-coalesced global memory (highest impact, especially D≥8)**
- Both kernels use AoS layout: `p1[i*dim + d]`. For a 32-thread warp, accesses are strided by `dim` floats → `dim` cache-line fetches per warp per dimension (16× overhead at D=16).
- Fix: transpose to SoA (`p1[d*P + i]`); also transpose input array on Python side (`pts_np.T.copy().flatten()`).
- Measure with `ncu` metric `l1tex__t_sectors / l1tex__t_requests`; ideal ratio = 1, bad > 4.

**B — Low SM occupancy (bruteforce kernel)**
- `TiledBruteforceNDKernel`: 128 threads/block × D × 4 bytes smem. At D=16: 8 KB/block → max 6 blocks/SM → 37.5% A100 occupancy.
- Fix option 1: reduce threads 128→64 in `no_grid_frnn.cu` launch config (halves smem/block).
- Fix option 2: `cudaFuncSetAttribute(..., cudaFuncAttributePreferredSharedMemoryCarveout, 75)`.
- Measure with `ncu` metric `sm__warps_active.avg.pct_of_peak_sustained_active`.

**C — Register pressure (both kernels)**
- `local_dists[128]` and `local_idxs[128]` are always sized to `MAX_K_CAPACITY=128` even for K=16, wasting 224 slots/thread.
- Fix: template kernels on `K_STATIC`; instantiate for K ∈ {16,32,64}; dispatch via switch in `run_bruteforce()`.
- Measure: `nvcc --ptxas-info -c no_grid_frnn.cu | grep registers`; target < 64/thread.

**D — Shared memory bank conflicts (bruteforce tile loader)**
- `tile[threadIdx.x * dim + d]` with power-of-2 dim causes 2-way bank conflicts for threads 0,2,4,...
- Fix: pad smem allocation by +1 float per row; index with `tile[tx * (dim+1) + d]`.
- Measure with `ncu` metric `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st`.
