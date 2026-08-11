#!/usr/bin/env python3
"""
diagnose_crash.py — Pinpoint the cudaErrorIllegalAddress crash in benchmark_master.py.

Each step prints before running; the step that doesn't print OK is the culprit.
Run:
  PYTHONPATH=. python3 Tests/scripts/diagnose_crash.py 2>&1 | tee diagnose.log
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
for _p in (os.path.join(ROOT, "xju2_frnn", "FRNN"),
           os.path.join(ROOT, "xju2_frnn", "prefix_sum"),
           os.path.join(ROOT, "new_xju2_frnn")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

def step(n, desc):
    print(f"[{n:02d}] {desc} ...", flush=True)

def ok(extra=""):
    print(f"      OK{(' — ' + extra) if extra else ''}", flush=True)


# ── 1. Same imports as benchmark_master.py, same order ────────────────────────
step(1, "import torch")
import torch
ok()

step(2, "import frnn_cuda")
import frnn_cuda
ok()

step(3, "import pynvml + nvmlInit()")
import pynvml
pynvml.nvmlInit()
ok()

step(4, "tiny GPU tensor + synchronize (CUDA context init)")
_ = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
del _
ok()

# ── 2. warmup_gpu() exactly as written in benchmark_master.py ─────────────────
step(5, "warmup_gpu: allocate 2048×2048 on GPU")
a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
ok()

step(6, "warmup_gpu: tanh(a @ a) loop — 3 seconds")
t0 = time.perf_counter()
iters = 0
while time.perf_counter() - t0 < 3.0:
    a = torch.tanh(a @ a)
    iters += 1
ok(f"{iters} iters")

step(7, "warmup_gpu: torch.cuda.synchronize()")
torch.cuda.synchronize()
ok()

step(8, "warmup_gpu: del a")
del a
ok()

step(9, "warmup_gpu: torch.cuda.empty_cache()")
torch.cuda.empty_cache()
ok()

# ── 3. FRNNEngine lifecycle (constructor + search + destructor) ────────────────
for N, D, R in [(100_000, 3, 0.05), (100_000, 16, 2.0), (500_000, 3, 0.03), (500_000, 16, 2.0)]:
    step(10, f"FRNNEngine N={N} D={D}: construct")
    eng = frnn_cuda.FRNNEngine(max_points=N)
    ok()

    step(11, f"FRNNEngine N={N} D={D}: search_gpu ×3")
    pts = np.random.rand(N, D).astype(np.float32)
    pts_t   = torch.tensor(pts, device="cuda")
    pts_soa = pts_t.T.contiguous().reshape(-1)
    for _ in range(3):
        eng.search_gpu(pts_soa.data_ptr(), N, D, 16, R)
    torch.cuda.synchronize()
    ok()

    step(12, f"FRNNEngine N={N} D={D}: del (destructor cudaFree)")
    del eng, pts_t, pts_soa
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    ok()

# ── 4. lowrank data generation (benchmark_master.py default DIST) ─────────────
step(13, "lowrank gen N=500K D=16")
from math import pi, gamma, ceil
rng = np.random.RandomState(1234)
N, D = 500_000, 16
INTRINSIC, NOISE = 3, 0.02
core  = rng.rand(N, INTRINSIC)
embed = rng.standard_normal((INTRINSIC, D))
Q, _  = np.linalg.qr(rng.standard_normal((D, D)))
pts   = (core @ embed) @ Q + NOISE * rng.standard_normal((N, D))
mn, mx = pts.min(0, keepdims=True), pts.max(0, keepdims=True)
pts = ((pts - mn) / np.maximum(mx - mn, 1e-9)).astype(np.float32)
ok(f"shape={pts.shape}")

# ── 5. FAISS GPU (first baseline in run_baselines) ────────────────────────────
step(14, "import faiss + StandardGpuResources")
try:
    import faiss, faiss.contrib.torch_utils
    _res = faiss.StandardGpuResources()
    ok()

    step(15, "FAISS range search N=100K D=16")
    sub = pts[:100_000].copy()
    sub_t = torch.tensor(sub, device="cuda")
    idx = faiss.GpuIndexFlatL2(_res, D)
    idx.add(sub)
    _, __ = idx.search(sub, 16)
    torch.cuda.synchronize()
    del sub_t, idx
    torch.cuda.empty_cache()
    ok()
except Exception as e:
    print(f"      SKIP: {e}", flush=True)

pynvml.nvmlShutdown()
print("\n=== All steps passed — crash is not reproducible in isolation ===")
print("If benchmark_master.py still crashes, it may be specific to the")
print("subprocess isolation path (_run_frnn_isolated.py). Try:")
print("  N_SWEEP=100000 D_SWEEP=3 PYTHONPATH=. python3 Tests/scripts/benchmark_master.py")
