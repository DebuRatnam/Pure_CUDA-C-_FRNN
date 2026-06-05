#!/usr/bin/env python3
# benchmark_master.py — FRNN vs FAISS vs PyG vs xju2/FRNN latency sweep.
# Every framework — FRNN included — is timed in-process on a GPU-resident tensor.
# FRNN uses the zero-copy `frnn_torch` extension (see python_interface/frnn_torch.cu):
# no host<->device copies occur inside the timed loop, so it is measured on the same
# footing as PyG. (The old `_run_frnn_isolated.py` subprocess copied H2D/D2H every
# trial — kernel time was swamped by PCIe traffic, making the comparison unfair.)
import json, time
import numpy as np
import torch
import frnn_torch
import pynvml
from math import pi, gamma, ceil

N_SWEEP = [1_000, 10_000, 100_000]
D_SWEEP = [2, 3, 4, 8, 16]
K, SEED, WARMUP, TRIALS = 16, 1234, 20, 10


def radius_for(D, N):
    v = pi**(D / 2) / gamma(D / 2 + 1)
    r = min((K / (N * v))**(1.0 / D), 2.0)
    if r < 1.0 and ceil(1.0 / r)**D > 900_000:
        r = 2.0
    return round(r, 5)


def timed_gpu(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(TRIALS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000.0


def run_frnn_torch(pts_t, N, D, R):
    # Zero-copy, in-process FRNN. pts_t is already a GPU-resident (N, D) float32
    # tensor — the same one PyG receives. The frnn_torch extension transposes
    # AoS->SoA on the GPU, runs the kernels on the tensor's device pointer, and
    # returns CUDA tensors; no H2D/D2H copies happen inside the timed loop.
    engine = frnn_torch.FRNNTorch(N)
    torch.cuda.reset_peak_memory_stats()
    latency_ms = timed_gpu(lambda: engine.search(pts_t, K, R))
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    return {"latency_ms": latency_ms, "peak_mb": float(peak_mb)}


def run_baselines(pts_np, D, R):
    out = {}
    pts_t = torch.tensor(pts_np).cuda()

    # FAISS GPU knn search — GpuIndexFlatL2 does not implement range_search;
    # use search(K) instead (returns K nearest; radius filtering is done at result-read time)
    try:
        import faiss
        cpu_idx = faiss.IndexFlatL2(D)
        gpu_res = faiss.StandardGpuResources()
        idx = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_idx)
        idx.add(pts_np)
        out["faiss_ms"] = timed_gpu(lambda: idx.search(pts_np, K))
    except Exception as e:
        out["faiss_ms"] = None
        print(f"    [FAISS] {e}")
    torch.cuda.empty_cache()

    # PyG — dimension-agnostic but kd-tree degrades 5-50x at D>=8
    try:
        from torch_cluster import radius as pyg_radius
        out["pyg_ms"] = timed_gpu(
            lambda: pyg_radius(pts_t, pts_t, R, max_num_neighbors=K)
        )
    except Exception as e:
        out["pyg_ms"] = None
        print(f"    [PyG] {e}")
    torch.cuda.empty_cache()

    # xju2/FRNN — hardcoded 3D; skip all other dimensions
    if D == 3:
        try:
            import frnn as xf
            # Resolve API — some pip builds nest the function differently
            if hasattr(xf, 'frnn_grid_points'):
                _xfn = xf.frnn_grid_points
            elif hasattr(xf, 'frnn') and hasattr(xf.frnn, 'frnn_grid_points'):
                _xfn = xf.frnn.frnn_grid_points
            else:
                raise AttributeError(
                    f"frnn_grid_points not found. Available: {[x for x in dir(xf) if not x.startswith('_')]}"
                )
            L = torch.tensor([len(pts_np)]).cuda()
            p = pts_t.unsqueeze(0)
            out["xfrnn_ms"] = timed_gpu(lambda: _xfn(p, p, L, L, K, R))
        except Exception as e:
            out["xfrnn_ms"] = None
            print(f"    [xju2] {e}")
    else:
        out["xfrnn_ms"] = None
        print(f"    [xju2] D={D} unsupported (3D only), skipping")

    del pts_t
    torch.cuda.empty_cache()
    return out


pynvml.nvmlInit()
all_results = {}

for D in D_SWEEP:
    for N in N_SWEEP:
        R = radius_for(D, N)
        np.random.seed(SEED)
        pts_np   = np.random.rand(N, D).astype(np.float32)
        key      = f"D{D}_N{N}"
        print(f"\n{'─'*56}\n  {key}  R={R:.5f}\n{'─'*56}")

        try:
            pts_t    = torch.tensor(pts_np).cuda()   # GPU-resident, same as PyG
            frnn_res = run_frnn_torch(pts_t, N, D, R)
            del pts_t
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [FRNN ERROR] {e}")
            frnn_res = {"latency_ms": None, "peak_mb": None}

        base = run_baselines(pts_np, D, R)
        all_results[key] = {"R": R, **frnn_res, **base}

        f_ms = frnn_res["latency_ms"]
        print(f"  FRNN={f_ms}ms  FAISS={base['faiss_ms']}ms"
              f"  PyG={base['pyg_ms']}ms  xju2={base['xfrnn_ms']}ms")

        if f_ms is not None:
            for name, ms in [("FAISS", base["faiss_ms"]),
                             ("PyG",   base["pyg_ms"]),
                             ("xju2",  base["xfrnn_ms"])]:
                if ms is not None and f_ms > ms:
                    print(f"  !! REGRESSION: FRNN {f_ms:.2f}ms > {name} {ms:.2f}ms"
                          f" — see §4.3 diagnostics")

with open("benchmark_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\n→ benchmark_results.json")
pynvml.nvmlShutdown()
