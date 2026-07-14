#!/usr/bin/env python3
# benchmark_master.py — FRNN vs FAISS vs FlashLib vs xju2/FRNN latency sweep.
# Every framework — FRNN included — is timed in-process on a GPU-resident array.
# FRNN uses the zero-copy `frnn_cupy` wrapper (frnn_cupy.py): AoS->SoA transpose
# is done on the GPU via CuPy, so no host<->device copies occur inside the timed
# loop. FRNN is measured on the same footing as FAISS.
import os, sys, json, time
import numpy as np
import cupy as cp
import pynvml
from math import pi, gamma, ceil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frnn_cupy import FRNNCuPy

# xju2/FRNN baseline path setup (used in run_baselines, guarded with try/except).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "xju2_frnn", "FRNN"),
           os.path.join(_ROOT, "xju2_frnn", "prefix_sum")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

N_SWEEP = [100_000, 200_000, 300_000, 400_000, 500_000]
D_SWEEP = [int(d) for d in os.environ.get("D_SWEEP", "3,16").split(",")]
K, SEED, WARMUP, TRIALS = 16, 1234, 20, 10


def radius_for(D, N):
    v = pi**(D / 2) / gamma(D / 2 + 1)
    r = min((K / (N * v))**(1.0 / D), 2.0)
    if r < 1.0 and ceil(1.0 / r)**D > 900_000:
        r = 2.0
    return round(r, 5)


LOWRANK       = int(os.environ.get("LOWRANK", "0"))
LOWRANK_NOISE = float(os.environ.get("LOWRANK_NOISE", "0.02"))


def gen_points(N, D, seed):
    rng = np.random.RandomState(seed)
    if LOWRANK <= 0 or LOWRANK >= D:
        return rng.rand(N, D).astype(np.float32)
    core  = rng.rand(N, LOWRANK)
    embed = rng.standard_normal((LOWRANK, D))
    Q, _  = np.linalg.qr(rng.standard_normal((D, D)))
    pts   = (core @ embed) @ Q + LOWRANK_NOISE * rng.standard_normal((N, D))
    mn, mx = pts.min(0, keepdims=True), pts.max(0, keepdims=True)
    return ((pts - mn) / np.maximum(mx - mn, 1e-9)).astype(np.float32)


def calibrate_radius(pts, target=K, sample=256, lo=1e-4, hi=2.0, iters=20):
    rng = np.random.RandomState(0)
    qi = rng.choice(len(pts), size=min(sample, len(pts)), replace=False)
    qs = pts[qi]
    def avg_nbr(R):
        r2 = R * R
        return sum(int((((pts - qs[i]) ** 2).sum(1) <= r2).sum()) for i in range(len(qi))) / len(qi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if avg_nbr(mid) < target: lo = mid
        else:                     hi = mid
    return round(0.5 * (lo + hi), 6)


def warmup_gpu(seconds=3.0):
    # Spin the GPU under load so clocks reach boost before any cell is timed.
    a = cp.random.standard_normal((2048, 2048)).astype(cp.float32)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = cp.tanh(a @ a)
    cp.cuda.Device().synchronize()
    del a
    cp.get_default_memory_pool().free_all_blocks()


def timed_gpu(fn):
    for _ in range(WARMUP):
        fn()
    cp.cuda.Device().synchronize()
    times = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        cp.cuda.Device().synchronize()
        fn()
        cp.cuda.Device().synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000.0


def run_frnn_cupy(pts_cp, N, D, R):
    # Zero-copy, in-process FRNN. pts_cp is a GPU-resident (N, D) CuPy float32
    # array. FRNNCuPy transposes AoS->SoA on the GPU, runs the kernels, and
    # returns CuPy arrays — no H2D/D2H copies in the timed loop.
    engine = FRNNCuPy(N)
    cp.get_default_memory_pool().free_all_blocks()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    latency_ms = timed_gpu(lambda: engine.search_projected(pts_cp, K, R))
    peak_mb = max(0, pynvml.nvmlDeviceGetMemoryInfo(handle).used - mem_before) / 1024**2
    return {"latency_ms": latency_ms, "peak_mb": float(peak_mb)}


def run_baselines(pts_np, D, R):
    out = {}

    # FAISS GPU — uses numpy directly for add/search; no PyTorch needed.
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
    cp.get_default_memory_pool().free_all_blocks()

    # FlashLib — attempt with CuPy array (DLPack interop). Falls back to None
    # if FlashLib requires PyTorch tensors.
    try:
        from flashlib import flash_knn
        pts_cp = cp.asarray(pts_np)
        out["flash_ms"] = timed_gpu(lambda: flash_knn(pts_cp, pts_cp, K))
        del pts_cp
    except Exception as e:
        out["flash_ms"] = None
        print(f"    [FlashLib] {e}")
    cp.get_default_memory_pool().free_all_blocks()

    # xju2/lxxue FRNN — requires PyTorch tensors internally; will fail gracefully.
    try:
        import frnn as xf
        if hasattr(xf, 'frnn_grid_points'):
            _xfn = xf.frnn_grid_points
        elif hasattr(xf, 'frnn') and hasattr(xf.frnn, 'frnn_grid_points'):
            _xfn = xf.frnn.frnn_grid_points
        else:
            raise AttributeError(
                f"frnn_grid_points not found. Available: {[x for x in dir(xf) if not x.startswith('_')]}"
            )
        import torch
        pts_t = torch.tensor(pts_np).cuda()
        L = torch.tensor([len(pts_np)]).cuda()
        p = pts_t.unsqueeze(0)
        out["xfrnn_ms"] = timed_gpu(lambda: _xfn(p, p, L, L, K, R))
        del pts_t
    except Exception as e:
        out["xfrnn_ms"] = None
        print(f"    [xju2] {e}")

    cp.get_default_memory_pool().free_all_blocks()
    return out


def plot_results(results, path="benchmark_comparison.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot] matplotlib unavailable ({e}); skipping graph")
        return

    methods = [("FRNN",     "latency_ms", "o", "-",  "#1f77b4"),
               ("FAISS",    "faiss_ms",   "s", "--", "#d62728"),
               ("FlashLib", "flash_ms",   "^", "--", "#ff7f0e"),
               ("xju2",     "xfrnn_ms",   "D", "-.", "#2ca02c")]

    by_d = {}
    for key, v in results.items():
        D = int(key.split("_")[0][1:])
        N = int(key.split("_N")[1])
        by_d.setdefault(D, []).append((N, v))
    Ds = sorted(by_d)
    if not Ds:
        return

    fig, axes = plt.subplots(1, len(Ds), figsize=(6.5 * len(Ds), 5.0),
                             squeeze=False, sharey=True)
    for ax, D in zip(axes[0], Ds):
        cells = sorted(by_d[D])
        Ns = [n for n, _ in cells]
        for name, field, mk, ls, col in methods:
            xs = [n for n, v in cells if v.get(field) is not None]
            ys = [v[field] for n, v in cells if v.get(field) is not None]
            if ys:
                ax.plot(xs, ys, marker=mk, ls=ls, color=col, lw=1.8, ms=6, label=name)
        ax.set_yscale("log")
        ax.set_title(f"D = {D}")
        ax.set_xlabel("N (points)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.set_xticks(Ns)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))
        ax.tick_params(axis="x", rotation=45)
    axes[0][0].set_ylabel("Latency (ms) — log scale")
    axes[0][0].legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    fig.suptitle("Fixed-radius KNN latency vs N (lower is better)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


if "--plot-only" in sys.argv:
    with open("benchmark_results.json") as f:
        plot_results(json.load(f))
    sys.exit(0)


pynvml.nvmlInit()
warmup_gpu()
all_results = {}

for D in D_SWEEP:
    for N in N_SWEEP:
        pts_np = gen_points(N, D, SEED)
        R = calibrate_radius(pts_np) if LOWRANK > 0 and LOWRANK < D else radius_for(D, N)
        key = f"D{D}_N{N}"
        print(f"\n{'─'*56}\n  {key}  R={R:.5f}\n{'─'*56}")

        try:
            pts_cp   = cp.asarray(pts_np)
            frnn_res = run_frnn_cupy(pts_cp, N, D, R)
            del pts_cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"  [FRNN ERROR] {e}")
            frnn_res = {"latency_ms": None, "peak_mb": None}

        base = run_baselines(pts_np, D, R)
        all_results[key] = {"R": R, **frnn_res, **base}

        f_ms = frnn_res["latency_ms"]
        print(f"  FRNN={f_ms}ms  FAISS={base['faiss_ms']}ms"
              f"  FlashLib={base['flash_ms']}ms  xju2={base['xfrnn_ms']}ms")

        if f_ms is not None:
            for name, ms in [("FAISS",    base["faiss_ms"]),
                             ("FlashLib", base["flash_ms"]),
                             ("xju2",     base["xfrnn_ms"])]:
                if ms is not None and f_ms > ms:
                    print(f"  !! REGRESSION: FRNN {f_ms:.2f}ms > {name} {ms:.2f}ms"
                          f" — see §4.3 diagnostics")

with open("benchmark_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\n→ benchmark_results.json")
plot_results(all_results)
pynvml.nvmlShutdown()
