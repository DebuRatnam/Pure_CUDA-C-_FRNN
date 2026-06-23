#!/usr/bin/env python3
# benchmark_master.py — FRNN vs FAISS vs PyG vs xju2/FRNN latency sweep.
# Every framework — FRNN included — is timed in-process on a GPU-resident tensor.
# FRNN uses the zero-copy `frnn_torch` extension (see python_interface/frnn_torch.cu):
# no host<->device copies occur inside the timed loop, so it is measured on the same
# footing as PyG. (The old `_run_frnn_isolated.py` subprocess copied H2D/D2H every
# trial — kernel time was swamped by PCIe traffic, making the comparison unfair.)
import os, sys, json, time
import numpy as np
import torch
import frnn_torch
import pynvml
from math import pi, gamma, ceil

# Make `import frnn` resolve to the xju2/FRNN baseline package (and its prefix_sum
# dependency) instead of this repo's local ./frnn source dir, which would otherwise
# shadow it as an empty namespace package. Both must be built for the running Python.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "xju2_frnn", "FRNN"),
           os.path.join(_ROOT, "xju2_frnn", "prefix_sum")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# D=3 exercises the grid path; D=16 exercises the high-D brute-force path. xju2 only
# supports D in {2,3}, so it shows up on the D=3 panel only (skip-logged at D=16).
# Dense N grid up to 200K for smooth scaling curves (D=16 high-N cells are slow: O(N^2)).
N_SWEEP = [10_000, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000]
D_SWEEP = [int(d) for d in os.environ.get("D_SWEEP", "3,16").split(",")]
K, SEED, WARMUP, TRIALS = 16, 1234, 20, 10


def radius_for(D, N):
    v = pi**(D / 2) / gamma(D / 2 + 1)
    r = min((K / (N * v))**(1.0 / D), 2.0)
    if r < 1.0 and ceil(1.0 / r)**D > 900_000:
        r = 2.0
    return round(r, 5)


def warmup_gpu(seconds=3.0):
    # Spin the GPU under sustained load so its clocks reach boost before any cell is
    # timed. The per-cell WARMUP loop runs only at N=1000 first, which is too small/
    # brief to ramp clocks — so without this the first row (D2_N1000) is measured at
    # idle clocks and comes out ~10x inflated across every framework. tanh keeps the
    # repeated matmul bounded (no inf/nan).
    a = torch.randn(2048, 2048, device="cuda")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a = torch.tanh(a @ a)
    torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()


def timed_gpu(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        torch.cuda.synchronize()
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

    # xju2/FRNN — the lxxue grid kernel only supports D in {2, 3}; higher dims
    # are genuinely unsupported by the library, so they stay None (skip-logged).
    if D in (3, 16):
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
        print(f"    [xju2] D={D} unsupported (lxxue FRNN is 2D/3D only), skipping")

    del pts_t
    torch.cuda.empty_cache()
    return out


def plot_results(results, path="benchmark_comparison.png"):
    # Latency vs N, one panel per D, four methods. The y-axis is LOG so FRNN's
    # sub-millisecond line stays readable while FAISS/PyG climb ~300x at high N.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot] matplotlib unavailable ({e}); skipping graph")
        return

    methods = [("FRNN",  "latency_ms", "o", "-",  "#1f77b4"),
               ("FAISS", "faiss_ms",   "s", "--", "#d62728"),
               ("PyG",   "pyg_ms",     "^", "--", "#ff7f0e"),
               ("xju2",  "xfrnn_ms",   "D", "-.", "#2ca02c")]

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
        ax.set_yscale("log")                       # span the ~300x dynamic range
        ax.set_title(f"D = {D}")
        ax.set_xlabel("N (points)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.set_xticks(Ns)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}K"))
        ax.tick_params(axis="x", rotation=45)
    axes[0][0].set_ylabel("Latency (ms) — log scale")
    axes[0][0].legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    fig.suptitle("Fixed-radius KNN latency vs N (lower is better)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))      # reserve top strip for the suptitle
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


if "--plot-only" in sys.argv:        # regenerate the graph from existing results
    with open("benchmark_results.json") as f:
        plot_results(json.load(f))
    sys.exit(0)


pynvml.nvmlInit()
warmup_gpu()          # boost GPU clocks so the first timed cell isn't throttled
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
plot_results(all_results)
pynvml.nvmlShutdown()
