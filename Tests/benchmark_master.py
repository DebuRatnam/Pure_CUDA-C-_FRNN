#!/usr/bin/env python3
# benchmark_master.py — FRNN vs FAISS vs FlashLib vs xju2/FRNN latency sweep.
# FRNN is timed in-process on a GPU-resident PyTorch tensor via search_gpu()
# (raw device pointer, SoA layout) — zero-copy, same footing as FAISS/xju2.
import os, sys, json, time
import numpy as np
import torch
import frnn_cuda
import pynvml
from math import pi, gamma, ceil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# xju2/FRNN baseline path setup (used in run_baselines, guarded with try/except).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "xju2_frnn", "FRNN"),
           os.path.join(_ROOT, "xju2_frnn", "prefix_sum")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

N_SWEEP = [int(n) for n in os.environ.get(
    "N_SWEEP", "100000,200000,300000,400000,500000").split(",")]
D_SWEEP = [int(d) for d in os.environ.get("D_SWEEP", "3,16").split(",")]
# Singular probe cells appended to the D_SWEEP×N_SWEEP grid (env "D:N,D:N,...").
# Default: one D=12, N=200K point exercising the projection path at mid-D.
EXTRA_CELLS = [tuple(int(x) for x in c.split(":"))
               for c in os.environ.get("EXTRA_CELLS", "12:200000").split(",") if c.strip()]
K, SEED, WARMUP, TRIALS = 16, 1234, 20, 10
PROJ_K = 3   # target dimension for the projection variant

# Data distribution, env-selectable. uniform: iid in [0,1]^D (uses analytic
# radius_for). lowrank: INTRINSIC-dim structure linearly embedded in D + noise —
# the realistic non-uniform regime and projection's win case (uses a calibrated
# radius, see below).
DIST          = os.environ.get("DIST", "lowrank").lower()
INTRINSIC     = int(os.environ.get("INTRINSIC", str(PROJ_K)))
LOWRANK_NOISE = float(os.environ.get("LOWRANK_NOISE", "0.02"))


def radius_for(D, N):
    v = pi**(D / 2) / gamma(D / 2 + 1)
    r = min((K / (N * v))**(1.0 / D), 2.0)
    if r < 1.0 and ceil(1.0 / r)**D > 900_000:
        r = 2.0
    return round(r, 5)


def gen_points(N, D, seed):
    """Generate a point cloud per DIST. All outputs are float32 in [0,1]^D."""
    rng = np.random.RandomState(seed)
    if DIST == "uniform":
        return rng.rand(N, D).astype(np.float32)

    if DIST == "lowrank":
        # INTRINSIC-dim core linearly embedded into D, rotated, plus isotropic noise.
        if INTRINSIC <= 0 or INTRINSIC >= D:
            return rng.rand(N, D).astype(np.float32)
        core  = rng.rand(N, INTRINSIC)
        embed = rng.standard_normal((INTRINSIC, D))
        Q, _  = np.linalg.qr(rng.standard_normal((D, D)))
        pts   = (core @ embed) @ Q + LOWRANK_NOISE * rng.standard_normal((N, D))
        mn, mx = pts.min(0, keepdims=True), pts.max(0, keepdims=True)
        return ((pts - mn) / np.maximum(mx - mn, 1e-9)).astype(np.float32)

    raise ValueError(f"unknown DIST={DIST!r} (expected uniform|lowrank)")


def calibrate_radius(pts, target=K, sample=256, lo=1e-4, hi=2.0, iters=20):
    """Bisection for a radius giving ~`target` neighbors on average (non-uniform data).
    The analytic radius_for assumes uniformity; on structured clouds it over/under-shoots
    and can collapse the (projected) grid to res=1, so calibrate empirically."""
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
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
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


def run_frnn_gpu(pts_t, N, D, R):
    # pts_t: torch (N, D) float32 on CUDA, AoS layout.
    # Transpose to SoA (D, N) so search_gpu gets the expected d*N+i flat layout.
    pts_soa = pts_t.T.contiguous().reshape(-1)
    engine = frnn_cuda.FRNNEngine(max_points=N)
    torch.cuda.empty_cache()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    latency_ms = timed_gpu(lambda: engine.search_gpu(pts_soa.data_ptr(), N, D, K, R))
    peak_mb = max(0, pynvml.nvmlDeviceGetMemoryInfo(handle).used - mem_before) / 1024**2
    return {"latency_ms": latency_ms, "peak_mb": float(peak_mb)}


def run_frnn_projected(pts_np, pts_t, N, D, R):
    """PCA project to PROJ_K dims, run our FRNN at inflated radius, verify in full D on GPU.

    Only runs when D > PROJ_K. PCA is precomputed on CPU (not timed); the timed
    region is FRNN-on-projected-coords + full-D candidate verification via PyTorch.
    """
    if D <= PROJ_K:
        return None

    # PCA: project to PROJ_K dims (CPU preprocessing, not timed).
    centered = pts_np - pts_np.mean(0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:PROJ_K].T.astype(np.float32)            # (D, PROJ_K)
    proj_np = (centered @ basis).astype(np.float32)     # (N, PROJ_K)

    # Inflate radius: an orthonormal projection is contractive, so true-D distances
    # are >= projected distances. Inflate by sqrt(D / PROJ_K) as a conservative bound.
    R_proj = float(min(R * (D / PROJ_K) ** 0.5, 2.0))
    K_over = min(K * 4, 128)   # oversample in projected space before full-D filter

    proj_t   = torch.tensor(proj_np, device="cuda")
    proj_soa = proj_t.T.contiguous().reshape(-1)
    engine_p = frnn_cuda.FRNNEngine(max_points=N)
    # warm up the projection engine + verify path
    for _ in range(WARMUP):
        engine_p.search_gpu(proj_soa.data_ptr(), N, PROJ_K, K_over, R_proj)
    torch.cuda.synchronize()

    handle     = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    times = []
    r2 = float(R * R)
    for _ in range(TRIALS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Step 1: FRNN in projected space.
        engine_p.search_gpu(proj_soa.data_ptr(), N, PROJ_K, K_over, R_proj)
        # Step 2: retrieve candidate indices (D2H — part of pipeline cost).
        _, idxs_raw = engine_p.get_results(N, K_over)
        idxs_t = torch.tensor(
            np.array(idxs_raw, dtype=np.int64).reshape(N, K_over), device="cuda"
        )
        # Step 3: full-D verification on GPU via PyTorch.
        valid = idxs_t.clamp(min=0)          # replace -1 sentinels with 0 (masked below)
        cands = pts_t[valid]                  # (N, K_over, D)
        diff  = pts_t.unsqueeze(1) - cands   # (N, K_over, D)
        d2    = (diff * diff).sum(-1)         # (N, K_over)
        keep  = (d2 <= r2) & (idxs_t >= 0)  # (N, K_over) bool mask

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    latency_ms = float(np.median(times)) * 1000.0
    peak_mb    = max(0, pynvml.nvmlDeviceGetMemoryInfo(handle).used - mem_before) / 1024**2
    del proj_t, proj_soa, idxs_t, cands, diff, d2, keep
    torch.cuda.empty_cache()
    return {"latency_ms": latency_ms, "peak_mb": float(peak_mb)}


def run_baselines(pts_np, D, R):
    out = {}

    # FAISS GPU — queries pre-transferred to GPU so H2D copy is outside the
    # timed loop, matching the same footing as FRNN/FlashLib/xju2.
    try:
        import faiss
        import faiss.contrib.torch_utils
        cpu_idx = faiss.IndexFlatL2(D)
        gpu_res = faiss.StandardGpuResources()
        idx = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_idx)
        idx.add(pts_np)
        pts_t = torch.tensor(pts_np).cuda()
        out["faiss_ms"] = timed_gpu(lambda: idx.search(pts_t, K))
        del pts_t
    except Exception as e:
        out["faiss_ms"] = None
        print(f"    [FAISS] {e}")
    torch.cuda.empty_cache()

    # FlashLib — requires PyTorch tensors internally; pre-transfer to GPU so
    # H2D copy is outside the timed loop, matching FRNN/FAISS footing.
    try:
        from flashlib import flash_knn
        pts_t = torch.tensor(pts_np).cuda()
        out["flash_ms"] = timed_gpu(lambda: flash_knn(pts_t, pts_t, K))
        del pts_t
    except Exception as e:
        out["flash_ms"] = None
        print(f"    [FlashLib] {e}")
    torch.cuda.empty_cache()

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
        pts_t = torch.tensor(pts_np).cuda()
        L = torch.tensor([len(pts_np)]).cuda()
        p = pts_t.unsqueeze(0)
        out["xfrnn_ms"] = timed_gpu(lambda: _xfn(p, p, L, L, K, R))
        del pts_t
    except Exception as e:
        out["xfrnn_ms"] = None
        print(f"    [xju2] {e}")

    torch.cuda.empty_cache()
    return out


def plot_results(results, path="benchmark_comparison.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot] matplotlib unavailable ({e}); skipping graph")
        return

    methods = [("FRNN",          "latency_ms",   "o", "-",  "#1f77b4"),
               ("FRNN-proj",    "frnn_proj_ms", "P", "--", "#17becf"),
               ("FAISS",        "faiss_ms",     "s", "--", "#d62728"),
               ("FlashLib",     "flash_ms",     "^", "--", "#ff7f0e"),
               ("xju2",         "xfrnn_ms",     "D", "-.", "#2ca02c")]

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

print(f"\n=== distribution: {DIST}"
      + (f" (intrinsic={INTRINSIC}, noise={LOWRANK_NOISE})" if DIST == "lowrank"
         else "") + " ===")

CELLS = [(D, N) for D in D_SWEEP for N in N_SWEEP]
for c in EXTRA_CELLS:
    if c not in CELLS:
        CELLS.append(c)

for D, N in CELLS:
    pts_np = gen_points(N, D, SEED)
    # Uniform uses the analytic radius; non-uniform calibrates so the (projected)
    # grid stays non-degenerate and neighbor counts are comparable across cells.
    if DIST == "uniform":
        R, r_mode = radius_for(D, N), "analytic"
    else:
        R, r_mode = calibrate_radius(pts_np, target=K), "calibrated"
    key    = f"D{D}_N{N}"
    print(f"\n{'─'*56}\n  {key}  R={R:.5f} ({r_mode}, {DIST})\n{'─'*56}")

    pts_t = torch.tensor(pts_np, device="cuda")

    try:
        frnn_res = run_frnn_gpu(pts_t, N, D, R)
    except Exception as e:
        print(f"  [FRNN ERROR] {e}")
        frnn_res = {"latency_ms": None, "peak_mb": None}

    try:
        proj_res = run_frnn_projected(pts_np, pts_t, N, D, R)
    except Exception as e:
        print(f"  [FRNN-proj ERROR] {e}")
        proj_res = None

    del pts_t
    torch.cuda.empty_cache()

    base = run_baselines(pts_np, D, R)
    proj_ms = proj_res["latency_ms"] if proj_res else None
    all_results[key] = {"R": R, "dist": DIST, "radius_mode": r_mode, **frnn_res,
                        "frnn_proj_ms": proj_ms, **base}

    f_ms = frnn_res["latency_ms"]
    print(f"  FRNN={f_ms}ms  FRNN-proj={proj_ms}ms"
          f"  FAISS={base['faiss_ms']}ms"
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
