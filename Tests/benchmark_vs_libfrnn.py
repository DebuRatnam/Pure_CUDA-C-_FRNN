#!/usr/bin/env python3
# benchmark_vs_libfrnn.py — Our FRNN engine vs xju2/libFRNN head-to-head.
#
# Both engines are timed end-to-end: CPU numpy in → GPU search → results back on CPU.
# libFRNN's Python API only accepts numpy (H2D+D2H always included; no device path).
# Our engine matches that: run_frnn_e2e() does H2D + search_gpu() + get_results() + D2H.
#
# run_frnn_gpu() (kernel-only, data pre-transferred) is also reported as a reference.
#
# Build libFRNN first (GPU node):
#   module load pytorch/2.8.0
#   bash new_xju2_frnn/build_libfrnn.sh
#
# Run:
#   PYTHONPATH=. python3 Tests/benchmark_vs_libfrnn.py 2>&1 | tee vs_libfrnn.log
#   grep "REGRESSION" vs_libfrnn.log

import os, sys, json, time
import numpy as np
import torch
import frnn_cuda
import pynvml
from math import pi, gamma, ceil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NEW_XJU2_DIR = os.path.join(_ROOT, "new_xju2_frnn")
if _NEW_XJU2_DIR not in sys.path:
    sys.path.insert(0, _NEW_XJU2_DIR)

N_SWEEP = [int(n) for n in os.environ.get(
    "N_SWEEP", "100000,200000,300000,400000,500000").split(",")]
D_SWEEP = [int(d) for d in os.environ.get("D_SWEEP", "3,16").split(",")]
EXTRA_CELLS = [tuple(int(x) for x in c.split(":"))
               for c in os.environ.get("EXTRA_CELLS", "12:200000").split(",") if c.strip()]
K, SEED, WARMUP, TRIALS = 16, 1234, 20, 10
PROJ_K = 3

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
    rng = np.random.RandomState(seed)
    if DIST == "uniform":
        return rng.rand(N, D).astype(np.float32)
    if DIST == "lowrank":
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
    """Our engine: GPU kernel-only, data already on device (no H2D/D2H in timed region).
    Reported as a reference; use run_frnn_e2e for the fair head-to-head."""
    pts_soa = pts_t.T.contiguous().reshape(-1)
    engine = frnn_cuda.FRNNEngine(max_points=N)
    torch.cuda.empty_cache()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    latency_ms = timed_gpu(lambda: engine.search_gpu(pts_soa.data_ptr(), N, D, K, R))
    peak_mb = max(0, pynvml.nvmlDeviceGetMemoryInfo(handle).used - mem_before) / 1024**2
    return {"latency_ms": latency_ms, "peak_mb": float(peak_mb)}


def run_frnn_e2e(pts_np, N, D, R):
    """Our engine end-to-end: H2D + search + D2H — matches libFRNN's timing scope."""
    engine = frnn_cuda.FRNNEngine(max_points=N)
    for _ in range(WARMUP):
        pts_t_ = torch.tensor(pts_np, device="cuda")
        pts_soa_ = pts_t_.T.contiguous().reshape(-1)
        engine.search_gpu(pts_soa_.data_ptr(), N, D, K, R)
        engine.get_results(N, K)
        del pts_t_, pts_soa_
    torch.cuda.empty_cache()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    times = []
    for _ in range(TRIALS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pts_t_ = torch.tensor(pts_np, device="cuda")
        pts_soa_ = pts_t_.T.contiguous().reshape(-1)
        engine.search_gpu(pts_soa_.data_ptr(), N, D, K, R)
        engine.get_results(N, K)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        del pts_t_, pts_soa_
    peak_mb = max(0, pynvml.nvmlDeviceGetMemoryInfo(handle).used - mem_before) / 1024**2
    torch.cuda.empty_cache()
    return {"latency_ms": float(np.median(times)) * 1000.0, "peak_mb": float(peak_mb)}


def run_frnn_projected(pts_np, pts_t, N, D, R):
    """Our projection path: PROJ_K-dim grid search + full-D GPU verify. D > PROJ_K only."""
    if D <= PROJ_K:
        return None
    centered = pts_np - pts_np.mean(0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:PROJ_K].T.astype(np.float32)
    proj_np = (centered @ basis).astype(np.float32)
    R_proj = float(min(R * (D / PROJ_K) ** 0.5, 2.0))
    K_over = min(K * 4, 128)
    proj_t   = torch.tensor(proj_np, device="cuda")
    proj_soa = proj_t.T.contiguous().reshape(-1)
    engine_p = frnn_cuda.FRNNEngine(max_points=N)
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
        engine_p.search_gpu(proj_soa.data_ptr(), N, PROJ_K, K_over, R_proj)
        _, idxs_raw = engine_p.get_results(N, K_over)
        idxs_t = torch.tensor(
            np.array(idxs_raw, dtype=np.int64).reshape(N, K_over), device="cuda"
        )
        valid = idxs_t.clamp(min=0)
        cands = pts_t[valid]
        diff  = pts_t.unsqueeze(1) - cands
        d2    = (diff * diff).sum(-1)
        keep  = (d2 <= r2) & (idxs_t >= 0)  # noqa: F841
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    latency_ms = float(np.median(times)) * 1000.0
    peak_mb    = max(0, pynvml.nvmlDeviceGetMemoryInfo(handle).used - mem_before) / 1024**2
    del proj_t, proj_soa, idxs_t, cands, diff, d2, keep
    torch.cuda.empty_cache()
    return {"latency_ms": latency_ms, "peak_mb": float(peak_mb)}


def run_libfrnn(pts_np, D, R):
    """xju2/libFRNN standalone. Includes H2D + search + D2H (synchronous API).
    Returns latency_ms or None if the .so is not built or D > 32."""
    try:
        import _frnn as _lf
    except ImportError:
        return None
    if D > 32:
        return None
    for _ in range(WARMUP):
        _lf.build_edges(pts_np, radius=R, max_neighbors=K, exclude_self=False)
    times = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        _lf.build_edges(pts_np, radius=R, max_neighbors=K, exclude_self=False)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000.0


def plot_results(results, path="vs_libfrnn_comparison.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot] matplotlib unavailable ({e}); skipping")
        return

    methods = [
        ("FRNN (H2D+D2H)",       "frnn_e2e_ms",  "o", "-",  "#1f77b4"),
        ("FRNN (kernel-only)",   "latency_ms",   "s", "--", "#aec7e8"),
        ("FRNN-proj",            "frnn_proj_ms", "P", "--", "#17becf"),
        ("libFRNN (H2D+D2H)",    "libfrnn_ms",   "v", "-.", "#9467bd"),
    ]

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
    fig.suptitle("Our FRNN vs xju2/libFRNN  (both timed H2D + search + D2H)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


if "--plot-only" in sys.argv:
    fname = "vs_libfrnn_results.json"
    with open(fname) as f:
        plot_results(json.load(f))
    sys.exit(0)


pynvml.nvmlInit()
warmup_gpu()
all_results = {}

print(f"\n=== FRNN vs xju2/libFRNN  |  distribution: {DIST}"
      + (f" (intrinsic={INTRINSIC}, noise={LOWRANK_NOISE})" if DIST == "lowrank" else "")
      + " ===")
print("  Both engines timed end-to-end: numpy in → GPU search → results back on CPU.")
print("  FRNN(H2D+D2H) uses run_frnn_e2e(); kernel-only is reported as a reference.\n")

CELLS = [(D, N) for D in D_SWEEP for N in N_SWEEP]
for c in EXTRA_CELLS:
    if c not in CELLS:
        CELLS.append(c)

for D, N in CELLS:
    pts_np = gen_points(N, D, SEED)
    if DIST == "uniform":
        R, r_mode = radius_for(D, N), "analytic"
    else:
        R, r_mode = calibrate_radius(pts_np, target=K), "calibrated"
    key = f"D{D}_N{N}"
    print(f"\n{'─'*56}\n  {key}  R={R:.5f} ({r_mode}, {DIST})\n{'─'*56}")

    pts_t = torch.tensor(pts_np, device="cuda")

    try:
        frnn_res = run_frnn_gpu(pts_t, N, D, R)
    except Exception as e:
        print(f"  [FRNN kernel-only ERROR] {e}")
        frnn_res = {"latency_ms": None, "peak_mb": None}

    try:
        proj_res = run_frnn_projected(pts_np, pts_t, N, D, R)
    except Exception as e:
        print(f"  [FRNN-proj ERROR] {e}")
        proj_res = None

    del pts_t
    torch.cuda.empty_cache()

    try:
        e2e_res = run_frnn_e2e(pts_np, N, D, R)
    except Exception as e:
        print(f"  [FRNN e2e ERROR] {e}")
        e2e_res = {"latency_ms": None, "peak_mb": None}

    try:
        libfrnn_ms = run_libfrnn(pts_np, D, R)
        if libfrnn_ms is None:
            print(f"  [libFRNN] not available (D={D}>32 or .so not built)")
    except Exception as e:
        libfrnn_ms = None
        print(f"  [libFRNN] {e}")

    proj_ms = proj_res["latency_ms"] if proj_res else None
    all_results[key] = {
        "R": R, "dist": DIST, "radius_mode": r_mode,
        **frnn_res,
        "frnn_e2e_ms": e2e_res["latency_ms"],
        "frnn_proj_ms": proj_ms,
        "libfrnn_ms": libfrnn_ms,
    }

    f_ms   = frnn_res["latency_ms"]
    e2e_ms = e2e_res["latency_ms"]
    print(f"  FRNN(kernel-only)={f_ms}ms  FRNN(H2D+D2H)={e2e_ms}ms"
          f"  FRNN-proj={proj_ms}ms  libFRNN(H2D+D2H)={libfrnn_ms}ms")

    if e2e_ms is not None and libfrnn_ms is not None:
        if e2e_ms > libfrnn_ms:
            print(f"  !! REGRESSION: FRNN(H2D+D2H) {e2e_ms:.2f}ms"
                  f" > libFRNN(H2D+D2H) {libfrnn_ms:.2f}ms"
                  f" — see §4.3 diagnostics")
        else:
            ratio = libfrnn_ms / e2e_ms
            print(f"  FRNN is {ratio:.2f}x faster than libFRNN (fair H2D+D2H vs H2D+D2H)")

with open("vs_libfrnn_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\n→ vs_libfrnn_results.json")
plot_results(all_results)
pynvml.nvmlShutdown()
