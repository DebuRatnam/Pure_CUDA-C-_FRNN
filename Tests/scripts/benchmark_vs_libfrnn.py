#!/usr/bin/env python3
# benchmark_vs_libfrnn.py — Our FRNN engine vs xju2/libFRNN kernel-only head-to-head.
#
# Both engines are timed kernel-only: data already on device, results stay on device.
# No H2D or D2H in the timed region. Both use pre-allocated workspaces so there is
# no cudaMalloc/Free overhead per call.
#
# Stored mode (default) benchmarks only FRNN and compares it with the immutable
# libFRNN measurements below. To deliberately remeasure libFRNN, set
# LIBFRNN_MODE=live and build it first (GPU node):
#   module load pytorch/2.8.0
#   bash new_xju2_frnn/build_libfrnn.sh
#
# Run:
#   PYTHONPATH=. python3 Tests/scripts/benchmark_vs_libfrnn.py
#   grep "REGRESSION" vs_libfrnn.log

import os, sys, json, time
import numpy as np
import torch
import frnn_cuda
import pynvml
from math import pi, gamma, ceil
from types import MappingProxyType

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TEST_ROOT = os.path.dirname(_SCRIPT_DIR)
_ROOT = os.path.dirname(_TEST_ROOT)
sys.path.insert(0, _ROOT)
_NEW_XJU2_DIR = os.path.join(_ROOT, "new_xju2_frnn")
if _NEW_XJU2_DIR not in sys.path:
    sys.path.insert(0, _NEW_XJU2_DIR)

N_SWEEP = [int(n) for n in os.environ.get(
    "N_SWEEP", "200000,500000,750000,1000000,1250000,1500000"
).split(",") if n.strip()]
D_SWEEP = [int(d) for d in os.environ.get("D_SWEEP", "3,12,16").split(",") if d.strip()]
EXTRA_CELLS = [tuple(int(x) for x in c.split(":"))
               for c in os.environ.get("EXTRA_CELLS", "").split(",") if c.strip()]
K, SEED, WARMUP, TRIALS = 16, 1234, 20, 10
PROJ_K = 3

DIST          = os.environ.get("DIST", "lowrank").lower()
INTRINSIC     = int(os.environ.get("INTRINSIC", str(PROJ_K)))
LOWRANK_NOISE = float(os.environ.get("LOWRANK_NOISE", "0.02"))

JSON_DIR = os.path.join(_TEST_ROOT, "json_results")
PNG_DIR = os.path.join(_TEST_ROOT, "png_results")
RESULTS_PATH = os.path.join(JSON_DIR, "vs_libfrnn_results.json")
PLOT_PATH = os.path.join(PNG_DIR, "vs_libfrnn_comparison.png")

# Fixed A100 measurements for the default lowrank/K=16 sweep. MappingProxyType
# prevents accidental in-process mutation. LIBFRNN_MODE=stored (the default)
# runs only our FRNN implementation and uses these values for comparison.
STORED_LIBFRNN_MS = MappingProxyType({
    "D3_N200000": 1.6875634901225567,
    "D3_N500000": 4.220264032483101,
    "D3_N750000": 6.43969897646457,
    "D3_N1000000": 8.501932490617037,
    "D3_N1250000": 11.319194512907416,
    "D3_N1500000": 13.493136502802372,
    "D12_N200000": 5.978823988698423,
    "D12_N500000": 21.653761039488018,
    "D12_N750000": 60.55587547598407,
    "D12_N1000000": 68.0998710449785,
    "D12_N1250000": 100.64674098975956,
    "D12_N1500000": 268.1538970209658,
    "D16_N200000": 9.88839496858418,
    "D16_N500000": 28.637330513447523,
    "D16_N750000": 41.95324803004041,
    "D16_N1000000": 90.10353952180594,
    "D16_N1250000": 162.11757005658,
    "D16_N1500000": 178.52352902991697,
})
LIBFRNN_MODE = os.environ.get("LIBFRNN_MODE", "stored").lower()
if LIBFRNN_MODE not in {"stored", "live", "none"}:
    raise ValueError("LIBFRNN_MODE must be stored, live, or none")
if LIBFRNN_MODE == "stored" and (
    DIST != "lowrank" or INTRINSIC != 3 or LOWRANK_NOISE != 0.02
):
    raise ValueError(
        "stored libFRNN values require DIST=lowrank, INTRINSIC=3, "
        "and LOWRANK_NOISE=0.02; use LIBFRNN_MODE=live or none otherwise"
    )

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

    # Distances do not change during radius bisection. Cache them once instead
    # of recomputing all sample-to-point distances for every iteration. At the
    # largest requested cell (N=1.5M), this uses about 1.43 GiB of host memory
    # and removes 19 of the previous 20 full distance-computation passes.
    distances_sq = np.empty((len(qi), len(pts)), dtype=np.float32)
    for i, query in enumerate(qs):
        diff = pts - query
        distances_sq[i] = (diff * diff).sum(axis=1)
        if (i + 1) % 32 == 0 or i + 1 == len(qi):
            print(f"  [radius] cached {i + 1}/{len(qi)} sampled queries", flush=True)

    target_total = target * len(qi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if int((distances_sq <= mid * mid).sum()) < target_total:
            lo = mid
        else:
            hi = mid
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
        torch.cuda.synchronize()   # GPU idle before clock starts
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()   # all GPU work done before clock stops
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000.0


def run_frnn_gpu(pts_t, N, D, R):
    """Our engine: data already on device, results stay on device. No H2D/D2H."""
    pts_soa = pts_t.T.contiguous().reshape(-1)
    engine = frnn_cuda.FRNNEngine(max_points=N)
    torch.cuda.empty_cache()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    latency_ms = timed_gpu(lambda: engine.search_gpu(pts_soa.data_ptr(), N, D, K, R))
    peak_mb = max(0, pynvml.nvmlDeviceGetMemoryInfo(handle).used - mem_before) / 1024**2
    return {"latency_ms": latency_ms, "peak_mb": float(peak_mb)}


def _libfrnn_session(N, D):
    """Return a pre-allocated _frnn.Session, or None if unavailable."""
    try:
        import _frnn as _lf
    except ImportError:
        return None, None
    if D > 32 or not hasattr(_lf, "Session"):
        return None, None
    return _lf.Session(N, D, K), _lf


def run_libfrnn_gpu(pts_t, N, D, R):
    """libFRNN kernel-only: AoS data already on device, pre-allocated workspace.
    Uses the same timed_gpu() helper as run_frnn_gpu() for identical sync semantics."""
    session, _ = _libfrnn_session(N, D)
    if session is None:
        return {"latency_ms": None, "peak_mb": None}
    pts_aos = pts_t.contiguous()
    dev_ptr = pts_aos.data_ptr()
    torch.cuda.empty_cache()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
    latency_ms = timed_gpu(lambda: session.search_device(dev_ptr, R))
    peak_mb = max(0, pynvml.nvmlDeviceGetMemoryInfo(handle).used - mem_before) / 1024**2
    return {"latency_ms": latency_ms, "peak_mb": float(peak_mb)}


def plot_results(results, path=PLOT_PATH):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot] matplotlib unavailable ({e}); skipping")
        return

    methods = [
        ("FRNN",    "latency_ms",     "o", "-",  "#1f77b4"),
        ("libFRNN", "libfrnn_gpu_ms", "^", "--", "#d62728"),
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
    fig.suptitle(
        "FRNN vs libFRNN — kernel-only (device ptr in, results on device)\n"
        "pre-allocated workspace · 20 warmup + 10 timed trials · median latency",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


if "--plot-only" in sys.argv:
    with open(RESULTS_PATH) as f:
        plot_results(json.load(f))
    sys.exit(0)


pynvml.nvmlInit()
warmup_gpu()
all_results = {}

print(f"\n=== FRNN vs xju2/libFRNN  |  kernel-only  |  distribution: {DIST}"
      + (f" (intrinsic={INTRINSIC}, noise={LOWRANK_NOISE})" if DIST == "lowrank" else "")
      + " ===")
print("  Device ptr in, results on device. Pre-allocated workspace. No H2D/D2H.")
print(f"  libFRNN mode: {LIBFRNN_MODE}\n")

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
        print(f"  [FRNN ERROR] {e}")
        frnn_res = {"latency_ms": None, "peak_mb": None}

    if LIBFRNN_MODE == "live":
        try:
            lf_res = run_libfrnn_gpu(pts_t, N, D, R)
        except Exception as e:
            print(f"  [libFRNN ERROR] {e}")
            lf_res = {"latency_ms": None, "peak_mb": None}
    elif LIBFRNN_MODE == "stored":
        lf_res = {"latency_ms": STORED_LIBFRNN_MS.get(key), "peak_mb": None}
        if lf_res["latency_ms"] is None:
            print(f"  [libFRNN] no stored result for {key}")
    else:
        lf_res = {"latency_ms": None, "peak_mb": None}

    del pts_t
    torch.cuda.empty_cache()

    f_ms  = frnn_res["latency_ms"]
    lf_ms = lf_res["latency_ms"]

    all_results[key] = {
        "R": R, "dist": DIST, "radius_mode": r_mode,
        **frnn_res,
        "libfrnn_gpu_ms": lf_ms,
    }

    print(f"  FRNN:{f_ms}ms  libFRNN:{lf_ms}ms")

    if f_ms is not None and lf_ms is not None:
        if f_ms > lf_ms:
            print(f"  !! REGRESSION: FRNN {f_ms:.2f}ms"
                  f" > libFRNN {lf_ms:.2f}ms — see §4.3 diagnostics")
        else:
            print(f"  FRNN is {lf_ms/f_ms:.2f}x faster")

os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n→ {RESULTS_PATH}")
plot_results(all_results)
pynvml.nvmlShutdown()
