#!/usr/bin/env python3
"""compare_projection.py — head-to-head: D=16 brute force ("old") vs D=16->D=3
projection two-stage ("proj"), on the A100. THIS is the file to run on Perlmutter.

Runs both methods in isolated subprocesses (clean CUDA timing, no torch -- CLAUDE.md s1)
over two data regimes, so the win/lose boundary is visible in numbers rather than asserted:

  uniform : full-rank uniform [0,1]^16. Projection has ~no selectivity here (R is large
            from volume concentration) -> expected PROJECTION LOSES.
  lowrank : points near a low-dim manifold embedded in 16D + noise. PCA->3D is
            near-isometric -> selective filter -> expected PROJECTION WINS, recall ~= 1.

Metrics per (dataset, N): latency old vs proj (+ speedup), proj per-stage breakdown,
recall vs the exact old result, precision, mean candidates/query (selectivity),
saturation fraction, peak GPU mem. Writes projection_comparison.json.

Usage:
  PYTHONPATH=. python3 compare_projection.py
  PYTHONPATH=. python3 compare_projection.py --N 100000 200000 --datasets lowrank \
      --intrinsic 3 --noise 0.02 --oversample 128 --mode pca
"""
import argparse, subprocess, sys, json, os, tempfile
import numpy as np
from math import pi, gamma, ceil

K, SEED, WARMUP, TRIALS = 16, 1234, 20, 10
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_run_projection_isolated.py")


def radius_for(D, N):
    """Radius giving ~K expected neighbors for uniform [0,1]^D (from benchmark_master.py)."""
    v = pi**(D / 2) / gamma(D / 2 + 1)
    r = min((K / (N * v))**(1.0 / D), 2.0)
    if r < 1.0 and ceil(1.0 / r)**D > 900_000:
        r = 2.0
    return round(r, 5)


# --------------------------------------------------------------------------- data
def gen_uniform(N, D, seed):
    rng = np.random.RandomState(seed)
    return rng.rand(N, D).astype(np.float32)


def gen_lowrank(N, D, intrinsic, noise, seed):
    """N points near an `intrinsic`-dim linear manifold in D-space + gaussian noise,
    then min-max normalized to [0,1]^D. Low intrinsic dim => PCA->3D near-isometric.
    Built in float64 (avoids spurious float32-BLAS warnings) then cast to float32."""
    rng = np.random.RandomState(seed)
    core  = rng.rand(N, intrinsic)
    embed = rng.standard_normal((intrinsic, D))
    Q, _  = np.linalg.qr(rng.standard_normal((D, D)))          # random rotation
    pts   = (core @ embed) @ Q + noise * rng.standard_normal((N, D))
    mn, mx = pts.min(0, keepdims=True), pts.max(0, keepdims=True)
    return ((pts - mn) / np.maximum(mx - mn, 1e-9)).astype(np.float32)


def avg_neighbor_count(pts, R, sample=256, seed=0):
    """Brute-force mean neighbor count on a random query sample (sanity on R)."""
    rng = np.random.RandomState(seed)
    qi = rng.choice(len(pts), size=min(sample, len(pts)), replace=False)
    r2 = R * R
    tot = 0
    for i in qi:
        d2 = ((pts - pts[i]) ** 2).sum(1)
        tot += int((d2 <= r2).sum())            # includes self
    return tot / len(qi)


def calibrate_radius(pts, target=K, sample=256, lo=1e-4, hi=2.0, iters=20):
    """Bisect R so the mean neighbor count ~= target for THIS dataset.

    radius_for() assumes uniform [0,1]^D; on low-rank data the same R captures
    orders of magnitude more neighbors. Calibrating per-dataset keeps each query
    at ~K true neighbors, so the projection filter's selectivity is meaningful."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if avg_neighbor_count(pts, mid, sample) < target:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 6)


# --------------------------------------------------------------------------- worker
def run_worker(pts, N, D, R, method, out_idx_path, oversample, mode):
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        pts_path = f.name
    np.save(pts_path, pts.reshape(-1))
    spec = dict(pts_path=pts_path, N=N, D=D, K=K, R=R, method=method,
                oversample=oversample, mode=mode, warmup=WARMUP, trials=TRIALS,
                out_idx_path=out_idx_path)
    try:
        p = subprocess.run([sys.executable, WORKER], input=json.dumps(spec),
                           capture_output=True, text=True)
    finally:
        os.unlink(pts_path)
    if p.returncode != 0:
        print(f"  [{method}] worker FAILED rc={p.returncode}\n{p.stderr}", file=sys.stderr)
        return None
    return json.loads(p.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- recall
def recall_precision(old_idx_path, proj_idx_path):
    """Per-query set overlap of proj vs the exact old neighbors. -1 entries ignored."""
    old = np.load(old_idx_path)                  # (N, K)
    prj = np.load(proj_idx_path)
    N = old.shape[0]
    rec_sum = prec_sum = rec_n = prec_n = 0.0
    for i in range(N):
        t = set(int(x) for x in old[i] if x >= 0)
        p = set(int(x) for x in prj[i] if x >= 0)
        if t:
            rec_sum += len(p & t) / len(t); rec_n += 1
        if p:
            prec_sum += len(p & t) / len(p); prec_n += 1
    return (rec_sum / rec_n if rec_n else 1.0,
            prec_sum / prec_n if prec_n else 1.0)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[100_000, 200_000, 500_000])
    ap.add_argument("--datasets", nargs="+", default=["uniform", "lowrank"],
                    choices=["uniform", "lowrank"])
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--intrinsic", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--oversample", type=int, default=128)
    ap.add_argument("--mode", default="pca", choices=["pca", "random"])
    ap.add_argument("--radius", type=float, default=None,
                    help="fixed R override; default = per-dataset calibration to ~K neighbors")
    ap.add_argument("--out", default="projection_comparison.json")
    args = ap.parse_args()

    results = []
    hdr = (f"{'dataset':<8}{'N':>8}{'R':>9}{'avgNbr':>8}{'old_ms':>10}{'proj_ms':>10}"
           f"{'speedup':>9}{'recall':>8}{'prec':>7}{'cand/q':>9}{'sat':>7}"
           f"  (proj/stg1/vrfy)")
    print(hdr); print("-" * len(hdr))

    for ds in args.datasets:
        for N in args.N:
            if ds == "uniform":
                pts = gen_uniform(N, args.D, SEED)
            else:
                pts = gen_lowrank(N, args.D, args.intrinsic, args.noise, SEED)
            R = args.radius if args.radius is not None else calibrate_radius(pts, target=K)
            avgn = avg_neighbor_count(pts, R)

            tmp = tempfile.mkdtemp()
            old_idx = os.path.join(tmp, "old.npy")
            prj_idx = os.path.join(tmp, "proj.npy")
            old = run_worker(pts, N, args.D, R, "old", old_idx, args.oversample, args.mode)
            prj = run_worker(pts, N, args.D, R, "proj", prj_idx, args.oversample, args.mode)
            if old is None or prj is None:
                continue

            recall = precision = float("nan")
            if os.path.exists(old_idx) and os.path.exists(prj_idx):
                recall, precision = recall_precision(old_idx, prj_idx)

            speedup = old["latency_ms"] / prj["latency_ms"] if prj["latency_ms"] else float("nan")
            row = dict(dataset=ds, N=N, R=R, avg_neighbors=avgn,
                       old_ms=old["latency_ms"], proj_ms=prj["latency_ms"], speedup=speedup,
                       recall=recall, precision=precision,
                       mean_cand=prj.get("mean_cand"), sat_frac=prj.get("sat_frac"),
                       project_ms=prj.get("project_ms"), stage1_ms=prj.get("stage1_ms"),
                       verify_ms=prj.get("verify_ms"),
                       old_peak_mb=old["peak_mb"], proj_peak_mb=prj["peak_mb"])
            results.append(row)

            print(f"{ds:<8}{N:>8}{R:>9.4f}{avgn:>8.1f}{old['latency_ms']:>10.2f}"
                  f"{prj['latency_ms']:>10.2f}{speedup:>8.2f}x{recall:>8.3f}{precision:>7.3f}"
                  f"{(prj.get('mean_cand') or 0):>9.1f}{(prj.get('sat_frac') or 0):>7.3f}"
                  f"  ({prj.get('project_ms',0):.2f}/{prj.get('stage1_ms',0):.2f}/{prj.get('verify_ms',0):.2f})")

            verdict = "PROJECTION WINS" if speedup > 1.0 else "PROJECTION LOSES"
            print(f"         -> {verdict} (latency)"
                  + ("" if recall >= 0.999 else f"   !! RECALL {recall:.3f} < 1.0 (not exact)"))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out} ({len(results)} rows)")


if __name__ == "__main__":
    main()
