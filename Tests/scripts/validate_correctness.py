#!/usr/bin/env python3
# validate_correctness.py — is our FRNN returning the *right* neighbors?
#
# Validates against an exact float64 brute-force ground truth on identical clouds
# at every swept D. No dependency on xju2 or any other library.
#
# Dispatch paths exercised by default sweep:
#   D=3              → full-D grid (always feasible)
#   D=12, D=16       → first-d grid (grid_dim=4) + inline full-D ranking
#
# The first-d grid guarantees no false negatives at the grid step: first-d distance
# is always <= full-D distance (fewer positive terms), so every true full-D neighbor
# lies in one of the scanned grid cells. Each scanned point is ranked in full D.
#
# Two precision points that matter:
#   * Truth distances are computed in float64. The kernels compute Sum (qi-pi)^2 directly;
#     the textbook |q|^2+|p|^2-2 q.p form catastrophically cancels in float32 at small d^2,
#     so a float32 "truth" is actually *less* accurate than the kernel.
#   * Points within float32 epsilon of the radius are genuinely ambiguous (in or out), so a
#     small boundary band (TAU) is treated as don't-care for boundary points only.
#
#   Run from the repo root:  PYTHONPATH=. python3 Tests/scripts/validate_correctness.py
import os, sys
import numpy as np
import frnn_cuda
from math import pi, gamma, ceil

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _p in (os.path.join(_ROOT, "xju2_frnn", "FRNN"),
           os.path.join(_ROOT, "xju2_frnn", "prefix_sum")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

K, SEED = 16, 1234
D_SWEEP = [3, 12, 16]
N_SWEEP = [1_000, 10_000, 50_000, 100_000, 200_000]
N_SAMPLE = 2_000
RTOL, ATOL = 1e-3, 1e-6
TAU = 1e-4

LOWRANK       = int(os.environ.get("LOWRANK", "4"))
LOWRANK_NOISE = float(os.environ.get("LOWRANK_NOISE", "0.02"))


def radius_for(D, N):
    v = pi**(D / 2) / gamma(D / 2 + 1)
    r = min((K / (N * v))**(1.0 / D), 2.0)
    if r < 1.0 and ceil(1.0 / r)**D > 900_000:
        r = 2.0
    return round(r, 5)


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


def ours(pts_np, R):
    N = len(pts_np)
    engine = frnn_cuda.FRNNEngine(max_points=N)
    idxs_flat, dists_flat = engine.search(pts_np.reshape(-1).tolist(), K, float(R))
    idx  = np.asarray(idxs_flat,  dtype=np.int32).reshape(N, K)
    dist = np.asarray(dists_flat, dtype=np.float32).reshape(N, K)
    return idx, dist


def valid_set(idx_row, N):
    return set(int(i) for i in idx_row if 0 <= int(i) < N)


def valid_sorted_dists(idx_row, dist_row, N):
    return np.sort([float(d) for i, d in zip(idx_row, dist_row)
                    if 0 <= int(i) < N and np.isfinite(d)]).astype(np.float64)


def knn_match(a, b, r2_lo):
    n = min(len(a), len(b))
    if n:
        bad = ~np.isclose(a[:n], b[:n], rtol=RTOL, atol=ATOL)
        if np.any(a[:n][bad] < r2_lo) or np.any(b[:n][bad] < r2_lo):
            return False
    return all(x >= r2_lo for x in list(a[n:]) + list(b[n:]))


print(f"Validating FRNN correctness  (K={K}, sample={N_SAMPLE} queries/cell)\n")

all_pass = True
for D in D_SWEEP:
    for N in N_SWEEP:
        pts_np = gen_points(N, D, SEED)
        R = calibrate_radius(pts_np) if LOWRANK > 0 and LOWRANK < D else radius_for(D, N)

        o_idx, o_dist = ours(pts_np, R)

        # Exact truth (float64) for a sample of queries — computed in numpy.
        rng = np.random.default_rng(SEED)
        S = min(N_SAMPLE, N)
        qs = rng.choice(N, S, replace=False)
        Pd = pts_np.astype(np.float64)
        Qd = Pd[qs]
        sqd = (Pd * Pd).sum(1)
        d2  = (Qd * Qd).sum(1)[:, None] + sqd[None, :] - 2.0 * (Qd @ Pd.T)
        d2  = np.maximum(d2, 0.0)   # float64, shape (S, N)

        r2_lo, r2_hi = R * R * (1 - TAU), R * R * (1 + TAU)
        ours_truth = 0
        ours_invalid = 0
        for r, q in enumerate(qs):
            row = d2[r]
            inrad = set(np.where(row <= r2_hi)[0].tolist())
            true_knn = np.sort(row[row <= r2_hi])[:K]

            o_set = valid_set(o_idx[q], N)
            o_d   = valid_sorted_dists(o_idx[q], o_dist[q], N)
            ours_truth   += knn_match(o_d, true_knn, r2_lo)
            ours_invalid += not o_set.issubset(inrad)

        cell_ok = (ours_truth == S and ours_invalid == 0)
        all_pass &= cell_ok
        print(f"  D{D}_N{N:<6} R={R:.5f}")
        print(f"    ours vs brute-force truth (K-nearest): {ours_truth}/{S}"
              f"   {'OK' if ours_truth == S else 'FAIL'}")
        print(f"    out-of-radius neighbors: {ours_invalid}")
        print(f"    => {'PASS' if cell_ok else 'FAIL'}\n")

print("=" * 70)
if all_pass:
    print("ALL PASS — our FRNN returns the exact K-nearest neighbors in radius\n"
          "(matches the float64 brute-force truth for all swept D and N).")
else:
    print("FAILURES above — investigate.")
sys.exit(0 if all_pass else 1)
