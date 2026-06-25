#!/usr/bin/env python3
# validate_correctness.py — is our FRNN returning the *right* neighbors?
#
# Runs our FRNN (projection two-stage, auto-dispatched), xju2/lxxue FRNN, and an exact
# brute-force ground truth on identical clouds at every swept D, and cross-checks them.
# Set LOWRANK=k to validate on low intrinsic-dim data (the regime where projection engages).
#
# The crucial subtlety this validator gets right:
#   Our FRNN returns the K *nearest* points within the radius (textbook fixed-radius KNN).
#   xju2 returns *some* K points within the radius — when more than K are in range it does
#   NOT guarantee the nearest ones. So a naive "do the neighbor lists match?" check FAILS
#   on dense queries even though both are valid. The correct checks are:
#     1. ours vs brute-force truth: must match (proves we return the true K-nearest).
#     2. sparse queries (<=K points in radius, answer unambiguous): ours == xju2 (== truth).
#     3. dense queries (>K in radius): both must return only in-radius points; ours must
#        still match truth. The nearest-vs-any difference there is by design, not a bug.
#
# Two precision points that matter:
#   * Truth distances are computed in float64. The kernels compute Sum (qi-pi)^2 directly;
#     the textbook |q|^2+|p|^2-2 q.p form catastrophically cancels in float32 at small d^2,
#     so a float32 "truth" is actually *less* accurate than the kernel.
#   * Points within float32 epsilon of the radius are genuinely ambiguous (in or out), so a
#     small boundary band (TAU) is treated as don't-care.
#
#   Truth is computed on a SAMPLE of query points (distances to all N points), so the check
#   scales to any N without an N x N matrix.
#
#   Run from the repo root:  PYTHONPATH=. python3 Tests/validate_correctness.py
import os, sys
import numpy as np
import torch
import frnn_torch
from math import pi, gamma, ceil

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
for _p in (os.path.join(_ROOT, "xju2_frnn", "FRNN"),
           os.path.join(_ROOT, "xju2_frnn", "prefix_sum")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import frnn as xju2
from projection_frnn_torch import frnn_search_torch   # validate the projection algorithm

K, SEED = 16, 1234
D_SWEEP = [3, 16]                   # D=3 grid path, D=16 brute-force / projection path
N_SWEEP = [1_000, 10_000, 50_000, 100_000]
N_SAMPLE = 2_000                    # query points spot-checked against exact truth per cell
RTOL, ATOL = 1e-3, 1e-6
TAU = 1e-4                          # radius boundary band (relative), float32-ambiguous zone

# Match benchmark_master's data regime so the validator checks the SAME thing that gets
# timed: uniform (default) -> projection auto-falls-back to BF; LOWRANK=k -> projection engages.
LOWRANK       = int(os.environ.get("LOWRANK", "0"))
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


def ours(pts, R):
    # The actual algorithm the benchmark times: projection two-stage, auto-dispatched.
    idx, dist = frnn_search_torch(frnn_torch.FRNNTorch(pts.shape[0]), pts, K, R)
    torch.cuda.synchronize()
    return idx.cpu().numpy(), dist.cpu().numpy()


def xju2_search(pts, R):
    N = pts.shape[0]
    L = torch.tensor([N], device="cuda")
    d, i, _, _ = xju2.frnn_grid_points(pts.unsqueeze(0), pts.unsqueeze(0), L, L, K, R)
    torch.cuda.synchronize()
    return i[0].cpu().numpy(), d[0].cpu().numpy()


def valid_set(idx_row, N):
    return set(int(i) for i in idx_row if 0 <= int(i) < N)


def valid_sorted_dists(idx_row, dist_row, N):
    return np.sort([float(d) for i, d in zip(idx_row, dist_row)
                    if 0 <= int(i) < N and np.isfinite(d)]).astype(np.float64)


def knn_match(a, b, r2_lo):
    # a (ours), b (truth): sorted ascending squared distances of the K-nearest in radius.
    # Mismatches are allowed only among boundary entries (>= r2_lo), where in/out is
    # float32-ambiguous; any disagreement on a clearly-inside distance is a real failure.
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
        pts = torch.tensor(pts_np, device="cuda")

        o_idx, o_dist = ours(pts, R)
        try:                                    # xju2 works at any D; guard a missing build
            x_idx, x_dist = xju2_search(pts, R)
            has_xju2 = True
        except Exception as e:
            has_xju2 = False
            print(f"    [xju2] unavailable: {e}")

        # Exact truth (float64) for a sample of queries: distances to ALL N points.
        rng = np.random.default_rng(SEED)
        S = min(N_SAMPLE, N)
        qs = rng.choice(N, S, replace=False)
        Pd = pts.double()
        Qd = Pd[qs]
        sqd = (Pd * Pd).sum(1)
        d2 = (Qd * Qd).sum(1)[:, None] + sqd[None, :] - 2.0 * (Qd @ Pd.T)
        d2 = d2.clamp_(min=0).cpu().numpy()

        r2_lo, r2_hi = R * R * (1 - TAU), R * R * (1 + TAU)
        ours_truth = sparse = dense = sparse_ok = 0
        ours_invalid = xju2_invalid = 0
        for r, q in enumerate(qs):
            row = d2[r]
            core = set(np.where(row <= r2_lo)[0].tolist())                 # definitely in
            inrad = core | set(np.where(row <= r2_hi)[0].tolist())         # in + boundary
            true_knn = np.sort(row[row <= r2_hi])[:K]

            o_set = valid_set(o_idx[q], N)
            o_d = valid_sorted_dists(o_idx[q], o_dist[q], N)
            ours_truth += knn_match(o_d, true_knn, r2_lo)
            ours_invalid += not o_set.issubset(inrad)
            sparse += (len(inrad) <= K)
            dense += (len(inrad) > K)
            if has_xju2:               # unambiguous queries: ours and xju2 must agree (== truth)
                x_set = valid_set(x_idx[q], N)
                xju2_invalid += not x_set.issubset(inrad)
                if len(inrad) <= K:
                    sparse_ok += (core <= o_set <= inrad) and (core <= x_set <= inrad)

        # Correctness gate: ours always checked vs the brute-force truth; the xju2 cross-check
        # applies whenever the xju2 build is present (it runs at any D).
        cell_ok = (ours_truth == S and ours_invalid == 0
                   and (not has_xju2 or (sparse_ok == sparse and xju2_invalid == 0)))
        all_pass &= cell_ok
        print(f"  D{D}_N{N:<6} R={R:.5f}")
        print(f"    ours vs brute-force truth (K-nearest): {ours_truth}/{S}"
              f"   {'OK' if ours_truth == S else 'FAIL'}")
        if has_xju2:
            print(f"    sparse (<=K in radius): {sparse:>4}   ours==xju2(==truth): {sparse_ok}/{sparse}"
                  f"   {'OK' if sparse_ok == sparse else 'FAIL'}")
            print(f"    dense  ( >K in radius): {dense:>4}   (ours=nearest-K matches truth; "
                  f"xju2=any-K — by design)")
            print(f"    out-of-radius neighbors:  ours={ours_invalid}  xju2={xju2_invalid}")
        else:
            print(f"    xju2: unavailable — ours validated against brute-force truth")
            print(f"    out-of-radius neighbors:  ours={ours_invalid}")
        print(f"    => {'PASS' if cell_ok else 'FAIL'}\n")

        del pts
        torch.cuda.empty_cache()

print("=" * 70)
if all_pass:
    print("ALL PASS — our FRNN returns the exact K-nearest neighbors in radius\n"
          "(matches the float64 brute-force truth, and matches xju2 wherever the\n"
          "answer is unambiguous). The only ours-vs-xju2 differences are dense\n"
          "queries where xju2 returns any-K-in-radius rather than the nearest-K.")
else:
    print("FAILURES above — investigate.")
sys.exit(0 if all_pass else 1)
