#!/usr/bin/env python3
# validate_bf16_wmma.py — does the tensor-core (WMMA TF32) D=16 kernel return the
# SAME neighbors as the default TiledBruteforce16Kernel?
#
# The WMMA kernel computes d2 = ||q||^2 + ||t||^2 - 2 (q.t) with the q.t Gram
# product on the A100 tensor cores in TF32, then RECOMPUTES every kept neighbor's
# distance exactly (direct sum_(q-t)^2 from global) and drops any whose exact
# distance is >= r^2. So reported distances are the same exact fp32 the direct
# kernel produces; only the in/out radius decision in the TF32 boundary band can
# differ -- which the boundary mask below treats as ambiguous.
#
# Method: run engine.search at D=16 (auto-dispatch -> brute-force) twice in one
# process, toggling FRNN_BF16_WMMA. CPython writes os.environ through putenv(), so
# the C getenv() in run_bruteforce() sees the change on the next call. Compare each
# query's neighbor list as a distance-sorted (dist, idx) multiset.
#
# Run from repo root:  PYTHONPATH=. python3 Tests/validate_bf16_wmma.py
# Exit 0 = WMMA matches default on every cell.
import os, sys
import numpy as np
import torch
import frnn_torch
from math import pi, gamma, ceil

K, SEED = 16, 1234
D = 16                                   # the kernel under test is the D=16 specialization
N_SWEEP = [1_000, 10_000, 50_000, 100_000, 200_000]


def radius_for(D, N):
    v = pi**(D / 2) / gamma(D / 2 + 1)
    r = min((K / (N * v))**(1.0 / D), 2.0)
    if r < 1.0 and ceil(1.0 / r)**D > 900_000:
        r = 2.0
    return round(r, 5)


def gen_points(N, D, seed):
    rng = np.random.RandomState(seed)
    return rng.rand(N, D).astype(np.float32)


def run(engine, pts_t, K, R, wmma):
    # Toggle the kernel via env; getenv() in run_bruteforce reads it per launch.
    if wmma:
        os.environ["FRNN_BF16_WMMA"] = "1"
    else:
        os.environ.pop("FRNN_BF16_WMMA", None)
    idx, dist = engine.search(pts_t, K, float(R))
    torch.cuda.synchronize()
    return idx.cpu().numpy(), dist.cpu().numpy()


def sorted_rows(idx, dist):
    # Per row, order neighbors by (dist, idx) so tie-order differences don't cause
    # false mismatches. Padding entries (idx == -1) sort to the end via +inf dist.
    d = dist.copy()
    d[idx < 0] = np.inf
    order = np.lexsort((idx, d), axis=1)        # primary key d, secondary idx
    rows = np.arange(idx.shape[0])[:, None]
    return idx[rows, order], d[rows, order]


def compare(name, idx_a, dist_a, idx_b, dist_b, R):
    ia, da = sorted_rows(idx_a, dist_a)
    ib, db = sorted_rows(idx_b, dist_b)

    # Boundary band: neighbors within float32 eps of r^2 are genuinely ambiguous
    # (in or out), so the TF32 in/out decision flipping there is not a bug.
    r2 = R * R
    band = 1e-4 * max(r2, 1.0)
    ambiguous = (np.abs(da - r2) < band) | (np.abs(db - r2) < band)

    # Distances are exact-recomputed in the WMMA kernel, so they should match the
    # direct kernel tightly (RTOL=1e-3, ATOL=1e-6, same as validate_correctness).
    dist_close = np.isclose(da, db, rtol=1e-3, atol=1e-6) | ambiguous | (ia < 0) | (ib < 0)

    # An index difference is only a real bug if the distance at that rank also
    # differs (equal-d2 tie swaps are legal) and we are not in the boundary band.
    idx_mismatch = (ia != ib) & ~dist_close & ~ambiguous & (ia >= 0) & (ib >= 0)

    n_idx_bad = int(idx_mismatch.sum())
    n_dist_bad = int((~dist_close).sum())
    ok = (n_idx_bad == 0) and (n_dist_bad == 0)
    tag = "OK  " if ok else "FAIL"
    print(f"  [{tag}] {name}: idx_mismatch={n_idx_bad}  dist_mismatch={n_dist_bad}")
    if not ok:
        bad = np.argwhere(idx_mismatch | ~dist_close)[:5]
        for q, k in bad:
            print(f"        q={q} k={k}: default(idx={ia[q,k]},d2={da[q,k]:.6g}) "
                  f"wmma(idx={ib[q,k]},d2={db[q,k]:.6g})")
    return ok


def main():
    if not torch.cuda.is_available():
        print("no CUDA device"); return 1
    print(f"D={D} K={K}  (toggling FRNN_BF16_WMMA)")
    all_ok = True
    for N in N_SWEEP:
        R = radius_for(D, N)
        pts = gen_points(N, D, SEED)
        pts_t = torch.from_numpy(pts).cuda().contiguous()
        engine = frnn_torch.FRNNTorch(N)

        idx_def, dist_def = run(engine, pts_t, K, R, wmma=False)
        idx_wma, dist_wma = run(engine, pts_t, K, R, wmma=True)

        ok = compare(f"N={N:>7} R={R}", idx_def, dist_def, idx_wma, dist_wma, R)
        all_ok = all_ok and ok

        del pts_t, engine
        torch.cuda.empty_cache()

    os.environ.pop("FRNN_BF16_WMMA", None)
    print("ALL PASS" if all_ok else "MISMATCH — WMMA kernel differs from default")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
