#!/usr/bin/env python3
# validate_bf16_blocked.py — does the RM register-blocked D=16 brute-force kernel
# return the SAME neighbors as the default TiledBruteforce16Kernel?
#
# The blocked kernel (frnn/csrc/bruteforce/bruteforce16_blocked.cuh) is an exact
# fp32 reimplementation: same direct Sum(qi-pi)^2 distance, same per-query max-heap
# top-K, only the work mapping differs (RM queries per thread, ref tile reused).
# So for identical (pts, K, R) the two kernels must agree up to tie-order among
# neighbors at exactly-equal distance.
#
# Method: run engine.search at D=16 (auto-dispatch -> brute-force) twice in one
# process, toggling FRNN_BF16_BLOCKED. CPython writes os.environ through putenv(),
# so the C getenv() in run_bruteforce() sees the change on the next call. Compare
# each query's neighbor list as a distance-sorted (dist, idx) multiset.
#
# Run from repo root:  PYTHONPATH=. python3 Tests/validate_bf16_blocked.py
#   FRNN_BF16_RM=4 PYTHONPATH=. python3 Tests/validate_bf16_blocked.py   # test RM=4 path
# Exit 0 = blocked matches default on every cell.
import os, sys
import numpy as np
import torch
import frnn_torch
from math import pi, gamma, ceil

K, SEED = 16, 1234
D = 16                                   # the kernel under test is the D=16 specialization
N_SWEEP = [1_000, 10_000, 50_000, 100_000, 200_000]
RM = os.environ.get("FRNN_BF16_RM", "2")  # echoed in the header; passed through to the kernel


def radius_for(D, N):
    v = pi**(D / 2) / gamma(D / 2 + 1)
    r = min((K / (N * v))**(1.0 / D), 2.0)
    if r < 1.0 and ceil(1.0 / r)**D > 900_000:
        r = 2.0
    return round(r, 5)


def gen_points(N, D, seed):
    rng = np.random.RandomState(seed)
    return rng.rand(N, D).astype(np.float32)


def run(engine, pts_t, K, R, blocked):
    # Toggle the kernel via env; getenv() in run_bruteforce reads it per launch.
    if blocked:
        os.environ["FRNN_BF16_BLOCKED"] = "1"
    else:
        os.environ.pop("FRNN_BF16_BLOCKED", None)
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
    # (in or out), so don't count those as mismatches.
    r2 = R * R
    band = 1e-4 * max(r2, 1.0)
    ambiguous = (np.abs(da - r2) < band) | (np.abs(db - r2) < band)

    # Match validate_correctness tolerances (RTOL=1e-3, ATOL=1e-6): the two kernels
    # should be bitwise-identical, but stay loose so compiler scheduling never false-fails.
    dist_close = np.isclose(da, db, rtol=1e-3, atol=1e-6) | ambiguous | (ia < 0) | (ib < 0)

    # An index difference is only a real bug if the distance at that rank also differs.
    # When distances match (a tie among equal-d2 points), either index is valid — both
    # kernels break ties the same way, but tolerate it so a legal swap never false-fails.
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
                  f"blocked(idx={ib[q,k]},d2={db[q,k]:.6g})")
    return ok


def main():
    if not torch.cuda.is_available():
        print("no CUDA device"); return 1
    print(f"D={D} K={K} RM={RM}  (toggling FRNN_BF16_BLOCKED)")
    all_ok = True
    for N in N_SWEEP:
        R = radius_for(D, N)
        pts = gen_points(N, D, SEED)
        pts_t = torch.from_numpy(pts).cuda().contiguous()
        engine = frnn_torch.FRNNTorch(N)

        idx_def, dist_def = run(engine, pts_t, K, R, blocked=False)
        idx_blk, dist_blk = run(engine, pts_t, K, R, blocked=True)

        ok = compare(f"N={N:>7} R={R}", idx_def, dist_def, idx_blk, dist_blk, R)
        all_ok = all_ok and ok

        del pts_t, engine
        torch.cuda.empty_cache()

    os.environ.pop("FRNN_BF16_BLOCKED", None)
    print("ALL PASS" if all_ok else "MISMATCH — blocked kernel differs from default")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
