#!/usr/bin/env python3
"""projection_frnn_torch.py — torch-native D->3 projection FRNN for the in-process
benchmark (benchmark_master.py uses the zero-copy `frnn_torch` extension on GPU tensors).

This is the SAME two-stage algorithm as projection_frnn.py (cupy worker version), but
written in torch so it drops into the benchmark's GPU-resident timing loop with no
cupy / no host copies.

`frnn_search_torch` is a drop-in replacement for `FRNNTorch.search`: same (idx, dist)
(N, K) tensor outputs, but auto-dispatched so it is EXACT and never regresses:

    D <= k_proj            -> existing grid path (projection pointless at low D)
    D > k_proj, low-rank   -> projection two-stage (PCA->3D filter + full-D verify)
    D > k_proj, full-rank  -> existing brute-force path (exact; projection would be
                              slower AND lossy here, so we don't use it)

"low-rank" = the top-k_proj PCA directions capture >= var_thresh of the total variance.
That is exactly the condition under which the contractive PCA projection stays selective,
so the projection filter is both fast and (after the full-D verify) exact.
"""
import torch


def _pca(pts):
    """Return (top3 basis (D,3), centered pts X (N,D), var_ratio of top-3)."""
    X = pts - pts.mean(0, keepdim=True)
    cov = (X.t() @ X) / X.shape[0]                  # (D, D), small
    evals, evecs = torch.linalg.eigh(cov)           # ascending eigenvalues
    var_ratio = float(evals[-3:].sum() / evals.sum().clamp_min(1e-12))
    return evecs[:, -3:].contiguous(), X, var_ratio


def _verify(pts, cand_idx, K, R, chunk=100_000):
    """Stage 2 (exact, full-D): recompute true distances to candidates, keep K nearest <= R.

    pts: (N, D). cand_idx: (N, O) original ids (-1 = empty). Returns (N,K) idx/dist.
    Chunked over queries to bound the transient (chunk, O, D) gather.
    """
    N, D = pts.shape
    O = cand_idx.shape[1]
    r2 = R * R
    out_idx = torch.full((N, K), -1, dtype=torch.int32, device=pts.device)
    out_dst = torch.full((N, K), float("inf"), dtype=torch.float32, device=pts.device)

    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        cc = cand_idx[c0:c1].long()                 # (M, O)
        valid = cc >= 0
        safe = cc.clamp_min(0)
        cand_xyz = pts[safe.reshape(-1)].reshape(c1 - c0, O, D)   # (M, O, D)
        q = pts[c0:c1].unsqueeze(1)                  # (M, 1, D)
        d2 = ((cand_xyz - q) ** 2).sum(-1)           # (M, O)
        d2 = torch.where(valid & (d2 <= r2), d2, torch.full_like(d2, float("inf")))

        kk = min(K, O)
        dk, pk = torch.topk(d2, kk, dim=1, largest=False)        # (M, kk) SQUARED distances
        ids = torch.gather(cc, 1, pk).to(torch.int32)
        keep = torch.isfinite(dk)
        out_idx[c0:c1, :kk] = torch.where(keep, ids, torch.full_like(ids, -1))
        # Return SQUARED distance to match the native engine convention (kernels store d^2).
        out_dst[c0:c1, :kk] = torch.where(keep, dk, torch.full_like(dk, float("inf")))
    return out_idx, out_dst


def frnn_search_torch(engine, pts, K, R, k_proj=3, oversample=128, var_thresh=0.9):
    """Drop-in for engine.search(pts, K, R). Auto-dispatched, exact (see module docstring).

    engine : frnn_torch.FRNNTorch
    pts    : CUDA float32 (N, D) AoS tensor.  Returns (idx, dist) (N, K) tensors.
    """
    N, D = pts.shape
    if D <= k_proj:
        return engine.search(pts, K, float(R))       # low D: native grid path

    basis, X, var_ratio = _pca(pts)
    if var_ratio < var_thresh:
        # Full-rank: projection has no selectivity and would drop neighbors -> use exact BF.
        return engine.search(pts, K, float(R))

    # --- projection two-stage (data is low intrinsic-dim) ---
    proj = X @ basis                                 # (N, 3); contractive projection
    mn = proj.min(0).values
    rng = (proj.max(0).values - mn).max().clamp_min(1e-12)
    s = float(1.0 / rng)                             # isotropic scale (preserves superset guarantee)
    proj01 = ((proj - mn) * s).contiguous()          # (N, 3) in [0,1]

    # Stage 1: fast 3D grid search -> candidate ids. radius R*s matches the isotropic scale.
    osamp = min(oversample, 128)                     # engine heap cap (MAX_K_CAPACITY)
    cand_idx, _ = engine.search(proj01, osamp, float(R * s))      # (N, osamp) ids / -1

    # Stage 2: exact full-D verify.
    return _verify(pts, cand_idx, K, R)
