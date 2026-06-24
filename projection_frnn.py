#!/usr/bin/env python3
"""projection_frnn.py — D=16->D=3 projection FRNN sub-algorithm (A100, GPU-resident).

Two-stage exact fixed-radius neighbor search:

  Stage 0 (project)  : project the D-dim points onto an orthonormal k-dim basis
                       (PCA by default). Orthonormal projection is contractive:
                       ||P x - P y|| <= ||x - y||. A radius-R search in the
                       projection therefore returns a SUPERSET of the true D-dim
                       R-neighbors -- no false negatives.
  Stage 1 (filter)   : run the engine's fast k=3 grid path on the projected,
                       isotropically-rescaled coords to get candidate ids.
  Stage 2 (verify)   : recompute true D-dim distances to the candidates, keep
                       those <= R, take the K nearest. This makes the result EXACT.

Everything runs on the GPU via cupy + the engine's device-pointer entry
`FRNNEngine.search_gpu`; data never leaves the A100 between stages. No torch
(it would contaminate CUDA timing -- see CLAUDE.md s1).

The win regime is low intrinsic-dimension data, where the projection is
near-isometric so the k=3 filter is selective and Stage 2 is cheap. On full-rank
uniform data the filter has ~no selectivity and this is slower than brute force --
the comparison harness measures exactly that.
"""
import cupy as cp


# ---------------------------------------------------------------------------
# Projection bases
# ---------------------------------------------------------------------------
def pca_basis(pts_gpu, k=3):
    """Top-k PCA axes of pts_gpu (N, D) on the GPU -> orthonormal basis (D, k).

    Columns are the eigenvectors of the covariance with the k largest
    eigenvalues. Orthonormal => the projection is contractive (exact two-stage).
    """
    X = pts_gpu - pts_gpu.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / X.shape[0]                 # (D, D), tiny for D=16
    evals, evecs = cp.linalg.eigh(cov)           # ascending eigenvalues
    basis = evecs[:, -k:]                         # k largest -> (D, k), orthonormal
    return cp.ascontiguousarray(basis.astype(cp.float32))


def random_basis(D, k, seed=0):
    """Orthonormalized Gaussian basis (D, k). APPROXIMATE filter only.

    A random projection is near distance-preserving (Johnson-Lindenstrauss) but
    is NOT contractive at k=3, so it can drop true neighbors. Recall is measured,
    not guaranteed. Provided only as a baseline against PCA.
    """
    rng = cp.random.RandomState(seed)
    G = rng.standard_normal((D, k), dtype=cp.float32)
    Q, _ = cp.linalg.qr(G)                        # orthonormal columns
    return cp.ascontiguousarray(Q.astype(cp.float32))


def project_and_rescale(pts_gpu, basis):
    """Project (N, D) onto basis (D, k), then isotropically map into [0,1]^k.

    Returns (proj01 (N,k) in [0,1], s) where s is the distance scale factor:
    the rescale multiplies every pairwise distance by exactly s, so a Stage-2
    radius R becomes R*s in the rescaled space. A SINGLE global divisor is used
    (not per-axis) so the map stays a similarity transform and the contractive
    superset guarantee is preserved.
    """
    proj = pts_gpu @ basis                        # (N, k); centering cancels in diffs
    mn = proj.min(axis=0, keepdims=True)
    mx = proj.max(axis=0, keepdims=True)
    global_range = float((mx - mn).max())
    if global_range <= 0.0:
        global_range = 1.0                        # degenerate (all points identical)
    s = 1.0 / global_range
    proj01 = ((proj - mn) * s).astype(cp.float32)  # in [0,1]^k, isotropically scaled
    return cp.ascontiguousarray(proj01), s


# ---------------------------------------------------------------------------
# Stage 1: candidate generation via the engine's k=3 grid path
# ---------------------------------------------------------------------------
def _wrap_device_int(ptr, n):
    """Wrap a raw device int32* (returned by search_gpu) as a cupy view, no copy."""
    mem = cp.cuda.UnownedMemory(ptr, n * 4, owner=None)
    return cp.ndarray((n,), dtype=cp.int32, memptr=cp.cuda.MemoryPointer(mem, 0))


def candidate_ids(engine, proj01, N, k_proj, R_search, oversample):
    """Stage 1: k=3 grid search on projected coords -> candidate ids (oversample, N).

    Feeds the engine its required SoA layout (coord d, point p at index d*N+p) and
    reads back d_idxs as the SoA block idxs[k*N + p]. Values are ORIGINAL point ids
    (the engine maps neighbor ids back through its sort), or -1 for empty slots.
    """
    proj_soa = cp.ascontiguousarray(proj01.T.reshape(-1))   # (k*N,) SoA, float32
    idx_ptr, _dist_ptr = engine.search_gpu(
        int(proj_soa.data.ptr), int(N), int(k_proj), int(oversample), float(R_search))
    cand_flat = _wrap_device_int(idx_ptr, N * oversample)
    return cand_flat.reshape(oversample, N), proj_soa  # keep proj_soa alive for the caller


# ---------------------------------------------------------------------------
# Stage 2: exact D-dim verify
# ---------------------------------------------------------------------------
def verify_exact(pts_gpu, cand, K, R, chunk=50_000):
    """Stage 2: recompute true D-dim distances to candidates, keep K nearest <= R.

    cand: (oversample, N) original ids (-1 = empty). pts_gpu: (N, D) AoS.
    Returns (idxs (N,K) int32, dists (N,K) float32, saturated (N,) bool).
    Chunked over queries to bound the transient (oversample, chunk, D) gather.
    'saturated' = every candidate slot was filled AND all within R -> the
    oversample-nearest-in-projection cut may have dropped a true neighbor (recall risk).
    """
    O, N = cand.shape
    D = pts_gpu.shape[1]
    r2 = R * R
    out_idx = cp.full((N, K), -1, dtype=cp.int32)
    out_dst = cp.full((N, K), cp.inf, dtype=cp.float32)
    saturated = cp.zeros(N, dtype=cp.bool_)

    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        cc = cand[:, c0:c1]                         # (O, M) ids, -1 = empty
        M = c1 - c0
        valid = cc >= 0
        safe = cp.where(valid, cc, 0)               # clamp -1 so the gather is in-range
        cand_xyz = pts_gpu[safe.reshape(-1)].reshape(O, M, D)   # (O, M, D)
        q = pts_gpu[c0:c1][cp.newaxis, :, :]        # (1, M, D) query coords
        diff = cand_xyz - q
        d2 = cp.sum(diff * diff, axis=2)            # (O, M)
        d2 = cp.where(valid & (d2 <= r2), d2, cp.inf)

        # K nearest of the O candidates per query
        kk = min(K, O)
        part = cp.argpartition(d2, kth=kk - 1, axis=0)[:kk]      # (kk, M)
        cols = cp.arange(M)
        part_d2 = d2[part, cols]                                 # (kk, M)
        order = cp.argsort(part_d2, axis=0)                      # sort the kk
        sel = cp.take_along_axis(part, order, axis=0)            # (kk, M) row indices into O
        sel_d2 = cp.take_along_axis(part_d2, order, axis=0)      # (kk, M)
        keep = cp.isfinite(sel_d2)
        sel_ids = cp.where(keep, cp.take_along_axis(cc, sel, axis=0), -1)

        out_idx[c0:c1, :kk] = sel_ids.T
        out_dst[c0:c1, :kk] = cp.where(keep, cp.sqrt(sel_d2), cp.inf).T
        saturated[c0:c1] = cp.all(valid, axis=0)    # no empty slot -> potential miss

    return out_idx, out_dst, saturated


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def projected_frnn(engine, pts16_gpu, K, R, k_proj=3, oversample=128,
                   mode="pca", seed=0, time_stages=False):
    """Exact (for orthonormal mode) two-stage projection FRNN, fully on GPU.

    pts16_gpu: (N, D) float32 cupy array. Returns (idxs (N,K), dists (N,K),
    saturated (N,)). If time_stages, also returns a dict of per-stage ms.
    """
    N, D = pts16_gpu.shape

    def _ev():
        e = cp.cuda.Event(); e.record(); return e

    t = {}
    e0 = _ev()
    basis = pca_basis(pts16_gpu, k_proj) if mode == "pca" else random_basis(D, k_proj, seed)
    proj01, s = project_and_rescale(pts16_gpu, basis)
    e1 = _ev()
    cand, _proj_soa = candidate_ids(engine, proj01, N, k_proj, R * s, oversample)
    e2 = _ev()
    idxs, dists, saturated = verify_exact(pts16_gpu, cand, K, R)
    e3 = _ev()

    if time_stages:
        e3.synchronize()
        t["project_ms"] = cp.cuda.get_elapsed_time(e0, e1)
        t["stage1_ms"]  = cp.cuda.get_elapsed_time(e1, e2)
        t["verify_ms"]  = cp.cuda.get_elapsed_time(e2, e3)
        # selectivity: mean Stage-1 candidates per query (lower = filter more useful)
        t["mean_cand"]  = float(cp.mean((cand >= 0).sum(axis=0).astype(cp.float32)).get())
        return idxs, dists, saturated, t
    return idxs, dists, saturated
