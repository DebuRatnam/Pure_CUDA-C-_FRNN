#!/usr/bin/env python3
"""frnn_cupy.py — zero-copy CuPy interface to frnn_cuda.FRNNEngine.

Replaces frnn_torch.FRNNTorch without requiring PyTorch. Accepts and returns
CuPy (N, D) float32 arrays; no host<->device copies in the hot path.

search_gpu() expects SoA input (D*N flat) and returns SoA output pointers
into the engine's internal buffers (K*N each). This class handles the
AoS<->SoA transposes on the GPU via cp.ascontiguousarray.
"""
import cupy as cp
import frnn_cuda
from projection_frnn import pca_basis, project_and_rescale, candidate_ids, verify_exact


def _wrap_soa(ptr, K, N, dtype):
    """Wrap a raw SoA device pointer (K*N elements) as a (K, N) CuPy view."""
    nbytes = K * N * cp.dtype(dtype).itemsize
    mem = cp.cuda.UnownedMemory(ptr, nbytes, owner=None)
    arr = cp.ndarray((K * N,), dtype=dtype, memptr=cp.cuda.MemoryPointer(mem, 0))
    return arr.reshape(K, N)


class FRNNCuPy:
    """Zero-copy CuPy wrapper around frnn_cuda.FRNNEngine.

    Mirrors the FRNNTorch API: search() and search_projected() accept CuPy
    (N, D) float32 arrays and return (idx, dist) (N, K) CuPy arrays.
    """

    def __init__(self, max_points):
        self._engine = frnn_cuda.FRNNEngine(max_points=max_points)

    def search(self, pts, K, radius):
        """Zero-copy FRNN search.

        pts: CuPy (N, D) float32 AoS. Transposes to SoA, calls search_gpu
        (full grid/BF auto-dispatch), transposes results back to (N, K) AoS.
        Returns (idx (N,K) int32, dist (N,K) float32 squared). No H2D/D2H.
        """
        N, D = int(pts.shape[0]), int(pts.shape[1])
        # AoS (N, D) -> SoA (D, N) flat; search_gpu expects SoA layout.
        pts_soa = cp.ascontiguousarray(pts.T).reshape(-1)
        idx_ptr, dist_ptr = self._engine.search_gpu(
            int(pts_soa.data.ptr), N, D, K, float(radius))
        # Results live in engine's internal SoA buffers (K, N). Copy immediately
        # so they survive the next search_gpu call.
        idx_soa  = _wrap_soa(idx_ptr,  K, N, cp.int32).copy()
        dist_soa = _wrap_soa(dist_ptr, K, N, cp.float32).copy()
        # SoA (K, N) -> AoS (N, K)
        return cp.ascontiguousarray(idx_soa.T), cp.ascontiguousarray(dist_soa.T)

    def search_projected(self, pts, K, radius, oversample=128, var_thresh=0.9):
        """Auto-dispatched exact projection two-stage search.

        Mirrors FRNNTorch.search_projected: PCA-projects high-D data to 3D,
        runs a fast grid search for candidates, then verifies in full D-dim.
        Falls back to plain search() when data is full-rank (no selectivity).
        Returns (idx (N,K) int32, dist (N,K) float32 squared).
        """
        N, D = int(pts.shape[0]), int(pts.shape[1])
        k_proj = 3
        if D <= k_proj:
            return self.search(pts, K, radius)

        basis = pca_basis(pts, k_proj)
        X = pts - pts.mean(axis=0, keepdims=True)
        cov = (X.T @ X) / N
        evals, _ = cp.linalg.eigh(cov)
        var_ratio = float(evals[-k_proj:].sum() / evals.sum().clip(1e-12))

        if var_ratio < var_thresh:
            # Full-rank: projection has no selectivity; exact BF via search().
            return self.search(pts, K, radius)

        proj01, s = project_and_rescale(pts, basis)
        O = min(oversample, 128)
        cand, _proj_soa = candidate_ids(self._engine, proj01, N, k_proj, radius * s, O)
        idx, dist, _ = verify_exact(pts, cand, K, radius)
        return idx, dist
