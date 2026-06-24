#!/usr/bin/env python3
"""_run_projection_isolated.py — isolated timing worker for the projection comparison.

Times ONE method (`old` brute-force D=16, or `proj` two-stage projection) in a clean
process. No torch (it contaminates the CUDA context / sync -- CLAUDE.md s1); the proj
path uses cupy. Reads a JSON spec from stdin, writes one JSON line to stdout.

stdin JSON:
  pts_path    : .npy of float32 points, shape (N*D,) row-major AoS  (required)
  N, D, K, R  : ints / float                                        (required)
  method      : "old" | "proj"                                      (required)
  oversample  : Stage-1 candidate count for proj (<=128, default 128)
  mode        : "pca" | "random"  (proj basis, default "pca")
  out_idx_path: where to .npy-dump the (N,K) neighbor ids for recall check (optional)
  warmup,trials: timing loop sizes (default 20 / 10)

stdout JSON: latency_ms, peak_mb, [project_ms, stage1_ms, verify_ms], [sat_frac], out_idx_path
"""
import sys, json, time, ctypes
import numpy as np
import frnn_cuda
import pynvml

data     = json.load(sys.stdin)
pts_flat = np.load(data["pts_path"]).astype(np.float32)   # (N*D,)
N, D, K  = int(data["N"]), int(data["D"]), int(data["K"])
R        = float(data["R"])
method   = data["method"]
WARMUP   = int(data.get("warmup", 20))
TRIALS   = int(data.get("trials", 10))

_cu  = ctypes.CDLL("libcudart.so")
sync = _cu.cudaDeviceSynchronize

engine = frnn_cuda.FRNNEngine(max_points=N)
result = {}

if method == "old":
    # Current D=16 path (brute force inside the engine). Also the exact ground truth.
    def run():
        return engine.search(pts_flat, K=K, radius=R)

    for _ in range(WARMUP):
        run()
    sync()

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem0 = pynvml.nvmlDeviceGetMemoryInfo(h).used
    times = []
    for _ in range(TRIALS):
        sync(); t0 = time.perf_counter()
        run()
        sync(); times.append(time.perf_counter() - t0)
    peak_mb = (pynvml.nvmlDeviceGetMemoryInfo(h).used - mem0) / 1024**2
    pynvml.nvmlShutdown()
    result["latency_ms"] = float(np.median(times)) * 1000.0
    result["peak_mb"]    = float(max(peak_mb, 0.0))

    if data.get("out_idx_path"):
        idxs, _ = run()                                   # untimed, for recall check
        np.save(data["out_idx_path"], np.asarray(idxs, dtype=np.int32).reshape(N, K))
        result["out_idx_path"] = data["out_idx_path"]

elif method == "proj":
    import cupy as cp
    import projection_frnn as pf

    oversample = int(data.get("oversample", 128))
    mode       = data.get("mode", "pca")
    pts_gpu    = cp.asarray(pts_flat.reshape(N, D))       # (N, D) on device, once

    def run(time_stages=False):
        return pf.projected_frnn(engine, pts_gpu, K, R, k_proj=3,
                                 oversample=oversample, mode=mode,
                                 time_stages=time_stages)

    for _ in range(WARMUP):
        run()
    sync()

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem0 = pynvml.nvmlDeviceGetMemoryInfo(h).used
    times = []
    for _ in range(TRIALS):
        sync(); t0 = time.perf_counter()
        run()
        sync(); times.append(time.perf_counter() - t0)
    peak_mb = (pynvml.nvmlDeviceGetMemoryInfo(h).used - mem0) / 1024**2
    pynvml.nvmlShutdown()
    result["latency_ms"] = float(np.median(times)) * 1000.0
    result["peak_mb"]    = float(max(peak_mb, 0.0))

    # One extra run for per-stage breakdown + saturation + idx dump (all untimed).
    idxs, dists, sat, stages = run(time_stages=True)
    result.update({k: float(v) for k, v in stages.items()})   # includes mean_cand
    result["sat_frac"] = float(cp.mean(sat.astype(cp.float32)).get())
    if data.get("out_idx_path"):
        np.save(data["out_idx_path"], cp.asnumpy(idxs).astype(np.int32))
        result["out_idx_path"] = data["out_idx_path"]

else:
    sys.stderr.write(f"unknown method: {method}\n")
    sys.exit(2)

print(json.dumps(result))
