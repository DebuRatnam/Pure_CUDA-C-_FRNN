#!/usr/bin/env python3
# _run_frnn_isolated.py — pure-CUDA FRNN timing worker. No PyTorch in this process.
import sys, json, time, ctypes, numpy as np, frnn_cuda, pynvml

data     = json.load(sys.stdin)
pts_flat = np.load(data["pts_path"])          # float32, shape (N*D,)
N, D, K, R = data["N"], data["D"], data["K"], data["R"]
WARMUP   = data.get("warmup", 20)
TRIALS   = data.get("trials", 10)

_cu  = ctypes.CDLL("libcudart.so")
sync = _cu.cudaDeviceSynchronize

engine = frnn_cuda.FRNNEngine(max_points=N)
for _ in range(WARMUP):
    engine.search(pts_flat, K=K, radius=R)
sync()

pynvml.nvmlInit()
handle     = pynvml.nvmlDeviceGetHandleByIndex(0)
mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used

times = []
for _ in range(TRIALS):
    sync()
    t0 = time.perf_counter()
    engine.search(pts_flat, K=K, radius=R)
    sync()
    times.append(time.perf_counter() - t0)

peak_mb = (pynvml.nvmlDeviceGetMemoryInfo(handle).used - mem_before) / 1024**2
pynvml.nvmlShutdown()

print(json.dumps({
    "latency_ms": float(np.median(times)) * 1000.0,
    "peak_mb":    float(max(peak_mb, 0.0)),
}))
