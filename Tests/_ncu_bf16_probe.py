#!/usr/bin/env python3
# _ncu_bf16_probe.py — single-launch driver for ncu to profile the D=16
# brute-force kernel (TiledBruteforce16Kernel). No PyTorch in this process.
#
# Purpose: settle whether the D=16 BF path is FP32-ALU-bound (tensor cores
# have headroom) or latency/occupancy-bound (tensor cores won't help at D=16).
#
# R is irrelevant to the measurement: BF computes all N*N distances regardless;
# the radius only gates heap inserts, not the dominant distance loop. We warm up
# OUTSIDE the profiled region, then do exactly ONE timed search so ncu captures
# a single clean launch (use --launch-count 1).
import ctypes, numpy as np, frnn_cuda

N, D, K, R = 100_000, 16, 16, 0.5
rng = np.random.default_rng(1234)
# SoA flat layout (d-major): the kernel reads p1[d*N + i]; engine expects N*D flat.
pts = rng.random((N, D), dtype=np.float32)
pts_flat = pts.reshape(-1)  # engine auto-detects dim = size/max_points

_cu = ctypes.CDLL("libcudart.so")
sync = _cu.cudaDeviceSynchronize

engine = frnn_cuda.FRNNEngine(max_points=N)
for _ in range(5):                 # warm up engine + driver (NOT profiled)
    engine.search(pts_flat, K=K, radius=R)
sync()

engine.search(pts_flat, K=K, radius=R)   # <-- the one launch ncu profiles
sync()
print("done")
