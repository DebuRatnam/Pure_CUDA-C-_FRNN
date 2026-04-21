import frnn_cuda
import numpy as np
import time
import pynvml
import ctypes
import json
import math

_cudart = ctypes.CDLL('libcudart.so')

def run_scaling_benchmark():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except:
        handle = None
        print("[WARN] NVML not found. Hardware metrics will be skipped.")

    # 2. Benchmark Parameters
    try:
        user_dim = int(input("Enter Dimension for sweep (e.g., 3, 8, 16): "))
    except ValueError:
        user_dim = 3

    # Define K and the Auto-Radius Logic
    user_k = 16 
    MAX_TOTAL_CELLS = 2**18 
    
    # Mathematical Radius Scaling
    base_r = 1.0 / (MAX_TOTAL_CELLS ** (1.0 / user_dim))
    density_boost = math.sqrt(user_dim / 3.0) 
    user_r = max(0.01, min(0.8, base_r * density_boost))
    
    n_counts = [2**i for i in range(14, 21)]
    num_trials = 7
    results = []

    print(f"\n" + "="*60)
    print(f"LHC FRNN SCALING TEST ({user_dim}D)")
    print("="*60)
    print(f"Fixed Parameters: K={user_k}, Radius={user_r:.4f}")
    print(f"Methodology: {num_trials} trials per N, Median result recorded")

    # 3. Warmup Phase
    print("\n[Phase 1/2] Performing Burn-in run...")
    warmup_n = 16384
    engine_warmup = frnn_cuda.FRNNEngine(max_points=warmup_n)
    warmup_hits = np.random.rand(warmup_n * user_dim).astype(np.float32)
    engine_warmup.search(warmup_hits, K=user_k, radius=user_r)
    _cudart.cudaDeviceSynchronize()
    del engine_warmup 

    # 4. The Sweep
    print("[Phase 2/2] Starting Particle Sweep...")
    for n in n_counts:
        print(f"  Testing N = {n:<8}...", end=" ", flush=True)
        
        if handle:
            _cudart.cudaDeviceSynchronize()
            m_baseline = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        
        engine = frnn_cuda.FRNNEngine(max_points=n)
        
        if handle:
            _cudart.cudaDeviceSynchronize()
            m_peak = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            mem_usage = (m_peak - m_baseline) / 1024**2 
        
        hits = np.random.rand(n * user_dim).astype(np.float32)
        trial_times = []

        for t in range(num_trials):
            _cudart.cudaDeviceSynchronize()
            start = time.perf_counter()
            engine.search(hits, K=user_k, radius=user_r)
            _cudart.cudaDeviceSynchronize()
            end = time.perf_counter()
            trial_times.append((end - start) * 1000)

        median_time = np.median(trial_times)
        results.append({
            "n": n,
            "latency_ms": median_time,
            "memory_mb": max(0.0, mem_usage) if handle else 0.0
        })
        print(f"Median: {median_time:.3f} ms | Mem: {mem_usage:.1f} MB")
        del engine

    # 5. Final Report
    print("\n" + "="*60)
    print(f"{'N (Particles)':<15} | {'Median Latency (ms)':<20} | {'Peak Mem (MB)':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['n']:<15} | {r['latency_ms']:<20.4f} | {r['memory_mb']:<15.2f}")
    print("="*60)
    
    log_data = {
        "x_particles": [int(r['n']) for r in results],
        "y_latency": [float(r['latency_ms']) for r in results],
        "y_memory": [float(r['memory_mb']) for r in results],
        "config": {"dim": user_dim, "radius": user_r, "k": user_k}
    }
    print("\nRAW DATA FOR PLOTTING (JSON):")
    print(json.dumps(log_data))

    if handle:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    run_scaling_benchmark()