import frnn_cuda
import numpy as np
import time
import pynvml
import ctypes

# Load CUDA runtime to allow manual synchronization without Torch
_cudart = ctypes.CDLL('libcudart.so')

def run_deterministic_test():
    # 1. Setup hardware monitoring
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"NVML Init failed: {e}. Hardware metrics may be unavailable.")
        handle = None
    
    # 2. Parameters & Determinism
    np.random.seed(1234)
    num_particles = 1000
    user_dim = 3
    user_k = 10
    user_r = 0.5
    
    print("--- FRNN Deterministic Performance Test (No-Torch) ---")
    print(f"Points: {num_particles} | K: {user_k} | Radius: {user_r} | Seed: 1234")

    # 3. Data Generation (3D coordinates)
    hits = np.random.rand(num_particles * user_dim).astype(np.float32)

    # 4. Initialize Engine
    engine = frnn_cuda.FRNNEngine(max_points=num_particles)

    # 5. Search with Metrics
    if handle:
        mem_info_start = pynvml.nvmlDeviceGetMemoryInfo(handle)
    
    # Ensure GPU is idle/ready
    _cudart.cudaDeviceSynchronize()
    
    start_time = time.perf_counter()
    
    # CORE SEARCH CALL
    indices, distances = engine.search(hits, K=user_k, radius=user_r)
    
    # Force CPU to wait for GPU to finish before stopping clock
    _cudart.cudaDeviceSynchronize()
    end_time = time.perf_counter()
    
    duration_ms = (end_time - start_time) * 1000

    # 6. Gather Hardware Stats
    if handle:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info_end = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_util = f"{util.gpu}%"
        gpu_mem = f"{(mem_info_end.used - mem_info_start.used) / 1024**2:.2f} MB"
    else:
        gpu_util = "N/A"
        gpu_mem = "N/A"

    # 7. Results Display
    indices_res = np.array(indices).reshape(num_particles, user_k)
    distances_res = np.array(distances).reshape(num_particles, user_k)

    print("\n" + "="*50)
    print("METRICS REPORT")
    print("="*50)
    print(f"Search Latency:         {duration_ms:.4f} ms")
    print(f"Avg GPU Utilization:    {gpu_util}")
    print(f"Max GPU Memory Delta:   {gpu_mem}")
    print("="*50)

    print("\nDETERMINISTIC CHECK (First 2 particles):")
    for i in range(2):
        print(f"Pt {i} Indices:   {indices_res[i].tolist()}")
        # Check if any neighbors were found (indices != -1)
        valid_dist = distances_res[i][indices_res[i] != -1]
        print(f"Pt {i} Distances: {np.round(valid_dist, 4).tolist()}")
    
    print("="*50)
    
    if handle:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    run_deterministic_test()