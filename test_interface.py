import frnn_cuda
import numpy as np
import time
import sys

def run_interactive_lhc_test():
    print("====================================================")
    print("   LHC FRNN CUDA Engine - Interactive Tester")
    print("====================================================\n")

    # 1. Get User Parameters
    try:
        user_k = int(input("Enter K (number of neighbors, e.g., 16): "))
        user_r = float(input("Enter Radius (search distance, e.g., 0.05): "))
        num_particles = int(input("Enter number of particles (e.g., 100000): "))
    except ValueError:
        print("\n[ERROR] Invalid input. Please enter numbers only (Int for K, Float for Radius).")
        return

    # 2. Setup Data and Engine
    print(f"\n[1/3] Initializing A100 Engine for {num_particles} points...")
    try:
        engine = frnn_cuda.FRNNEngine(max_points=num_particles)
    except Exception as e:
        print(f"Failed to initialize GPU engine: {e}")
        return

    print(f"[2/3] Generating random particle hits on CPU...")
    # Hits are flattened: [x0, y0, z0, x1, y1, z1...]
    hits = np.random.rand(num_particles * 3).astype(np.float32)
    indices, distances = engine.search(hits, K=user_k, radius=user_r)

    # 3. Execution and Timing
    print(f"[3/3] Running GPU Search (K={user_k}, R={user_r})...")
    hits = np.random.rand(num_particles * 3).astype(np.float32)
    
    # Start the clock ONLY for the GPU work
    start_time = time.perf_counter()
    
    indices, distances = engine.search(hits, K=user_k, radius=user_r)
    
    # Synchronize/End clock
    end_time = time.perf_counter()
    
    duration_ms = (end_time - start_time) * 1000

    # 4. Results Formatting
    print("\n" + "="*50)
    print(f"RESULTS FOR {num_particles} PARTICLES")
    print("="*50)
    print(f"Total GPU Execution Time: {duration_ms:.3f} ms")
    
    # Reshape indices to see neighbors per particle
    indices_res = np.array(indices).reshape(num_particles, user_k)
    
    # Check first particle
    p0_neighbors = indices_res[0]
    valid_count = np.sum(p0_neighbors != -1)
    
    print(f"Particle 0: Found {valid_count} neighbors within radius {user_r}")
    if valid_count > 0:
        # Show first few non -1 neighbors
        print(f"Sample Neighbor IDs: {p0_neighbors[p0_neighbors != -1][:5]}")
    
    print("="*50)

if __name__ == "__main__":
    run_interactive_lhc_test()