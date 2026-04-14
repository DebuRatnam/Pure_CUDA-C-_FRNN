import frnn_cuda
import numpy as np
import time

def run_lhc_test():
    # 1. Configuration
    num_particles = 100000
    K = 16
    radius = 0.05
    
    print(f"--- Initializing A100 Engine for {num_particles} particles ---")
    # This calls your C++ Constructor and runs cudaMalloc
    engine = frnn_cuda.FRNNEngine(max_points=num_particles)

    # 2. Create synthetic particle data
    # We flatten it because our C++ interface expects a flat vector
    # [x1, y1, z1, x2, y2, z2...]
    print("Generating synthetic particle hits...")
    hits = np.random.rand(num_particles * 3).astype(np.float32)

    # 3. The Big Moment: Run the GPU Search
    print(f"Running FRNN search (Radius={radius}, K={K})...")
    start_time = time.time()
    
    # This triggers the Host->Device copy and your 4 CUDA kernels
    indices, distances = engine.search(hits, K, radius)
    
    end_time = time.time()
    
    # 4. Results Analysis
    # Indices comes back as a flat list of (num_particles * K)
    indices_res = np.array(indices).reshape(num_particles, K)
    
    print("\n--- Search Complete ---")
    print(f"Execution Time on A100: {(end_time - start_time)*1000:.2f} ms")
    
    # Spot check the first particle
    first_particle_neighbors = indices_res[0]
    valid_neighbors = first_particle_neighbors[first_particle_neighbors != -1]
    
    print(f"Particle 0 found {len(valid_neighbors)} neighbors.")
    if len(valid_neighbors) > 0:
        print(f"First 3 neighbor IDs: {valid_neighbors[:3]}")
    else:
        print("No neighbors found for Particle 0. (Try increasing radius?)")

if __name__ == "__main__":
    run_lhc_test()