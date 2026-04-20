import frnn_cuda
import numpy as np
import time

def run_interactive_lhc_test():
    print("====================================================")
    print("   LHC FRNN CUDA Engine - ML Latent Space Tester")
    print("====================================================\n")

    try:
        num_particles = int(input("Enter number of particles: "))
        user_dim = int(input("Enter Dimensions (e.g., 3 or 8): "))
        engine = frnn_cuda.FRNNEngine(max_points=num_particles)
    except Exception as e:
        print(f"[ERROR] Init failed: {e}")
        return

    # Loop for search parameters to prevent crashes
    while True:
        try:
            print(f"\n--- New Search Configuration ({user_dim}D) ---")
            user_k = int(input("  Enter K: "))
            user_r = float(input("  Enter Radius: "))
            
            print(f"  [2/3] Generating random {user_dim}D data...")
            hits = np.random.rand(num_particles * user_dim).astype(np.float32)

            print(f"  [3/3] Running GPU Search...")
            start_time = time.perf_counter()
            indices, distances = engine.search(hits, K=user_k, radius=user_r)
            end_time = time.perf_counter()
            
            # If we reached here, the search succeeded!
            duration_ms = (end_time - start_time) * 1000
            print(f"\nSUCCESS: Search took {duration_ms:.3f} ms")
            break # Exit the loop

        except RuntimeError as e:
            print(f"\n[GPU CONFIG ERROR]: {e}")
            print("Tip: In high dimensions, use a LARGER radius to reduce grid complexity.")
            choice = input("Adjust parameters and try again? (y/n): ")
            if choice.lower() != 'y':
                break

if __name__ == "__main__":
    run_interactive_lhc_test()