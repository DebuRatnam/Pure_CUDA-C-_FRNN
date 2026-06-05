import numpy as np
import frnn_cuda

def run_search(pts_flat, N, D, K, R):
    # EVOLVE-BLOCK START
    # Default search paradigm to be evolved via AdaEvolve based on 2602.pdf context
    engine = frnn_cuda.FRNNEngine(max_points=N)
    idx, dists = engine.search(pts_flat, K=K, radius=R)
    # EVOLVE-BLOCK END
    return idx, dists
