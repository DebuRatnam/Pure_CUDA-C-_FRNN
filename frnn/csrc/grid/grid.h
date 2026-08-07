#ifndef GRID_H
#define GRID_H

/*
 * =============================================================================
 * grid.h — Shared types and helpers for the uniform-grid FRNN path
 * =============================================================================
 *
 * This header is included by both insert_points.cu and find_nbrs.cu, which
 * together implement the two-stage grid FRNN algorithm:
 *
 *   Stage 1 (insert_points.cu): assign each point to a hypercell, count points
 *   per cell, prefix-scan the counts into offsets, and sort the point set into
 *   cell order so spatially adjacent points are contiguous in memory.
 *
 *   Stage 2 (find_nbrs.cu): for each query, iterate over the shell of
 *   (2*cell_radius+1)^D neighboring cells, compute squared L2 distances to
 *   candidate points inside those cells, and maintain a max-heap of the K
 *   nearest neighbors within radius r.
 *
 * The grid is a D-dimensional hypercube with uniform cell edge length
 * `cell_size = radius` (cell_radius = 1 → 3^D cell shell). For large D or
 * small radius, where 3^D ≥ total_cells, the engine auto-dispatches to the
 * brute-force path in no_grid_frnn.cu instead.
 *
 * All point arrays use Structure-of-Arrays (SoA) layout: p[d*P + i] stores
 * coordinate d of point i. This ensures consecutive threads in a warp read
 * consecutive memory addresses (coalesced loads) when iterating over points
 * with fixed d.
 */

#include <cuda_runtime.h>

/*
 * GridParams — configuration for one FRNN search call.
 * Constructed in FRNNEngine::search() / search_gpu() and passed to every kernel.
 */
struct GridParams {
    int dim;             // Number of dimensions (e.g., 3, 16, 128)
    float radius;        // The search radius (defines the cell size)
    float min_val;       // The minimum value in the data (origin of the hypergrid)
    float max_val;       // The maximum value in the data
    int res;             // Calculated: ceil((max_val - min_val) / cell_size)
    long long total_cells; // Total hyper-cells (res ^ dim)
    float cell_size;     // Grid cell edge length = radius / radius_cell_ratio.
                         // Finer cells (ratio > 1) tighten the candidate set per query,
                         // cutting redundant distance checks that the old cell=radius
                         // (3^dim shell) layout incurred — especially at higher dim.
    int cell_radius;     // Cell-steps each side needed to cover the search radius,
                         // = ceil(radius / cell_size). The neighbor loop scans
                         // (2*cell_radius+1)^dim cells. cell_radius=1 reproduces 3^dim.
};

/**
 * K-Nearest Neighbor Constraints
 * We keep these to ensure the local registers in find_nbrs.cu 
 * don't overflow the GPU's memory per thread.
 */
constexpr int MAX_K_CAPACITY = 128; // Max neighbors per particle
constexpr int MAX_DIM_SUPPORTED = 128; // Max latent space dimensions

/*
 * calculate_cell_hash — maps an N-D integer grid coordinate to a scalar cell index
 * using a row-major (C-order) linearisation: hash = coords[0] + res*coords[1] + ...
 * Used in CountPointsNDKernel to assign each point to a cell and in FindNbrsNDKernel
 * to address neighboring cells.
 *
 * Key variables:
 *   coords[dim]  — per-axis integer grid position of the point (0 ≤ coords[d] < res)
 *   res          — grid resolution (cells per axis), same for all axes
 *   stride       — accumulated multiplier res^d; updated each iteration
 */
__device__ __host__ inline long long calculate_cell_hash(int* coords, int res, int dim) {
    long long hash = 0;
    long long stride = 1;
    for (int i = 0; i < dim; i++) {
        hash += (long long)coords[i] * stride;
        stride *= res;
    }
    return hash;
}

#endif // GRID_H