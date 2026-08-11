cmake_minimum_required(VERSION 3.18)
project(frnn_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Default to A100 (Perlmutter). Override: cmake -DCMAKE_CUDA_ARCHITECTURES=90
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# High-dimensional grid launch policy.  These are cache variables so occupancy
# experiments can rebuild a single policy without editing CUDA source.
set(FRNN_GRID_D12_THREADS 128 CACHE STRING "Threads/block for the D=12 grid kernel")
set(FRNN_GRID_D12_MIN_BLOCKS 8 CACHE STRING "launch_bounds minimum blocks/SM for D=12")
set(FRNN_GRID_D16_THREADS 128 CACHE STRING "Threads/block for the D=16 grid kernel")
set(FRNN_GRID_D16_MIN_BLOCKS 8 CACHE STRING "launch_bounds minimum blocks/SM for D=16")
set(FRNN_GRID_GENERIC_THREADS 128 CACHE STRING "Threads/block for the generic high-D grid kernel")
set(FRNN_GRID_GENERIC_MIN_BLOCKS 4 CACHE STRING "launch_bounds minimum blocks/SM for generic high-D")
option(FRNN_PTXAS_VERBOSE "Print CUDA register, stack, and spill usage while compiling" OFF)

set(FRNN_SOURCES
    python_interface/frnn_engine.cu
    frnn/csrc/grid/insert_points.cu
    frnn/csrc/grid/find_nbrs.cu
    frnn/csrc/no_grid_frnn/no_grid_frnn.cu
    python_interface/nanobind_module.cu
)

nanobind_add_module(frnn_cuda NB_STATIC ${FRNN_SOURCES})

target_include_directories(frnn_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/python_interface
    ${CMAKE_SOURCE_DIR}/frnn/csrc/grid
    ${CMAKE_SOURCE_DIR}/frnn/csrc/no_grid_frnn
)

target_compile_options(frnn_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${FRNN_PTXAS_VERBOSE}>>:-Xptxas=-v>
)

target_compile_definitions(frnn_cuda PRIVATE
    FRNN_GRID_D12_THREADS=${FRNN_GRID_D12_THREADS}
    FRNN_GRID_D12_MIN_BLOCKS=${FRNN_GRID_D12_MIN_BLOCKS}
    FRNN_GRID_D16_THREADS=${FRNN_GRID_D16_THREADS}
    FRNN_GRID_D16_MIN_BLOCKS=${FRNN_GRID_D16_MIN_BLOCKS}
    FRNN_GRID_GENERIC_THREADS=${FRNN_GRID_GENERIC_THREADS}
    FRNN_GRID_GENERIC_MIN_BLOCKS=${FRNN_GRID_GENERIC_MIN_BLOCKS}
)

install(TARGETS frnn_cuda LIBRARY DESTINATION .)
