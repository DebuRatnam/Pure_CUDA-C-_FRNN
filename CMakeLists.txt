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

set(FRNN_SOURCES
    python_interface/frnn_engine.cu
    frnn/csrc/grid/insert_points.cu
    frnn/csrc/grid/find_nbrs.cu
    frnn/csrc/no_grid_frnn/no_grid_frnn.cu
    frnn/csrc/projection/project.cu
    frnn/csrc/projection/verify.cu
    python_interface/nanobind_module.cu
)

nanobind_add_module(frnn_cuda NB_STATIC ${FRNN_SOURCES})

target_include_directories(frnn_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/python_interface
    ${CMAKE_SOURCE_DIR}/frnn/csrc/grid
    ${CMAKE_SOURCE_DIR}/frnn/csrc/no_grid_frnn
    ${CMAKE_SOURCE_DIR}/frnn/csrc/projection
)

target_compile_options(frnn_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>
)

install(TARGETS frnn_cuda LIBRARY DESTINATION .)
