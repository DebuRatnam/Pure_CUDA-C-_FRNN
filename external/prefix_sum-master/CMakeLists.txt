cmake_minimum_required(VERSION 3.18)
project(CERN_PrefixSum LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80) 

# Added utils.cpp here to fix the linker errors
add_library(prefix_sum_core STATIC
    parallel-scan/kernels.cu 
    parallel-scan/scan.cu 
    parallel-scan/utils.cpp
    prefix_sum.cu
) 

target_include_directories(prefix_sum_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/parallel-scan
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Added the test executable back in
add_executable(run_test parallel-scan/Main.cpp) 
target_link_libraries(run_test PRIVATE prefix_sum_core)

