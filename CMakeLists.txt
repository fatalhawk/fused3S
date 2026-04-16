cmake_minimum_required(VERSION 3.24)
project(Fused3S_GraphTransformer LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES "89")

# Performance Optimization Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -O3 --use_fast_math")

# Find CUDA
find_package(CUDA REQUIRED)

# add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the baseline executable
add_executable(test_baselines 
    tests/test_baselines.cu
    baselines/cpu_baseline.cpp
    baselines/gpu_baseline.cu
)

# v0: Original per-block softmax kernel
add_executable(test_fused_3s_v0
    tests/test_fused_3s_v0.cu
    src/converter.cu
    src/fused3s_v0.cu
)

# v1: Online softmax + 2D thread blocks
add_executable(test_fused_3s_v1
    tests/test_fused_3s_v1.cu
    src/converter.cu
    src/fused3s_v1.cu
)

# Link CUDA libraries
target_link_libraries(test_baselines PRIVATE cudart_static cuda)

# v2: Register-resident O + fragment rescaling
add_executable(test_fused_3s_v2
    tests/test_fused_3s_v2.cu
    src/converter.cu
    src/fused3s_v2.cu
)
