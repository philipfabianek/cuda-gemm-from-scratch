cmake_minimum_required(VERSION 3.18)

project(CudaGemmFromScratch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Adjust this if necessary
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDA REQUIRED)

# --- Executables ---

add_executable(gemm_runner src/main.cu)
target_include_directories(gemm_runner PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(gemm_runner ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
target_compile_options(gemm_runner PRIVATE --generate-line-info)