cmake_minimum_required(VERSION 3.18)

project(CudaGemmFromScratch LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)

# Adjust this if necessary
set(CUDA_COMPUTE_CAPABILITY 86)

# --- Executables ---

add_executable(gemm_runner src/main.cu)
target_include_directories(gemm_runner PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(gemm_runner ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
set_target_properties(gemm_runner PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_compile_options(gemm_runner PRIVATE --generate-line-info)