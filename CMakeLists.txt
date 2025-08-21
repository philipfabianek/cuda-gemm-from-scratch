cmake_minimum_required(VERSION 3.18)

project(CudaSgemmFromScratch LANGUAGES CXX CUDA)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)

# Adjust this if necessary
set(CUDA_COMPUTE_CAPABILITY 86)

# --- Executables ---

add_executable(sgemm_runner src/main.cu src/utils.cu)
target_link_libraries(sgemm_runner ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
set_target_properties(sgemm_runner PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})