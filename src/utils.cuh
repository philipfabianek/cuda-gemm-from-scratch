#pragma once

#include <chrono>
#include <random>
#include <vector>

#include "kernels/verify.cuh"
#include "types.cuh"

#define CUDA_CHECK(err)                                                        \
  {                                                                            \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err),    \
              __FILE__, __LINE__);                                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

template <typename T>
void initialize_matrix(std::vector<T> &matrix, int rows, int cols) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  for (int i = 0; i < rows * cols; ++i) {
    if constexpr (std::is_same_v<T, float>) {
      matrix[i] = distribution(generator);
    } else if constexpr (std::is_same_v<T, bf16>) {
      matrix[i] = __float2bfloat16(distribution(generator));
    }
  }
}

template <typename T>
void verify_with_cublas_reference(int M, int N, const T *d_C_result,
                                  const T *d_C_reference) {
  printf("Verifying results against cuBLAS...\n");

  int n_elements = M * N;

  // Define variable to hold error (mismatch) count
  int h_error_count = 0;
  int *d_error_count;
  CUDA_CHECK(cudaMalloc(&d_error_count, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_error_count, 0, sizeof(int)));

  // Configure and launch comparison kernel
  int threads_per_block = 256;
  int num_blocks = (n_elements + threads_per_block - 1) / threads_per_block;
  comparison_kernel<T><<<num_blocks, threads_per_block>>>(
      d_C_result, d_C_reference, n_elements, d_error_count);

  // Copy the final error count back from device to host
  CUDA_CHECK(cudaMemcpy(&h_error_count, d_error_count, sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_error_count));

  if (h_error_count == 0) {
    printf("SUCCESS: All values match cuBLAS reference.\n");
  } else {
    printf("FAILURE: %d mismatches found against cuBLAS reference.\n",
           h_error_count);
  }
}