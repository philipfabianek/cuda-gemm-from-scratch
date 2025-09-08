#pragma once

#include "kernels/00_cublas.cuh"
#include "kernels/01_naive.cuh"
#include "kernels/02_tiled.cuh"
#include "kernels/03_1D_coarsened.cuh"
#include "kernels/04_2D_coarsened.cuh"
#include "kernels/05_transposed.cuh"
#include "kernels/06_warptiling.cuh"
#include "kernels/07_naive_wmma.cuh"
#include "types.cuh"

template <typename T>
void run_kernel(int kernel_id, cublasHandle_t handle, int M, int N, int K,
                float alpha, T *d_A, T *d_B, float beta, T *d_C) {
  // By adding a dependence on T using a condition that
  // is always false, we can check that this is never instantiated.
  static_assert(sizeof(T) == 0,
                "run_kernel is not specialized for this data type");
}

// fp32 specialization
template <>
void run_kernel(int kernel_id, cublasHandle_t handle, int M, int N, int K,
                float alpha, float *d_A, float *d_B, float beta, float *d_C) {
  switch (kernel_id) {
  case 0:
    run_cublas_kernel(handle, M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 1:
    run_naive_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 2:
    run_tiled_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 3:
    run_1D_coarsened_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 4:
    run_2D_coarsened_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 5:
    run_transposed_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 6:
    run_warptiling_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  }
  // No default needed, main() validates the kernel_id
}

// bf16 specialization
template <>
void run_kernel<bf16>(int kernel_id, cublasHandle_t handle, int M, int N, int K,
                      float alpha, bf16 *d_A, bf16 *d_B, float beta,
                      bf16 *d_C) {
  switch (kernel_id) {
  case 0:
    run_cublas_kernel(handle, M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 7:
    run_naive_wmma_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  }
  // No default needed, main() validates the kernel_id
}