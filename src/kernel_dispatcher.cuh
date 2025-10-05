#pragma once

#include "kernels/00_cublas.cuh"
#include "kernels/01_naive.cuh"
#include "kernels/02_tiled.cuh"
#include "kernels/03_1D_coarsened.cuh"
#include "kernels/04_2D_coarsened.cuh"
#include "kernels/05_transposed.cuh"
#include "kernels/06_warptiling.cuh"
#include "kernels/07_naive_wmma.cuh"
#include "kernels/08_naive_mma.cuh"
#include "kernels/09_hierarchical_mma.cuh"
#include "kernels/10_vectorized_mma.cuh"
#include "kernels/11_memory_swizzling.cuh"
#include "kernels/12_buffered_gmem.cuh"
#include "types.cuh"

template <typename InputType, typename AccumType>
void run_kernel(int kernel_id, cublasHandle_t handle, int M, int N, int K,
                float alpha, InputType *d_A, InputType *d_B, float beta,
                AccumType *d_C) {
  // By adding a dependence on InputType using a condition that
  // is always false, we can check that this is never instantiated.
  static_assert(sizeof(InputType) == 0,
                "run_kernel is not specialized for this data type");
}

// fp32 specialization
template <>
void run_kernel(int kernel_id, cublasHandle_t handle, int M, int N, int K,
                float alpha, float *d_A, float *d_B, float beta, float *d_C) {
  switch (kernel_id) {
  case 0:
    k0::run_cublas_kernel<float, float>(handle, M, N, K, alpha, d_A, d_B, beta,
                                        d_C);
    break;
  case 1:
    k1::run_naive_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 2:
    k2::run_tiled_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 3:
    k3::run_1D_coarsened_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 4:
    k4::run_2D_coarsened_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 5:
    k5::run_transposed_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 6:
    k6::run_warptiling_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  }
  // No default needed, main() validates the kernel_id
}

// fp16 specialization
template <>
void run_kernel(int kernel_id, cublasHandle_t handle, int M, int N, int K,
                float alpha, half *d_A, half *d_B, float beta, float *d_C) {
  switch (kernel_id) {
  case 0:
    k0::run_cublas_kernel<half, float>(handle, M, N, K, alpha, d_A, d_B, beta,
                                       d_C);
    break;
  case 7:
    k7::run_naive_wmma_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 8:
    k8::run_naive_mma_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 9:
    k9::run_hierarchical_mma_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 10:
    k10::run_vectorized_mma_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 11:
    k11::run_memory_swizzling_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 12:
    k12::run_buffered_gmem_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  }
  // No default needed, main() validates the kernel_id
}