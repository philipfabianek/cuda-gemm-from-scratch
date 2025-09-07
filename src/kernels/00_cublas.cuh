#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>

template <typename T>
void run_cublas_kernel(cublasHandle_t handle, int M, int N, int K, float alpha,
                       const T *d_A, const T *d_B, float beta, T *d_C) {
  // cuBLAS uses column-major order, we can use that
  // C_rowmajor = alpha * A_rowmajor * B_rowmajor + beta * C_rowmajor
  // is equivalent to
  // C_colmajor^T = alpha * B_colmajor^T * A_colmajor^T + beta * C_colmajor^T.

  // Hence it suffices to call cublasGemmEx with swapped matrices and
  // dimensions. We also need to supply leading dimensions, which are the number
  // of rows of the transposed matrices.

  // Since we are using the general cublasGemmEx function,
  // we need to specify the data types, the compute type and the algorithm.
  cudaDataType_t matrix_type;
  cublasComputeType_t compute_type;
  cublasGemmAlgo_t algo;

  if constexpr (std::is_same_v<T, float>) {
    // fp32 matrices
    matrix_type = CUDA_R_32F;
    // fp32 compute type
    compute_type = CUBLAS_COMPUTE_32F;
    // This means cuBLAS will choose the most suitable algorithm
    // for fp32 based on the hardware and input sizes.
    algo = CUBLAS_GEMM_DEFAULT;

  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    // bf16 matrices
    matrix_type = CUDA_R_16BF;
    // Compute type is still fp32 for mixed precision!
    compute_type = CUBLAS_COMPUTE_32F;
    // This means cuBLAS will choose the most suitable algorithm
    // for bf16 based on the hardware and input sizes.
    algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }

  // No transposition from cuBLAS's perspective.
  cublasOperation_t trans = CUBLAS_OP_N;

  int lda = K; // A is M x K -> transpose is K x M -> leading dimension is K
  int ldb = N; // B is K x N -> transpose is N x K -> leading dimension is N
  int ldc = N; // C is M x N -> transpose is N x M -> leading dimension is N

  cublasGemmEx(handle, trans, trans, N, M, K, &alpha, d_B, matrix_type, ldb,
               d_A, matrix_type, lda, &beta, d_C, matrix_type, ldc,
               compute_type, algo);
}