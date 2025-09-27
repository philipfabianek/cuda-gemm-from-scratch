#pragma once

#include <cublas_v2.h>

#include "types.cuh"

template <typename InputType, typename AccumType>
void run_cublas_kernel(cublasHandle_t handle, int M, int N, int K, float alpha,
                       const InputType *d_A, const InputType *d_B, float beta,
                       AccumType *d_C) {
  // cuBLAS uses column-major order, we can use that
  // C_rowmajor = alpha * A_rowmajor * B_rowmajor + beta * C_rowmajor
  // is equivalent to
  // C_colmajor^T = alpha * B_colmajor^T * A_colmajor^T + beta * C_colmajor^T.

  // Hence it suffices to call cublasGemmEx with swapped matrices and
  // dimensions. We also need to supply leading dimensions, which are the number
  // of rows of the transposed matrices.

  // Since we are using the general cublasGemmEx function,
  // we need to specify the data types, the compute type and the algorithm.
  cudaDataType_t input_type, output_type;
  cublasComputeType_t compute_type;
  cublasGemmAlgo_t algo;

  if constexpr (std::is_same_v<InputType, float>) {
    // fp32 matrices
    input_type = CUDA_R_16F;
    output_type = CUDA_R_32F;
    // fp32 compute type
    compute_type = CUBLAS_COMPUTE_32F;
    // This means cuBLAS will choose the most suitable algorithm
    // for fp32 based on the hardware and input sizes.
    algo = CUBLAS_GEMM_DEFAULT;

  } else if constexpr (std::is_same_v<InputType, half>) {
    // fp16 input matrices and fp32 output matrix
    input_type = CUDA_R_16F;
    output_type = CUDA_R_32F;
    // Compute type is still fp32 for mixed precision!
    compute_type = CUBLAS_COMPUTE_32F;
    // This means cuBLAS will choose the most suitable algorithm
    // for fp16 based on the hardware and input sizes.
    algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }

  // No transposition from cuBLAS's perspective.
  cublasOperation_t trans = CUBLAS_OP_N;

  int lda = K; // A is M x K -> transpose is K x M -> leading dimension is K
  int ldb = N; // B is K x N -> transpose is N x K -> leading dimension is N
  int ldc = N; // C is M x N -> transpose is N x M -> leading dimension is N

  cublasGemmEx(handle, trans, trans, N, M, K, &alpha, d_B, input_type, ldb, d_A,
               input_type, lda, &beta, d_C, output_type, ldc, compute_type,
               algo);
}