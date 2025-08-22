#pragma once

#include <cublas_v2.h>

void run_cublas_kernel(cublasHandle_t handle, int M, int N, int K, float alpha,
                       const float *d_A, const float *d_B, float beta,
                       float *d_C) {
  // cuBLAS uses column-major order, we can use that
  // C_rowmajor = alpha * A_rowmajor * B_rowmajor + beta * C_rowmajor
  // is equivalent to
  // C_colmajor^T = alpha * B_colmajor^T * A_colmajor^T + beta * C_colmajor^T.

  // Hence it suffices to call cublasSgemm with swapped matrices and dimensions.
  // We also need to supply leading dimensions, which are the number of rows of
  // the transposed matrices.

  // B is K x N -> transpose is N x K -> leading dimension is N
  // A is M x K -> transpose is K x M -> leading dimension is K
  // C is M x N -> transpose is N x M -> leading dimension is N

  // CUBLAS_OP_N means we don't want any transpose from cuBLAS's perspective.
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K,
              &beta, d_C, N);
}