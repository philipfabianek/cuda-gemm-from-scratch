#pragma once

#include <mma.h>

#include "types.cuh"

using namespace nvcuda;

namespace k7 {

template <const int WARPSIZE, const int WMMA_M, const int WMMA_N,
          const int WMMA_K>
__global__ void gemm_naive_wmma_kernel(int M, int N, int K, float alpha,
                                       const half *d_A, const half *d_B,
                                       float beta, float *d_C) {
  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  int tile_row = blockIdx.y * WMMA_M;
  int tile_col = blockIdx.x * WMMA_N;

  // Define fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

  // Fill the accumulator fragment with zeros
  wmma::fill_fragment(acc_frag, 0.0f);

  // Iterate over the necessary tiles from A and B and accumulate results
  for (int k_tile = 0; k_tile < K; k_tile += WMMA_K) {
    // Compute pointers to the relevant input tiles
    const half *A_tile_ptr = d_A + (tile_row * lda) + k_tile;
    const half *B_tile_ptr = d_B + (k_tile * ldb) + tile_col;

    // Load the inputs
    wmma::load_matrix_sync(a_frag, A_tile_ptr, lda);
    wmma::load_matrix_sync(b_frag, B_tile_ptr, ldb);

    // Perform the matrix multiplication
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Compute pointer to the relevant output tile
  float *C_ptr = d_C + tile_row * ldc + tile_col;

  // Load the output
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::load_matrix_sync(c_frag, C_ptr, ldc, wmma::mem_row_major);

  // Epilogue
  for (int i = 0; i < c_frag.num_elements; i++) {
    c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
  }

  wmma::store_matrix_sync(C_ptr, c_frag, ldc, wmma::mem_row_major);
}

void run_naive_wmma_kernel(int M, int N, int K, float alpha, const half *d_A,
                           const half *d_B, float beta, float *d_C) {
  const int WARPSIZE = 32;

  // One warp per block
  dim3 block_dim(WARPSIZE);

  // One warp computes a WMMA_M x WMMA_N output tile,
  // we use the most common 16x16x16 (m16n16k16) setup,
  // my code works only with this one
  const int WMMA_M = 16;
  const int WMMA_N = 16;
  const int WMMA_K = 16;

  dim3 grid_dim((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);

  gemm_naive_wmma_kernel<WARPSIZE, WMMA_M, WMMA_N, WMMA_K>
      <<<grid_dim, block_dim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}

} // namespace k7