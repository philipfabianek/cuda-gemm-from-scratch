#pragma once

#include <mma.h>

#include "types.cuh"

using namespace nvcuda;

template <const int WARPSIZE, const int WMMA_M, const int WMMA_N,
          const int WMMA_K>
__global__ void gemm_naive_wmma_kernel(int M, int N, int K, float alpha,
                                       const bf16 *d_A, const bf16 *d_B,
                                       float beta, bf16 *d_C) {
  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  int tile_row = blockIdx.y * WMMA_M;
  int tile_col = blockIdx.x * WMMA_N;

  // Define fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

  // Fill the accumulator fragment with zeros
  wmma::fill_fragment(acc_frag, 0.0f);

  // Iterate over the necessary tiles from A and B and accumulate results
  for (int k_tile = 0; k_tile < K; k_tile += WMMA_K) {
    // Compute pointers to the relevant input tiles
    const bf16 *A_tile_ptr = d_A + (tile_row * lda) + k_tile;
    const bf16 *B_tile_ptr = d_B + (k_tile * ldb) + tile_col;

    // Load the inputs
    wmma::load_matrix_sync(a_frag, A_tile_ptr, lda);
    wmma::load_matrix_sync(b_frag, B_tile_ptr, ldb);

    // Perform the matrix multiplication
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Compute pointer to the relevant output tile
  bf16 *C_ptr = d_C + tile_row * ldc + tile_col;

  // Load existing C tile from global memory,
  // to use bf16 it is necessary to use 'matrix_a' here
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, bf16, wmma::row_major>
      c_tile_bf16_frag;
  wmma::load_matrix_sync(c_tile_bf16_frag, C_ptr, ldc);

  // Perform the final output computation in fp32
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_final_frag;
  for (int i = 0; i < c_final_frag.num_elements; i++) {
    c_final_frag.x[i] =
        alpha * acc_frag.x[i] + beta * __bfloat162float(c_tile_bf16_frag.x[i]);
  }

  // Final result is in fp32 but d_C in in bf16,
  // use a shared memory buffer to store the results
  // before writing them to global memory,
  // leading dimension for this tile is WMMA_N
  __shared__ float C_tile_temp[WMMA_M * WMMA_N];
  wmma::store_matrix_sync(C_tile_temp, c_final_frag, WMMA_N,
                          wmma::mem_row_major);

  __syncthreads();

  // Write output tile result from shared memory to global memory
  for (int i = threadIdx.x; i < WMMA_M * WMMA_N; i += WARPSIZE) {
    int row = i / WMMA_N;
    int col = i % WMMA_N;
    C_ptr[row * N + col] = __float2bfloat16(C_tile_temp[i]);
  }
}

void run_naive_wmma_kernel(int M, int N, int K, float alpha, const bf16 *d_A,
                           const bf16 *d_B, float beta, bf16 *d_C) {
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