#pragma once

namespace k3 {

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1D_coarsened_kernel(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  // Move pointers to the beginning (top-left corner) of the initial tiles
  // (C stays the same, A and B will iteratively move by BK in their respective
  // directions)
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // I was previously using 2D vectors (same performance)
  // but here I switched to 1D and explicit indexing for consistency
  __shared__ float A_tile[BM * BK];
  __shared__ float B_tile[BK * BN];

  // Compute initial thread indices inside the tiles,
  // these are used only for loading data into smem,
  // this effectively created a 1D to 2D mapping from our 1D blocks
  // to the 2D smem tiles, in this case this is a 1-to-1 mapping
  // (one thread loads one element into the shared memory)
  const int A_smem_col = threadIdx.x % BK;
  const int A_smem_row = threadIdx.x / BK;
  const int B_smem_col = threadIdx.x % BN;
  const int B_smem_row = threadIdx.x / BN;

  // Compute initial coordinates of every thread inside the C tile,
  // these are initial because one thread computes a vertical strip
  // (<TM> consecutive elements in one column)
  const int C_inner_col = B_smem_col;
  const int C_inner_row = B_smem_row;

  // Each thread needs to store the sums for the full vertical strip
  float sums[TM] = {0.0};

  // Iterate over the necessary tiles from A and B to compute the output C tile
  for (int k_tile = 0; k_tile < K; k_tile += BK) {
    // Load values from A and B into smem in a coalesced manner
    A_tile[A_smem_row * BK + A_smem_col] = A[A_smem_row * K + A_smem_col];
    B_tile[B_smem_row * BN + B_smem_col] = B[B_smem_row * N + B_smem_col];

    // Synchronize so that all tile values are loaded
    __syncthreads();

    // Advance pointers to the next tile
    A += BK;
    B += BK * N;

    // Move in the direction of the matrix multiplication
    for (int k = 0; k < BK; ++k) {
      // Store this in a register and reuse it with
      // A elements from multiple rows
      float B_val = B_tile[k * BN + C_inner_col];

      // One thread works with a vertical strip from the A tile
      // (<TM> consecutive elements in one column)
      for (int i = 0; i < TM; ++i) {
        sums[i] += A_tile[(C_inner_row * TM + i) * BK + k] * B_val;
      }
    }

    // Synchronize so that all calculations using the tile
    // values finish before loading the next tile values
    __syncthreads();
  }

  // Each thread saves the full vertical strip
  for (int i = 0; i < TM; ++i) {
    C[(C_inner_row * TM + i) * N + C_inner_col] =
        alpha * sums[i] + beta * C[(C_inner_row * TM + i) * N + C_inner_col];
  }
}

void run_1D_coarsened_kernel(int M, int N, int K, float alpha, const float *d_A,
                             const float *d_B, float beta, float *d_C) {
  // Number of rows of smem A tile
  const int BM = 64;
  // Number of columns of smem B tile
  const int BN = 64;
  // Number of columns of smem A tile and number of rows of smem B tile
  const int BK = 4;
  // Number of rows of smem A tile that every thread works
  // with, this is the coarsening factor
  const int TM = 16;

  // 2D output tiles
  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
  // 1D blocks, in such kernels it is better to create manual 1D to 2D mappings
  // because the kernels work with multiple 2D tiles (e.g. smem A and B tiles
  // which have different dimensions), the output tile has BM * BN elements but
  // each thread computes TM elements so we need (BM * BN) / TM threads
  dim3 blockDim((BM * BN) / TM);

  // The shared tiles have sizes BM * BK and BN * BK,
  // and we have (BM * BN) / TM threads in each block,
  // so we need BM = BN and BM * BK = BN * BK = (BM * BN) / TM,
  // hence BK = BM / TM = BN / TM or equivalently BK * TM = BM = BN
  static_assert(BM == BN);
  static_assert(BK * TM == BM);

  sgemm_1D_coarsened_kernel<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}

} // namespace k3