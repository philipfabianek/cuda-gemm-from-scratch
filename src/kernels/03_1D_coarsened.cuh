#pragma once

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_1D_coarsened_kernel(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  // I was previously using 2D vectors (same performance)
  // but here I switched to 1D and explicit indexing for consistency
  __shared__ float A_tile[BM * BK];
  __shared__ float B_tile[BK * BN];

  // Move pointers to the beginning (top-left corner) of the initial tiles
  // (C stays the same, A and B will iteratively move by BK in their respective
  // directions)
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // Compute initial thread indices inside the tiles,
  // this effectively created a 1D to 2D mapping from our 1D blocks
  // to the 2D tiles, this is a 1-to-1 mapping for the A and B tiles
  // (one thread loads one element into the shared memory) but inside
  // the C tile one thread computes <TM> consecutive elements in one column
  const int A_inner_col = threadIdx.x % BK;
  const int A_inner_row = threadIdx.x / BK;
  const int B_inner_col = threadIdx.x % BN;
  const int B_inner_row = threadIdx.x / BN;
  const int C_inner_col = B_inner_col;
  const int C_inner_row = B_inner_row;

  float sums[TM] = {0.0};

  // Iterate over the necessary tiles from A and B to compute the output C tile
  for (int i = 0; i < K; i += BK) {
    A_tile[A_inner_row * BK + A_inner_col] = A[A_inner_row * K + A_inner_col];
    B_tile[B_inner_row * BN + B_inner_col] = B[B_inner_row * N + B_inner_col];

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

      // One thread works with <TM> consecutive rows from the A tile
      for (int j = 0; j < TM; ++j) {
        sums[j] += A_tile[(C_inner_row * TM + j) * BK + k] * B_val;
      }
    }

    // Synchronize so that all calculations using the tile
    // values finish before loading the next tile values
    __syncthreads();
  }

  // Each thread saves <TM> consecutive elements in one column of C
  for (int j = 0; j < TM; ++j) {
    C[(C_inner_row * TM + j) * N + C_inner_col] =
        alpha * sums[j] + beta * C[(C_inner_row * TM + j) * N + C_inner_col];
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
  // 1D blocks, in such kernels it is better to create a manual 1D to 2D mapping
  // because the smem A and B tiles have different dimensions and there
  // is no longer a natural 2D to 2D mapping between threads in a block and the
  // tiles, output tile has BM * BN elements but each thread computes TM
  // elements so we need (BM * BN) / TM threads
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