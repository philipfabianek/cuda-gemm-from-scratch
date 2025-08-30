#pragma once

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_2D_coarsened_kernel(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  // Move pointers to the beginning of the initial tiles
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // Define shared memory tiles
  __shared__ float A_tile[BM * BK];
  __shared__ float B_tile[BK * BN];

  // Compute initial thread indices inside the tiles,
  // these are used only for loading data into smem
  const int A_smem_col = threadIdx.x % BK;
  const int A_smem_row = threadIdx.x / BK;
  const int B_smem_col = threadIdx.x % BN;
  const int B_smem_row = threadIdx.x / BN;

  // There aren't enough threads to load the entire tile in one go,
  // so we compute how many rows we fully cover and use that
  // as the row stride when loading the values into smem
  // (we load both tiles in a coalesced manner)
  const int num_threads = BM * BN / (TM * TN);
  const int A_row_stride = num_threads / BK;
  const int B_row_stride = num_threads / BN;

  // Compute initial coordinates of every thread inside the C tile,
  // (BN / TN) is the number of minitiles per row so by dividing by it
  // we obtain the minitile coordinates and then by multiplying by TN and TM
  // we obtain the actual element coordinates
  const int C_inner_col = (threadIdx.x % (BN / TN)) * TN;
  const int C_inner_row = (threadIdx.x / (BN / TN)) * TM;

  // Each thread needs to store the sums for the full minitile
  float sums[TM * TN] = {0.0};

  // These are the values that we reuse, interestingly, omitting this explicit
  // setup does not actually reduce performance which means the compiler is
  // smart enough to optimize it away
  float A_reg[TM] = {0.0};
  float B_reg[TN] = {0.0};

  // Iterate over the necessary tiles from A and B to compute the output C tile
  for (int k_tile = 0; k_tile < K; k_tile += BK) {
    // Load values from A into smem in a coalesced manner,
    // each thread is responsible for loading multiple elements
    // depending on A_row_stride (see above)
    for (int A_row_offset = 0; A_row_offset < BM;
         A_row_offset += A_row_stride) {
      A_tile[(A_smem_row + A_row_offset) * BK + A_smem_col] =
          A[(A_smem_row + A_row_offset) * K + A_smem_col];
    }

    // Load values from B into smem in a coalesced manner
    for (int B_row_offset = 0; B_row_offset < BK;
         B_row_offset += B_row_stride) {
      B_tile[(B_smem_row + B_row_offset) * BN + B_smem_col] =
          B[(B_smem_row + B_row_offset) * N + B_smem_col];
    }

    // Synchronize so that all tile values are loaded
    __syncthreads();

    // Advance pointers to the next tile
    A += BK;
    B += BK * N;

    // Move in the direction of the matrix multiplication
    for (int k = 0; k < BK; ++k) {
      // Load <TM> consecutive elements within one column (vertical slice) from
      // the A tile and store them in registers
      for (int i = 0; i < TM; i++) {
        A_reg[i] = A_tile[(C_inner_row + i) * BK + k];
      }

      // Load <TN> consecutive elements within one row (horizontal slice) from
      // the B tile and store them in registers
      for (int j = 0; j < TN; j++) {
        B_reg[j] = B_tile[k * BN + C_inner_col + j];
      }

      // Compute products of all pairs from the respective slices
      for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
          sums[i * TN + j] += A_reg[i] * B_reg[j];
        }
      }
    }

    // Synchronize so that all calculations using the tile
    // values finish before loading the next tile values
    __syncthreads();
  }

  // Each thread saves a minitile of size TM x TN
  for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j++) {
      C[(C_inner_row + i) * N + (C_inner_col + j)] =
          alpha * sums[i * TN + j] +
          beta * C[(C_inner_row + i) * N + (C_inner_col + j)];
    }
  }
}

void run_2D_coarsened_kernel(int M, int N, int K, float alpha, const float *d_A,
                             const float *d_B, float beta, float *d_C) {
  // Number of rows of smem A tile
  const int BM = 128;
  // Number of columns of smem B tile
  const int BN = 128;
  // Number of columns of smem A tile and number of rows of smem B tile
  const int BK = 32;
  // Number of rows of smem A tile that every thread works with
  const int TM = 8;
  // Number of columns of smem B tile that every thread works with
  const int TN = 8;

  // 2D output tiles
  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
  // Each thread computes a "minitile" of size TM x TN
  const int num_threads = (BM * BN) / (TM * TN);
  dim3 blockDim(num_threads);

  // Sizes of A and B smem tiles must be divisible
  // by thread counts due to the way they are being loaded
  static_assert((BM * BK) % num_threads == 0);
  static_assert((BK * BN) % num_threads == 0);

  sgemm_2D_coarsened_kernel<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}