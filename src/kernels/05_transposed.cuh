#pragma once

namespace k5 {

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_transposed_kernel(int M, int N, int K, float alpha,
                                        float *A, float *B, float beta,
                                        float *C) {
  // Move pointers to the beginning of the initial tiles
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // A_tile has BK rows and BM columns since we are transposing the tile
  __shared__ float A_tile[BK * BM];
  __shared__ float B_tile[BK * BN];

  // One thread loads chunks of 4 elements into smem
  const int A_smem_col = (threadIdx.x % (BK / 4)) * 4;
  const int A_smem_row = threadIdx.x / (BK / 4);
  const int B_smem_col = (threadIdx.x % (BN / 4)) * 4;
  const int B_smem_row = threadIdx.x / (BN / 4);

  // Multiply by 4 because of the adjusted loading
  constexpr int num_threads = BM * BN / (TM * TN);
  constexpr int A_row_stride = (num_threads * 4) / BK;
  constexpr int B_row_stride = (num_threads * 4) / BN;

  // Compute initial coordinates of every thread inside the C tile
  const int C_inner_col = (threadIdx.x % (BN / TN)) * TN;
  const int C_inner_row = (threadIdx.x / (BN / TN)) * TM;

  // Each thread needs to store the sums for the full minitile
  float sums[TM * TN] = {0.0};

  // These are the values that every thread reuses
  float A_reg[TM] = {0.0};
  float B_reg[TN] = {0.0};

  // Iterate over the necessary tiles from A and B to compute the output C tile
  for (int k_tile = 0; k_tile < K; k_tile += BK) {
    // Load values from A into smem using a vectorized load,
    // each thread is responsible for loading chunks of 4 consecutive elements
    // depending on A_row_stride, notice the transposed loading into A_tile
    for (int A_row_offset = 0; A_row_offset < BM;
         A_row_offset += A_row_stride) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(A_smem_row + A_row_offset) * K + A_smem_col])[0];
      A_tile[(A_smem_col + 0) * BM + (A_smem_row + A_row_offset)] = tmp.x;
      A_tile[(A_smem_col + 1) * BM + (A_smem_row + A_row_offset)] = tmp.y;
      A_tile[(A_smem_col + 2) * BM + (A_smem_row + A_row_offset)] = tmp.z;
      A_tile[(A_smem_col + 3) * BM + (A_smem_row + A_row_offset)] = tmp.w;
    }

    // Load values from B into smem using a vectorized load
    for (int B_row_offset = 0; B_row_offset < BK;
         B_row_offset += B_row_stride) {
      reinterpret_cast<float4 *>(
          &B_tile[(B_smem_row + B_row_offset) * BN + B_smem_col])[0] =
          reinterpret_cast<float4 *>(
              &B[(B_smem_row + B_row_offset) * N + B_smem_col])[0];
    }

    // Synchronize so that all tile values are loaded
    __syncthreads();

    // Advance pointers to the next tile
    A += BK;
    B += BK * N;

    // Move in the direction of the matrix multiplication
    for (int k = 0; k < BK; ++k) {
      // Load <TM> consecutive elements within one row (horizontal slice) from
      // the A tile and store them in registers
      for (int i = 0; i < TM; i += 4) {
        // This explicit vectorization does not improve performance
        // which means the loads were already vectorized
        reinterpret_cast<float4 *>(&A_reg[i])[0] =
            reinterpret_cast<float4 *>(&A_tile[k * BM + C_inner_row + i])[0];
      }

      // Load <TN> consecutive elements within one row (horizontal slice) from
      // the B tile and store them in registers
      for (int j = 0; j < TN; j += 4) {
        // This explicit vectorization does not improve performance
        // which means the loads were already vectorized
        reinterpret_cast<float4 *>(&B_reg[j])[0] =
            reinterpret_cast<float4 *>(&B_tile[k * BN + C_inner_col + j])[0];
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
    for (int j = 0; j < TN; j += 4) {
      // Use vectorization for global memory writes
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(C_inner_row + i) * N + (C_inner_col + j)])[0];
      tmp.x = alpha * sums[i * TN + j + 0] + beta * tmp.x;
      tmp.y = alpha * sums[i * TN + j + 1] + beta * tmp.y;
      tmp.z = alpha * sums[i * TN + j + 2] + beta * tmp.z;
      tmp.w = alpha * sums[i * TN + j + 3] + beta * tmp.w;
      reinterpret_cast<float4 *>(
          &C[(C_inner_row + i) * N + (C_inner_col + j)])[0] = tmp;
    }
  }
}

void run_transposed_kernel(int M, int N, int K, float alpha, float *d_A,
                           float *d_B, float beta, float *d_C) {
  // Number of rows of smem A tile
  const int BM = 128;
  // Number of columns of smem B tile
  const int BN = 128;
  // Number of columns of smem A tile and number of rows of smem B tile
  const int BK = 16;
  // Number of columns of smem A tile that every thread works with
  const int TM = 8;
  // Number of columns of smem B tile that every thread works with
  const int TN = 8;

  // 2D output tiles
  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
  // Each thread computes a "minitile" of size TM x TN
  const int num_threads = (BM * BN) / (TM * TN);
  dim3 blockDim(num_threads);

  // Constraints
  static_assert(TM % 4 == 0);
  static_assert(TN % 4 == 0);
  static_assert((num_threads * 4) % BK == 0);
  static_assert((num_threads * 4) % BN == 0);
  static_assert((BM * BK) % (num_threads * 4) == 0);
  static_assert((BK * BN) % (num_threads * 4) == 0);

  sgemm_transposed_kernel<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}

} // namespace k5