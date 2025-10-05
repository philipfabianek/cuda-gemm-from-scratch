#pragma once

namespace k6 {

template <const int NUM_THREADS, const int BM, const int BN, const int BK,
          const int TM, const int TN, const int WM, const int WN,
          const int NUM_SM, const int NUM_SN, const int SUBTILE_M,
          const int SUBTILE_N>
__global__ void sgemm_warptiling_kernel(int M, int N, int K, float alpha,
                                        float *A, float *B, float beta,
                                        float *C) {
  const int warp_idx = threadIdx.x / 32;

  // Map warp indexes within the blocktile to a 2D starting
  // coordinate within the first warptile
  const int warptile_row = (warp_idx / (BN / WN)) * WM;
  const int warptile_col = (warp_idx % (BN / WN)) * WN;

  // Move gmem pointers to the start of blocktile
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;

  // Move pointer to the start of warptile
  C += (blockIdx.y * BM + warptile_row) * N + (blockIdx.x * BN + warptile_col);

  // Define smem tiles
  __shared__ float A_tile[BK * BM];
  __shared__ float B_tile[BK * BN];

  // One thread loads chunks of 4 elements into smem
  const int A_smem_col = (threadIdx.x % (BK / 4)) * 4;
  const int A_smem_row = threadIdx.x / (BK / 4);
  const int B_smem_col = (threadIdx.x % (BN / 4)) * 4;
  const int B_smem_row = threadIdx.x / (BN / 4);

  // Strides for threads to cooperatively load the entire smem tiles
  constexpr int A_row_stride = (NUM_THREADS * 4) / BK;
  constexpr int B_row_stride = (NUM_THREADS * 4) / BN;

  // Map thread indexes within the warp to a 2D starting
  // coordinate within the first subtile
  const int thread_warp_idx = threadIdx.x % 32;
  const int thread_subtile_row = (thread_warp_idx / (SUBTILE_N / TN)) * TM;
  const int thread_subtile_col = (thread_warp_idx % (SUBTILE_N / TN)) * TN;

  // Each thread computes one threadtile in all warptile subtiles (yeah)
  // so it needs to store NUM_SM * NUM_SN threadtile values
  float sums[NUM_SM * NUM_SN * TM * TN] = {0.0};

  // These are the values that every thread reuses
  float A_reg[NUM_SM * TM] = {0.0};
  float B_reg[NUM_SN * TN] = {0.0};

  // Iterate over the necessary tiles from A and B to compute the output C tile
  for (int k_tile = 0; k_tile < K; k_tile += BK) {
    // Load A smem tile (transposed)
    for (int A_row_offset = 0; A_row_offset < BM;
         A_row_offset += A_row_stride) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(A_smem_row + A_row_offset) * K + A_smem_col])[0];
      A_tile[(A_smem_col + 0) * BM + (A_smem_row + A_row_offset)] = tmp.x;
      A_tile[(A_smem_col + 1) * BM + (A_smem_row + A_row_offset)] = tmp.y;
      A_tile[(A_smem_col + 2) * BM + (A_smem_row + A_row_offset)] = tmp.z;
      A_tile[(A_smem_col + 3) * BM + (A_smem_row + A_row_offset)] = tmp.w;
    }

    // Load B smem tile
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
    for (int bk_idx = 0; bk_idx < BK; bk_idx++) {
      // Load A smem values into registers for all subtiles
      for (int sm_idx = 0; sm_idx < NUM_SM; sm_idx++) {
        for (int tm_idx = 0; tm_idx < TM; tm_idx += 4) {
          reinterpret_cast<float4 *>(&A_reg[sm_idx * TM + tm_idx])[0] =
              reinterpret_cast<float4 *>(
                  &A_tile[bk_idx * BM + warptile_row + (sm_idx * SUBTILE_M) +
                          thread_subtile_row + tm_idx])[0];
        }
      }

      // Load B smem values into registers for all subtiles
      for (int sn_idx = 0; sn_idx < NUM_SN; sn_idx++) {
        for (int tn_idx = 0; tn_idx < TN; tn_idx += 4) {
          reinterpret_cast<float4 *>(&B_reg[sn_idx * TN + tn_idx])[0] =
              reinterpret_cast<float4 *>(
                  &B_tile[bk_idx * BN + warptile_col + (sn_idx * SUBTILE_N) +
                          thread_subtile_col + tn_idx])[0];
        }
      }

      // Compute products of all element pairs from all pairs of subtiles
      for (int wm_offset = 0; wm_offset < NUM_SM * TM; wm_offset += TM) {
        for (int wn_offset = 0; wn_offset < NUM_SN * TN; wn_offset += TN) {
          for (int tm_idx = 0; tm_idx < TM; tm_idx++) {
            for (int tn_idx = 0; tn_idx < TN; tn_idx++) {
              sums[(wm_offset + tm_idx) * (NUM_SN * TN) + (wn_offset) +
                   tn_idx] +=
                  A_reg[wm_offset + tm_idx] * B_reg[wn_offset + tn_idx];
            }
          }
        }
      }
    }

    // Synchronize so that all calculations using the tile
    // values finish before loading the next tile values
    __syncthreads();
  }

  // Each thread saves NUM_SM x NUM_SN threadtiles of size TM x TN
  for (int sm_idx = 0; sm_idx < NUM_SM; sm_idx++) {
    for (int sn_idx = 0; sn_idx < NUM_SN; sn_idx++) {
      float *C_subtile_ptr = C + (thread_subtile_row + sm_idx * SUBTILE_M) * N +
                             (thread_subtile_col + sn_idx * SUBTILE_N);
      for (int tm_idx = 0; tm_idx < TM; tm_idx++) {
        for (int tn_idx = 0; tn_idx < TN; tn_idx += 4) {
          // Use vectorization for global memory writes
          float4 tmp = reinterpret_cast<float4 *>(
              &C_subtile_ptr[tm_idx * N + tn_idx])[0];
          const int idx =
              (sm_idx * TM + tm_idx) * (NUM_SN * TN) + sn_idx * TN + tn_idx;
          tmp.x = alpha * sums[idx + 0] + beta * tmp.x;
          tmp.y = alpha * sums[idx + 1] + beta * tmp.y;
          tmp.z = alpha * sums[idx + 2] + beta * tmp.z;
          tmp.w = alpha * sums[idx + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(&C_subtile_ptr[tm_idx * N + tn_idx])[0] =
              tmp;
        }
      }
    }
  }
}

void run_warptiling_kernel(int M, int N, int K, float alpha, float *d_A,
                           float *d_B, float beta, float *d_C) {
  // A smem tile has dimensions BK x BM (it is transposed),
  // B smem tile has dimensions BK x BN
  const int BM = 128;
  const int BN = 128;
  const int BK = 8;
  // Coarsening factor just like before but one thread now
  // works with several subtiles in one warptile (see below),
  // TM x TN is called a threadtile in this kernel
  const int TM = 8;
  const int TN = 4;

  // Same blocktiles as before
  dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
  // Threads per block, using 128 seems to be the most performant
  const int NUM_THREADS = 128;
  dim3 blockDim(NUM_THREADS);

  // Warptile dimensions, one warptile is computed by one warp
  // using NUM_SM x NUM_SN iterations
  const int WM = 64;
  const int WN = (NUM_THREADS * TM * TN) / WM;

  // Number of subtiles in one warptile in the M direction,
  // covering all warptile rows with one subtile seems to be the most performant
  constexpr int NUM_SM = 1;
  // Number of subtiles in one warptile in the N direction
  constexpr int NUM_SN = (WM * WN) / (32 * TM * TN * NUM_SM);
  // One warptile contains NUM_SM x NUM_SN subtiles

  // One subtile has dimensions SUBTILE_M x SUBTILE_N,
  // it is computed by 1 warp in 1 iteration,
  // subtile is not a commonly used term but I like using it in this context
  constexpr int SUBTILE_M = WM / NUM_SM;
  constexpr int SUBTILE_N = WN / NUM_SN;

  // Warptile has to fit in blocktile
  static_assert(BM % WM == 0);
  static_assert(BN % WN == 0);
  // Number of warptiles has to be equal to number of warps
  static_assert((BN / WN) * (BM / WM) == NUM_THREADS / 32);

  // Constraints from previous kernels, some are probably
  // redundant but it is better to have more than less
  static_assert(TM % 4 == 0);
  static_assert(TN % 4 == 0);
  static_assert((NUM_THREADS * 4) % BK == 0);
  static_assert((NUM_THREADS * 4) % BN == 0);
  static_assert((BM * BK) % (NUM_THREADS * 4) == 0);
  static_assert((BK * BN) % (NUM_THREADS * 4) == 0);
  static_assert(BM % (16 * TM) == 0);
  static_assert(BN % (16 * TN) == 0);

  sgemm_warptiling_kernel<NUM_THREADS, BM, BN, BK, TM, TN, WM, WN, NUM_SM,
                          NUM_SN, SUBTILE_M, SUBTILE_N>
      <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}

} // namespace k6