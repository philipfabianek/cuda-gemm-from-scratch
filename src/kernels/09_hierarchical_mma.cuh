#pragma once

#include <cstdint>

#include "types.cuh"

template <const int num_threads, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WK, const int MMA_M,
          const int MMA_N, const int MMA_K>
__global__ void hierarchical_mma_kernel(int M, int N, int K, float alpha,
                                        const half *d_A, const half *d_B,
                                        float beta, float *d_C) {
  d_A += blockIdx.y * BM * K;
  d_B += blockIdx.x * BN;
  d_C += blockIdx.y * BM * N + blockIdx.x * BN;

  // This describes the number of MMA instructions
  // within one warptile iteration in the k dimension,
  // registers are reused across these iterations
  constexpr int TILES_M = WM / MMA_M;
  constexpr int TILES_N = WN / MMA_N;
  constexpr int TILES_K = WK / MMA_K;

  const int warp_idx = threadIdx.x / 32;
  const int warptile_row = (warp_idx / (BN / WN)) * WM;
  const int warptile_col = (warp_idx % (BN / WN)) * WN;

  const int lane_id = threadIdx.x % 32;
  const int group_id = lane_id >> 2;
  const int thread_id_in_group = lane_id % 4;

  // Values are loaded into registers outside
  // the inner warptile MMA loop
  uint32_t a_regs[TILES_M][TILES_K][2];
  uint32_t b_regs[TILES_K][TILES_N][1];
  float c_regs[TILES_M][TILES_N][4];

  // Initialize the C registers to zero
  for (int i = 0; i < TILES_M; ++i) {
    for (int j = 0; j < TILES_N; ++j) {
      c_regs[i][j][0] = 0.0f;
      c_regs[i][j][1] = 0.0f;
      c_regs[i][j][2] = 0.0f;
      c_regs[i][j][3] = 0.0f;
    }
  }

  __shared__ half a_shmem[BM][BK];
  __shared__ half b_shmem[BK][BN];

  for (int block_k = 0; block_k < K; block_k += BK) {
    // Load A tile into shared memory
    for (int i = threadIdx.x; i < BM * BK; i += num_threads) {
      const int row = i / BK;
      const int col = i % BK;
      a_shmem[row][col] = d_A[row * K + (block_k + col)];
    }

    // Load B tile into shared memory
    for (int i = threadIdx.x; i < BK * BN; i += num_threads) {
      const int row = i / BN;
      const int col = i % BN;
      b_shmem[row][col] = d_B[(block_k + row) * N + col];
    }

    __syncthreads();

    // This loop level only affects register usage
    for (int warp_k = 0; warp_k < BK; warp_k += WK) {
      // Load values from shared memory A tile to registers
      for (int m_tile = 0; m_tile < TILES_M; m_tile++) {
        for (int k_frag = 0; k_frag < TILES_K; k_frag++) {
          const int a_row_offset =
              warptile_row + m_tile * MMA_M + (lane_id % 16);
          const int a_col_offset = warp_k + k_frag * MMA_K;

          asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                       : "=r"(a_regs[m_tile][k_frag][0]),
                         "=r"(a_regs[m_tile][k_frag][1])
                       : "l"(&a_shmem[a_row_offset][a_col_offset]));
        }
      }

      // Load values from shared memory B tile to registers
      for (int k_frag = 0; k_frag < TILES_K; k_frag++) {
        for (int n_tile = 0; n_tile < TILES_N; n_tile++) {
          const int b_row_offset = (lane_id % 8) + warp_k + k_frag * MMA_K;
          const int b_col_offset = warptile_col + n_tile * MMA_N;

          asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.b16 {%0}, [%1];"
                       : "=r"(b_regs[k_frag][n_tile][0])
                       : "l"(&b_shmem[b_row_offset][b_col_offset]));
        }
      }

      // Perform MMA
      for (int k_frag = 0; k_frag < TILES_K; k_frag++) {
        for (int m_tile = 0; m_tile < TILES_M; m_tile++) {
          for (int n_tile = 0; n_tile < TILES_N; n_tile++) {
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                         "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
                         : "=f"(c_regs[m_tile][n_tile][0]),
                           "=f"(c_regs[m_tile][n_tile][1]),
                           "=f"(c_regs[m_tile][n_tile][2]),
                           "=f"(c_regs[m_tile][n_tile][3])
                         : "r"(a_regs[m_tile][k_frag][0]),
                           "r"(a_regs[m_tile][k_frag][1]),
                           "r"(b_regs[k_frag][n_tile][0]),
                           "f"(c_regs[m_tile][n_tile][0]),
                           "f"(c_regs[m_tile][n_tile][1]),
                           "f"(c_regs[m_tile][n_tile][2]),
                           "f"(c_regs[m_tile][n_tile][3]));
          }
        }
      }
    }

    __syncthreads();
  }

  // Epilogue
  for (int m_tile = 0; m_tile < TILES_M; m_tile++) {
    for (int n_tile = 0; n_tile < TILES_N; n_tile++) {
      const int row_offset = warptile_row + m_tile * MMA_M + group_id;
      const int col_offset =
          warptile_col + n_tile * MMA_N + thread_id_in_group * 2;

      d_C[row_offset * N + col_offset + 0] =
          alpha * c_regs[m_tile][n_tile][0] +
          beta * d_C[row_offset * N + col_offset + 0];
      d_C[row_offset * N + col_offset + 1] =
          alpha * c_regs[m_tile][n_tile][1] +
          beta * d_C[row_offset * N + col_offset + 1];
      d_C[(row_offset + 8) * N + col_offset + 0] =
          alpha * c_regs[m_tile][n_tile][2] +
          beta * d_C[(row_offset + 8) * N + col_offset + 0];
      d_C[(row_offset + 8) * N + col_offset + 1] =
          alpha * c_regs[m_tile][n_tile][3] +
          beta * d_C[(row_offset + 8) * N + col_offset + 1];
    }
  }
}

void run_hierarchical_mma_kernel(int M, int N, int K, float alpha,
                                 const half *d_A, const half *d_B, float beta,
                                 float *d_C) {
  // Blocktile sizes
  const int BM = 128;
  const int BN = 128;
  const int BK = 8;

  // Warptile sizes
  const int WM = 128;
  const int WN = 32;
  const int WK = 8;

  // Number of warps needs to be equal to the number of warptiles
  const int num_warps = (BN / WN) * (BM / WM);
  const int num_threads = 32 * num_warps;

  // The shape of the MMA we are using
  // (mma.sync.aligned.m16n8k8)
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 8;

  static_assert(MMA_M == 16);
  static_assert(MMA_N == 8);
  static_assert(MMA_K == 8);

  static_assert(BM % WM == 0);
  static_assert(BN % WN == 0);
  static_assert(BK % WK == 0);

  static_assert(WM % MMA_M == 0);
  static_assert(WN % MMA_N == 0);
  static_assert(WK % MMA_K == 0);

  dim3 block_dim(num_threads);
  dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM);

  hierarchical_mma_kernel<num_threads, BM, BN, BK, WM, WN, WK, MMA_M, MMA_N,
                          MMA_K>
      <<<grid_dim, block_dim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}