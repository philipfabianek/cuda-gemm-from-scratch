#pragma once

#include <cstdint>

#include "types.cuh"

// The shape of the MMA we are using
// (mma.sync.aligned.m16n8k8)
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 8;

template <const int num_threads, const int BM, const int BN, const int BK>
__global__ void naive_mma_kernel(int M, int N, int K, float alpha,
                                 const half *d_A, const half *d_B, float beta,
                                 float *d_C) {
  d_A += blockIdx.y * BM * K;
  d_B += blockIdx.x * BN;
  d_C += blockIdx.y * BM * N + blockIdx.x * BN;

  constexpr int TILES_M = BM / MMA_M;
  constexpr int TILES_N = BN / MMA_N;

  // Lane ID is the thread index within the warp
  const int lane_id = threadIdx.x;
  const int group_id = lane_id >> 2;
  const int thread_id_in_group = lane_id % 4;

  // Thread needs to store 4 x fp16 values for the A matrix,
  // which fit into two 32-bit registers
  uint32_t a_regs[2];

  // Thread needs to store 2 x fp16 values for the B matrix,
  // which fit into one 32-bit register
  uint32_t b_regs[1];

  // Thread needs to store 4 x fp32 values for the C matrix,
  // which fit into four 32-bit registers, since this is for accumulation
  // we need the registers for each tile
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

  for (int k_tile = 0; k_tile < K; k_tile += BK) {
    // Load A tile into shared memory
    for (int i = lane_id; i < BM * BK; i += 32) {
      const int row = i / BK;
      const int col = i % BK;
      a_shmem[row][col] = d_A[row * K + (k_tile + col)];
    }

    // Load B tile into shared memory
    for (int i = lane_id; i < BK * BN; i += 32) {
      const int row = i / BN;
      const int col = i % BN;
      b_shmem[row][col] = d_B[(k_tile + row) * N + col];
    }

    __syncthreads();

    for (int k_step = 0; k_step < BK; k_step += MMA_K) {
      for (int m_tile = 0; m_tile < TILES_M; m_tile++) {
        for (int n_tile = 0; n_tile < TILES_N; n_tile++) {
          // Obtain shared memory offsets for A and B matrices
          // which can be used for the ldmatrix instructions,
          // each thread calculates a pointer to the start of the matrix row it's responsible for
          const int a_row_offset = m_tile * MMA_M + (lane_id % 16);
          const int a_col_offset = k_step;
          const int b_row_offset = (lane_id % 8) + k_step;
          const int b_col_offset = n_tile * MMA_N;

          // Load values from shared memory to registers
          asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                       : "=r"(a_regs[0]), "=r"(a_regs[1])
                       : "l"(&a_shmem[a_row_offset][a_col_offset]));
          asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.b16 {%0}, [%1];"
                       : "=r"(b_regs[0])
                       : "l"(&b_shmem[b_row_offset][b_col_offset]));

          // Perform the MMA operation
          // A fragment distribution: https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-1688-a-f16
          // B fragment distribution: https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-1688-b-f16
          // C fragment distribution: https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-1688-c-f16-f32
          asm volatile(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              : "=f"(c_regs[m_tile][n_tile][0]),
                "=f"(c_regs[m_tile][n_tile][1]),
                "=f"(c_regs[m_tile][n_tile][2]), "=f"(c_regs[m_tile][n_tile][3])
              : "r"(a_regs[0]), "r"(a_regs[1]), "r"(b_regs[0]),
                "f"(c_regs[m_tile][n_tile][0]), "f"(c_regs[m_tile][n_tile][1]),
                "f"(c_regs[m_tile][n_tile][2]), "f"(c_regs[m_tile][n_tile][3]));
        }
      }
    }

    __syncthreads();
  }

  // Epilogue, for indexing details see the previously linked C fragment distribution
  for (int m_tile = 0; m_tile < TILES_M; m_tile++) {
    for (int n_tile = 0; n_tile < TILES_N; n_tile++) {
      const int row_offset = m_tile * MMA_M + group_id;
      const int col_offset = n_tile * MMA_N + thread_id_in_group * 2;

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

void run_naive_mma_kernel(int M, int N, int K, float alpha, const half *d_A,
                          const half *d_B, float beta, float *d_C) {
  const int BM = 64;
  const int BN = 32;
  const int BK = 8;

  static_assert(BM % MMA_M == 0);
  static_assert(BN % MMA_N == 0);
  static_assert(BK % MMA_K == 0);

  // This kernel is called "naive" because one block consists of only one warp
  // and so blocktile and warptile sizes are identical
  const int num_threads = 32;
  static_assert(num_threads == 32);

  dim3 block_dim(num_threads);
  dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM);

  naive_mma_kernel<num_threads, BM, BN, BK>
      <<<grid_dim, block_dim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}