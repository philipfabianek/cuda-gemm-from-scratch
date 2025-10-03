#pragma once

#include <cstdint>

#include "types.cuh"
#include "utils.cuh"

template <const int num_threads, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WK, const int MMA_M,
          const int MMA_N, const int MMA_K>
__global__ void memory_swizzling_kernel(int M, int N, int K, float alpha,
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

  // Setup shared memory
  __shared__ half a_shmem[BM * BK];
  __shared__ half b_shmem[BK * BN];

  constexpr int a_shmem_row_stride = (num_threads * 8) / BK;
  constexpr int a_shmem_iters = BM / a_shmem_row_stride;

  constexpr int b_shmem_row_stride = (num_threads * 8) / BN;
  constexpr int b_shmem_iters = BK / b_shmem_row_stride;

  // Swizzling parameters, mask is shifted by 3 bits because of the vectorization
  // and then by a number of bits needed for column index representation,
  // the _bytes masks are used for ldmatrix instructions because they use byte offsets
  // instead of element offsets, half is 2 bytes hence the << 1 shift
  constexpr int a_swizzle_bits = int_log2(BK / 8);
  constexpr int a_swizzle_mask = 0b111 << (3 + a_swizzle_bits);
  constexpr int a_swizzle_mask_bytes = a_swizzle_mask << 1;

  constexpr int b_swizzle_bits = int_log2(BN / 8);
  constexpr int b_swizzle_mask = 0b111 << (3 + b_swizzle_bits);
  constexpr int b_swizzle_mask_bytes = b_swizzle_mask << 1;

  for (int block_k = 0; block_k < K; block_k += BK) {
    int a_shmem_row = (threadIdx.x * 8) / BK;
    const int a_shmem_col = (threadIdx.x * 8) % BK;

    // Load swizzled A tile into shared memory
    for (int i = 0; i < a_shmem_iters; i++) {
      const int gmem_idx = a_shmem_row * K + (block_k + a_shmem_col);
      int smem_idx = a_shmem_row * BK + a_shmem_col;

      // Swizzle
      smem_idx = smem_idx ^ ((smem_idx & a_swizzle_mask) >> a_swizzle_bits);

      reinterpret_cast<float4 *>(&a_shmem[smem_idx])[0] =
          reinterpret_cast<const float4 *>(&d_A[gmem_idx])[0];
      a_shmem_row += a_shmem_row_stride;
    }

    int b_shmem_row = (threadIdx.x * 8) / BN;
    const int b_shmem_col = (threadIdx.x * 8) % BN;

    // Load swizzled B tile into shared memory
    for (int i = 0; i < b_shmem_iters; i++) {
      const int gmem_idx = (block_k + b_shmem_row) * N + b_shmem_col;
      int smem_idx = b_shmem_row * BN + b_shmem_col;

      // Swizzle
      smem_idx = smem_idx ^ ((smem_idx & b_swizzle_mask) >> b_swizzle_bits);

      reinterpret_cast<float4 *>(&b_shmem[smem_idx])[0] =
          reinterpret_cast<const float4 *>(&d_B[gmem_idx])[0];
      b_shmem_row += b_shmem_row_stride;
    }

    __syncthreads();

    // This loop level only affects register usage
    for (int warp_k = 0; warp_k < BK; warp_k += WK) {
      uint32_t a_shmem_byte_offset =
          cvta_to_shared_u32(&a_shmem[warptile_row * BK + warp_k]);

      // Load values from shared memory A tile to registers
      for (int m_tile = 0; m_tile < TILES_M; m_tile++) {
        for (int k_frag = 0; k_frag < TILES_K; k_frag++) {
          const int thread_byte_offset =
              ((m_tile * MMA_M + lane_id) * BK + k_frag * MMA_K) * sizeof(half);
          int total_byte_offset = a_shmem_byte_offset + thread_byte_offset;

          // Swizzle
          total_byte_offset =
              total_byte_offset ^
              ((total_byte_offset & a_swizzle_mask_bytes) >> a_swizzle_bits);

          // Providing shared memory offset is actually a little faster
          // than not using the .shared modifier and providing a generic pointer,
          // hence this setup
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
              : "=r"(a_regs[m_tile][k_frag][0]), "=r"(a_regs[m_tile][k_frag][1])
              : "r"(total_byte_offset));
        }
      }

      uint32_t b_shmem_byte_offset =
          cvta_to_shared_u32(&b_shmem[warp_k * BN + warptile_col]);

      // Load values from shared memory B tile to registers
      for (int k_frag = 0; k_frag < TILES_K; k_frag++) {
        for (int n_tile = 0; n_tile < TILES_N; n_tile++) {
          const int thread_byte_offset =
              ((k_frag * MMA_K + threadIdx.x % MMA_K) * BN + n_tile * MMA_N) *
              sizeof(half);
          int total_byte_offset = b_shmem_byte_offset + thread_byte_offset;

          // Swizzle
          total_byte_offset =
              total_byte_offset ^
              ((total_byte_offset & b_swizzle_mask_bytes) >> b_swizzle_bits);

          // Providing shared memory offset is actually a little faster
          // than not using the .shared modifier and providing a generic pointer,
          // hence this setup
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
              : "=r"(b_regs[k_frag][n_tile][0])
              : "r"(total_byte_offset));
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

void run_memory_swizzling_kernel(int M, int N, int K, float alpha,
                                 const half *d_A, const half *d_B, float beta,
                                 float *d_C) {
  // Blocktile sizes
  const int BM = 128;
  const int BN = 128;
  const int BK = 64;

  // Warptile sizes
  const int WM = 64;
  const int WN = 64;
  const int WK = 16;

  // Number of warps needs to be equal to the number of warptiles
  constexpr int num_warps = (BN / WN) * (BM / WM);
  constexpr int num_threads = 32 * num_warps;

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

  // Vectorized loads of 8 halfs (128 bits) at a time
  static_assert((BM * BK) % (8 * num_threads) == 0);
  static_assert((BK * BN) % (8 * num_threads) == 0);

  dim3 block_dim(num_threads);
  dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM);

  memory_swizzling_kernel<num_threads, BM, BN, BK, WM, WN, WK, MMA_M, MMA_N,
                          MMA_K>
      <<<grid_dim, block_dim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}