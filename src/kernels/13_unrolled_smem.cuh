#pragma once

#include <cstdint>

#include "types.cuh"
#include "utils.cuh"

namespace k13 {

template <const int num_threads, const int smem_rows, const int smem_cols>
__device__ __forceinline__ void load_from_gmem(const int gmem_cols,
                                               const half *src, float4 *dst) {
  constexpr int smem_row_stride = (num_threads * 8) / smem_cols;
  constexpr int iters = smem_rows / smem_row_stride;

  const int smem_col = (threadIdx.x * 8) % smem_cols;
  int smem_row = (threadIdx.x * 8) / smem_cols;

  for (int i = 0; i < iters; i++) {
    const int gmem_idx = smem_row * gmem_cols + smem_col;
    dst[i] = reinterpret_cast<const float4 *>(&src[gmem_idx])[0];
    smem_row += smem_row_stride;
  }
}

template <const int num_threads, const int smem_rows, const int smem_cols>
__device__ __forceinline__ void store_to_smem(const float4 *src, half *dst) {
  constexpr int swizzle_bits = int_log2(smem_cols / 8);
  constexpr int swizzle_mask = 0b111 << (3 + swizzle_bits);

  constexpr int smem_row_stride = (num_threads * 8) / smem_cols;
  constexpr int iters = smem_rows / smem_row_stride;

  const int smem_col = (threadIdx.x * 8) % smem_cols;
  int smem_row = (threadIdx.x * 8) / smem_cols;

  for (int i = 0; i < iters; i++) {
    int smem_idx = smem_row * smem_cols + smem_col;

    // Swizzle
    smem_idx = smem_idx ^ ((smem_idx & swizzle_mask) >> swizzle_bits);

    reinterpret_cast<float4 *>(&dst[smem_idx])[0] = src[i];
    smem_row += smem_row_stride;
  }
}

template <const int BK, const int mma_tiles_m, const int mma_tiles_k>
__device__ __forceinline__ void
load_a_from_smem(const half *warptile_a_smem,
                 uint32_t (&a_regs)[mma_tiles_m][mma_tiles_k][2]) {
  int thread_offset = (threadIdx.x % 32) * BK;

  // Swizzle
  constexpr int a_swizzle_bits = int_log2(BK / 8);
  constexpr int a_swizzle_mask = 0b111 << (3 + a_swizzle_bits);

  thread_offset =
      thread_offset ^ ((thread_offset & a_swizzle_mask) >> a_swizzle_bits);

  // Compute shared memory offset for the thread
  uint32_t total_byte_offset =
      cvta_to_shared_u32(warptile_a_smem + thread_offset);

  // The 'ldmatrix...m8n8.x4' instruction handles 32 rows hence this value
  constexpr int ldmatrix_stride = 32 * BK * sizeof(half);

  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(a_regs[0][0][0]), "=r"(a_regs[0][0][1]), "=r"(a_regs[1][0][0]),
        "=r"(a_regs[1][0][1])
      : "r"(total_byte_offset));

  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(a_regs[2][0][0]), "=r"(a_regs[2][0][1]), "=r"(a_regs[3][0][0]),
        "=r"(a_regs[3][0][1])
      : "r"(total_byte_offset + ldmatrix_stride));

  // Adjust for the next set of columns
  total_byte_offset ^= 0b1 << int_log2(8 * sizeof(half));

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, "
               "%3}, [%4];"
               : "=r"(a_regs[0][1][0]), "=r"(a_regs[0][1][1]),
                 "=r"(a_regs[1][1][0]), "=r"(a_regs[1][1][1])
               : "r"(total_byte_offset));

  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(a_regs[2][1][0]), "=r"(a_regs[2][1][1]), "=r"(a_regs[3][1][0]),
        "=r"(a_regs[3][1][1])
      : "r"(total_byte_offset + ldmatrix_stride));

  // Adjust for the next set of columns,
  // we want the initial offset again but with the next bit set,
  // this achieves just that using the previous modified offset
  total_byte_offset ^= 0b11 << int_log2(8 * sizeof(half));

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, "
               "%3}, [%4];"
               : "=r"(a_regs[0][2][0]), "=r"(a_regs[0][2][1]),
                 "=r"(a_regs[1][2][0]), "=r"(a_regs[1][2][1])
               : "r"(total_byte_offset));

  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(a_regs[2][2][0]), "=r"(a_regs[2][2][1]), "=r"(a_regs[3][2][0]),
        "=r"(a_regs[3][2][1])
      : "r"(total_byte_offset + ldmatrix_stride));

  // This is sufficient given the previous modified offset
  total_byte_offset ^= 0b1 << int_log2(8 * sizeof(half));

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, "
               "%3}, [%4];"
               : "=r"(a_regs[0][3][0]), "=r"(a_regs[0][3][1]),
                 "=r"(a_regs[1][3][0]), "=r"(a_regs[1][3][1])
               : "r"(total_byte_offset));

  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(a_regs[2][3][0]), "=r"(a_regs[2][3][1]), "=r"(a_regs[3][3][0]),
        "=r"(a_regs[3][3][1])
      : "r"(total_byte_offset + ldmatrix_stride));
}

template <const int BN, const int mma_tiles_n, const int mma_tiles_k>
__device__ __forceinline__ void
load_b_from_smem(const half *warptile_b_smem,
                 uint32_t (&b_regs)[mma_tiles_k][mma_tiles_n][1]) {
  int thread_offset = ((threadIdx.x % 8) * BN + ((threadIdx.x % 32) / 8) * 8);

  // Swizzle
  constexpr int b_swizzle_bits = int_log2(BN / 8);
  constexpr int b_swizzle_mask = 0b111 << (3 + b_swizzle_bits);

  thread_offset =
      thread_offset ^ ((thread_offset & b_swizzle_mask) >> b_swizzle_bits);

  // Compute shared memory offset for the thread
  uint32_t total_byte_offset =
      cvta_to_shared_u32(warptile_b_smem + thread_offset);

  // The 'ldmatrix...m8n8.x4.trans' instruction handles 32 columns hence this value
  constexpr int ldmatrix_xor = 0b1 << int_log2(32 * sizeof(half));

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, "
               "%2, %3}, [%4];"
               : "=r"(b_regs[0][0][0]), "=r"(b_regs[0][1][0]),
                 "=r"(b_regs[0][2][0]), "=r"(b_regs[0][3][0])
               : "r"(total_byte_offset));

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, "
               "%2, %3}, [%4];"
               : "=r"(b_regs[0][4][0]), "=r"(b_regs[0][5][0]),
                 "=r"(b_regs[0][6][0]), "=r"(b_regs[0][7][0])
               : "r"(total_byte_offset ^ ldmatrix_xor));

  // Move to the next set of rows
  total_byte_offset += 8 * BN * sizeof(half);

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, "
               "%2, %3}, [%4];"
               : "=r"(b_regs[1][0][0]), "=r"(b_regs[1][1][0]),
                 "=r"(b_regs[1][2][0]), "=r"(b_regs[1][3][0])
               : "r"(total_byte_offset));

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, "
               "%2, %3}, [%4];"
               : "=r"(b_regs[1][4][0]), "=r"(b_regs[1][5][0]),
                 "=r"(b_regs[1][6][0]), "=r"(b_regs[1][7][0])
               : "r"(total_byte_offset ^ ldmatrix_xor));

  // Move to the next set of rows
  total_byte_offset += 8 * BN * sizeof(half);

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, "
               "%2, %3}, [%4];"
               : "=r"(b_regs[2][0][0]), "=r"(b_regs[2][1][0]),
                 "=r"(b_regs[2][2][0]), "=r"(b_regs[2][3][0])
               : "r"(total_byte_offset));

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, "
               "%2, %3}, [%4];"
               : "=r"(b_regs[2][4][0]), "=r"(b_regs[2][5][0]),
                 "=r"(b_regs[2][6][0]), "=r"(b_regs[2][7][0])
               : "r"(total_byte_offset ^ ldmatrix_xor));

  // Move to the next set of rows
  total_byte_offset += 8 * BN * sizeof(half);

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, "
               "%2, %3}, [%4];"
               : "=r"(b_regs[3][0][0]), "=r"(b_regs[3][1][0]),
                 "=r"(b_regs[3][2][0]), "=r"(b_regs[3][3][0])
               : "r"(total_byte_offset));

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, "
               "%2, %3}, [%4];"
               : "=r"(b_regs[3][4][0]), "=r"(b_regs[3][5][0]),
                 "=r"(b_regs[3][6][0]), "=r"(b_regs[3][7][0])
               : "r"(total_byte_offset ^ ldmatrix_xor));
}

template <const int num_threads, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WK, const int MMA_M,
          const int MMA_N, const int MMA_K>
__global__ void unrolled_smem_kernel(int M, int N, int K, float alpha,
                                     const half *d_A, const half *d_B,
                                     float beta, float *d_C) {
  d_A += blockIdx.y * BM * K;
  d_B += blockIdx.x * BN;
  d_C += blockIdx.y * BM * N + blockIdx.x * BN;

  // These numbers describe the number of MMA instructions
  // within one warptile iteration
  constexpr int mma_tiles_m = WM / MMA_M;
  constexpr int mma_tiles_n = WN / MMA_N;
  constexpr int mma_tiles_k = WK / MMA_K;

  // This code manually manages the ldmatrix instructions
  // while were previously inside a loop so these are static now
  static_assert(mma_tiles_m == 4);
  static_assert(mma_tiles_n == 8);
  static_assert(mma_tiles_k == 4);

  const int warp_idx = threadIdx.x / 32;
  const int warptile_row = (warp_idx / (BN / WN)) * WM;
  const int warptile_col = (warp_idx % (BN / WN)) * WN;

  const int lane_id = threadIdx.x % 32;
  const int group_id = lane_id >> 2;
  const int thread_id_in_group = lane_id % 4;

  // Values are loaded into registers outside
  // the inner warptile MMA loop
  uint32_t a_regs[mma_tiles_m][mma_tiles_k][2];
  uint32_t b_regs[mma_tiles_k][mma_tiles_n][1];
  float c_regs[mma_tiles_m][mma_tiles_n][4];

  // Initialize the C registers to zero
  for (int i = 0; i < mma_tiles_m; ++i) {
    for (int j = 0; j < mma_tiles_n; ++j) {
      c_regs[i][j][0] = 0.0f;
      c_regs[i][j][1] = 0.0f;
      c_regs[i][j][2] = 0.0f;
      c_regs[i][j][3] = 0.0f;
    }
  }

  // Setup shared memory
  __shared__ half a_smem[BM * BK];
  __shared__ half b_smem[BK * BN];

  constexpr int a_smem_row_stride = (num_threads * 8) / BK;
  constexpr int a_smem_iters = BM / a_smem_row_stride;

  constexpr int b_smem_row_stride = (num_threads * 8) / BN;
  constexpr int b_smem_iters = BK / b_smem_row_stride;

  // Registers for loading values from gmem to smem
  float4 tmp_a[a_smem_iters];
  float4 tmp_b[b_smem_iters];

  // Load values from gmem to smem for the first tile
  load_from_gmem<num_threads, BM, BK>(K, d_A, tmp_a);
  load_from_gmem<num_threads, BK, BN>(N, d_B, tmp_b);
  store_to_smem<num_threads, BM, BK>(tmp_a, a_smem);
  store_to_smem<num_threads, BK, BN>(tmp_b, b_smem);

  for (int block_k = 0; block_k < K; block_k += BK) {
    __syncthreads();

    // Start loading values from gmem for the next tile
    load_from_gmem<num_threads, BM, BK>(K, d_A + (block_k + BK), tmp_a);
    load_from_gmem<num_threads, BK, BN>(N, d_B + (block_k + BK) * N, tmp_b);

    // This loop level only affects register usage
    for (int warp_k = 0; warp_k < BK; warp_k += WK) {
      // Load values from shared memory to registers
      const half *warptile_a_smem = &a_smem[warptile_row * BK + warp_k];
      load_a_from_smem<BK, mma_tiles_m, mma_tiles_k>(warptile_a_smem, a_regs);
      const half *warptile_b_smem = &b_smem[warp_k * BN + warptile_col];
      load_b_from_smem<BN, mma_tiles_n, mma_tiles_k>(warptile_b_smem, b_regs);

      // Perform MMA
      for (int k_frag = 0; k_frag < mma_tiles_k; k_frag++) {
        for (int m_tile = 0; m_tile < mma_tiles_m; m_tile++) {
          for (int n_tile = 0; n_tile < mma_tiles_n; n_tile++) {
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

    // Store the next tile to shared memory
    store_to_smem<num_threads, BM, BK>(tmp_a, a_smem);
    store_to_smem<num_threads, BK, BN>(tmp_b, b_smem);
  }

  // Epilogue
  for (int m_tile = 0; m_tile < mma_tiles_m; m_tile++) {
    for (int n_tile = 0; n_tile < mma_tiles_n; n_tile++) {
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

void run_unrolled_smem_kernel(int M, int N, int K, float alpha, const half *d_A,
                              const half *d_B, float beta, float *d_C) {
  // Blocktile sizes
  const int BM = 128;
  const int BN = 128;
  const int BK = 32;

  // Warptile sizes
  const int WM = 64;
  const int WN = 64;
  const int WK = 32;

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

  unrolled_smem_kernel<num_threads, BM, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K>
      <<<grid_dim, block_dim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}

} // namespace k13