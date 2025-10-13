#pragma once

#include <cstdint>

#include "types.cuh"
#include "utils.cuh"

namespace k12 {

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

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WK, const int MMA_M, const int MMA_N, const int MMA_K,
          const int MMA_TILES_M, const int MMA_TILES_N, const int MMA_TILES_K>
__device__ __forceinline__ void
compute_warptile_mma(const int warptile_row, const int warptile_col,
                     const int lane_id, half *a_smem, half *b_smem,
                     uint32_t a_regs[MMA_TILES_M][MMA_TILES_K][2],
                     uint32_t b_regs[MMA_TILES_K][MMA_TILES_N][1],
                     float c_regs[MMA_TILES_M][MMA_TILES_N][4]) {
  // Swizzling parameters
  constexpr int a_swizzle_bits = int_log2(BK / 8);
  constexpr int a_swizzle_mask = 0b111 << (3 + a_swizzle_bits);
  constexpr int a_swizzle_mask_bytes = a_swizzle_mask << 1;

  constexpr int b_swizzle_bits = int_log2(BN / 8);
  constexpr int b_swizzle_mask = 0b111 << (3 + b_swizzle_bits);
  constexpr int b_swizzle_mask_bytes = b_swizzle_mask << 1;

  // This loop level only affects register usage
  for (int warp_k = 0; warp_k < BK; warp_k += WK) {
    uint32_t a_smem_byte_offset =
        cvta_to_shared_u32(&a_smem[warptile_row * BK + warp_k]);

    // Load values from shared memory A tile to registers
    for (int m_tile = 0; m_tile < MMA_TILES_M; m_tile++) {
      for (int k_frag = 0; k_frag < MMA_TILES_K; k_frag++) {
        const int thread_byte_offset =
            ((m_tile * MMA_M + lane_id) * BK + k_frag * MMA_K) * sizeof(half);
        int total_byte_offset = a_smem_byte_offset + thread_byte_offset;

        // Swizzle
        total_byte_offset =
            total_byte_offset ^
            ((total_byte_offset & a_swizzle_mask_bytes) >> a_swizzle_bits);

        // Use 'ldmatrix' with shared memory offset
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                     : "=r"(a_regs[m_tile][k_frag][0]),
                       "=r"(a_regs[m_tile][k_frag][1])
                     : "r"(total_byte_offset));
      }
    }

    uint32_t b_smem_byte_offset =
        cvta_to_shared_u32(&b_smem[warp_k * BN + warptile_col]);

    // Load values from shared memory B tile to registers
    for (int k_frag = 0; k_frag < MMA_TILES_K; k_frag++) {
      for (int n_tile = 0; n_tile < MMA_TILES_N; n_tile++) {
        const int thread_byte_offset =
            ((k_frag * MMA_K + threadIdx.x % MMA_K) * BN + n_tile * MMA_N) *
            sizeof(half);
        int total_byte_offset = b_smem_byte_offset + thread_byte_offset;

        // Swizzle
        total_byte_offset =
            total_byte_offset ^
            ((total_byte_offset & b_swizzle_mask_bytes) >> b_swizzle_bits);

        // Use 'ldmatrix' with shared memory offset
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
            : "=r"(b_regs[k_frag][n_tile][0])
            : "r"(total_byte_offset));
      }
    }

    // Perform MMA
    for (int k_frag = 0; k_frag < MMA_TILES_K; k_frag++) {
      for (int m_tile = 0; m_tile < MMA_TILES_M; m_tile++) {
        for (int n_tile = 0; n_tile < MMA_TILES_N; n_tile++) {
          asm volatile(
              "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
              "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
              : "=f"(c_regs[m_tile][n_tile][0]),
                "=f"(c_regs[m_tile][n_tile][1]),
                "=f"(c_regs[m_tile][n_tile][2]), "=f"(c_regs[m_tile][n_tile][3])
              : "r"(a_regs[m_tile][k_frag][0]), "r"(a_regs[m_tile][k_frag][1]),
                "r"(b_regs[k_frag][n_tile][0]), "f"(c_regs[m_tile][n_tile][0]),
                "f"(c_regs[m_tile][n_tile][1]), "f"(c_regs[m_tile][n_tile][2]),
                "f"(c_regs[m_tile][n_tile][3]));
        }
      }
    }
  }
}

template <const int num_threads, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WK, const int MMA_M,
          const int MMA_N, const int MMA_K, const int MMA_TILES_M,
          const int MMA_TILES_N, const int MMA_TILES_K>
__global__ void buffered_gmem_kernel(int M, int N, int K, float alpha,
                                     const half *d_A, const half *d_B,
                                     float beta, float *d_C) {
  d_A += blockIdx.y * BM * K;
  d_B += blockIdx.x * BN;
  d_C += blockIdx.y * BM * N + blockIdx.x * BN;

  const int warp_idx = threadIdx.x / 32;
  const int warptile_row = (warp_idx / (BN / WN)) * WM;
  const int warptile_col = (warp_idx % (BN / WN)) * WN;

  const int lane_id = threadIdx.x % 32;
  const int group_id = lane_id >> 2;
  const int thread_id_in_group = lane_id % 4;

  // Values are loaded into registers outside
  // the inner warptile MMA loop
  uint32_t a_regs[MMA_TILES_M][MMA_TILES_K][2];
  uint32_t b_regs[MMA_TILES_K][MMA_TILES_N][1];
  float c_regs[MMA_TILES_M][MMA_TILES_N][4];

  // Initialize the C registers to zero
  for (int i = 0; i < MMA_TILES_M; ++i) {
    for (int j = 0; j < MMA_TILES_N; ++j) {
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

  // Compute MMA for all tiles but the last one
  for (int block_k = BK; block_k < K; block_k += BK) {
    // Call '__syncthreads' here because 'store_to_smem' calls
    // are at the end of the loop, this is also compatible
    // with the code above which loads the first tile
    __syncthreads();

    // Start loading values from gmem for the next tile,
    // the 'store_to_smem' calls are at the end of the loop
    // so that the computation can overlap with gmem loading
    load_from_gmem<num_threads, BM, BK>(K, d_A + block_k, tmp_a);
    load_from_gmem<num_threads, BK, BN>(N, d_B + block_k * N, tmp_b);

    // Compute the MMA for this warptile
    compute_warptile_mma<BM, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K,
                         MMA_TILES_M, MMA_TILES_N, MMA_TILES_K>(
        warptile_row, warptile_col, lane_id, a_smem, b_smem, a_regs, b_regs,
        c_regs);

    // Call '__syncthreads' here to finish the computation
    // before overwriting the shared memory values
    __syncthreads();

    store_to_smem<num_threads, BM, BK>(tmp_a, a_smem);
    store_to_smem<num_threads, BK, BN>(tmp_b, b_smem);
  }

  // Compute MMA for the last tile,
  // gmem loading and smem storing is not needed here
  {
    __syncthreads();

    compute_warptile_mma<BM, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K,
                         MMA_TILES_M, MMA_TILES_N, MMA_TILES_K>(
        warptile_row, warptile_col, lane_id, a_smem, b_smem, a_regs, b_regs,
        c_regs);

    __syncthreads();
  }

  // Epilogue
  for (int m_tile = 0; m_tile < MMA_TILES_M; m_tile++) {
    for (int n_tile = 0; n_tile < MMA_TILES_N; n_tile++) {
      const int row_offset = warptile_row + m_tile * MMA_M + group_id;
      const int col_offset =
          warptile_col + n_tile * MMA_N + thread_id_in_group * 2;

      float2 *c_ptr1 =
          reinterpret_cast<float2 *>(&d_C[row_offset * N + col_offset]);
      float2 *c_ptr2 =
          reinterpret_cast<float2 *>(&d_C[(row_offset + 8) * N + col_offset]);

      const float2 old_c1 = c_ptr1[0];
      const float2 old_c2 = c_ptr2[0];

      c_ptr1[0] =
          make_float2(alpha * c_regs[m_tile][n_tile][0] + beta * old_c1.x,
                      alpha * c_regs[m_tile][n_tile][1] + beta * old_c1.y);
      c_ptr2[0] =
          make_float2(alpha * c_regs[m_tile][n_tile][2] + beta * old_c2.x,
                      alpha * c_regs[m_tile][n_tile][3] + beta * old_c2.y);
    }
  }
}

void run_buffered_gmem_kernel(int M, int N, int K, float alpha, const half *d_A,
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

  // These numbers describe the number of MMA instructions
  // within one warptile iteration
  constexpr int MMA_TILES_M = WM / MMA_M;
  constexpr int MMA_TILES_N = WN / MMA_N;
  constexpr int MMA_TILES_K = WK / MMA_K;

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

  buffered_gmem_kernel<num_threads, BM, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K,
                       MMA_TILES_M, MMA_TILES_N, MMA_TILES_K>
      <<<grid_dim, block_dim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}

} // namespace k12