#pragma once

template <const int TILE_DIM>
__global__ void sgemm_tiled_kernel(int M, int N, int K, float alpha,
                                   const float *d_A, const float *d_B,
                                   float beta, float *d_C) {
  int col = blockIdx.x * TILE_DIM + threadIdx.x;
  int row = blockIdx.y * TILE_DIM + threadIdx.y;

  int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;
  __shared__ float A_tile[TILE_DIM][TILE_DIM];
  __shared__ float B_tile[TILE_DIM][TILE_DIM];

  float sum = 0;
  for (int i = 0; i < num_tiles; i++) {
    int A_read_row = row;
    int A_read_col = TILE_DIM * i + threadIdx.x;

    if (A_read_row < M && A_read_col < K) {
      A_tile[threadIdx.y][threadIdx.x] = d_A[A_read_row * K + A_read_col];
    } else {
      A_tile[threadIdx.y][threadIdx.x] = 0;
    }

    int B_read_row = TILE_DIM * i + threadIdx.y;
    int B_read_col = col;

    if (B_read_row < K && B_read_col < N) {
      B_tile[threadIdx.y][threadIdx.x] = d_B[B_read_row * N + B_read_col];
    } else {
      B_tile[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    for (int k = 0; k < TILE_DIM; k++) {
      sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    d_C[row * N + col] = alpha * sum + beta * d_C[row * N + col];
  }
}

void run_tiled_kernel(int M, int N, int K, float alpha, const float *d_A,
                      const float *d_B, float beta, float *d_C) {
  const int TILE_DIM = 16;
  dim3 threads_per_block(TILE_DIM, TILE_DIM);
  dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x,
                  (M + threads_per_block.y - 1) / threads_per_block.y);

  sgemm_tiled_kernel<TILE_DIM>
      <<<num_blocks, threads_per_block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}