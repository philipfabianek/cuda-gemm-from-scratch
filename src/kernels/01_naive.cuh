#pragma once

__global__ void sgemm_naive_kernel(int M, int N, int K, float alpha,
                                   const float *d_A, const float *d_B,
                                   float beta, float *d_C) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float sum = 0;
    for (int k = 0; k < K; k++) {
      sum += d_A[row * K + k] * d_B[k * N + col];
    }
    d_C[row * N + col] = alpha * sum + beta * d_C[row * N + col];
  }
}

void run_naive_kernel(int M, int N, int K, float alpha, const float *d_A,
                      const float *d_B, float beta, float *d_C) {
  dim3 block_dim(16, 16);
  dim3 grid_dim((N + 16 - 1) / 16, (M + 16 - 1) / 16);

  sgemm_naive_kernel<<<grid_dim, block_dim>>>(M, N, K, alpha, d_A, d_B, beta,
                                              d_C);
}
