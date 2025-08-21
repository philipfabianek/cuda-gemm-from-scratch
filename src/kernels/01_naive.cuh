#pragma once

__global__ void sgemm_naive_kernel(int M, int N, int K, float alpha, float *d_A,
                                   float *d_B, float beta, float *d_C) {
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float sum = 0;
    for (int k = 0; k < K; k++) {
      sum += d_A[row * K + k] * d_B[k * N + col];
    }
    d_C[row * N + col] = alpha * sum + beta * d_C[row * N + col];
  }
}

void run_naive_kernel(int M, int K, int N, float alpha, float *d_A, float *d_B,
                      float beta, float *d_C) {
  dim3 threads_per_block(16, 16);
  dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x,
                  (M + threads_per_block.y - 1) / threads_per_block.y);

  sgemm_naive_kernel<<<num_blocks, threads_per_block>>>(M, K, N, alpha, d_A,
                                                        d_B, beta, d_C);
}