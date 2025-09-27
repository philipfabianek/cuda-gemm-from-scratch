#pragma once

#include <vector>

#include "kernel_dispatcher.cuh"
#include "utils.cuh"

template <typename InputType, typename AccumType>
void run_warmup_and_verify(int kernel_id, cublasHandle_t handle, int M, int N,
                           int K, float alpha, InputType *d_A, InputType *d_B,
                           float beta, AccumType *d_C) {
  size_t c_size = (size_t)M * N * sizeof(AccumType);

  // Store initial C matrix to reset after each run
  AccumType *d_C_initial;
  CUDA_CHECK(cudaMalloc(&d_C_initial, c_size));
  CUDA_CHECK(cudaMemcpy(d_C_initial, d_C, c_size, cudaMemcpyDeviceToDevice));

  // Generate cuBLAS reference result
  AccumType *d_C_reference;
  CUDA_CHECK(cudaMalloc(&d_C_reference, c_size));
  CUDA_CHECK(
      cudaMemcpy(d_C_reference, d_C_initial, c_size, cudaMemcpyDeviceToDevice));
  run_cublas_kernel(handle, M, N, K, alpha, d_A, d_B, beta, d_C_reference);

  // Warm-up run, check for errors, verify results and reset d_C afterwards
  run_kernel<InputType, AccumType>(kernel_id, handle, M, N, K, alpha, d_A, d_B,
                                   beta, d_C);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  verify_with_cublas_reference(M, N, d_C, d_C_reference);

  CUDA_CHECK(cudaMemcpy(d_C, d_C_initial, c_size, cudaMemcpyDeviceToDevice));

  CUDA_CHECK(cudaFree(d_C_initial));
  CUDA_CHECK(cudaFree(d_C_reference));
}

template <typename InputType, typename AccumType>
void run_and_benchmark(int kernel_id, int size, int repeats,
                       cublasHandle_t handle) {
  // Benchmark configuration
  int M = size;
  int N = size;
  int K = size;
  float alpha = 2.0f;
  float beta = 0.5f;

  // Prepare host matrices
  std::vector<InputType> h_A(M * K);
  std::vector<InputType> h_B(K * N);
  std::vector<AccumType> h_C(M * N);

  initialize_matrix(h_A, M, K);
  initialize_matrix(h_B, K, N);
  initialize_matrix(h_C, M, N);

  // Prepare device variables
  InputType *d_A, *d_B;
  AccumType *d_C;
  size_t a_size = h_A.size() * sizeof(InputType);
  size_t b_size = h_B.size() * sizeof(InputType);
  size_t c_size = h_C.size() * sizeof(AccumType);

  CUDA_CHECK(cudaMalloc(&d_A, a_size));
  CUDA_CHECK(cudaMalloc(&d_B, b_size));
  CUDA_CHECK(cudaMalloc(&d_C, c_size));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), c_size, cudaMemcpyHostToDevice));

  // Do a warm-up run and verify results against cublas
  run_warmup_and_verify(kernel_id, handle, M, N, K, alpha, d_A, d_B, beta, d_C);

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Execute the kernel, no need to check errors here
  for (int i = 0; i < repeats; ++i) {
    run_kernel<InputType, AccumType>(kernel_id, handle, M, N, K, alpha, d_A,
                                     d_B, beta, d_C);
  }

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Measure performance
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  float avg_milliseconds = milliseconds / repeats;
  long long total_ops = (long long)2 * M * K * N;
  double gflops = (double)total_ops / (avg_milliseconds / 1000.0) / 1e9;
  printf("Kernel ID %d - Average time: (%f) ms, performance: (%.2f) GFLOPS, "
         "size: (%d).\n",
         kernel_id, avg_milliseconds, gflops, size);

  // Free memory and destroy events
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}