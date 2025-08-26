#include <cstdio>
#include <vector>

#include "arg_parser.cuh"
#include "kernels/00_cublas.cuh"
#include "kernels/01_naive.cuh"
#include "kernels/02_tiled.cuh"
#include "kernels/03_1D_coarsened.cuh"
#include "kernels/04_2D_coarsened.cuh"
#include "utils.cuh"

void run_kernel(int kernel_id, cublasHandle_t handle, int M, int N, int K,
                float alpha, const float *d_A, const float *d_B, float beta,
                float *d_C) {
  switch (kernel_id) {
  case 0:
    run_cublas_kernel(handle, M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 1:
    run_naive_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 2:
    run_tiled_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 3:
    run_1D_coarsened_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  case 4:
    run_2D_coarsened_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  default:
    fprintf(stderr, "Error: Invalid kernel ID.\n");
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv) {
  ArgParser parser(argc, argv);

  if (parser.cmd_option_exists("--help")) {
    printf("Usage: %s --kernel <id> [--size N] [--repeats N]\n", argv[0]);
    return 0;
  }

  // Make kernel_id a named, required argument
  if (!parser.cmd_option_exists("--kernel")) {
    fprintf(stderr, "Error: Missing required argument --kernel <id>\n");
    printf("Usage: %s --kernel <id> [--size N] [--repeats N]\n", argv[0]);
    return 1;
  }

  // Load CLI arguments
  int kernel_id = parser.get_cmd_option<int>("--kernel", 1);
  int size = parser.get_cmd_option<int>("--size", 1024);
  int repeats = parser.get_cmd_option<int>("--repeats", 100);

  // Benchmark configuration
  int M = size;
  int N = size;
  int K = size;
  float alpha = 2.0f;
  float beta = 0.5f;

  // Create cublas handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Prepare host matrices
  std::vector<float> h_A(M * K);
  std::vector<float> h_B(K * N);
  std::vector<float> h_C(M * N);

  initialize_matrix(h_A, M, K);
  initialize_matrix(h_B, K, N);
  initialize_matrix(h_C, M, N);

  // Store initial C matrix for verification
  std::vector<float> h_C_initial = h_C;

  // Prepare device variables
  float *d_A, *d_B, *d_C, *d_C_reference;
  size_t a_size = h_A.size() * sizeof(float);
  size_t b_size = h_B.size() * sizeof(float);
  size_t c_size = h_C.size() * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_A, a_size));
  CUDA_CHECK(cudaMalloc(&d_B, b_size));
  CUDA_CHECK(cudaMalloc(&d_C, c_size));
  CUDA_CHECK(cudaMalloc(&d_C_reference, c_size));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), c_size, cudaMemcpyHostToDevice));

  // Generate cublas reference, reset d_C afterwards
  CUDA_CHECK(
      cudaMemcpy(d_C_reference, h_C.data(), c_size, cudaMemcpyHostToDevice));
  run_cublas_kernel(handle, M, N, K, alpha, d_A, d_B, beta, d_C_reference);
  CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), c_size, cudaMemcpyHostToDevice));

  // Warm-up run, check for errors, verify results and reset d_C afterwards
  run_kernel(kernel_id, handle, M, N, K, alpha, d_A, d_B, beta, d_C);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  verify_with_cublas_reference(M, N, d_C, d_C_reference);
  CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), c_size, cudaMemcpyHostToDevice));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Execute the kernel, no need to check errors here
  for (int i = 0; i < repeats; ++i) {
    run_kernel(kernel_id, handle, M, N, K, alpha, d_A, d_B, beta, d_C);
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
  CUDA_CHECK(cudaFree(d_C_reference));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  cublasDestroy(handle);

  return 0;
}