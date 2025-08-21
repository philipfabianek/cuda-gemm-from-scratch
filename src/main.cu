#include <cstdio>
#include <vector>
#include "utils.cuh"
#include "kernels/01_naive.cuh"

void run_kernel(int kernel_id, int M, int N, int K, float alpha, float *d_A,
                float *d_B, float beta, float *d_C)
{
  switch (kernel_id)
  {
  case 1:
    run_naive_kernel(M, N, K, alpha, d_A, d_B, beta, d_C);
    break;
  default:
    fprintf(stderr, "Error: Invalid kernel ID.\n");
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    fprintf(stderr, "Usage: %s <kernel_id>\n", argv[0]);
    return 1;
  }
  int kernel_id = std::stoi(argv[1]);

  // Benchmark configuration
  bool random_initialization = true;
  int size = 2048;
  int M = size;
  int N = size;
  int K = size;
  float alpha = 2.0f;
  float beta = 0.5f;

  // Prepare host matrices
  std::vector<float> h_A(M * K);
  std::vector<float> h_B(K * N);
  std::vector<float> h_C(M * N);

  initialize_matrix(h_A, M, K, random_initialization);
  initialize_matrix(h_B, K, N, random_initialization);
  initialize_matrix(h_C, M, N, random_initialization);

  // Store initial C matrix for verification
  std::vector<float> h_C_initial = h_C;

  // Prepare device variables
  float *d_A, *d_B, *d_C;
  size_t a_size = h_A.size() * sizeof(float);
  size_t b_size = h_B.size() * sizeof(float);
  size_t c_size = h_C.size() * sizeof(float);

  CUDA_CHECK(cudaMalloc(&d_A, a_size));
  CUDA_CHECK(cudaMalloc(&d_B, b_size));
  CUDA_CHECK(cudaMalloc(&d_C, c_size));

  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), c_size, cudaMemcpyHostToDevice));

  // Warm-up run, reset d_C afterwards!
  run_kernel(kernel_id, M, N, K, alpha, d_A, d_B, beta, d_C);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), c_size, cudaMemcpyHostToDevice));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Execute the kernel, no need to check errors now
  run_kernel(kernel_id, M, N, K, alpha, d_A, d_B, beta, d_C);

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel: Naive\n");
  printf("Matrix Size: %dx%d and %dx%d\n", M, K, K, N);
  printf("Execution time: %f ms\n", milliseconds);

  // Calculate TFLOPS
  long long total_ops = (long long)2 * M * K * N;
  double tflops = (double)total_ops / (milliseconds / 1000.0) / 1e12;
  printf("Performance: %.2f TFLOPS\n", tflops);

  // Verify results
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_size, cudaMemcpyDeviceToHost));
  verify_on_cpu(M, K, N, alpha, h_A, h_B, beta, h_C_initial, h_C);

  // Free memory and destroy events
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}