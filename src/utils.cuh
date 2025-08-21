#pragma once

#include <vector>
#include <chrono>
#include <random>

#define CUDA_CHECK(err)                                                                          \
  {                                                                                              \
    if (err != cudaSuccess)                                                                      \
    {                                                                                            \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

void initialize_matrix(std::vector<float> &matrix, int rows, int cols, bool is_random);

int verify_on_cpu(
    int M, int K, int N,
    float alpha,
    const std::vector<float> &A,
    const std::vector<float> &B,
    float beta,
    const std::vector<float> &C_initial,
    const std::vector<float> &C_result);