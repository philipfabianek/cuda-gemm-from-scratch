#pragma once

#include <chrono>
#include <random>
#include <vector>

#define CUDA_CHECK(err)                                                        \
  {                                                                            \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err),    \
              __FILE__, __LINE__);                                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

void initialize_matrix(std::vector<float> &matrix, int rows, int cols);

void verify_with_cublas_reference(int M, int N, const float *d_C_result,
                                  const float *d_C_reference);