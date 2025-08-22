#pragma once

#include <cmath>

constexpr float VERIFICATION_TOLERANCE = 1e-4f;

__global__ void comparison_kernel(const float *result, const float *reference,
                                  int n_elements, int *error_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n_elements) {
    if (fabs(result[idx] - reference[idx]) > VERIFICATION_TOLERANCE) {
      atomicAdd(error_count, 1);
    }
  }
}