#pragma once

#include <cmath>

constexpr float VERIFICATION_TOLERANCE = 1e-3f;

__global__ void comparison_kernel(const float *result, const float *reference,
                                  int n_elements, int *error_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n_elements) {
    float diff = result[idx] - reference[idx];
    if (isnan(diff) || fabs(diff) > VERIFICATION_TOLERANCE) {
      atomicAdd(error_count, 1);
    }
  }
}