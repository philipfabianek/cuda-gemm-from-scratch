#pragma once

#include <cmath>

#include "types.cuh"

template <typename T>
__global__ void comparison_kernel(const T *result, const T *reference,
                                  int n_elements, int *error_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n_elements) {
    float val_result;
    float val_reference;
    float tolerance;

    if constexpr (std::is_same_v<T, float>) {
      val_result = result[idx];
      val_reference = reference[idx];
      tolerance = 1e-3f;
    } else if constexpr (std::is_same_v<T, half>) {
      val_result = __half2float(result[idx]);
      val_reference = __half2float(reference[idx]);
      tolerance = 1e-3f;
    }

    float diff = val_result - val_reference;
    if (isnan(diff) || fabs(diff) > tolerance) {
      atomicAdd(error_count, 1);
    }
  }
}