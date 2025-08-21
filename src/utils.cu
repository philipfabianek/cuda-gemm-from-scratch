#include "utils.cuh"

void initialize_matrix(std::vector<float> &matrix, int rows, int cols,
                       bool is_random) {
  if (is_random) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
    for (int i = 0; i < rows * cols; ++i) {
      matrix[i] = distribution(generator);
    }
  } else {
    for (int i = 0; i < rows * cols; ++i) {
      matrix[i] = 1.0f;
    }
  }
}

int verify_on_cpu(int M, int K, int N, float alpha, const std::vector<float> &A,
                  const std::vector<float> &B, float beta,
                  const std::vector<float> &C_initial,
                  const std::vector<float> &C_result) {
  int samples_to_check = 1000;

  for (int s = 0; s < samples_to_check; ++s) {
    int row = rand() % M;
    int col = rand() % N;

    float dot_product = 0.0f;
    for (int k = 0; k < K; ++k) {
      dot_product += A[row * K + k] * B[k * N + col];
    }

    float expected_value =
        alpha * dot_product + beta * C_initial[row * N + col];
    float gpu_result = C_result[row * N + col];

    if (fabs(gpu_result - expected_value) > 1e-4) {
      printf("Mismatch at (%d, %d): CPU expected %f, GPU produced %f\n", row,
             col, expected_value, gpu_result);
      return 1;
    }
  }

  printf("All (sampled) values match!\n");
  return 0;
}