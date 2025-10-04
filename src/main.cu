#include <cstdio>
#include <vector>

#include "arg_parser.cuh"
#include "benchmark_runner.cuh"
#include "types.cuh"

int main(int argc, char **argv) {
  ArgParser parser(argc, argv);

  if (parser.cmd_option_exists("--help")) {
    printf("Usage: %s"
           "--kernel <id>"
           " [--size N]"
           " [--repeats N]"
           " [--precision <fp32|fp16>]\n",
           argv[0]);
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
  int size = parser.get_cmd_option<int>("--size", 2048);
  int repeats = parser.get_cmd_option<int>("--repeats", 100);
  std::string precision =
      parser.get_cmd_option<std::string>("--precision", "fp32");

  if (precision != "fp32" && precision != "fp16") {
    fprintf(stderr, "Error: Invalid precision, choose 'fp32' or 'fp16'\n");
    exit(EXIT_FAILURE);
  }

  // Get device properties
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);

  // Check if fp16 is supported
  if (precision == "fp16" && props.major < 8) {
    fprintf(stderr,
            "Error: fp16 precision requires a GPU with Compute Capability "
            "8.0 or higher\n");
    exit(EXIT_FAILURE);
  }

  // Create cublas handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  if (precision == "fp32") {
    switch (kernel_id) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
      run_and_benchmark<float, float>(kernel_id, size, repeats, handle);
      break;
    default:
      fprintf(
          stderr,
          "Error: Kernel ID %d does not exist or it does not support the fp32 "
          "precision.\n",
          kernel_id);
      cublasDestroy(handle);
      exit(EXIT_FAILURE);
    }
  } else if (precision == "fp16") {
    switch (kernel_id) {
    case 0:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
      run_and_benchmark<half, float>(kernel_id, size, repeats, handle);
      break;
    default:
      fprintf(
          stderr,
          "Error: Kernel ID %d does not exist or it does not support the fp16 "
          "precision.\n",
          kernel_id);
      cublasDestroy(handle);
      exit(EXIT_FAILURE);
    }
  }

  cublasDestroy(handle);

  return 0;
}
