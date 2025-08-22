# CUDA SGEMM From Scratch

This repository is a step-by-step implementation and optimization of (single-precision general) matrix multiplication (SGEMM) in CUDA. Starting from a naive kernel, this project progressively applies advanced techniques to approach the performance of NVIDIA's cuBLAS library.

The project is heavily inspired by the article [CUDA Matrix Multiply](https://siboehm.com/articles/22/CUDA-MMM) and serves as a deep-dive into a single algorithm, one of the most important for deep learning.

Each kernel is profiled on **NVIDIA GeForce RTX 3070**.

## Building the Project

This repository uses CMake to build the main runner executable.

### 1. Prerequisites

- A C++ compiler (g++, clang++, etc.)
- The NVIDIA CUDA Toolkit (nvcc)
- CMake (version 3.18 or higher)

### 2. Build Steps

From the root of the repository, run the following commands:

```bash
mkdir build
cd build
cmake ..
make -j
```

This will compile the project. The executable `sgemm_runner` will be located in the `build/` directory.

### 3. Running a Kernel

You can run a specific kernel by passing its ID as a command-line argument:

```bash
# Run the naive kernel (ID 1)
./build/sgemm_runner --kernel 1

# Run with a different size and number of repeats
./build/sgemm_runner --kernel 1 --size 4096 --repeats 50
```

## License

This project is licensed under the MIT License.
