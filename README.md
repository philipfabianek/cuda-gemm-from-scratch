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
./build/sgemm_runner --kernel 1 --size 2048 --repeats 50
```

## Performance Overview

Performance for a 2048x2048 matrix multiplication on an NVIDIA GeForce RTX 3070.

| ID  | Kernel     |      GFLOPS | Performance vs. cuBLAS |
| --- | :--------- | ----------: | :--------------------- |
| 0   | **cuBLAS** | `~11,613.7` | 100.00%                |
| 1   | **Naive**  |  `~1,343.8` | 11.6%                  |
| 2   | **Tiled**  |  `~1,805.8` | 15.5%                  |

## Kernel Explanations

### 0: [cuBLAS](./src/kernels/00_cublas.cuh)

Reference implementation created by NVIDIA. Highly optimized, state-of-the-art, extremely performant.

### 1: [Naive](./src/kernels/01_naive.cuh)

Simple matrix multiplication kernel with coalesced memory access. Threads in a warp read the same elements from the first matrix and consecutive elements from the second matrix (`threadIdx.x` is mapped to the columns of the second matrix). This results in memory coalescing. The main bottleneck is global memory usage.

### 2: [Tiled](./src/kernels/02_tiled.cuh)

Each block in this kernel computes an output tile by loading tiles from the input matrices into shared memory and then performing the calculations using the loaded tiles. This reduces global memory usage. Global memory access is still coalesced for threads in a warp.

## License

This project is licensed under the MIT License.
