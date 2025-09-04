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

Before building, adjust `CUDA_COMPUTE_CAPABILITY` in CMakeLists to fit your GPU. Then, from the root of the repository, run the following commands:

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

| ID  | Kernel           |      GFLOPS | Performance vs. cuBLAS |
| --- | :--------------- | ----------: | :--------------------- |
| 0   | **cuBLAS**       | `~11,613.7` | 100.0%                 |
| 1   | **Naive**        |  `~1,343.8` | 11.6%                  |
| 2   | **Tiled**        |  `~1,805.8` | 15.5%                  |
| 3   | **1D Coarsened** |  `~5,865.9` | 50.5%                  |
| 4   | **2D Coarsened** | `~10,836.8` | 93.3%                  |
| 5   | **Transposed**   | `~11,711.2` | 100.8%                 |
| 6   | **Warptiling**   | `~12,621.1` | 108.7%                 |

## Kernel Explanations

### 0: [cuBLAS](./src/kernels/00_cublas.cuh)

Reference implementation created by NVIDIA. Highly optimized, state-of-the-art, extremely performant.

### 1: [Naive](./src/kernels/01_naive.cuh)

Simple matrix multiplication kernel with coalesced memory access where each thread computes one output element. Threads in a warp read the same elements from the first matrix and consecutive elements from the second matrix (`threadIdx.x` is mapped to the columns of the second matrix). This results in memory coalescing. The main bottleneck is global memory usage.

### 2: [Tiled](./src/kernels/02_tiled.cuh)

Each block in this kernel computes an output tile by loading tiles from the input matrices into shared memory and then performing the calculations using the loaded tiles. This reduces global memory usage. Global memory access is still coalesced for threads in a warp and each thread computes one output element.

### 3: [1D Coarsened](./src/kernels/03_1D_coarsened.cuh)

This kernel increases the work per thread. Each thread is now responsible for computing a `TM x 1` vertical slice of the output tile. This improves arithmetic intensity. The kernel also uses a 1D thread block and uses several mappings to map the threads to various 2D tiles, e.g. to load values into the shared memory. The implementation details, parameter constraints, and indexing logic can be found in the source file.

### 4: [2D Coarsened](./src/kernels/04_2D_coarsened.cuh)

In this kernel, each thread computes a "minitile" of size `TM x TN`, which increases the work per thread and arithmetic intensity. Many more details can be found in the source file.

### 5: [Transposed](./src/kernels/05_transposed.cuh)

This kernel uses a transposed A tile to eliminate bank conflicts when loading values from shared memory into registers. Initially, this introduced bank conflicts when writing into the shared memory. These were resolved by making threads load and write chunks of 4 elements. Finally, explicit vectorization of all global memory reads and writes improved performance even more.

### 6: [Warptiling](./src/kernels/06_warptiling.cuh)

In this kernel, one warp computes one "warptile" in several iterations, which adds another level of parallelism. After adjusting the parameters, this change further improves the performance. A lot more details can be found in the source file.

## License

This project is licensed under the MIT License.
