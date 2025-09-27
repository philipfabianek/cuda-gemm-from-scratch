# CUDA GEMM From Scratch

This repository contains several implementations of a general matrix multiplication (GEMM) kernel in CUDA, supporting both single-precision (FP32) and mixed-precision (FP16 inputs, FP32 accumulation) operations.
It starts with a slow naive kernel and applies several optimizations to approach and surpass (at least on my GPU) the performance of NVIDIA's cuBLAS library.

The project is heavily inspired by [this blog post](https://siboehm.com/articles/22/CUDA-MMM) written by [Simon Boehm](https://siboehm.com/). Compared to the blog post, I focused more on [analysis-driven optimizations](https://developer.nvidia.com/blog/analysis-driven-optimization-preparing-for-analysis-with-nvidia-nsight-compute-part-1/) and profiling with Nsight Compute.

## Hardware Requirements

This project supports mixed precision to leverage tensor cores.
This requires a compatible GPU with compute capability 8.0 or higher.
The code will check your GPU at runtime and produce an error if you attempt to use `--precision fp16` on incompatible hardware.
All fp32 kernels should run on older architectures.

## Building the Project

This repository uses CMake to build the main runner executable.

### 1. Prerequisites

- A C++ compiler (g++, clang++, etc.)
- The NVIDIA CUDA Toolkit (nvcc)
- CMake (version 3.18 or higher)

### 2. Build Steps

Before building, adjust `CMAKE_CUDA_ARCHITECTURES` in [`CMakeLists.txt`](./CMakeLists.txt) to fit your GPU. Then, from the root of the repository, run the following commands:

```bash
mkdir build
cd build
cmake ..
make -j
```

This will compile the project. The executable `gemm_runner` will be located in the `build/` directory.

### 3. Running a Kernel

You can run a specific kernel by passing its ID as a command-line argument:

```bash
# Run the warptiling FP32 kernel (ID 1)
./build/gemm_runner --kernel 6 --precision fp32 --repeats 1000

# Run the FP16 cuBLAS kernel (ID 0)
./build/gemm_runner --kernel 0 --precision fp16 --repeats 1000

# Run the naive WMMA FP16 kernel (ID 7)
./build/gemm_runner --kernel 7 --precision fp16 --repeats 1000
```

## FP32 Performance Overview

Performance for a 2048x2048 FP32 matrix multiplication on an NVIDIA GeForce RTX 3070.

| ID  | Kernel           |      GFLOPS | Performance vs. cuBLAS |
| --- | :--------------- | ----------: | :--------------------- |
| 0   | **cuBLAS**       | `~11,613.7` | 100.0%                 |
| 1   | **Naive**        |  `~1,343.8` | 11.6%                  |
| 2   | **Tiled**        |  `~1,805.8` | 15.5%                  |
| 3   | **1D Coarsened** |  `~5,865.9` | 50.5%                  |
| 4   | **2D Coarsened** | `~10,836.8` | 93.3%                  |
| 5   | **Transposed**   | `~11,711.2` | 100.8%                 |
| 6   | **Warptiling**   | `~12,621.1` | 108.7%                 |

## FP16 Performance Overview

Performance for a 2048x2048 FP16 matrix multiplication on an NVIDIA GeForce RTX 3070.

| ID  | Kernel         |      GFLOPS | Performance vs. cuBLAS |
| --- | :------------- | ----------: | :--------------------- |
| 0   | **cuBLAS**     | `~39,071.4` | 100.0%                 |
| 7   | **Naive WMMA** |  `~9,941.5` | 25.4%                  |

## Kernel Explanations

### 0: [cuBLAS](./src/kernels/00_cublas.cuh)

Reference implementation created by NVIDIA. Highly optimized, state-of-the-art, extremely performant.

### 1: [Naive](./src/kernels/01_naive.cuh)

Simple matrix multiplication kernel with coalesced memory access where each thread computes one output element. Threads in a warp read the same elements from the first matrix and consecutive elements from the second matrix (`threadIdx.x` is mapped to the columns of the second matrix). This results in memory coalescing.

The kernel suffers from high global memory usage. By profiling in Nsight Compute (NCU), we can see that warps stall mostly due to global memory loads.

![Warp stalls](./profiling/01_warp_stalls.png)

### 2: [Tiled](./src/kernels/02_tiled.cuh)

In this kernel, we reduce global memory usage by utilizing shared memory. Each block in this kernel computes an output tile by loading tiles from the input matrices into shared memory and then performing the calculations using the loaded tiles. Global memory access is still coalesced for threads in a warp and each thread computes one output element.

Compared to the previous kernel, performance increases by ~33%. Using tiled algorithms can traditionally lead to even larger performance gains, however, the previous kernel had a high L1 cache hit rate of close to 90%. More performance gains can be found by increasing arithmetic intensity since the kernel is memory bound.

![Roofline model](./profiling/02_roofline.png)

### 3: [1D Coarsened](./src/kernels/03_1D_coarsened.cuh)

This kernel increases the work per thread. Each thread is now responsible for computing a `TM x 1` vertical slice of the output tile. This improves arithmetic intensity. The kernel also uses a 1D thread block and uses several mappings to map the threads to various 2D tiles, e.g. to load values into the shared memory. The implementation details, parameter constraints, and indexing logic can be found in the source file.

Compared to the previous kernel, performance increases by ~225%. While this kernel is now compute bound in NCU, its arithmetic intensity is only slightly past the ridge point, so increasing it further can still yield significant performance gains.

![Roofline model](./profiling/03_roofline.png)

### 4: [2D Coarsened](./src/kernels/04_2D_coarsened.cuh)

In this kernel, each thread computes a "minitile" of size `TM x TN`, which increases the work per thread and arithmetic intensity. Many more details can be found in the source file.

Compared to the previous kernel, performance increases by ~85%. However, this optimization introduced a few new problems. Each thread now loads a `TM x 1` vertical slice from the A tile and this access isn't ideal. Furthermore, since each thread now stores a full `TM x TN` minitile, the global memory writes are no longer coalesced.

![Loads and stores](./profiling/04_loads_stores.png)

### 5: [Transposed](./src/kernels/05_transposed.cuh)

This kernel uses a transposed A tile to make data loading from shared memory into registers more efficient. It also vectorizes all global memory reads and writes.

Compared to the previous kernel, performance increases by ~8%. Using transposed A tile results in vectorized shared memory reads but it initially introduced a new problem: the shared memory writes weren't coalesced and there were a lot more bank conflicts. This was resolved resolved by making threads load and write chunks of 4 elements. Finally, the global memory access vectorization alleviated the inefficient write problem.

Since bank conflicts still remain, we could try to improve performance by reducing the access inefficiencies even more, but there is a more traditional optimization to implement.

![Bank conflicts](./profiling/05_bank_conflicts.png)

### 6: [Warptiling](./src/kernels/06_warptiling.cuh)

In this kernel, one warp computes one "warptile" in several iterations. A lot more details can be found in the source file.

Compared to the previous kernel, performance increases by another ~8%, which is quite a lot given the previous kernel already outperformed cuBLAS on my GPU. Intuitively, using warptiling adds another level of parallelism. After careful tuning, this gets rid of all shared memory load bank conflicts by ensuring that all threads within a single warp access different memory banks.

![No bank conflicts](./profiling/06_no_bank_conflicts.png)

This image displays the roofline model with the cuBLAS kernel and all 6 custom kernels:

![Roofline model](./profiling/combined_roofline.png)

### 7: [Naive WMMA](./src/kernels/07_naive_wmma.cuh)

This is the first custom kernel which uses tensor cores and mixed precision (FP16 + FP32). It is programmed using the WMMA (warp matrix multiply accumulate) API. In this kernel, blocks consist of 1 warp and each warp computes a `16 x 16` tile of the output matrix. Despite the simple implenetation, it achieves almost 10 TFLOPS, a significant speedup over the naive FP32 kernel and approaches both the FP32 cuBLAS and warptiling kernels.

## License

This project is licensed under the MIT License.
