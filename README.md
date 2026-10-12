# CUDA GEMM From Scratch

This repository contains several implementations of a general matrix multiplication (GEMM) kernel in CUDA, supporting both single-precision (FP32) and mixed-precision (FP16 inputs, FP32 accumulation) operations.
It starts with a slow naive kernel and applies several optimizations to approach and surpass (at least on my GPU) the performance of NVIDIA's cuBLAS library.

The project is heavily inspired by the following blog posts:

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog
  ](https://siboehm.com/articles/22/CUDA-MMM) by [Simon Boehm](https://siboehm.com/)
- [How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html) by [Alex Armbruster](https://alexarmbr.github.io/)

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
# Run the warptiling FP32 kernel (ID 6)
./build/gemm_runner --kernel 6 --precision fp32 --repeats 1000

# Run the FP16 cuBLAS kernel (ID 0)
./build/gemm_runner --kernel 0 --precision fp16 --repeats 1000

# Run the naive WMMA FP16 kernel (ID 7)
./build/gemm_runner --kernel 7 --precision fp16 --repeats 1000
```

## FP32 Performance Overview

Performance for a 2048x2048 FP32 matrix multiplication on an NVIDIA GeForce RTX 3070.

| ID  | Kernel           |    GFLOPS | Performance vs. cuBLAS |
| --- | :--------------- | --------: | :--------------------- |
| 0   | **cuBLAS**       | `~11,613` | 100.0%                 |
| 1   | **Naive**        |  `~1,343` | 11.6%                  |
| 2   | **Tiled**        |  `~1,805` | 15.6%                  |
| 3   | **1D Coarsened** |  `~5,865` | 50.5%                  |
| 4   | **2D Coarsened** | `~10,836` | 93.3%                  |
| 5   | **Transposed**   | `~11,711` | 100.8%                 |
| 6   | **Warptiling**   | `~12,621` | 108.7%                 |

## FP16 Performance Overview

Performance for a 2048x2048 FP16 matrix multiplication on an NVIDIA GeForce RTX 3070.

| ID  | Kernel               |    GFLOPS | Performance vs. cuBLAS |
| --- | :------------------- | --------: | :--------------------- |
| 0   | **cuBLAS**           | `~39,071` | 100.0%                 |
| 7   | **Naive WMMA**       |  `~9,941` | 25.4%                  |
| 8   | **Naive MMA**        |  `~9,450` | 24.2%                  |
| 9   | **Hierarchical MMA** | `~18,153` | 46.5%                  |
| 10  | **Vectorized MMA**   | `~28,102` | 71.9%                  |
| 11  | **Memory swizzling** | `~34,133` | 87.4%                  |
| 12  | **Buffered GMEM**    | `~35,902` | 91.9%                  |
| 13  | **Unrolled SMEM**    | `~37,511` | 96.0%                  |

## Kernel Explanations

### 0: [cuBLAS](./src/kernels/00_cublas.cuh)

NVIDIA’s optimized matrix multiplication library, used as the performance reference.

### 1: [Naive](./src/kernels/01_naive.cuh)

Each thread computes one output element by multiplying a row of A by a column of B. Threads working on neighboring output columns read the same A value and consecutive B values, allowing their memory accesses to be combined.

For a 2048x2048 multiplication, the kernel executes about 537 million global-load instructions across all warps. These loads request about 34.38 GB of data, but caching reduces the amount read from VRAM to about 2.24 GB.

Caching helps, but the GPU still has to execute all those load instructions. Nsight Compute shows that the largest source of stalls is `LG Throttle`: warps are waiting for room in the queue for global-memory instructions. They also spend time waiting for loaded values to arrive (`Long Scoreboard`).

The kernel takes 15.87 ms in this profiling run.

![Warp stalls](./profiling/01_warp_stalls.png)

### 2: [Tiled](./src/kernels/02_tiled.cuh)

Each block computes a 16x16 output tile. Threads cooperate to load 16x16 tiles from A and B into shared memory, then reuse those values for their calculations. They repeat this along the K dimension until the output tile is complete. Each thread still computes one output element.

This reduces global-load instructions from about 537 million to 34 million, almost eliminating `LG Throttle` stalls. Requested global-load data drops from 34.38 GB to 4.27 GB.

However, the amount read from VRAM stays almost unchanged at about 2.25 GB. The naive kernel's caches were already serving most of the repeated requests. Tiling removes many of those requests, but adds shared-memory loads, stores, and synchronization between threads.

The largest source of stalls is now MIO Throttle, associated here with issuing shared-memory instructions. Threads also wait for each other at the block barriers.

The compiler also makes an important optimization: it loads four A values at once using `LDS.128` instructions. This reduces shared-load instructions from 32 to 20 per warp per tile.

With this vectorization, the tiled kernel takes 12.01 ms, down from 15.87 ms. Replacing those vectorized shared loads with scalar loads using CuAssembler increases the runtime to 17.78 ms—slower than the naive kernel.

Vectorizing the naive kernel directly gives 10.06 ms, beating this tiled version too. In this configuration, shared-memory tiling needs more work to pay off. The following kernels make better use of it by computing more outputs per thread and reusing values in registers.

![Warp stalls](./profiling/02_warp_stalls.png)

### 3: [1D Coarsened](./src/kernels/03_1D_coarsened.cuh)

Each thread now computes a vertical slice of `TM` output elements, with `TM = 16`. For each step along K, it loads 16 values from the shared A tile and one value from the shared B tile. That B value is kept in a register and reused for all 16 multiplications.

This reduces the shared-memory instructions needed for the same amount of arithmetic. Shared-load instructions drop from about 335.5 million to 83.9 million. There are also fewer global-memory instructions, and the amount read from VRAM decreases from about 2.26 GB to 0.41 GB.

The kernel still performs the same number of multiply-adds, but total executed instructions decrease from about 1.03 billion to 425 million. Warps spend less time waiting to issue shared-memory instructions and waiting at barriers.

Computing more outputs requires more registers: usage increases from 38 to 48 registers per thread, and achieved occupancy decreases from about 99% to 78%. The reduction in memory and instruction overhead more than makes up for this. Runtime decreases from 12.01 ms to 3.57 ms, making this kernel about 3.36x faster.

![Roofline model](./profiling/03_roofline.png)

### 4: [2D Coarsened](./src/kernels/04_2D_coarsened.cuh)

Each thread now computes an `8 x 8` output minitile. For each step along K, it loads eight A values and eight B values into registers, then multiplies every A value by every B value. Each loaded value is reused eight times.

These 16 input values support 64 multiply-adds. The previous kernel needed 17 input values for 16 multiply-adds. Computing a minitile therefore reduces the shared-memory loading needed for the same amount of arithmetic.

Shared-load instructions decrease by about 79%, and total executed instructions decrease from about 425 million to 320 million. Register usage increases from 48 to 128 per thread, reducing achieved occupancy from about 78% to 31%. Despite fewer active warps, runtime decreases from 3.57 ms to 1.86 ms, making this kernel about 1.92x faster.

There is still room to improve the memory accesses. Shared loads have bank conflicts, requiring extra work to serve them. In the final C loads and stores, neighboring threads access values eight floats apart. Each scalar instruction therefore uses only 4 bytes of each requested 32-byte sector, and subsequent instructions revisit those sectors.

The next kernel changes how A is stored in shared memory and explicitly groups memory accesses into vector instructions.

![Loads and stores](./profiling/04_loads_stores.png)

### 5: [Transposed](./src/kernels/05_transposed.cuh)

This kernel transposes A when storing it in shared memory. The A values a thread needs for one step along K are now consecutive, allowing four of them to be loaded together. The compiler already vectorized some A loads in the previous kernel, but those vectors contained values from different K steps. Transposition changes which values are grouped together.

The kernel also explicitly loads and stores four floats at once when accessing global memory. Global-load instructions decrease from about 4.33 million to 1.08 million. Each C load or store now uses 16 bytes of a requested 32-byte sector instead of four, reducing repeated sector requests. The amount of A/B data requested stays essentially unchanged.

Testing the changes separately gives an interesting result:

| Version | Duration |
| --- | ---: |
| Vectorization only | 1.88 ms |
| Transposition only | 2.21 ms |
| Vectorization and transposition | 1.73 ms |

All three variants use `BK = 16`. The two changes work better together because vectorized global loading also changes which A elements each thread stores into shared memory. This reduces the transposed stores' bank conflicts from 16-way to 4-way. The stores themselves remain scalar because the four values go to different rows of the transposed tile.

Adding transposition to the vectorized version reduces shared loads from 72 to 64 instructions per warp per K-tile. Shared stores become more expensive, and total instruction count slightly increases. However, the fraction of cycles with no eligible warp decreases from 37.25% to 31.68%, allowing instructions to be issued more frequently. In this configuration, the improvement outweighs the extra storing cost.

Compared to kernel 4, runtime decreases from 1.86 ms to 1.73 ms, about 7%. This comparison also includes reducing `BK` from 32 to 16. Shared-load and shared-store bank conflicts remain, leaving room to improve how work is distributed across threads.

![Bank conflicts](./profiling/05_bank_conflicts.png)

### 6: [Warptiling](./src/kernels/06_warptiling.cuh)

Each block still computes a `128 x 128` output tile, but now uses four warps instead of eight. Each warp computes a `64 x 64` warptile, divided into four `64 x 16` subtiles. Each thread handles an `8 x 4` piece of every subtile, accumulating 128 outputs instead of 64. All four pieces are computed during the same traversal of K.

For each K-step, a thread loads eight A values and 16 B values into registers. Each A value is now reused for 16 multiply-adds instead of eight. This increases the arithmetic performed per shared load.

The thread mapping also changes. Four neighboring lanes access four consecutive B values each, covering 16 consecutive values together. This removes the shared-load bank conflicts reported for the previous kernel. Combined with the greater register reuse, shared-load instructions decrease from about 16.8 million to 12.6 million, while shared-load wavefronts decrease from 83.9 million to 41.9 million.

The same mapping improves the final C loads and stores. Neighboring lanes now fill each requested 32-byte sector, compared to using only half of it in kernel 5. Requested C load bytes decrease from about 33.6 MB to 16.8 MB, without changing the amount of useful output data.

This configuration also reduces `BK` from 16 to 8. The changed cooperative loading pattern reduces transposed A store conflicts from 4-way to 2-way, and shared-store wavefronts decrease from about 10.5 million to 6.3 million.

Keeping more outputs in registers increases usage from 107 to 167 registers per thread, reducing achieved occupancy from about 31% to 23%. Despite fewer active warps, more warps are eligible to issue instructions on average. Runtime decreases from 1.73 ms to 1.56 ms, about 10%.

![Shared-memory accesses](./profiling/06_no_bank_conflicts.png)

The following roofline plot compares cuBLAS and the six custom kernels for this problem size and GPU:

![Roofline model](./profiling/combined_roofline.png)

### 7: [Naive WMMA](./src/kernels/07_naive_wmma.cuh)

This is the first custom kernel to use tensor cores. A and B now contain FP16 values, while accumulation and the output remain FP32. The kernel uses CUDA's WMMA (warp matrix multiply accumulate) API.

Each block contains one warp, which computes a `16 x 16` output tile. In each iteration, the warp loads a `16 x 16` tile from A and B directly from global memory, then calls `wmma::mma_sync` to multiply them and accumulate the result. It advances 16 elements along K and repeats until the output tile is complete.

All 32 threads cooperate on these matrix operations. The values are held in fragments distributed across the threads' registers, with WMMA handling their layout. After accumulation, the threads apply alpha and beta and store the output tile.

There is no shared-memory staging or larger output tile to increase reuse. Despite this simple implementation, the kernel achieves almost 10 TFLOPS in the benchmark. Comparisons with the earlier kernels now also include the change from FP32 to FP16 inputs.

Nsight Compute shows that warps spend most of their waiting time on global-memory loads (`Long Scoreboard`). Each scheduler has no warp ready to issue an instruction in about 89% of its cycles.

![Warp stalls](./profiling/07_warp_stalls.png)

### 8: [Naive MMA](./src/kernels/08_naive_mma.cuh)

This kernel uses explicit PTX instructions for loading matrix fragments and performing tensor-core calculations. The `mma.sync.aligned.m16n8k8` instruction multiplies a `16 x 8` A fragment by an `8 x 8` B fragment and accumulates into a `16 x 8` output fragment. The operation is performed collectively by the warp.

Each block still contains one warp, but now computes a larger `64 x 32` output tile. Threads first cooperate to load a `64 x 8` A tile and an `8 x 32` B tile into shared memory. These supply 16 MMA operations for each step of eight elements along K.

The `ldmatrix` instructions load values from shared memory into the register layout required by MMA. Each thread holds four accumulator values for each output fragment, giving 64 accumulated output values per thread.

The larger output tile and shared-memory reuse reduce requested global-load data from about 4.31 GB to 1.36 GB compared to kernel 7. However, total executed instructions increase from about 62.8 million to 249.3 million. Reducing global-memory requests comes with substantial additional instruction overhead in this implementation.

The current loop reloads both input fragments for each MMA operation. An A fragment is loaded again for each output fragment along N, and a B fragment is loaded again for each output fragment along M. NCU confirms 16.8 million shared-memory matrix-load instructions, matching two loads per MMA operation.

These loads also have bank conflicts. They require about 50.3 million shared-memory wavefronts, of which 25.2 million are additional wavefronts caused by bank conflicts.

The next kernel keeps input fragments in registers and reuses them across multiple MMA operations, reducing the number of shared loads.

![Shared-memory matrix loads](./profiling/08_shared_loads.png)

### 9: [Hierarchical MMA](./src/kernels/09_hierarchical_mma.cuh)

Each block now contains four warps and computes a `128 x 128` output tile. Each warp handles a `128 x 32` part of that tile. Threads cooperate to load the input tiles into shared memory, where the same A tile serves all four warps.

Within each warp, A and B fragments are loaded into registers for reuse across the MMA operations. For each step of eight elements along K, the warp loads eight A fragments and four B fragments, then computes all 32 pairs. Each A fragment is reused for four MMA operations, and each B fragment is reused for eight.

This requires eight A `ldmatrix` instructions and four B `ldmatrix` instructions for 32 MMA operations. The previous kernel used two `ldmatrix` instructions per MMA operation, which would require 64 loads for the same amount of arithmetic.

NCU confirms that shared-memory matrix-load instructions decrease from about 16.8 million to 3.15 million, an 81% reduction. Shared-load wavefronts decrease from about 50.3 million to 12.6 million. Bank conflicts remain in the B loads, but reusing the fragments means these accesses happen much less often.

The larger blocktile also reduces repeated global loads. Requested global-load data decreases from about 1.36 GB to 0.44 GB. Together with the reduced shared loading and other instruction overhead, total executed instructions decrease from about 249.3 million to 92.9 million.

The larger warptile doubles the accumulated outputs per thread from 64 to 128. Register usage increases from 125 to 165 registers per thread, and achieved occupancy decreases from about 30% to 23%. Blocktile and warptile dimensions can be tuned separately to balance input reuse against register usage.

With the chosen configuration, this kernel runs roughly twice as fast as the previous version in the benchmark. Sharing input tiles across warps and reusing fragments in registers substantially reduce the work needed to supply the same matrix calculations.

![Shared-memory matrix loads](./profiling/09_shared_loads.png)

### 10: [Vectorized MMA](./src/kernels/10_vectorized_mma.cuh)

This kernel keeps the same tile sizes and MMA calculations, but groups memory accesses into wider instructions. Each thread copies eight FP16 values at once when loading A and B into shared memory, using a 128-bit global load followed by a 128-bit shared store. The code uses `float4` to copy these 16 bytes without changing the values' FP16 representation.

For the A and B tiles, global-load instructions decrease from about 4.19 million to 0.524 million. Shared-store instructions decrease by the same amount. Each vector instruction replaces eight scalar accesses.

The epilogue also groups neighboring output values together. Each thread loads and stores two FP32 values at once using `float2`, halving the C load and store instruction counts. Neighboring threads now fill each requested 32-byte sector, reducing requested C load bytes from 33.6 MB to 16.8 MB. Requested C store bytes decrease by the same amount.

Including the epilogue, total global-load instructions decrease from about 4.33 million to 0.590 million. Requested global-load data changes only slightly, from about 0.435 GB to 0.419 GB. The main benefit is executing fewer instructions to move the data.

Vectorization also reduces the addressing and loop overhead involved in loading the input tiles. Total executed instructions decrease from about 92.9 million to 27.3 million, a 71% reduction. Register usage changes only slightly, from 165 to 167 registers per thread, and achieved occupancy remains about 23%.

This gives a substantial speedup in the benchmark. Shared-memory fragment loading remains unchanged: the kernel still executes about 3.15 million matrix-load instructions requiring 12.6 million wavefronts. The B loads still have bank conflicts, which the next kernel addresses.

![Vectorized memory accesses](./profiling/10_vectorized_memory.png)

### 11: [Memory swizzling](./src/kernels/11_memory_swizzling.cuh)

This kernel changes where values are stored in shared memory so that matrix-fragment loads spread their accesses across the memory banks.

Consider an `8 x 8` FP16 fragment from B. Each row contains 16 bytes, covering four banks. In the ordinary row-major layout, the distance between rows makes all eight rows use the same four banks. Loading the fragment therefore requires accessing those banks eight times.

Swizzling rearranges groups of eight FP16 values within each row, using an XOR with bits from the row index. The eight rows of a fragment then use eight different groups of four banks, covering all 32 banks. Each group remains contiguous and aligned, preserving the vectorized copies into shared memory.

The same mapping is applied when storing the input tiles and when loading fragments, so the MMA instructions receive the same logical matrix values. Both A and B use this layout.

The tile sizes are also adjusted. Each block still computes a `128 x 128` output tile using four warps, but each warp now handles `64 x 64` instead of `128 x 32`. `BK` increases from 8 to 64, and `WK` increases from 8 to 16, loading fragments for two MMA steps along K into registers at a time.

With this configuration, shared-memory matrix-load instructions remain at about 3.15 million, while shared-load wavefronts decrease from about 12.6 million to 4.22 million. Reported shared-load bank conflicts decrease from about 7.34 million to 29,000, a reduction of 99.6%.

Increasing `BK` reduces the number of tile iterations and block-wide synchronizations eightfold. It also lets neighboring threads load longer contiguous sections of A rows, filling the requested sectors. Requested global-load data decreases from about 0.419 GB to 0.285 GB.

These changes have costs too. Shared-memory usage increases from 4 KiB to 32 KiB per block, and total executed instructions increase from about 27.3 million to 32.7 million. Register usage stays at 167 per thread. The kernel also switches to explicit shared-memory offsets for `ldmatrix`. The overall performance change therefore includes swizzling, tile tuning, and changes to address calculation.

![Swizzled shared-memory accesses](./profiling/11_swizzled_memory.png)

### 12: [Buffered GMEM](./src/kernels/12_buffered_gmem.cuh)

This kernel starts loading the next input tiles before finishing the calculations on the current tiles. Temporary register arrays hold the incoming A and B values, giving the GPU independent calculations to execute while those global-memory loads are in progress.

The first input tiles are loaded into shared memory before the main loop. Each loop iteration then issues global loads for the next tiles and computes using the current tiles in shared memory. After all threads finish using the current tiles, they copy the buffered values from registers into shared memory. Another synchronization makes those values available for the next iteration. The final tiles are computed separately, without loading another set.

There is still only one shared-memory buffer for each input matrix. The overlap comes from keeping the current tiles in shared memory and the next tiles in registers. The global loads' results are only needed when those registers are copied into shared memory after the computation.

The configuration also changes: `BK` decreases from 64 to 32, while `WK` increases from 16 to 32. Each warp now loads the fragments for all four MMA steps along the current tile's K dimension before using them. Shared-memory usage halves from 32 KiB to 16 KiB per block, while the number of tile iterations doubles.

The global-load instruction count remains about 0.590 million, and the amount read from VRAM stays around 72 MB. NCU shows that `Long Scoreboard` stalls decrease from about 2.2 to 0.7 cycles per issued warp instruction. This is consistent with hiding more of the global-load latency, though the comparison also includes the tile-size changes.

Keeping the next tiles and more matrix fragments in registers increases register usage from 167 to 217 per thread. This reduces the maximum number of resident blocks per SM from three to two, and achieved occupancy decreases from about 21.6% to 15.6%.

Total executed instructions decrease from about 32.7 million to 27.9 million. The resulting performance reflects both the earlier global loads and the changed tile configuration, with additional register usage as the main tradeoff.

![Warp stalls with buffered global loads](./profiling/12_warp_stalls.png)

### 13: [Unrolled SMEM](./src/kernels/13_unrolled_smem.cuh)

This kernel keeps the previous tile sizes, swizzled shared-memory layout, and global-memory buffering. It replaces the shared-memory fragment-loading loops with manually written sequences in `load_a_from_smem` and `load_b_from_smem`.

The new sequences use `ldmatrix.x4`, which loads four `8 x 8` FP16 matrices in one warp instruction. For A, this supplies two `16 x 8` MMA fragments, replacing two `.x2` instructions. For B, one `.x4.trans` supplies four `8 x 8` fragments, replacing four `.x1.trans` instructions.

For each warp and each tile of 32 elements along K, the previous kernel needed 16 A loads and 32 B loads. This kernel needs eight of each, reducing the total from 48 matrix-load instructions to 16. The values loaded and their reuse in the MMA calculations stay the same.

The functions also calculate an initial swizzled address and update it using fixed offsets and XOR operations to reach the remaining fragments. This simplifies the repeated address calculations around the loads.

NCU confirms that shared-memory matrix-load instructions decrease from about 3.15 million to 1.05 million, a 67% reduction. Shared-load wavefronts remain about 4.19 million, with no reported bank conflicts. Each instruction handles more data, so fewer instructions still require essentially the same amount of shared-memory work.

Together with the changes to address calculation and control instructions, total executed instructions decrease from about 27.9 million to 14.0 million, a 50% reduction. Register usage changes from 217 to 218 per thread, achieved occupancy remains about 15.6%, and shared-memory usage stays at 16 KiB per block.

The manually written load sequences are specialized for this tile configuration. Changing the fragment counts would require updating those sequences.

![Wider shared-memory matrix loads](./profiling/13_shared_loads.png)

## License

This project is licensed under the MIT License.
