# Lab 3: Shared Memory Matrix Addition

This lab demonstrates the performance benefits of using shared memory in CUDA by comparing two implementations of matrix addition: one using only global memory and another using shared memory tiles.

## Objectives

- Understand the basic concept of shared memory in CUDA
- Learn how to use shared memory to improve kernel performance
- Observe the performance difference between global and shared memory access
- Practice using `__syncthreads()` for thread synchronization

## Code Overview

The code implements matrix addition in two different ways:

1. **Global Memory Version** (`matrixAddGlobal`):
   - Directly reads from and writes to global memory
   - Simple implementation with no optimizations

2. **Shared Memory Version** (`matrixAddShared`):
   - Loads data from global memory into shared memory tiles
   - Performs computation using the shared memory
   - Writes results back to global memory
   - Uses `__syncthreads()` to ensure proper synchronization

## Shared Memory Concept

Shared memory is:
- Much faster than global memory (typically 5-100x)
- Limited in size (typically 48KB-96KB per block, depending on GPU architecture)
- Shared among all threads in a thread block
- Not visible to threads in other blocks
- Not persistent beyond the lifetime of a block

## Key Code Sections

### 1. Shared Memory Declaration

```c
__shared__ float tileA[TILE_HEIGHT][TILE_WIDTH];
__shared__ float tileB[TILE_HEIGHT][TILE_WIDTH];
```

This declares two shared memory arrays that will be used as tiles to store portions of the input matrices.

### 2. Loading Data to Shared Memory

```c
if (row < height && col < width)
{
    tileA[threadIdx.y][threadIdx.x] = d_a[idx];
    tileB[threadIdx.y][threadIdx.x] = d_b[idx];
}
else
{
    // Initialize out-of-bounds elements to zero
    tileA[threadIdx.y][threadIdx.x] = 0.0f;
    tileB[threadIdx.y][threadIdx.x] = 0.0f;
}
```

Each thread loads one element from global memory into shared memory.

### 3. Thread Synchronization

```c
__syncthreads();
```

This barrier ensures all threads in the block have completed loading data into shared memory before any thread proceeds to the computation step.

### 4. Computation Using Shared Memory

```c
if (row < height && col < width)
{
    d_c[idx] = tileA[threadIdx.y][threadIdx.x] + tileB[threadIdx.y][threadIdx.x];
}
```

Threads perform the addition using data from shared memory and write the result to global memory.

## Expected Performance Gains

For this simple matrix addition example, the performance improvement of shared memory over global memory might be modest because:

1. Matrix addition is a memory-bound operation with a low arithmetic intensity
2. The shared memory version still requires the same number of global memory accesses
3. Each element is only accessed once

However, the shared memory version should still be faster due to:
1. Improved memory access patterns
2. Reduced global memory latency for neighboring threads
3. Potentially better memory coalescing

More complex operations that reuse data (like matrix multiplication) would show much more significant gains.

## How to Compile and Run

### Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C compiler compatible with CUDA
- NVIDIA HPC SDK (for Fortran version)

### Compilation

#### C Version
```bash
nvcc -o matrix_add_shared matrix_add_shared.cu
```

#### Fortran Version
```bash
nvfortran -o matrix_add_shared_fortran matrix_add_shared.cuf
```

### Running the Program

#### C Version
```bash
./matrix_add_shared
```

#### Fortran Version
```bash
./matrix_add_shared_fortran
```

## Expected Output

The program will display:
1. Matrix and thread configuration information
2. Execution time for both the global memory and shared memory versions
3. Verification that the results match
4. The speedup achieved by using shared memory

## Experiment Ideas

1. **Vary the Tile Size**: Change `TILE_WIDTH` and `TILE_HEIGHT` to see how different tile sizes affect performance
2. **Modify Matrix Size**: Try different matrix dimensions to see how the performance scales
3. **Memory Access Patterns**: Modify the code to experiment with different memory access patterns
4. **Measure Memory Bandwidth**: Calculate and display the effective memory bandwidth for each approach

## Discussion Questions

1. Why is shared memory faster than global memory?
2. In what types of algorithms would shared memory provide the greatest benefit?
3. What happens if we remove the `__syncthreads()` call? Why?
4. How does the tile size affect performance? Why is there an optimal tile size?
5. What are the limitations of shared memory?

## Common Issues

- **Shared Memory Bank Conflicts**: These can occur when multiple threads access the same memory bank
- **Synchronization Barriers**: Missing or incorrectly placed `__syncthreads()` can cause race conditions
- **Shared Memory Size Limits**: Requesting too much shared memory per block can reduce occupancy

## Going Further

For more advanced applications of shared memory:
1. Try implementing matrix multiplication, which benefits much more from shared memory
2. Experiment with more complex tiling strategies
3. Explore using shared memory for other purposes like inter-thread communication
