# Lab 5: Parallel Reduction and Atomic Operations

This lab explores efficient parallel reduction techniques and atomic operations in CUDA. We'll implement and compare different approaches to calculating array sums and other aggregate operations, with a focus on optimizing performance through various parallel reduction strategies.

## Learning Objectives

By the end of this lab, you will:

1. Understand parallel reduction algorithms and their implementation in CUDA
2. Learn how to use atomic operations for concurrent memory updates
3. Compare the performance of different reduction strategies
4. Gain experience with warp-level primitives for optimization
5. Apply synchronization techniques in parallel reductions

## Parallel Reduction Concepts

### Basic Parallel Reduction

- **Description**: A divide-and-conquer approach to aggregate data across multiple threads
- **Characteristics**:
  - Logarithmic time complexity O(log n)
  - Involves multiple synchronization points
  - Tree-based reduction pattern
- **Applications**: Sum, min, max, average calculations

### Atomic Operations

- **Description**: Hardware-supported operations that guarantee read-modify-write atomicity
- **Characteristics**:
  - Prevent race conditions during concurrent updates
  - May cause serialization at high contention points
  - Simpler to implement than manual reductions
- **Example Functions**: `atomicAdd()`, `atomicMin()`, `atomicMax()`

### Warp-Level Primitives

- **Description**: Specialized operations that leverage the SIMD nature of GPU warps
- **Characteristics**:
  - Avoid explicit synchronization within a warp
  - Utilize hardware shuffle instructions
  - Often significantly faster than global memory reductions
- **Example Functions**: `__shfl_down_sync()`, `__ballot_sync()`, `__any_sync()`

### Shared Memory Reduction

- **Description**: Uses on-chip shared memory for intermediate results
- **Characteristics**:
  - Much faster access than global memory
  - Limited by available shared memory size
  - Requires explicit synchronization between threads
- **Key Concepts**: Bank conflicts, padding, thread synchronization

## Code Structure

Our example code demonstrates multiple approaches to calculate the sum of elements in an array:

1. **Sequential CPU sum** (baseline)
2. **Naive global memory reduction**
3. **Shared memory reduction**
4. **Warp-level reduction**
5. **Atomic operations reduction**

For each approach, we measure:
1. Execution time
2. Correctness (compared to CPU result)
3. Speedup relative to the CPU baseline

## Detailed Code Walkthrough

### 1. Sequential CPU Sum (Baseline)

```c
/* Calculate sum on CPU for verification */
float cpuSum = 0.0f;
for (int i = 0; i < SIZE; i++) {
    cpuSum += h_data[i];
}
```

### 2. Naive Global Memory Reduction

```c
__global__ void globalMemoryReduction(float *input, float *output, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    extern __shared__ float sdata[];
    sdata[tid] = (i < size) ? input[i] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

### 3. Shared Memory Optimized Reduction

```c
__global__ void sharedMemoryReduction(float *input, float *output, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load input into shared memory (with sequential addressing)
    extern __shared__ float sdata[];
    sdata[tid] = (i < size) ? input[i] : 0;
    if (i + blockDim.x < size) sdata[tid] += input[i + blockDim.x];
    __syncthreads();
    
    // Reduction in shared memory with sequential addressing
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

### 4. Warp-Level Reduction

```c
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warpLevelReduction(float *input, float *output, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread loads one element
    float sum = (i < size) ? input[i] : 0;
    
    // Warp-level reduction
    sum = warpReduceSum(sum);
    
    // First thread in each warp writes result
    if ((tid & (warpSize - 1)) == 0) {
        atomicAdd(&output[0], sum);
    }
}
```

### 5. Atomic Operations Reduction

```c
__global__ void atomicReduction(float *input, float *output, int size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        atomicAdd(output, input[i]);
    }
}
```

## How to Compile and Run

### Requirements

- NVIDIA GPU supporting CUDA 
- CUDA Toolkit installed
- C compiler compatible with CUDA
- NVIDIA HPC SDK (for Fortran version)

### Compilation

#### C Version
```bash
nvcc -o parallel_reduction parallel_reduction.cu
```

#### Fortran Version
```bash
nvfortran -o parallel_reduction_fortran parallel_reduction.cuf
```

### Running the Program

The program takes three command-line arguments:
1. Array size
2. Block size (threads per block)
3. Number of runs for averaging performance

#### C Version
```bash
./parallel_reduction 16777216 256 10
```

#### Fortran Version
```bash
./parallel_reduction_fortran 16777216 256 10
```

This runs the program with an array of 16,777,216 elements, using 256 threads per block, and averages the performance over 10 runs.

## Expected Output

The program will display:
1. The calculated sum from each method
2. Execution time for each approach
3. Verification of correctness against CPU result
4. Speedup relative to CPU baseline

Example:
```
======= PARALLEL REDUCTION PERFORMANCE COMPARISON =======
Array Size: 16777216 elements

CPU Sum: 8388608.000000
Execution Time: 53.204678 ms

Naive Global Memory Reduction:
Sum: 8388608.000000 (correct)
Execution Time: 0.950336 ms
Speedup vs CPU: 55.98x

Shared Memory Reduction:
Sum: 8388608.000000 (correct)
Execution Time: 0.551392 ms
Speedup vs CPU: 96.49x

Warp-Level Reduction:
Sum: 8388608.000000 (correct)
Execution Time: 0.306176 ms
Speedup vs CPU: 173.77x

Atomic Reduction:
Sum: 8388608.000000 (correct)
Execution Time: 1.169408 ms
Speedup vs CPU: 45.50x
```

## Experiment Ideas

1. **Vary Array Sizes**: Try different array sizes to see how each approach scales
2. **Block Size Impact**: Experiment with different thread block sizes
3. **Reduction Operations**: Modify the code to perform min, max, or average operations
4. **Bank Conflict Analysis**: Add instrumentation to detect shared memory bank conflicts

## Discussion Questions

1. Which reduction approach performs best for small arrays? For large arrays?
2. Why does the warp-level reduction perform better than other approaches?
3. Under what circumstances would atomic operations be preferred despite potential performance penalties?
4. How could you further optimize the shared memory reduction kernel?
5. What are the tradeoffs between performance and code readability/complexity for each approach?

## Advanced Reduction Techniques

For further exploration, consider these advanced techniques:

1. **Complete Reduction**: Implement a multi-level reduction to get the final result in a single kernel launch
2. **Dynamic Parallelism**: Use CUDA dynamic parallelism to launch child kernels for hierarchical reduction
3. **Cooperative Groups**: Utilize CUDA Cooperative Groups for more flexible synchronization patterns
4. **Butterfly Reduction**: Implement alternative reduction patterns like butterfly reduction
5. **Multi-GPU Reduction**: Extend to perform reductions across multiple GPUs

## Common Issues

- **Synchronization Errors**: Missing `__syncthreads()` can lead to race conditions
- **Warp Divergence**: Conditional statements within warp-level operations can cause divergence
- **Bank Conflicts**: Unoptimized shared memory access patterns can lead to bank conflicts
- **Atomic Contention**: High contention on atomic operations can lead to serialization
- **Floating-Point Precision**: Reduction order can affect the precision of floating-point summation

## Going Further

For more advanced exploration:
1. Implement a parallel scan (prefix sum) algorithm
2. Compare different floating-point precisions (float vs. double)
3. Explore tensor core operations for matrix reductions on compatible hardware
4. Investigate dynamic shared memory allocation
5. Benchmark against optimized libraries like cuBLAS or Thrust
