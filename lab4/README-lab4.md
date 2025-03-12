# Lab 4: Understanding Memory Allocation Types

This lab explores the different memory allocation approaches available in CUDA. We'll compare four memory allocation strategies: pageable memory, pinned memory, mapped memory, and unified memory.

## Learning Objectives

By the end of this lab, you will:

1. Understand the different memory allocation techniques in CUDA
2. Compare the performance differences between these techniques
3. Learn how to choose the appropriate memory approach for different scenarios
4. Gain experience with the Grace Hopper memory architecture concepts

## Memory Allocation Types

### 1. Pageable Memory (Conventional)

- **Description**: Standard memory allocation using `malloc()` in host code and `cudaMalloc()` for device memory
- **Characteristics**: 
  - Two separate memory allocations (host and device)
  - Requires explicit data transfers between host and device
  - Data goes through system memory during transfers
- **Example Functions**: `malloc()`, `cudaMalloc()`, `cudaMemcpy()`

### 2. Pinned Memory

- **Description**: Non-pageable (pinned) host memory allocation that enables faster data transfers
- **Characteristics**:
  - Two memory allocations (host and device)
  - Direct access to host memory from GPU
  - No need for intermediate system memory copying during transfers
- **Example Functions**: `cudaMallocHost()`, `cudaFreeHost()`

### 3. Mapped Memory (Zero-Copy)

- **Description**: Memory that is mapped to both CPU and GPU address spaces
- **Characteristics**:
  - Single CPU memory allocation that is mapped to GPU address space
  - No explicit data transfers needed
  - Data accessed directly by GPU (may be slower for intensive computation)
- **Example Functions**: `cudaHostAlloc()` with `cudaHostAllocMapped` flag, `cudaHostGetDevicePointer()`

### 4. Unified Memory

- **Description**: Single memory allocation accessible by both CPU and GPU with automatic migration
- **Characteristics**:
  - Single allocation using unified memory
  - CUDA runtime automatically migrates data between CPU and GPU
  - Simplest programming model
- **Example Functions**: `cudaMallocManaged()`

## Code Structure

Our example code demonstrates all four memory allocation types for a simple task: filling an array with random numbers. For each approach, we measure:

1. Kernel execution time
2. Data transfer time (where applicable)
3. Total operation time

The results allow us to compare the performance characteristics of each approach.

## Detailed Code Walkthrough

### Common Elements

- Each section uses the same `randomArray` kernel to populate the array with random values
- We use CUDA events to time various operations
- Each approach prints the first 5 elements for verification

### 1. Pageable Memory Implementation

```c
/* Allocate the CPU array (pageable memory) */
float* cpu_array = (float*) malloc(SIZE * sizeof(float));

/* Allocate the GPU array */
float* gpu_array;
cudaMalloc((void**) &gpu_array, SIZE * sizeof(float));

/* Randomize the GPU array */
randomArray<<<gridsize, blocksize>>>(gpu_array, SIZE, seed);

/* Copy the GPU array to its CPU counterpart */
cudaMemcpy(cpu_array, gpu_array, SIZE*sizeof(float), cudaMemcpyDeviceToHost);

/* Deallocate memory */
cudaFree(gpu_array);
free(cpu_array);
```

### 2. Pinned Memory Implementation

```c
/* Allocate the CPU array with pinned memory */
cudaMallocHost((void **)&cpu_array, SIZE * sizeof(float));

/* Allocate the GPU array */
cudaMalloc((void**) &gpu_array, SIZE * sizeof(float));

/* Randomize the GPU array */
randomArray<<<gridsize, blocksize>>>(gpu_array, SIZE, seed);

/* Copy the GPU array to its CPU counterpart */
cudaMemcpy(cpu_array, gpu_array, SIZE*sizeof(float), cudaMemcpyDeviceToHost);

/* Deallocate memory */
cudaFree(gpu_array);
cudaFreeHost(cpu_array);
```

### 3. Mapped Memory Implementation

```c
/* Allocate the host array with mapped memory */
float* host_array = NULL;
cudaHostAlloc((void**) &host_array, SIZE*sizeof(float), cudaHostAllocMapped);

/* Assign the device pointer to the host array */
float* device_array;
cudaHostGetDevicePointer((void **) &device_array, (void *) host_array, 0);

/* Randomize the device array */
randomArray<<<gridsize, blocksize>>>(device_array, SIZE, seed);

/* Deallocate memory */
cudaFreeHost(host_array);
```

### 4. Unified Memory Implementation

```c
/* Allocate unified memory */
float* unified_array;
cudaMallocManaged(&unified_array, SIZE * sizeof(float));

/* Randomize the array directly using unified memory */
randomArray<<<gridsize, blocksize>>>(unified_array, SIZE, seed);

/* Deallocate memory */
cudaFree(unified_array);
```

## How to Compile and Run

### Requirements

- NVIDIA GPU supporting CUDA (ideally a recent generation for unified memory support)
- CUDA Toolkit installed
- C compiler compatible with CUDA
- NVIDIA HPC SDK (for Fortran version)

### Compilation

#### C Version
```bash
nvcc -o memory_types memory_types.cu
```

#### Fortran Version
```bash
nvfortran -o memory_types_fortran memory_types.cuf
```

### Running the Program

The program takes three command-line arguments:
1. Array size
2. Grid size (number of blocks)
3. Block size (threads per block)

#### C Version
```bash
./memory_types 1048576 128 256
```

#### Fortran Version
```bash
./memory_types_fortran 1048576 128 256
```

This runs the program with an array of 1,048,576 elements, using 128 blocks with 256 threads each.

## Expected Output

The program will display:
1. Execution times for each memory approach
2. First five generated values for verification
3. A summary comparing the performance of each method

Example:
```
======= CONVENTIONAL CUDA MEMORY ACCESS (Pageable) =======
Elapsed time of GPU kernel execution: 0.001234 seconds
Elapsed Time of the GPU-to-CPU data transfer: 0.004567 seconds
Total elapsed time: 0.005801 seconds
First 5 elements: 0.123456 0.234567 0.345678 0.456789 0.567890

...

======= MEMORY ACCESS PERFORMANCE SUMMARY =======
Pageable Memory:  0.005801 seconds (baseline)
Pinned Memory:    0.003456 seconds (1.68x vs pageable)
Mapped Memory:    0.004123 seconds (1.41x vs pageable)
Unified Memory:   0.002789 seconds (2.08x vs pageable)
```

## Experiment Ideas

1. **Vary Array Sizes**: Try different array sizes to see how each approach scales
2. **Memory Access Patterns**: Modify the kernel to use different access patterns
3. **Operation Complexity**: Change the kernel to perform more complex operations
4. **Data Transfer Direction**: Experiment with host-to-device transfers as well

## Discussion Questions

1. Which memory approach is fastest for small arrays? For large arrays?
2. When would you use pinned memory versus unified memory?
3. How does the access pattern affect the performance of mapped memory?
4. What are the trade-offs between programming simplicity and performance?
5. How might these different memory types benefit different kinds of applications?

## Understanding Grace Hopper Architecture

The Grace Hopper architecture introduces significant advancements in GPU memory management. This lab helps understand these concepts at a fundamental level:

1. **Coherent Memory Access**: Grace Hopper's coherent memory system simplifies programming by allowing both CPU and GPU to access the same memory coherently
2. **NVLink**: High-bandwidth, low-latency links between CPU and GPU memory improve data transfer rates
3. **Reduced Data Movement**: The unified and mapped memory approaches in this lab demonstrate the concept of reducing explicit data transfers

## Common Issues

- **Pinned Memory Limits**: System may have limits on how much pinned memory can be allocated
- **Mapped Memory Performance**: Mapped memory can be slower for compute-intensive operations with frequent access
- **Unified Memory Overhead**: There may be performance overhead for page faulting and migration

## Going Further

For more advanced exploration:
1. Implement a more complex algorithm to better see the performance differences
2. Experiment with explicit prefetching for unified memory
3. Explore using different streams with pinned memory for overlapping computation and data transfer
4. Investigate memory advise hints to further optimize unified memory performance
