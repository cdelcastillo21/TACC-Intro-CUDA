# Lab 1: Introduction to CUDA - Array Squaring

This lab introduces the fundamental concepts of CUDA programming through a simple example: squaring each element of an array. You'll learn about the basic structure of a CUDA program, including kernel definition, memory management, and execution configuration.

## Learning Objectives

By the end of this lab, you will be able to:

1. Understand the basic structure of a CUDA program
2. Allocate and manage memory on both the host (CPU) and device (GPU)
3. Write a simple CUDA kernel that operates on array data
4. Configure and launch a kernel with the appropriate thread/block structure
5. Measure and analyze the performance of GPU code

## Overview of the Code

The code demonstrates a simple "Hello CUDA" example that squares each element of an array:

1. Allocate memory on both the host (CPU) and device (GPU)
2. Initialize input data on the host
3. Transfer data from host to device
4. Execute the kernel (perform squaring of each element)
5. Transfer results back from device to host
6. Verify the computation
7. Measure and report timing information

## Key CUDA Concepts

### CUDA Kernel

The core computation is performed by a CUDA kernel function that squares each element:

```c
__global__ void squareArray(int *d_in, int *d_out, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        d_out[tid] = d_in[tid] * d_in[tid];
    }
}
```

The `__global__` keyword indicates that this function runs on the GPU but can be called from the CPU.

### Memory Management

CUDA requires explicit memory management between host and device:

```c
// Allocate device memory
cudaMalloc(&d_in, bytes);
cudaMalloc(&d_out, bytes);

// Copy host to device
cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

// Copy device to host
cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

// Free memory
cudaFree(d_in);
cudaFree(d_out);
```

### Thread and Block Configuration

To execute a GPU kernel, we must specify how many thread blocks to create and how many threads per block:

```c
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
squareArray<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
```

This configuration launches enough threads to process all N elements of the array.

## Performance Measurements

The code includes timing measurements for:
- Host to device data transfer
- Kernel execution time
- Device to host data transfer

This allows us to analyze where time is spent and understand the performance characteristics of GPU computing.

## How to Compile and Run

### C Version

1. Compile the code:
```bash
nvcc -o square square.cu
```

2. Run the executable:
```bash
./square
```

### Fortran Version

1. Compile the code:
```bash
nvfortran -o square_fortran square.cuf
```

2. Run the executable:
```bash
./square_fortran
```

## Expected Output

The program will display:

1. Thread/block configuration information
2. Timing data for memory transfers and computation
3. First 10 elements of the squared array for verification
4. Performance metrics including bandwidth and computational throughput

Example output:
```
CUDA kernel launch with 40 blocks of 256 threads
Array squaring completed successfully!

First 10 elements of squared array:
0² = 0
1² = 1
2² = 4
3² = 9
4² = 16
5² = 25
6² = 36
7² = 49
8² = 64
9² = 81

==== TIMING SUMMARY ====
Array size: 10000 integers (40000 bytes)
Host to Device Transfer: 0.123 ms
Kernel Execution:        0.056 ms
Device to Host Transfer: 0.098 ms
Total GPU Time:          0.277 ms

==== BANDWIDTH ====
Host to Device Bandwidth: 0.33 GB/s
Device to Host Bandwidth: 0.41 GB/s

==== THROUGHPUT ====
Computational Throughput: 178.57 MOPS
```

## Experiment Ideas

Here are some ways to experiment with the code to deepen your understanding:

1. **Vary array size**: Try different array sizes and observe how performance scales
2. **Adjust threads per block**: Change `threadsPerBlock` (try 128, 256, 512) and observe the impact
3. **Change the computation**: Modify the kernel to perform a different operation (e.g., cubing instead of squaring)
4. **Add error checking**: Add proper error handling for CUDA API calls

## Discussion Questions

1. Why does the kernel check if `tid < size`?
2. How does the data transfer time compare to the computation time?
3. How would you modify the code to handle larger arrays that exceed GPU memory?
4. For this simple operation, is using a GPU beneficial? At what data size does it become advantageous?

## Common Issues

- **Out of memory errors**: If the array is too large, `cudaMalloc` may fail
- **Kernel launch failures**: Check for invalid thread/block configurations
- **Incorrect results**: Make sure boundary conditions are handled correctly in the kernel

## Going Further

For a deeper exploration of CUDA programming, consider:

1. Implementing a more complex algorithm
2. Using shared memory to improve performance
3. Exploring different memory access patterns
4. Implementing error checking for all CUDA calls