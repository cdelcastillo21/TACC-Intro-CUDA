# CUDA Programming

This course provides a comprehensive introduction to CUDA programming, from basic concepts to some advanced optimization techniques. Through a series of hands-on labs, you'll learn how to harness the power of NVIDIA GPUs to accelerate computational tasks.

## Course Overview

This course is designed for programmers who want to develop high-performance applications using NVIDIA's CUDA parallel computing platform. Each lab builds on the previous one, gradually introducing more complex concepts and techniques.

## Prerequisites

- Basic knowledge of C/C++ or Fortran programming
- Understanding of fundamental algorithms and data structures
- Access to an NVIDIA GPU supporting CUDA
- CUDA Toolkit installed
- NVIDIA HPC SDK (for Fortran examples)

## Lab Structure

Each lab includes:
- A detailed README with theoretical background
- Complete source code in both C/C++ and Fortran
- Compilation and execution instructions
- Expected outputs and verification steps
- Discussion questions and experiment ideas
- Common issues and troubleshooting tips

## Labs Overview

### Lab 1: Introduction to CUDA - Array Squaring

An introduction to the fundamental concepts of CUDA programming through a simple example of squaring each element of an array.

**Key concepts:**
- Basic CUDA program structure
- Memory allocation and management
- Simple kernel implementation
- Thread/block configuration
- Performance measurement

### Lab 2: CUDA 2D Blocks and Matrix Operations

Demonstrates how to use 2D thread and block configurations to perform matrix operations efficiently.

**Key concepts:**
- 2D thread and block organization
- Global thread indexing in multiple dimensions
- Matrix addition and transposition
- Performance comparison

### Lab 3: Shared Memory Matrix Addition

Explores the performance benefits of using shared memory in CUDA by comparing two implementations of matrix addition.

**Key concepts:**
- Shared memory allocation and usage
- Thread synchronization with `__syncthreads()`
- Memory access patterns
- Performance comparison with global memory

### Lab 4: Grace Hopper Memory Allocation Types

Investigates different memory allocation approaches available in CUDA, with a focus on the memory architecture introduced in the NVIDIA Grace Hopper platform.

**Key concepts:**
- Pageable memory (conventional)
- Pinned memory
- Mapped memory (zero-copy)
- Unified memory
- Performance characteristics of different memory types

### Lab 5: Parallel Reduction and Atomic Operations

This lab explores efficient parallel reduction techniques and atomic operations in CUDA. You'll implement and compare different approaches to calculating array sums and other aggregate operations.

**Key concepts:**
- Parallel reduction algorithms
- Shared memory optimization strategies
- Warp-level primitives for efficient reduction
- Atomic operations for concurrent memory updates
- Performance analysis of different reduction strategies

**Implementation highlights:**
- Sequential CPU sum (baseline)
- Naive global memory reduction
- Shared memory optimized reduction
- Warp-level reduction using shuffle instructions
- Atomic operations-based reduction

**Performance considerations:**
- Memory access patterns and coalescing
- Bank conflict avoidance in shared memory
- Synchronization overhead reduction
- Warp divergence minimization
- Atomic operation contention management

### Lab 6: CUDA Streams and Asynchronous Execution

This lab focuses on CUDA streams and asynchronous execution to achieve greater performance through overlapping computation with data transfers.

**Key concepts:**
- CUDA streams for concurrent operations
- Asynchronous memory transfers
- Event-based synchronization
- Stream scheduling and priority
- Overlapping computation and data transfer

**Implementation highlights:**
- Sequential execution (baseline)
- Asynchronous transfers with pinned memory
- Multi-stream execution
- Fully overlapped pipeline implementation
- CUDA events for timing and synchronization

**Performance considerations:**
- Memory transfer bottlenecks
- Kernel launch overhead
- Stream synchronization points
- Optimal chunk sizes for streaming
- GPU resource utilization

## Compilation and Execution

### C/C++ Version

For C/C++ implementations:

```bash
# Compile
nvcc -o <executable_name> <source_file>.cu

# Execute
./<executable_name> [arguments]
```

### Fortran Version

For Fortran implementations:

```bash
# Compile
nvfortran -o <executable_name> <source_file>.cuf

# Execute
./<executable_name> [arguments]
```

## Advanced Topics and Further Study

After completing these labs, you may want to explore:

1. **CUDA Dynamic Parallelism**: Launch CUDA kernels from within CUDA kernels
2. **Multi-GPU Programming**: Scale your applications across multiple GPUs
3. **CUDA Graphs**: Define and execute graphs of operations
4. **Tensor Cores**: Utilize specialized hardware for matrix operations
5. **CUDA Libraries**: Leverage optimized libraries like cuBLAS, cuDNN, and Thrust

## Troubleshooting Common Issues

- **Compile Errors**: Ensure you have the correct CUDA Toolkit version installed
- **Memory Errors**: Check for proper memory allocation, deallocation, and boundary checks
- **Performance Issues**: Analyze memory access patterns and occupancy
- **Synchronization Bugs**: Be careful with thread synchronization and race conditions
- **Warp Divergence**: Minimize conditional branches within warps

## Resources

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/178)
