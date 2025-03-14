# Lab 6: CUDA Streams and Asynchronous Execution

This lab explores the concept of CUDA streams and asynchronous execution to achieve greater performance through overlapping computation with data transfers. We'll implement and benchmark different approaches to processing large datasets, from sequential execution to fully overlapped computation and communication.

## Learning Objectives

By the end of this lab, you will:

1. Understand CUDA streams and their role in concurrent execution
2. Learn how to overlap computation with data transfers
3. Implement event-based synchronization
4. Profile and benchmark asynchronous execution
5. Apply these techniques to real-world computational problems

## CUDA Streams Concepts

### What are CUDA Streams?

- **Description**: A CUDA stream is a sequence of operations that execute in issue-order on the GPU
- **Characteristics**:
  - Operations in different streams may execute concurrently
  - Operations within the same stream are executed sequentially
  - The default stream (stream 0) is synchronous with respect to other streams
- **Applications**: Overlapping computation and data transfers, task-level parallelism

### Types of CUDA Operations

1. **Host-to-Device Transfers**: Copying data from CPU memory to GPU memory
2. **Device-to-Host Transfers**: Copying data from GPU memory to CPU memory
3. **Kernel Execution**: Running computational code on the GPU
4. **Memory Operations**: Allocations, deallocations, and other memory management tasks

### Synchronization Methods

- **CUDA Events**: Lightweight points in a stream that can be used for timing and synchronization
- **Stream Synchronization**: Waiting for all operations in a specific stream to complete
- **Device Synchronization**: Waiting for all operations on the GPU to complete
- **Event-based Synchronization**: Using events to synchronize between streams

## Code Structure

Our example code demonstrates multiple approaches to process a large dataset:

1. **Sequential Execution**: Traditional approach with no overlap
2. **Asynchronous Transfers**: Using pinned memory for faster transfers
3. **Multi-Stream Execution**: Using multiple streams for overlapping operations
4. **Fully Overlapped Pipeline**: Implementing a pipeline to maximize GPU utilization

For each approach, we measure:
1. Total execution time
2. Effective throughput (operations per second)
3. GPU utilization
4. Speedup relative to the sequential baseline

## Detailed Execution Models

### 1. Sequential Execution (Baseline)

```
Time -->
|--- Host to Device Transfer ---|
                                |--- Kernel Execution ---|
                                                         |--- Device to Host Transfer ---|
```

### 2. Asynchronous Transfers

```
Time -->
|--- Host to Device Transfer ---|
                    |--- Kernel Execution ---|
                                    |--- Device to Host Transfer ---|
```

### 3. Multi-Stream Execution

```
Stream 1: |--- H2D 1 ---|--- Kernel 1 ---|--- D2H 1 ---|
Stream 2:      |--- H2D 2 ---|--- Kernel 2 ---|--- D2H 2 ---|
Stream 3:           |--- H2D 3 ---|--- Kernel 3 ---|--- D2H 3 ---|
```

### 4. Fully Overlapped Pipeline

```
Stream 1: |--- H2D 1 ---|--- Kernel 1 ---|--- D2H 1 ---|--- H2D 4 ---|--- Kernel 4 ---|
Stream 2:      |--- H2D 2 ---|--- Kernel 2 ---|--- D2H 2 ---|--- H2D 5 ---|--- Kernel 5 ---|
Stream 3:           |--- H2D 3 ---|--- Kernel 3 ---|--- D2H 3 ---|--- H2D 6 ---|--- Kernel 6 ---|
```

## Using CUDA Events for Profiling

CUDA events allow precise timing of specific operations:

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
// Operations to time
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

## How to Compile and Run

### Requirements

- NVIDIA GPU supporting CUDA 
- CUDA Toolkit installed
- C compiler compatible with CUDA
- NVIDIA HPC SDK (for Fortran version, provided separately)

### Compilation

```bash
nvcc -o stream_processing stream_processing.cu
```

### Running the Program

The program takes three command-line arguments:
1. Total size (in MB) of data to process
2. Chunk size (in KB) for each transfer and kernel operation
3. Number of streams to use

```bash
./stream_processing 1024 256 4
```

This runs the program processing 1GB of data in 256KB chunks using 4 streams.

## Expected Output

The program will display:
1. Configuration parameters
2. Execution time for each approach
3. Throughput metrics
4. Speedup compared to the sequential baseline
5. Timeline visualization of operations

Example:
```
======= CUDA STREAMS AND ASYNCHRONOUS EXECUTION =======
Total Data Size: 1024 MB
Chunk Size: 256 KB
Number of Streams: 4

Sequential Execution:
  Total Time: 253.45 ms
  Throughput: 4.04 GB/s

Asynchronous Transfers:
  Total Time: 157.82 ms
  Throughput: 6.49 GB/s
  Speedup vs Sequential: 1.61x

Multi-Stream Execution:
  Total Time: 98.64 ms
  Throughput: 10.38 GB/s
  Speedup vs Sequential: 2.57x

Fully Overlapped Pipeline:
  Total Time: 67.93 ms
  Throughput: 15.07 GB/s
  Speedup vs Sequential: 3.73x
```

## Visualization with NVIDIA Nsight Systems

The lab includes instructions for using NVIDIA Nsight Systems to visualize the execution timeline:

1. Install NVIDIA Nsight Systems from the CUDA Toolkit
2. Run the program with profiling enabled:
   ```bash
   nsys profile -o stream_profile ./stream_processing 1024 256 4
   ```
3. Open the profile in Nsight Systems:
   ```bash
   nsys-ui stream_profile.qdrep
   ```

## Experiment Ideas

1. **Vary Chunk Sizes**: Try different chunk sizes to find the optimal granularity
2. **Stream Count Impact**: Measure the impact of using different numbers of streams
3. **Kernel Complexity**: Modify the kernel to perform more complex operations
4. **Memory Dependency Patterns**: Create dependencies between operations in different streams

## Discussion Questions

1. What is the optimal chunk size for your GPU architecture, and why?
2. How does the number of streams affect performance, and is there a point of diminishing returns?
3. What factors limit the achievable overlap between computation and communication?
4. How would you design a real-world application to maximize the benefits of asynchronous execution?
5. What are the tradeoffs between code complexity and performance gains?

## Advanced Stream Techniques

For further exploration, consider these advanced techniques:

1. **Stream Priorities**: CUDA allows assigning priorities to streams (on supported hardware)
2. **Stream Callbacks**: Register host functions to be called when stream operations complete
3. **Stream Capture**: Record a sequence of operations for replay using CUDA graphs
4. **Persistent Kernels**: Long-running kernels that process data as it arrives
5. **Dynamic Parallelism**: Kernels that launch child kernels in separate streams

## Common Issues

- **Default Stream Synchronization**: Operations in the default stream (stream 0) synchronize with all other streams
- **Pinned Memory Requirements**: Asynchronous transfers require pinned memory
- **Resource Constraints**: Too many concurrent operations can exhaust GPU resources
- **Hidden Synchronization Points**: Some CUDA APIs may cause implicit synchronization
- **Driver Overhead**: Managing many small operations can increase driver overhead

## Going Further

For more advanced exploration:
1. Implement a producer-consumer pattern using CUDA streams
2. Explore CUDA graphs for capturing and replaying stream operations
3. Benchmark different memory types (pinned, managed, etc.) with streams
4. Implement a multi-GPU solution using streams for inter-GPU communication
5. Create a task scheduler that dynamically assigns work to different streams
