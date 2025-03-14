#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Initialize array with random values
void initArray(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = (float)rand() / RAND_MAX;
    }
}

// CPU baseline reduction (sum)
float cpuReduction(float* arr, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum;
}

// Kernel 1: Naive global memory reduction
__global__ void globalMemoryReduction(float *input, float *output, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    extern __shared__ float sdata[];
    sdata[tid] = (i < size) ? input[i] : 0;
    __syncthreads();
    
    // Reduction in shared memory (with potential bank conflicts)
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Kernel 2: Shared memory optimized reduction
__global__ void sharedMemoryReduction(float *input, float *output, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load input into shared memory (with sequential addressing)
    extern __shared__ float sdata[];
    
    // Each thread loads two elements
    sdata[tid] = 0.0f;
    
    if (i < size)
        sdata[tid] += input[i];
    
    if (i + blockDim.x < size)
        sdata[tid] += input[i + blockDim.x];
    
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

// Inline device function for warp-level reduction
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel 3: Warp-level reduction
__global__ void warpLevelReduction(float *input, float *output, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread loads one element
    float sum = (i < size) ? input[i] : 0;
    
    // Shared memory for partial sums (per warp)
    extern __shared__ float warpSum[];
    
    // Warp-level reduction
    int lane = tid % warpSize;
    int wid = tid / warpSize;
    
    // Each thread performs warp reduction
    sum = warpReduceSum(sum);
    
    // First thread in each warp writes the result
    if (lane == 0) {
        warpSum[wid] = sum;
    }
    
    __syncthreads();
    
    // Final reduction: first warp reduces all warp sums
    if (wid == 0) {
        sum = (tid < blockDim.x / warpSize) ? warpSum[lane] : 0;
        
        // Final warp reduction
        if (lane == 0) {
            atomicAdd(output, warpReduceSum(sum));
        }
    }
}

// Kernel 4: Atomic operations reduction
__global__ void atomicReduction(float *input, float *output, int size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        atomicAdd(output, input[i]);
    }
}

// Second-level reduction kernel for multi-block scenarios
__global__ void finalReduction(float *input, float *output, int size) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
    // Load input into shared memory
    sdata[tid] = (tid < size) ? input[tid] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) *output = sdata[0];
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    if (argc != 4) {
        printf("Usage: %s <array_size> <block_size> <num_runs>\n", argv[0]);
        printf("Example: %s 16777216 256 10\n", argv[0]);
        return -1;
    }
    
    unsigned int SIZE = atoi(argv[1]);
    unsigned int BLOCK_SIZE = atoi(argv[2]);
    unsigned int NUM_RUNS = atoi(argv[3]);
    
    printf("\n======= PARALLEL REDUCTION PERFORMANCE COMPARISON =======\n");
    printf("Array Size: %u elements\n", SIZE);
    printf("Block Size: %u threads\n", BLOCK_SIZE);
    printf("Performance averaged over %u runs\n\n", NUM_RUNS);
    
    // Set random seed
    srand(time(NULL));
    
    // Allocate memory
    float *h_data = (float*)malloc(SIZE * sizeof(float));
    float *h_result = (float*)malloc(sizeof(float));
    float *d_data, *d_result, *d_temp;
    
    // Initialize data
    initArray(h_data, SIZE);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_data, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_result, sizeof(float)));
    
    // Calculate grid dimensions
    unsigned int NUM_BLOCKS_GLOBAL = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int NUM_BLOCKS_SHARED = (SIZE + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);
    
    // For multi-block reduction, we need temporary storage
    CUDA_CHECK(cudaMalloc((void**)&d_temp, NUM_BLOCKS_GLOBAL * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // ----------------------------------------------------------------------
    // CPU Reduction (Baseline)
    // ----------------------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start));
    
    float cpu_sum = cpuReduction(h_data, SIZE);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float cpu_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&cpu_time, start, stop));
    
    printf("CPU Sum: %f\n", cpu_sum);
    printf("Execution Time: %f ms\n\n", cpu_time);
    
    // ----------------------------------------------------------------------
    // 1. Naive Global Memory Reduction
    // ----------------------------------------------------------------------
    float global_time = 0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        // Reset result
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_temp, 0, NUM_BLOCKS_GLOBAL * sizeof(float)));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        // First step: reduce each block
        globalMemoryReduction<<<NUM_BLOCKS_GLOBAL, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            d_data, d_temp, SIZE);
        
        // Second step: reduce the block results
        finalReduction<<<1, 1024, 1024 * sizeof(float)>>>(d_temp, d_result, NUM_BLOCKS_GLOBAL);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float temp_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        global_time += temp_time;
    }
    
    global_time /= NUM_RUNS;
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Naive Global Memory Reduction:\n");
    printf("Sum: %f %s\n", *h_result, (fabs(*h_result - cpu_sum) < 0.1f) ? "(correct)" : "(incorrect)");
    printf("Execution Time: %f ms\n", global_time);
    printf("Speedup vs CPU: %.2fx\n\n", cpu_time / global_time);
    
    // ----------------------------------------------------------------------
    // 2. Shared Memory Reduction
    // ----------------------------------------------------------------------
    float shared_time = 0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        // Reset result
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_temp, 0, NUM_BLOCKS_SHARED * sizeof(float)));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        // First step: reduce each block
        sharedMemoryReduction<<<NUM_BLOCKS_SHARED, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            d_data, d_temp, SIZE);
        
        // Second step: reduce the block results
        finalReduction<<<1, 1024, 1024 * sizeof(float)>>>(d_temp, d_result, NUM_BLOCKS_SHARED);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float temp_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        shared_time += temp_time;
    }
    
    shared_time /= NUM_RUNS;
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Shared Memory Reduction:\n");
    printf("Sum: %f %s\n", *h_result, (fabs(*h_result - cpu_sum) < 0.1f) ? "(correct)" : "(incorrect)");
    printf("Execution Time: %f ms\n", shared_time);
    printf("Speedup vs CPU: %.2fx\n\n", cpu_time / shared_time);
    
    // ----------------------------------------------------------------------
    // 3. Warp-Level Reduction
    // ----------------------------------------------------------------------
    float warp_time = 0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        // Reset result
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        unsigned int numWarps = (BLOCK_SIZE + warpSize - 1) / warpSize;
        warpLevelReduction<<<NUM_BLOCKS_GLOBAL, BLOCK_SIZE, numWarps * sizeof(float)>>>(
            d_data, d_result, SIZE);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float temp_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        warp_time += temp_time;
    }
    
    warp_time /= NUM_RUNS;
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Warp-Level Reduction:\n");
    printf("Sum: %f %s\n", *h_result, (fabs(*h_result - cpu_sum) < 0.1f) ? "(correct)" : "(incorrect)");
    printf("Execution Time: %f ms\n", warp_time);
    printf("Speedup vs CPU: %.2fx\n\n", cpu_time / warp_time);
    
    // ----------------------------------------------------------------------
    // 4. Atomic Reduction
    // ----------------------------------------------------------------------
    float atomic_time = 0;
    
    for (int run = 0; run < NUM_RUNS; run++) {
        // Reset result
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        atomicReduction<<<NUM_BLOCKS_GLOBAL, BLOCK_SIZE>>>(d_data, d_result, SIZE);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float temp_time = 0;
        CUDA_CHECK(cudaEventElapsedTime(&temp_time, start, stop));
        atomic_time += temp_time;
    }
    
    atomic_time /= NUM_RUNS;
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Atomic Reduction:\n");
    printf("Sum: %f %s\n", *h_result, (fabs(*h_result - cpu_sum) < 0.1f) ? "(correct)" : "(incorrect)");
    printf("Execution Time: %f ms\n", atomic_time);
    printf("Speedup vs CPU: %.2fx\n\n", cpu_time / atomic_time);
    
    // ----------------------------------------------------------------------
    // Performance Summary
    // ----------------------------------------------------------------------
    printf("======= PERFORMANCE SUMMARY =======\n");
    printf("CPU Baseline:            %f ms\n", cpu_time);
    printf("Naive Global Reduction:  %f ms (%.2fx speedup)\n", global_time, cpu_time / global_time);
    printf("Shared Memory Reduction: %f ms (%.2fx speedup)\n", shared_time, cpu_time / shared_time);
    printf("Warp-Level Reduction:    %f ms (%.2fx speedup)\n", warp_time, cpu_time / warp_time);
    printf("Atomic Reduction:        %f ms (%.2fx speedup)\n", atomic_time, cpu_time / atomic_time);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_temp));
    free(h_data);
    free(h_result);
    
    return 0;
}
