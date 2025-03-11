#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Global memory version of matrix addition
__global__ void matrixAddGlobal(float *d_a, float *d_b, float *d_c, int width, int height)
{
    // Calculate the row and column indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we don't go out of bounds
    if (row < height && col < width)
    {
        // Calculate the linear index for the matrices
        int idx = row * width + col;
        
        // Perform the addition directly from global memory
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}

// Shared memory version of matrix addition
__global__ void matrixAddShared(float *d_a, float *d_b, float *d_c, int width, int height)
{
    // Declare shared memory tiles for both input matrices
    __shared__ float tileA[TILE_HEIGHT][TILE_WIDTH];
    __shared__ float tileB[TILE_HEIGHT][TILE_WIDTH];

    // Calculate the row and column indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the linear index for the matrices in global memory
    int idx = row * width + col;
    
    // Load data from global memory to shared memory
    // Each thread loads one element
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
    
    // Make sure all threads have loaded their data before proceeding
    __syncthreads();
    
    // Perform the addition using the data in shared memory
    if (row < height && col < width)
    {
        d_c[idx] = tileA[threadIdx.y][threadIdx.x] + tileB[threadIdx.y][threadIdx.x];
    }
}

int main()
{
    // Matrix dimensions
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    
    // Total size of matrices in bytes
    size_t bytes = WIDTH * HEIGHT * sizeof(float);
    
    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float global_milliseconds = 0.0f;
    float shared_milliseconds = 0.0f;
    
    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c_global = (float*)malloc(bytes);
    float *h_c_shared = (float*)malloc(bytes);

    // Initialize matrices on host with random values
    for (int i = 0; i < HEIGHT * WIDTH; i++)
    {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c_global, *d_c_shared;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c_global, bytes);
    cudaMalloc(&d_c_shared, bytes);

    // Copy input data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 blocksPerGrid(
        (WIDTH + TILE_WIDTH - 1) / TILE_WIDTH,
        (HEIGHT + TILE_HEIGHT - 1) / TILE_HEIGHT
    );

    printf("Matrix dimensions: %d x %d\n", WIDTH, HEIGHT);
    printf("Block dimensions: %d x %d\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("Grid dimensions: %d x %d\n", blocksPerGrid.x, blocksPerGrid.y);

    // Execute global memory version
    printf("\nExecuting global memory version...\n");
    cudaEventRecord(start);
    matrixAddGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c_global, WIDTH, HEIGHT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&global_milliseconds, start, stop);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Global Memory Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_c_global, d_c_global, bytes, cudaMemcpyDeviceToHost);
    
    printf("Global memory version took %.3f ms\n", global_milliseconds);

    // Execute shared memory version
    printf("\nExecuting shared memory version...\n");
    cudaEventRecord(start);
    matrixAddShared<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c_shared, WIDTH, HEIGHT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&shared_milliseconds, start, stop);
    
    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Shared Memory Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_c_shared, d_c_shared, bytes, cudaMemcpyDeviceToHost);
    
    printf("Shared memory version took %.3f ms\n", shared_milliseconds);

    // Verify results match between the two versions
    bool resultsMatch = true;
    for (int i = 0; i < HEIGHT * WIDTH; i++)
    {
        // Using a small epsilon for floating point comparison
        if (fabs(h_c_global[i] - h_c_shared[i]) > 1e-5)
        {
            printf("Results do not match at index %d: global = %f, shared = %f\n", 
                   i, h_c_global[i], h_c_shared[i]);
            resultsMatch = false;
            break;
        }
    }
    
    if (resultsMatch)
        printf("\nResults from global and shared memory versions match!\n");
    
    // Calculate and display speedup
    if (global_milliseconds > 0)
    {
        float speedup = global_milliseconds / shared_milliseconds;
        printf("\nSpeedup from using shared memory: %.2fx\n", speedup);
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_global);
    cudaFree(d_c_shared);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c_global);
    free(h_c_shared);
    
    // Destroy timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
