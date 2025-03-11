#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix addition using 2D block and grid
__global__ void matrixAdd(int *d_a, int *d_b, int *d_c, int width, int height)
{
    // Calculate the row and column indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we don't go out of bounds
    if (row < height && col < width)
    {
        // Calculate the linear index for the matrices
        int idx = row * width + col;
        
        // Perform the addition
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}

// CUDA kernel for matrix transposition using 2D block and grid
__global__ void matrixTranspose(int *d_in, int *d_out, int width, int height)
{
    // Calculate the row and column indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we don't go out of bounds
    if (row < height && col < width)
    {
        // For transposition, rows become columns and vice versa
        int in_idx = row * width + col;
        int out_idx = col * height + row;  // Note the reversed indices
        
        // Perform the transposition
        d_out[out_idx] = d_in[in_idx];
    }
}

// Function to print a matrix
void printMatrix(int *matrix, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%4d ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main()
{
    // Matrix dimensions
    const int WIDTH = 16;
    const int HEIGHT = 8;
    
    // Total size of matrices in bytes
    size_t bytes = WIDTH * HEIGHT * sizeof(int);
    
    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0.0f;
    
    // Allocate host memory
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);            // For matrix addition result
    int *h_transposed = (int*)malloc(bytes);   // For transposed matrix

    // Initialize matrices on host
    for (int i = 0; i < HEIGHT; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            int idx = i * WIDTH + j;
            h_a[idx] = idx;          // Sequential numbers
            h_b[idx] = WIDTH * HEIGHT - idx;  // Reverse order
        }
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c, *d_transposed;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_transposed, bytes);

    // Copy from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(8, 8);  // 8x8 threads per block = 64 threads
    dim3 blocksPerGrid(
        (WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    printf("Matrix dimensions: %d x %d\n", WIDTH, HEIGHT);
    printf("Block dimensions: %d x %d\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("Grid dimensions: %d x %d\n", blocksPerGrid.x, blocksPerGrid.y);

    // Execute matrix addition kernel
    printf("\nExecuting matrix addition kernel...\n");
    cudaEventRecord(start);
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, WIDTH, HEIGHT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Addition Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    printf("Matrix addition took %.3f ms\n", milliseconds);

    // Execute matrix transposition kernel
    printf("\nExecuting matrix transposition kernel...\n");
    cudaEventRecord(start);
    matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_transposed, WIDTH, HEIGHT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Transposition Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_transposed, d_transposed, bytes, cudaMemcpyDeviceToHost);
    
    printf("Matrix transposition took %.3f ms\n", milliseconds);

    // Print the matrices (if they're small enough)
    if (WIDTH <= 16 && HEIGHT <= 16)
    {
        printf("\nMatrix A:\n");
        printMatrix(h_a, WIDTH, HEIGHT);
        
        printf("\nMatrix B:\n");
        printMatrix(h_b, WIDTH, HEIGHT);
        
        printf("\nMatrix C (A + B):\n");
        printMatrix(h_c, WIDTH, HEIGHT);
        
        printf("\nTransposed A:\n");
        printMatrix(h_transposed, HEIGHT, WIDTH);  // Note the swapped dimensions
    }

    // Verify matrix addition result
    bool additionCorrect = true;
    for (int i = 0; i < HEIGHT * WIDTH; i++)
    {
        if (h_c[i] != h_a[i] + h_b[i])
        {
            printf("Addition verification failed at index %d\n", i);
            additionCorrect = false;
            break;
        }
    }
    if (additionCorrect)
        printf("\nMatrix addition verification: SUCCESS\n");

    // Verify transposition result
    bool transposeCorrect = true;
    for (int i = 0; i < HEIGHT; i++)
    {
        for (int j = 0; j < WIDTH; j++)
        {
            int original_idx = i * WIDTH + j;
            int transposed_idx = j * HEIGHT + i;
            
            if (h_transposed[transposed_idx] != h_a[original_idx])
            {
                printf("Transpose verification failed at position (%d,%d)\n", i, j);
                transposeCorrect = false;
                break;
            }
        }
        if (!transposeCorrect) break;
    }
    if (transposeCorrect)
        printf("Matrix transposition verification: SUCCESS\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_transposed);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_transposed);
    
    // Destroy timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
