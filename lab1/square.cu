#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel to square each element of an array
__global__ void squareArray(int *d_in, int *d_out, int size)
{
    // Calculate the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (tid < size)
    {
        d_out[tid] = d_in[tid] * d_in[tid];
    }
}

int main()
{
    // Array size
    const int N = 10000;
    size_t bytes = N * sizeof(int);
    
    // Host arrays
    int *h_in = (int*)malloc(bytes);
    int *h_out = (int*)malloc(bytes);
    
    // Initialize input array on host
    for (int i = 0; i < N; i++)
    {
        h_in[i] = i;
    }
    
    // Device arrays
    int *d_in, *d_out;
    
    // Allocate memory on the device
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    
    // Copy input array from host to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    
    // Set up execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the kernel
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    squareArray<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < N; i++)
    {
        if (h_out[i] != h_in[i] * h_in[i])
        {
            printf("Verification failed at index %d: expected %d, got %d\n", 
                   i, h_in[i] * h_in[i], h_out[i]);
            break;
        }
    }
    printf("Array squaring completed successfully!\n");
    
    // Print first 10 elements for verification
    printf("First 10 elements of squared array:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%d² = %d\n", i, h_out[i]);
    }
    
    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
    
    // Free host memory
    free(h_in);
    free(h_out);
    
    return 0;
}
