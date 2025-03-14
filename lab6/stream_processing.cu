#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Constants
#define MB (1024 * 1024)
#define KB (1024)

// Simple kernel that performs vector squaring and adds a constant
__global__ void processData(float *input, float *output, int n, float addValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        val = val * val; // Square
        val += addValue; // Add constant
        output[idx] = val;
    }
}

// Fills buffer with random data
void fillRandomData(float *buffer, int size) {
    for (int i = 0; i < size; i++) {
        buffer[i] = (float)rand() / RAND_MAX;
    }
}

// Verifies that the output data is correct
bool verifyResults(float *input, float *output, int size, float addValue) {
    bool correct = true;
    for (int i = 0; i < size; i++) {
        float expected = input[i] * input[i] + addValue;
        if (fabs(output[i] - expected) > 1e-5) {
            printf("Verification failed at index %d: got %f, expected %f\n",
                   i, output[i], expected);
            correct = false;
            // Early exit after finding first error
            return false;
        }
    }
    return correct;
}

// Function to print a progress bar
void printProgressBar(float progress) {
    const int barWidth = 50;
    int pos = barWidth * progress;
    
    printf("[");
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %.1f%%\r", progress * 100.0f);
    fflush(stdout);
}

int main(int argc, char **argv) {
    // Parse command line arguments
    if (argc != 4) {
        printf("Usage: %s <total_size_MB> <chunk_size_KB> <num_streams>\n", argv[0]);
        printf("Example: %s 1024 256 4\n", argv[0]);
        return -1;
    }
    
    int totalSizeMB = atoi(argv[1]);
    int chunkSizeKB = atoi(argv[2]);
    int numStreams = atoi(argv[3]);
    
    // Check for valid arguments
    if (totalSizeMB <= 0 || chunkSizeKB <= 0 || numStreams <= 0) {
        printf("Error: All arguments must be positive integers\n");
        return -1;
    }
    
    // Calculate sizes in bytes
    size_t totalSizeBytes = (size_t)totalSizeMB * MB;
    size_t chunkSizeBytes = (size_t)chunkSizeKB * KB;
    int numElements = totalSizeBytes / sizeof(float);
    int chunkElements = chunkSizeBytes / sizeof(float);
    int numChunks = (numElements + chunkElements - 1) / chunkElements; // Ceiling division
    
    // Verify chunk size is valid
    if (chunkElements <= 0) {
        printf("Error: Chunk size too small\n");
        return -1;
    }
    
    // Print configuration information
    printf("\n======= CUDA STREAMS AND ASYNCHRONOUS EXECUTION =======\n");
    printf("Total Data Size: %d MB (%zu bytes, %d elements)\n", totalSizeMB, totalSizeBytes, numElements);
    printf("Chunk Size: %d KB (%zu bytes, %d elements)\n", chunkSizeKB, chunkSizeBytes, chunkElements);
    printf("Number of Chunks: %d\n", numChunks);
    printf("Number of Streams: %d\n\n", numStreams);
    
    // Set random seed
    srand((unsigned int)time(NULL));
    
    // Value to add in kernel
    const float addValue = 10.0f;
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Allocate host memory for full data
    float *h_input = (float*)malloc(totalSizeBytes);
    float *h_output = (float*)malloc(totalSizeBytes);
    
    if (!h_input || !h_output) {
        printf("Error: Failed to allocate host memory\n");
        return -1;
    }
    
    // Fill input with random data
    fillRandomData(h_input, numElements);
    
    // Calculate grid dimensions for kernel launch
    int threadsPerBlock = 256;
    int blocksPerChunk = (chunkElements + threadsPerBlock - 1) / threadsPerBlock;
    
    //--------------------------------------------------------------------------
    // Approach 1: Sequential Execution (Baseline)
    //--------------------------------------------------------------------------
    printf("Running Sequential Execution (Baseline)...\n");
    
    // Zero out output array
    memset(h_output, 0, totalSizeBytes);
    
    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input, chunkSizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, chunkSizeBytes));
    
    // Start timing
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int chunk = 0; chunk < numChunks; chunk++) {
        // Calculate current chunk size (for last chunk handling)
        int offset = chunk * chunkElements;
        int currentChunkElements = (offset + chunkElements <= numElements) ? 
                                   chunkElements : (numElements - offset);
        size_t currentChunkBytes = currentChunkElements * sizeof(float);
        
        // Copy input data (H2D)
        CUDA_CHECK(cudaMemcpy(d_input, h_input + offset, currentChunkBytes, cudaMemcpyHostToDevice));
        
        // Launch kernel
        processData<<<blocksPerChunk, threadsPerBlock>>>(d_input, d_output, currentChunkElements, addValue);
        CUDA_CHECK(cudaGetLastError()); // Check for async kernel errors
        
        // Wait for kernel to finish
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy output data (D2H)
        CUDA_CHECK(cudaMemcpy(h_output + offset, d_output, currentChunkBytes, cudaMemcpyDeviceToHost));
        
        // Print progress
        printProgressBar((float)(chunk + 1) / numChunks);
    }
    printf("\n");
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float sequentialTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&sequentialTime, start, stop));
    
    // Verify results
    bool sequentialCorrect = verifyResults(h_input, h_output, numElements, addValue);
    
    // Calculate throughput
    float sequentialThroughput = totalSizeBytes / (sequentialTime * 1.0e-3) / MB; // MB/s
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    //--------------------------------------------------------------------------
    // Approach 2: Asynchronous Transfers with Pinned Memory
    //--------------------------------------------------------------------------
    printf("\nRunning Asynchronous Transfers with Pinned Memory...\n");
    
    // Reset output array
    memset(h_output, 0, totalSizeBytes);
    
    // Allocate pinned memory on host
    float *h_input_pinned, *h_output_pinned;
    CUDA_CHECK(cudaMallocHost((void**)&h_input_pinned, chunkSizeBytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_output_pinned, chunkSizeBytes));
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_input, chunkSizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, chunkSizeBytes));
    
    // Create a default stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Start timing
    CUDA_CHECK(cudaEventRecord(start, stream));
    
    for (int chunk = 0; chunk < numChunks; chunk++) {
        // Calculate current chunk size
        int offset = chunk * chunkElements;
        int currentChunkElements = (offset + chunkElements <= numElements) ? 
                                   chunkElements : (numElements - offset);
        size_t currentChunkBytes = currentChunkElements * sizeof(float);
        
        // Copy chunk to pinned memory
        memcpy(h_input_pinned, h_input + offset, currentChunkBytes);
        
        // Copy input data (H2D)
        CUDA_CHECK(cudaMemcpyAsync(d_input, h_input_pinned, currentChunkBytes, cudaMemcpyHostToDevice, stream));
        
        // Launch kernel
        processData<<<blocksPerChunk, threadsPerBlock, 0, stream>>>(d_input, d_output, currentChunkElements, addValue);
        CUDA_CHECK(cudaGetLastError()); // Check for async kernel errors
        
        // Copy output data (D2H)
        CUDA_CHECK(cudaMemcpyAsync(h_output_pinned, d_output, currentChunkBytes, cudaMemcpyDeviceToHost, stream));
        
        // Synchronize to ensure this chunk is complete
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Copy from pinned memory back to regular host memory
        memcpy(h_output + offset, h_output_pinned, currentChunkBytes);
        
        // Print progress
        printProgressBar((float)(chunk + 1) / numChunks);
    }
    printf("\n");
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float asyncTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&asyncTime, start, stop));
    
    // Verify results
    bool asyncCorrect = verifyResults(h_input, h_output, numElements, addValue);
    
    // Calculate throughput
    float asyncThroughput = totalSizeBytes / (asyncTime * 1.0e-3) / MB; // MB/s
    
    // Free memory
    CUDA_CHECK(cudaFreeHost(h_input_pinned));
    CUDA_CHECK(cudaFreeHost(h_output_pinned));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    //--------------------------------------------------------------------------
    // Approach 3: Multi-Stream Execution
    //--------------------------------------------------------------------------
    printf("\nRunning Multi-Stream Execution (%d streams)...\n", numStreams);
    
    // Reset output array
    memset(h_output, 0, totalSizeBytes);
    
    // Create CUDA streams
    cudaStream_t *streams = (cudaStream_t*)malloc(numStreams * sizeof(cudaStream_t));
    if (!streams) {
        printf("Error: Failed to allocate memory for streams\n");
        return -1;
    }
    
    for (int i = 0; i < numStreams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Allocate pinned memory and device memory for each stream
    float **h_input_pinned_streams = (float**)malloc(numStreams * sizeof(float*));
    float **h_output_pinned_streams = (float**)malloc(numStreams * sizeof(float*));
    float **d_input_streams = (float**)malloc(numStreams * sizeof(float*));
    float **d_output_streams = (float**)malloc(numStreams * sizeof(float*));
    
    if (!h_input_pinned_streams || !h_output_pinned_streams || 
        !d_input_streams || !d_output_streams) {
        printf("Error: Failed to allocate memory for stream arrays\n");
        return -1;
    }
    
    for (int i = 0; i < numStreams; i++) {
        CUDA_CHECK(cudaMallocHost((void**)&h_input_pinned_streams[i], chunkSizeBytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_output_pinned_streams[i], chunkSizeBytes));
        CUDA_CHECK(cudaMalloc((void**)&d_input_streams[i], chunkSizeBytes));
        CUDA_CHECK(cudaMalloc((void**)&d_output_streams[i], chunkSizeBytes));
    }
    
    // Start timing
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int chunk = 0; chunk < numChunks; chunk++) {
        // Calculate stream index for this chunk
        int streamIdx = chunk % numStreams;
        
        // Calculate current chunk size
        int offset = chunk * chunkElements;
        int currentChunkElements = (offset + chunkElements <= numElements) ? 
                                   chunkElements : (numElements - offset);
        size_t currentChunkBytes = currentChunkElements * sizeof(float);
        
        // Copy chunk to pinned memory
        memcpy(h_input_pinned_streams[streamIdx], h_input + offset, currentChunkBytes);
        
        // Copy input data (H2D)
        CUDA_CHECK(cudaMemcpyAsync(d_input_streams[streamIdx], h_input_pinned_streams[streamIdx], 
                                 currentChunkBytes, cudaMemcpyHostToDevice, streams[streamIdx]));
        
        // Launch kernel
        processData<<<blocksPerChunk, threadsPerBlock, 0, streams[streamIdx]>>>(
            d_input_streams[streamIdx], d_output_streams[streamIdx], currentChunkElements, addValue);
        CUDA_CHECK(cudaGetLastError()); // Check for async kernel errors
        
        // Copy output data (D2H)
        CUDA_CHECK(cudaMemcpyAsync(h_output_pinned_streams[streamIdx], d_output_streams[streamIdx], 
                                 currentChunkBytes, cudaMemcpyDeviceToHost, streams[streamIdx]));
        
        // If we've filled all streams, synchronize the oldest stream and copy its data
        if ((chunk >= numStreams - 1) && (chunk % numStreams == numStreams - 1)) {
            for (int s = 0; s < numStreams; s++) {
                int completedChunk = chunk - numStreams + 1 + s;
                int completedOffset = completedChunk * chunkElements;
                int completedElements = (completedOffset + chunkElements <= numElements) ? 
                                      chunkElements : (numElements - completedOffset);
                size_t completedBytes = completedElements * sizeof(float);
                
                CUDA_CHECK(cudaStreamSynchronize(streams[s]));
                memcpy(h_output + completedOffset, h_output_pinned_streams[s], completedBytes);
            }
        }
        
        // Print progress
        printProgressBar((float)(chunk + 1) / numChunks);
    }
    
    // Synchronize remaining streams and copy final data
    int remainingChunks = numChunks % numStreams;
    if (remainingChunks == 0) remainingChunks = numStreams;
    
    for (int s = 0; s < numStreams; s++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        
        if (s < remainingChunks) {
            int finalChunk = numChunks - remainingChunks + s;
            int finalOffset = finalChunk * chunkElements;
            int finalElements = (finalOffset + chunkElements <= numElements) ? 
                              chunkElements : (numElements - finalOffset);
            size_t finalBytes = finalElements * sizeof(float);
            
            memcpy(h_output + finalOffset, h_output_pinned_streams[s], finalBytes);
        }
    }
    printf("\n");
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float multiStreamTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&multiStreamTime, start, stop));
    
    // Verify results
    bool multiStreamCorrect = verifyResults(h_input, h_output, numElements, addValue);
    
    // Calculate throughput
    float multiStreamThroughput = totalSizeBytes / (multiStreamTime * 1.0e-3) / MB; // MB/s
    
    //--------------------------------------------------------------------------
    // Approach 4: Fully Overlapped Pipeline
    //--------------------------------------------------------------------------
    printf("\nRunning Fully Overlapped Pipeline...\n");
    
    // Reset output array
    memset(h_output, 0, totalSizeBytes);
    
    // Start timing
    CUDA_CHECK(cudaEventRecord(start));
    
    // Create events for synchronization between iterations
    cudaEvent_t *events = (cudaEvent_t*)malloc(numChunks * sizeof(cudaEvent_t));
    if (!events) {
        printf("Error: Failed to allocate memory for events\n");
        return -1;
    }
    
    for (int i = 0; i < numChunks; i++) {
        CUDA_CHECK(cudaEventCreate(&events[i]));
    }
    
    // Process chunks in a pipelined fashion
    for (int chunk = 0; chunk < numChunks; chunk++) {
        // Calculate stream index for this chunk
        int streamIdx = chunk % numStreams;
        
        // Calculate current chunk size
        int offset = chunk * chunkElements;
        int currentChunkElements = (offset + chunkElements <= numElements) ? 
                                   chunkElements : (numElements - offset);
        size_t currentChunkBytes = currentChunkElements * sizeof(float);
        
        // Wait for previous iteration of this stream to finish D2H copy
        if (chunk >= numStreams) {
            CUDA_CHECK(cudaStreamWaitEvent(streams[streamIdx], events[chunk - numStreams], 0));
            
            // Copy completed data to output
            int completedChunk = chunk - numStreams;
            int completedOffset = completedChunk * chunkElements;
            int completedElements = (completedOffset + chunkElements <= numElements) ? 
                                  chunkElements : (numElements - completedOffset);
            size_t completedBytes = completedElements * sizeof(float);
            
            memcpy(h_output + completedOffset, h_output_pinned_streams[streamIdx], completedBytes);
        }
        
        // Copy chunk to pinned memory
        memcpy(h_input_pinned_streams[streamIdx], h_input + offset, currentChunkBytes);
        
        // Copy input data (H2D)
        CUDA_CHECK(cudaMemcpyAsync(d_input_streams[streamIdx], h_input_pinned_streams[streamIdx], 
                                 currentChunkBytes, cudaMemcpyHostToDevice, streams[streamIdx]));
        
        // Launch kernel
        processData<<<blocksPerChunk, threadsPerBlock, 0, streams[streamIdx]>>>(
            d_input_streams[streamIdx], d_output_streams[streamIdx], currentChunkElements, addValue);
        CUDA_CHECK(cudaGetLastError()); // Check for async kernel errors
        
        // Copy output data (D2H)
        CUDA_CHECK(cudaMemcpyAsync(h_output_pinned_streams[streamIdx], d_output_streams[streamIdx], 
                                 currentChunkBytes, cudaMemcpyDeviceToHost, streams[streamIdx]));
        
        // Record event after D2H copy
        CUDA_CHECK(cudaEventRecord(events[chunk], streams[streamIdx]));
        
        // Print progress
        printProgressBar((float)(chunk + 1) / numChunks);
    }
    
    // Process remaining data
    remainingChunks = min(numStreams, numChunks);
    for (int s = 0; s < remainingChunks; s++) {
        int finalChunk = numChunks - remainingChunks + s;
        CUDA_CHECK(cudaStreamSynchronize(streams[s % numStreams]));
        
        int finalOffset = finalChunk * chunkElements;
        int finalElements = (finalOffset + chunkElements <= numElements) ? 
                          chunkElements : (numElements - finalOffset);
        size_t finalBytes = finalElements * sizeof(float);
        
        memcpy(h_output + finalOffset, h_output_pinned_streams[s % numStreams], finalBytes);
    }
    printf("\n");
    
    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float pipelineTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&pipelineTime, start, stop));
    
    // Verify results
    bool pipelineCorrect = verifyResults(h_input, h_output, numElements, addValue);
    
    // Calculate throughput
    float pipelineThroughput = totalSizeBytes / (pipelineTime * 1.0e-3) / MB; // MB/s
    
    // Clean up events
    for (int i = 0; i < numChunks; i++) {
        CUDA_CHECK(cudaEventDestroy(events[i]));
    }
    free(events);
    
    //--------------------------------------------------------------------------
    // Performance Report
    //--------------------------------------------------------------------------
    printf("\n======= PERFORMANCE RESULTS =======\n");
    
    printf("\nApproach 1: Sequential Execution (Baseline)\n");
    printf("  Total Time: %.2f ms\n", sequentialTime);
    printf("  Throughput: %.2f MB/s\n", sequentialThroughput);
    printf("  Verification: %s\n", sequentialCorrect ? "PASSED" : "FAILED");
    
    printf("\nApproach 2: Asynchronous Transfers with Pinned Memory\n");
    printf("  Total Time: %.2f ms\n", asyncTime);
    printf("  Throughput: %.2f MB/s\n", asyncThroughput);
    printf("  Speedup vs Sequential: %.2fx\n", sequentialTime / asyncTime);
    printf("  Verification: %s\n", asyncCorrect ? "PASSED" : "FAILED");
    
    printf("\nApproach 3: Multi-Stream Execution (%d streams)\n", numStreams);
    printf("  Total Time: %.2f ms\n", multiStreamTime);
    printf("  Throughput: %.2f MB/s\n", multiStreamThroughput);
    printf("  Speedup vs Sequential: %.2fx\n", sequentialTime / multiStreamTime);
    printf("  Verification: %s\n", multiStreamCorrect ? "PASSED" : "FAILED");
    
    printf("\nApproach 4: Fully Overlapped Pipeline\n");
    printf("  Total Time: %.2f ms\n", pipelineTime);
    printf("  Throughput: %.2f MB/s\n", pipelineThroughput);
    printf("  Speedup vs Sequential: %.2fx\n", sequentialTime / pipelineTime);
    printf("  Verification: %s\n", pipelineCorrect ? "PASSED" : "FAILED");
    
    printf("\n======= SUMMARY =======\n");
    printf("Sequential:   %.2f ms (baseline)\n", sequentialTime);
    printf("Async:        %.2f ms (%.2fx speedup)\n", asyncTime, sequentialTime / asyncTime);
    printf("Multi-Stream: %.2f ms (%.2fx speedup)\n", multiStreamTime, sequentialTime / multiStreamTime);
    printf("Pipeline:     %.2f ms (%.2fx speedup)\n", pipelineTime, sequentialTime / pipelineTime);
    
    //--------------------------------------------------------------------------
    // Cleanup
    //--------------------------------------------------------------------------
    
    // Destroy streams
    for (int i = 0; i < numStreams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    free(streams);
    
    // Free pinned memory
    for (int i = 0; i < numStreams; i++) {
        CUDA_CHECK(cudaFreeHost(h_input_pinned_streams[i]));
        CUDA_CHECK(cudaFreeHost(h_output_pinned_streams[i]));
        CUDA_CHECK(cudaFree(d_input_streams[i]));
        CUDA_CHECK(cudaFree(d_output_streams[i]));
    }
    
    // Free memory arrays
    free(h_input_pinned_streams);
    free(h_output_pinned_streams);
    free(d_input_streams);
    free(d_output_streams);
    
    // Free host memory
    free(h_input);
    free(h_output);
    
    // Destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // Reset device
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}
