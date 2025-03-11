#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void randomArray(float* gpu_array, unsigned long SIZE, unsigned seed) {
    // Random number generator seed
    curandState gpu_curand_state;
    curand_init(seed, blockIdx.x, threadIdx.x, &gpu_curand_state);
    
    // Global_index: the global index in the gpu_array
    unsigned global_index;
    
    // niter: number of iterations for each GPU thread
    unsigned long niter = SIZE/(gridDim.x*blockDim.x);
    
    for(unsigned long i = 0; i < niter; i++) {
        global_index = (blockIdx.x*blockDim.x+threadIdx.x)*niter+i;
        if (global_index < SIZE) { // Safety check
            gpu_array[global_index] = curand_uniform(&gpu_curand_state);
        }
    }
}

int main(int argc, char *argv[]) {
    // Check command line arguments
    if (argc != 4) {
        printf("Usage: %s <array_size> <grid_size> <block_size>\n", argv[0]);
        printf("Example: %s 1048576 128 256\n", argv[0]);
        return -1;
    }
    
    /* Set the random seed as the current time */
    unsigned seed = time(0);
    
    /* Set the array size to the first command-line argument */
    unsigned long SIZE = (unsigned long) (atoi(argv[1]));
    printf("\nRandomizing an array with %ld elements\n", SIZE);
    
    /* Set the grid size to the second command-line argument */
    unsigned long gridsize = (unsigned long) (atoi(argv[2]));
    
    /* Set the block size to the third command-line argument */
    unsigned long blocksize = (unsigned long) (atoi(argv[3]));
    
    if (SIZE % (gridsize*blocksize) != 0) {
        printf("Warning: Array size of %ld is NOT divisible by %ld * %ld\n", 
               SIZE, gridsize, blocksize);
        printf("Some threads may be idle.\n");
    }
    
    //--------------------------------------------------------------------------
    // 1. CONVENTIONAL CUDA MEMORY ACCESS (Pageable Memory)
    //--------------------------------------------------------------------------
    printf("\n\n======= CONVENTIONAL CUDA MEMORY ACCESS (Pageable) =======\n");
    
    /* Starting time */
    clock_t t0 = clock();
    
    /* Allocate the CPU array (pageable memory) */
    float* cpu_array;
    cpu_array = (float*) malloc(SIZE * sizeof(float));
    
    /* Allocate the GPU array */
    float* gpu_array;
    cudaMalloc((void**) &gpu_array, SIZE * sizeof(float));
    
    /* Starting time of GPU kernel execution */
    clock_t t1 = clock();
    
    /* Randomize the GPU array */
    randomArray<<<gridsize, blocksize>>>(gpu_array, SIZE, seed);
    
    /* Synchronize GPU cores */
    cudaDeviceSynchronize();
    
    /* Ending time of GPU kernel execution */
    clock_t t2 = clock();
    
    /* Elapsed time of GPU kernel execution */
    double t2_t1 = (t2-t1)/(double) CLOCKS_PER_SEC;
    printf("Elapsed time of GPU kernel execution: %f seconds\n", t2_t1);
    
    /* Starting time of the GPU-to-CPU data transfer */
    clock_t t3_1 = clock();
    
    /* Copy the GPU array to its CPU counterpart */
    cudaMemcpy(cpu_array, gpu_array, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    
    /* Ending time of GPU-to-CPU data transfer */
    clock_t t4_1 = clock();
    
    /* Elapsed time of GPU-to-CPU data transfer */
    double t4_t3_1 = (t4_1-t3_1)/(double) CLOCKS_PER_SEC;
    printf("Elapsed Time of the GPU-to-CPU data transfer: %f seconds\n", t4_t3_1);
    
    /* Calculate and print total time */
    double total_pageable = (t4_1-t0)/(double) CLOCKS_PER_SEC;
    printf("Total elapsed time: %f seconds\n", total_pageable);
    
    /* Print first few elements for verification */
    printf("First 5 elements: %f %f %f %f %f\n", 
           cpu_array[0], cpu_array[1], cpu_array[2], cpu_array[3], cpu_array[4]);
    
    /* Deallocate the GPU array */
    cudaFree(gpu_array);
    
    /* Deallocate the CPU array */
    free(cpu_array);
    
    //--------------------------------------------------------------------------
    // 2. PINNED MEMORY ACCESS
    //--------------------------------------------------------------------------
    printf("\n\n======= PINNED MEMORY ACCESS =======\n");
    
    /* Starting time */
    clock_t tt0 = clock();
    
    /* Allocate the CPU array with pinned memory */
    cudaMallocHost((void **)&cpu_array, SIZE * sizeof(float));
    
    /* Allocate the GPU array */
    cudaMalloc((void**) &gpu_array, SIZE * sizeof(float));
    
    /* Starting time of GPU kernel execution */
    clock_t tt1 = clock();
    
    /* Randomize the GPU array */
    randomArray<<<gridsize, blocksize>>>(gpu_array, SIZE, seed);
    
    /* Synchronize GPU cores */
    cudaDeviceSynchronize();
    
    /* Ending time of GPU kernel execution */
    clock_t tt2 = clock();
    
    /* Elapsed time of GPU kernel execution */
    double tt2_tt1 = (tt2-tt1)/(double) CLOCKS_PER_SEC;
    printf("Elapsed time of GPU kernel execution: %f seconds\n", tt2_tt1);
    
    /* Starting time of the GPU-to-CPU data transfer */
    clock_t tt3_1 = clock();
    
    /* Copy the GPU array to its CPU counterpart */
    cudaMemcpy(cpu_array, gpu_array, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    
    /* Ending time of GPU-to-CPU data transfer */
    clock_t tt4_1 = clock();
    
    /* Elapsed time of GPU-to-CPU data transfer */
    double tt4_tt3_1 = (tt4_1-tt3_1)/(double) CLOCKS_PER_SEC;
    printf("Elapsed Time of the GPU-to-CPU data transfer: %f seconds\n", tt4_tt3_1);
    
    /* Calculate and print total time */
    double total_pinned = (tt4_1-tt0)/(double) CLOCKS_PER_SEC;
    printf("Total elapsed time: %f seconds\n", total_pinned);
    
    /* Print first few elements for verification */
    printf("First 5 elements: %f %f %f %f %f\n", 
           cpu_array[0], cpu_array[1], cpu_array[2], cpu_array[3], cpu_array[4]);
    
    /* Deallocate the GPU array */
    cudaFree(gpu_array);
    
    /* Deallocate the CPU array */
    cudaFreeHost(cpu_array);
    
    //--------------------------------------------------------------------------
    // 3. ZERO-COPY CUDA MEMORY ACCESS (Mapped Memory)
    //--------------------------------------------------------------------------
    printf("\n\n======= ZERO-COPY CUDA MEMORY ACCESS (Mapped) =======\n");
    
    /* Starting time */
    clock_t t5_start = clock();
    
    /* Allocate the host array with mapped memory */
    float* host_array = NULL;
    cudaHostAlloc((void**) &host_array, SIZE*sizeof(float), cudaHostAllocMapped);
    
    /* Assign the device pointer to the host array */
    float* device_array;
    cudaHostGetDevicePointer((void **) &device_array, (void *) host_array, 0);
    
    /* Starting time of GPU kernel execution */
    clock_t t5 = clock();
    
    /* Randomize the device array */
    randomArray<<<gridsize, blocksize>>>(device_array, SIZE, seed);
    
    /* Synchronize GPU cores */
    cudaDeviceSynchronize();
    
    /* Ending time of GPU kernel execution */
    clock_t t6 = clock();
    
    /* Elapsed time of GPU kernel execution */
    double t6_t5 = (t6-t5)/(double) CLOCKS_PER_SEC;
    printf("Elapsed Time of GPU kernel execution: %f seconds\n", t6_t5);
    
    /* Calculate and print total time */
    double total_mapped = (t6-t5_start)/(double) CLOCKS_PER_SEC;
    printf("Total elapsed time: %f seconds\n", total_mapped);
    
    /* Print first few elements for verification */
    printf("First 5 elements: %f %f %f %f %f\n", 
           host_array[0], host_array[1], host_array[2], host_array[3], host_array[4]);
    
    /* Deallocate the host array (device array is automatically freed) */
    cudaFreeHost(host_array);
    
    //--------------------------------------------------------------------------
    // 4. UNIFIED MEMORY ACCESS
    //--------------------------------------------------------------------------
    printf("\n\n======= UNIFIED MEMORY ACCESS =======\n");
    
    /* Starting time */
    clock_t t7_start = clock();
    
    /* Allocate unified memory */
    float* unified_array;
    cudaMallocManaged(&unified_array, SIZE * sizeof(float));
    
    /* Starting time of GPU kernel execution */
    clock_t t7 = clock();
    
    /* Randomize the array directly using unified memory */
    randomArray<<<gridsize, blocksize>>>(unified_array, SIZE, seed);
    
    /* Synchronize GPU cores */
    cudaDeviceSynchronize();
    
    /* Ending time of GPU kernel execution */
    clock_t t8 = clock();
    
    /* Elapsed time of GPU kernel execution */
    double t8_t7 = (t8-t7)/(double) CLOCKS_PER_SEC;
    printf("Elapsed Time of GPU kernel execution: %f seconds\n", t8_t7);
    
    /* Calculate and print total time */
    double total_unified = (t8-t7_start)/(double) CLOCKS_PER_SEC;
    printf("Total elapsed time: %f seconds\n", total_unified);
    
    /* Print first few elements for verification */
    printf("First 5 elements: %f %f %f %f %f\n", 
           unified_array[0], unified_array[1], unified_array[2], unified_array[3], unified_array[4]);
    
    /* Deallocate unified memory */
    cudaFree(unified_array);
    
    //--------------------------------------------------------------------------
    // SUMMARY
    //--------------------------------------------------------------------------
    printf("\n\n======= MEMORY ACCESS PERFORMANCE SUMMARY =======\n");
    printf("Pageable Memory:  %.6f seconds (baseline)\n", total_pageable);
    printf("Pinned Memory:    %.6f seconds (%.2fx vs pageable)\n", 
           total_pinned, total_pageable/total_pinned);
    printf("Mapped Memory:    %.6f seconds (%.2fx vs pageable)\n", 
           total_mapped, total_pageable/total_mapped);
    printf("Unified Memory:   %.6f seconds (%.2fx vs pageable)\n", 
           total_unified, total_pageable/total_unified);
    
    printf("\nGPU Kernel execution times only:\n");
    printf("Pageable:  %.6f seconds\n", t2_t1);
    printf("Pinned:    %.6f seconds\n", tt2_tt1);
    printf("Mapped:    %.6f seconds\n", t6_t5);
    printf("Unified:   %.6f seconds\n", t8_t7);
    
    return 0;
}
