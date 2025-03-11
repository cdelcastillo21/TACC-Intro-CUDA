# Lab 2: CUDA 2D Blocks and Matrix Operations

This lab demonstrates how to use 2D thread and block configurations in CUDA to perform matrix operations. We'll implement and compare matrix addition and matrix transposition using 2D grid structures.

## Code Overview

The provided code (`matrix_operations.cu`) implements two CUDA kernels:
1. `matrixAdd` - Adds two matrices element by element
2. `matrixTranspose` - Transposes a matrix (rows become columns, columns become rows)

### Key Concepts

#### 2D Thread and Block Organization
- The code uses 2D blocks (`dim3 threadsPerBlock(8, 8)`) with 8×8 threads per block
- The grid is also 2D, with dimensions calculated based on the matrix size
- Each thread handles one element of the matrix

#### Thread Indexing
- Global thread indices are calculated using both dimensions:
  ```c
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  ```
- The linear memory index is then computed: `idx = row * width + col`

#### Matrix Transposition
- The transposition kernel shows how to map between 2D and 1D indices
- For transposition, the input and output indices are swapped:
  ```c
  int in_idx = row * width + col;
  int out_idx = col * height + row;
  ```

## Code Breakdown

### Kernel Functions

#### Matrix Addition
```c
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
```

#### Matrix Transposition
```c
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
```

### Main Function Logic

1. **Setup**:
   - Define matrix dimensions (WIDTH × HEIGHT)
   - Allocate host and device memory
   - Initialize matrices with sample data

2. **Configure Thread and Block Dimensions**:
   ```c
   dim3 threadsPerBlock(8, 8);  // 8x8 threads per block = 64 threads
   dim3 blocksPerGrid(
       (WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
       (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y
   );
   ```

3. **Execute Kernels**:
   - Launch the matrix addition kernel
   - Launch the matrix transposition kernel
   - Measure and report execution time for each

4. **Verify Results**:
   - Check that addition was performed correctly
   - Check that transposition was performed correctly
   - Print matrices (if small enough) for visual verification

5. **Cleanup**:
   - Free device and host memory
   - Destroy CUDA timing events

## How to Compile and Run

### Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C compiler compatible with CUDA
- NVIDIA HPC SDK (for Fortran version)

### Compilation

#### C Version
```bash
nvcc -o matrix_operations matrix_operations.cu
```

#### Fortran Version
```bash
nvfortran -o matrix_operations_fortran matrix_operations.cuf
```

### Running the Program

#### C Version
```bash
./matrix_operations
```

#### Fortran Version
```bash
./matrix_operations_fortran
```

### Expected Output

The program will display:
1. Matrix dimensions and thread configuration
2. Execution time for each operation
3. The input matrices, result of addition, and transposed matrix (if small enough)
4. Verification results

## Experiment Ideas

1. Try different block sizes (e.g., 4×4, 16×16, 32×8) and observe performance differences
2. Increase matrix dimensions and observe scaling behavior
3. Try implementing element-wise multiplication as a new kernel
4. Experiment with non-square matrices to understand the impact on block/grid configuration

## Common Issues

- **Out of Bounds Access**: If thread indices exceed matrix dimensions, make sure boundary checks are correct
- **Incorrect Transposition**: Double-check the index mapping for transposition
- **Performance Issues**: Try different block sizes for optimal performance

## Understanding Thread Mapping

For a better intuition of how threads map to matrix elements:
- Block (0,0), Thread (0,0) processes element (0,0)
- Block (0,0), Thread (1,0) processes element (0,1)
- Block (0,0), Thread (0,1) processes element (1,0)
- Block (1,0), Thread (0,0) processes element (0,8) (assuming 8×8 blocks)
