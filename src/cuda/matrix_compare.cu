#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// CUDA kernel for matrix multiplication (naive)
// Each thread computes one element of the output matrix
__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CUDA kernel for matrix multiplication with shared memory tiling
#define TILE_WIDTH 16
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0.0;
    
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;
        
        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        
        __syncthreads();
    }
    
    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

// Function to initialize a matrix with values
void initializeMatrix(int n, float *matrix) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = (float)(i * n + j + 1);
        }
    }
}

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

float runKernel(int N, int kernelType) {
    // Allocate host memory
    size_t matrixSize = N * N * sizeof(float);
    float *h_A = (float *)malloc(matrixSize);
    float *h_B = (float *)malloc(matrixSize);
    float *h_C = (float *)malloc(matrixSize);
    
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        return -1.0f;
    }
    
    // Initialize matrices
    initializeMatrix(N, h_A);
    initializeMatrix(N, h_B);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, matrixSize));
    CUDA_CHECK(cudaMalloc((void **)&d_B, matrixSize));
    CUDA_CHECK(cudaMalloc((void **)&d_C, matrixSize));
    
    // Copy matrices from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    int blockSize = 16;
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Record start event
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch kernel based on type
    if (kernelType == 0) {
        // Naive kernel
        matrixMultiplyGPU<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    } else {
        // Tiled kernel
        matrixMultiplyTiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Record stop event and wait for completion
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    
    return milliseconds;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <matrix_size> <kernel_type>\n", argv[0]);
        fprintf(stderr, "  kernel_type: 0 = naive, 1 = tiled\n");
        fprintf(stderr, "Example: %s 512 0\n", argv[0]);
        return 1;
    }
    
    int N = atoi(argv[1]);
    int kernelType = atoi(argv[2]);
    
    if (N <= 0) {
        fprintf(stderr, "Error: Matrix size must be a positive integer\n");
        return 1;
    }
    
    if (kernelType != 0 && kernelType != 1) {
        fprintf(stderr, "Error: Kernel type must be 0 (naive) or 1 (tiled)\n");
        return 1;
    }
    
    // Check for CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found\n");
        return 1;
    }
    
    // Run kernel and get time
    float time_ms = runKernel(N, kernelType);
    
    if (time_ms < 0) {
        return 1;
    }
    
    // Output in format suitable for table
    const char *kernelName = (kernelType == 0) ? "Naive" : "Tiled";
    printf("%s CUDA: %.3f ms\n", kernelName, time_ms);
    
    return 0;
}
