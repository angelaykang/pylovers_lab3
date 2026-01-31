#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

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

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(1); \
        } \
    } while(0)

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        fprintf(stderr, "Example: %s 512\n", argv[0]);
        return 1;
    }
    
    int N = atoi(argv[1]);
    if (N <= 0) {
        fprintf(stderr, "Error: Matrix size must be a positive integer\n");
        return 1;
    }
    
    // Check for CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Matrix size: %d x %d\n\n", N, N);
    
    // Allocate host memory
    size_t matrixSize = N * N * sizeof(float);
    float *h_A = (float *)malloc(matrixSize);
    float *h_B = (float *)malloc(matrixSize);
    float *h_C = (float *)malloc(matrixSize);
    
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        return 1;
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
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Set cuBLAS to use Tensor Cores if available (optional, for better performance)
    // CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    
    // cuBLAS parameters
    // C = alpha * A * B + beta * C
    // We want: C = A * B, so alpha = 1.0, beta = 0.0
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Note: cuBLAS uses column-major order, but we're using row-major
    // To compute C = A * B in row-major, we compute C^T = B^T * A^T in column-major
    // So: cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N)
    // This computes: C^T = B^T * A^T, which is equivalent to C = A * B in row-major
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Record start event
    CUDA_CHECK(cudaEventRecord(start));
    
    // Perform matrix multiplication using cuBLAS
    // C = alpha * op(A) * op(B) + beta * C
    // CUBLAS_OP_N means no transpose
    // We compute: C = A * B (in row-major terms)
    // cuBLAS expects column-major, so we use: C^T = B^T * A^T
    CUBLAS_CHECK(cublasSgemm(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N,
                              &alpha,
                              d_B, N,
                              d_A, N,
                              &beta,
                              d_C, N));
    
    // Record stop event and wait for completion
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost));
    
    // Output results
    printf("Execution time: %.6f seconds (%.3f ms)\n", milliseconds / 1000.0f, milliseconds);
    printf("Result element [0][0]: %.2f\n", h_C[0]);
    
    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
