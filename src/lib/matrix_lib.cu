#include <cuda_runtime.h>
#include <stdio.h>

#ifdef _WIN32
#define CUDA_LIB_EXPORT __declspec(dllexport)
#else
#define CUDA_LIB_EXPORT
#endif

#define TILE_WIDTH 16

// Matrix multiplication kernel with shared memory tiling
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

// Convolution kernel
__global__ void convolutionGPU(unsigned int *image, float *filter, unsigned int *result, 
                                int image_size, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < image_size && y < image_size) {
        int pad = filter_size / 2;
        float sum = 0.0f;
        
        for (int fy = 0; fy < filter_size; fy++) {
            for (int fx = 0; fx < filter_size; fx++) {
                int img_y = y + fy - pad;
                int img_x = x + fx - pad;
                
                // Zero-padding: if out of bounds, use 0
                unsigned int pixel_value = 0;
                if (img_y >= 0 && img_y < image_size && img_x >= 0 && img_x < image_size) {
                    pixel_value = image[img_y * image_size + img_x];
                }
                
                sum += pixel_value * filter[fy * filter_size + fx];
            }
        }
        
        // Clamp result to valid range [0, 255]
        if (sum < 0) sum = 0;
        if (sum > 255) sum = 255;
        result[y * image_size + x] = (unsigned int)sum;
    }
}

// Exposed C function for matrix multiplication (dllexport needed on Windows for DLL)
extern "C" CUDA_LIB_EXPORT void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;
    
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    
    matrixMultiplyTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Exposed C function for convolution (dllexport needed on Windows for DLL)
extern "C" CUDA_LIB_EXPORT void gpu_convolution(unsigned int *h_image, float *h_filter, unsigned int *h_result, 
                                int image_size, int filter_size) {
    size_t imageSize = image_size * image_size * sizeof(unsigned int);
    size_t filterSize = filter_size * filter_size * sizeof(float);
    
    unsigned int *d_image, *d_result;
    float *d_filter;
    
    cudaMalloc((void**)&d_image, imageSize);
    cudaMalloc((void**)&d_result, imageSize);
    cudaMalloc((void**)&d_filter, filterSize);
    
    cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice);
    
    int blockSize = 16;
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((image_size + blockSize - 1) / blockSize, 
                   (image_size + blockSize - 1) / blockSize);
    
    convolutionGPU<<<numBlocks, threadsPerBlock>>>(d_image, d_filter, d_result, 
                                                    image_size, filter_size);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_result, d_result, imageSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_image);
    cudaFree(d_result);
    cudaFree(d_filter);
}
