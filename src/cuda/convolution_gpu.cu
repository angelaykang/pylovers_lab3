#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

// CUDA kernel for convolution
// Each thread computes one pixel of the output image
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

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <image_size_M> <filter_size_N> <filter_type>\n", argv[0]);
        fprintf(stderr, "  image_size_M: Size of input image (M x M)\n");
        fprintf(stderr, "  filter_size_N: Size of filter kernel (N x N)\n");
        fprintf(stderr, "  filter_type: sobel_x, sobel_y, gaussian, sharpen, laplacian\n");
        fprintf(stderr, "\nExample: %s 512 3 sobel_x\n", argv[0]);
        return 1;
    }
    
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    char *filter_type = argv[3];
    
    if (M <= 0 || N <= 0 || N % 2 == 0) {
        fprintf(stderr, "Error: Image size M and filter size N must be positive, and N must be odd\n");
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
    printf("Image size: %d x %d\n", M, M);
    printf("Filter size: %d x %d\n", N, N);
    printf("Filter type: %s\n\n", filter_type);
    
    // Allocate filter
    float *h_filter = (float*)malloc(N * N * sizeof(float));
    
    // Create filter based on type (simplified - you can expand this)
    if (strcmp(filter_type, "sobel_x") == 0) {
        float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        for (int i = 0; i < N * N && i < 9; i++) {
            h_filter[i] = sobel_x[i];
        }
    } else if (strcmp(filter_type, "sobel_y") == 0) {
        float sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
        for (int i = 0; i < N * N && i < 9; i++) {
            h_filter[i] = sobel_y[i];
        }
    } else if (strcmp(filter_type, "gaussian") == 0) {
        float sigma = N / 3.0f;
        float sum = 0.0f;
        int center = N / 2;
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                float dx = x - center;
                float dy = y - center;
                float value = expf(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                h_filter[y * N + x] = value;
                sum += value;
            }
        }
        for (int i = 0; i < N * N; i++) {
            h_filter[i] /= sum;
        }
    } else if (strcmp(filter_type, "sharpen") == 0) {
        float sharpen[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
        for (int i = 0; i < N * N && i < 9; i++) {
            h_filter[i] = sharpen[i];
        }
    } else if (strcmp(filter_type, "laplacian") == 0) {
        float laplacian[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
        for (int i = 0; i < N * N && i < 9; i++) {
            h_filter[i] = laplacian[i];
        }
    } else {
        fprintf(stderr, "Error: Unknown filter type: %s\n", filter_type);
        free(h_filter);
        return 1;
    }
    
    // Allocate host memory for image
    size_t imageSize = M * M * sizeof(unsigned int);
    size_t filterSize = N * N * sizeof(float);
    
    unsigned int *h_image = (unsigned int*)malloc(imageSize);
    unsigned int *h_result = (unsigned int*)malloc(imageSize);
    
    // Initialize test image (gradient pattern)
    for (int i = 0; i < M * M; i++) {
        h_image[i] = (unsigned int)(i % 256);
    }
    
    // Allocate device memory
    unsigned int *d_image, *d_result;
    float *d_filter;
    
    CUDA_CHECK(cudaMalloc((void **)&d_image, imageSize));
    CUDA_CHECK(cudaMalloc((void **)&d_result, imageSize));
    CUDA_CHECK(cudaMalloc((void **)&d_filter, filterSize));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_image, h_image, imageSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    int blockSize = 16;
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((M + blockSize - 1) / blockSize, (M + blockSize - 1) / blockSize);
    
    printf("Launch configuration: %dx%d blocks, %dx%d threads per block\n", 
           numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Record start event
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch kernel
    convolutionGPU<<<numBlocks, threadsPerBlock>>>(d_image, d_filter, d_result, M, N);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Record stop event and wait for completion
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_result, d_result, imageSize, cudaMemcpyDeviceToHost));
    
    // Output results
    printf("Execution time: %.6f seconds (%.3f ms)\n", milliseconds / 1000.0f, milliseconds);
    printf("Result pixel [0][0]: %u\n", h_result[0]);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_filter));
    free(h_image);
    free(h_result);
    free(h_filter);
    
    return 0;
}
