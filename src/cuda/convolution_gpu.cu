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

// Helper function to compute binomial coefficient C(n, k)
static int binomial_coeff(int n, int k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;

    int result = 1;
    for (int i = 0; i < k; i++) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

// Create Gaussian blur filter
static void create_gaussian_filter(float *filter, int size) {
    float sigma = size / 3.0f;
    float sum = 0.0f;
    int center = size / 2;

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = (float)(x - center);
            float dy = (float)(y - center);
            float value = expf(-(dx * dx + dy * dy) / (2 * sigma * sigma));
            filter[y * size + x] = value;
            sum += value;
        }
    }

    // Normalize
    for (int i = 0; i < size * size; i++) {
        filter[i] /= sum;
    }
}

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
    
    // Create filter based on type using dynamic sizing
    int center = N / 2;
    int n = N - 1;

    if (strcmp(filter_type, "sobel_x") == 0) {
        // Sobel X: Uses binomial coefficients for smoothing and gradient for derivative
        float *smooth = (float*)malloc(N * sizeof(float));
        float *deriv = (float*)malloc(N * sizeof(float));

        for (int i = 0; i < N; i++) {
            smooth[i] = (float)binomial_coeff(n, i);
            deriv[i] = (float)(i - center);
        }

        // Outer product: smooth (column) * deriv (row) for horizontal gradient
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                h_filter[y * N + x] = smooth[y] * deriv[x];
            }
        }

        free(smooth);
        free(deriv);
    } else if (strcmp(filter_type, "sobel_y") == 0) {
        // Sobel Y: Uses binomial coefficients for smoothing and gradient for derivative
        float *smooth = (float*)malloc(N * sizeof(float));
        float *deriv = (float*)malloc(N * sizeof(float));

        for (int i = 0; i < N; i++) {
            smooth[i] = (float)binomial_coeff(n, i);
            deriv[i] = (float)(i - center);
        }

        // Outer product: deriv (column) * smooth (row) for vertical gradient
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                h_filter[y * N + x] = deriv[y] * smooth[x];
            }
        }

        free(smooth);
        free(deriv);
    } else if (strcmp(filter_type, "gaussian") == 0) {
        // Gaussian blur filter
        create_gaussian_filter(h_filter, N);
    } else if (strcmp(filter_type, "sharpen") == 0) {
        // Sharpen filter using unsharp masking: identity + amount * (identity - gaussian)
        float *gaussian = (float*)malloc(N * N * sizeof(float));
        create_gaussian_filter(gaussian, N);

        // Create identity kernel (all zeros except center = 1)
        float *identity = (float*)malloc(N * N * sizeof(float));
        for (int i = 0; i < N * N; i++) {
            identity[i] = 0.0f;
        }
        identity[center * N + center] = 1.0f;

        // Sharpen = identity + amount * (identity - gaussian)
        float amount = 1.0f;
        for (int i = 0; i < N * N; i++) {
            h_filter[i] = identity[i] + amount * (identity[i] - gaussian[i]);
        }

        free(gaussian);
        free(identity);
    } else if (strcmp(filter_type, "laplacian") == 0) {
        // Laplacian of Gaussian (LoG) filter
        float sigma = N / 6.0f;
        float sigma2 = sigma * sigma;

        float sum = 0.0f;
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                float dx = (float)(x - center);
                float dy = (float)(y - center);
                float r2 = dx * dx + dy * dy;

                // Laplacian of Gaussian (LoG) formula
                h_filter[y * N + x] = ((r2 - 2 * sigma2) / (sigma2 * sigma2)) *
                                      expf(-r2 / (2 * sigma2));
                sum += h_filter[y * N + x];
            }
        }

        // Normalize so the sum is zero (characteristic of Laplacian filters)
        float mean = sum / (N * N);
        for (int i = 0; i < N * N; i++) {
            h_filter[i] -= mean;
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
