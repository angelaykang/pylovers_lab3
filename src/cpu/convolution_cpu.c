#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Structure to hold image data
typedef struct {
    unsigned int *data;
    int width;
    int height;
    int max_value;
} Image;

// Function to allocate image
Image* allocate_image(int width, int height) {
    Image *img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->max_value = 255;
    img->data = (unsigned int*)malloc(width * height * sizeof(unsigned int));
    return img;
}

// Function to free image
void free_image(Image *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

// Function to read PGM image (P5 format - binary)
Image* read_pgm(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }
    
    char magic[3];
    if (fscanf(file, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: Not a valid PGM file (P5 format required)\n");
        fclose(file);
        return NULL;
    }
    
    int width, height, max_value;
    if (fscanf(file, "%d %d %d", &width, &height, &max_value) != 3) {
        fprintf(stderr, "Error: Invalid PGM header\n");
        fclose(file);
        return NULL;
    }
    
    fgetc(file); // Skip newline
    
    Image *img = allocate_image(width, height);
    img->max_value = max_value;
    
    // Read binary data
    unsigned char *buffer = (unsigned char*)malloc(width * height);
    fread(buffer, 1, width * height, file);
    fclose(file);
    
    // Convert to unsigned int
    for (int i = 0; i < width * height; i++) {
        img->data[i] = (unsigned int)buffer[i];
    }
    
    free(buffer);
    return img;
}

// Function to write PGM image
int write_pgm(const char *filename, Image *img) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return 0;
    }
    
    fprintf(file, "P5\n%d %d\n%d\n", img->width, img->height, img->max_value);
    
    // Write binary data
    for (int i = 0; i < img->width * img->height; i++) {
        unsigned char val = (unsigned char)(img->data[i] > 255 ? 255 : img->data[i]);
        fwrite(&val, 1, 1, file);
    }
    
    fclose(file);
    return 1;
}

// Function to create a test image (gradient pattern)
Image* create_test_image(int width, int height) {
    Image *img = allocate_image(width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Create a checkerboard/gradient pattern
            int value = ((x + y) % 256);
            img->data[y * width + x] = value;
        }
    }
    return img;
}

// Convolution function
// Applies N×N filter to M×M image
// filter: N×N filter matrix
// image: M×M input image
// result: M×M output image
void convolution(Image *image, float *filter, int filter_size, Image *result) {
    int M = image->width;  // Assuming square image
    int N = filter_size;
    int pad = N / 2;
    
    // Initialize result
    for (int i = 0; i < M * M; i++) {
        result->data[i] = 0;
    }
    
    // Apply convolution
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            float sum = 0.0f;
            
            for (int fy = 0; fy < N; fy++) {
                for (int fx = 0; fx < N; fx++) {
                    int img_y = y + fy - pad;
                    int img_x = x + fx - pad;
                    
                    // Handle boundary: use zero-padding
                    if (img_y >= 0 && img_y < M && img_x >= 0 && img_x < M) {
                        sum += image->data[img_y * M + img_x] * filter[fy * N + fx];
                    }
                }
            }
            
            // Clamp result to valid range
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            result->data[y * M + x] = (unsigned int)sum;
        }
    }
}

// Edge detection filter (Sobel operator - horizontal)
void create_sobel_x_filter(float *filter, int size) {
    // 3x3 Sobel X filter
    float sobel_x[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    
    for (int i = 0; i < size * size; i++) {
        if (i < 9) {
            filter[i] = sobel_x[i];
        } else {
            filter[i] = 0.0f;
        }
    }
}

// Edge detection filter (Sobel operator - vertical)
void create_sobel_y_filter(float *filter, int size) {
    // 3x3 Sobel Y filter
    float sobel_y[9] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };
    
    for (int i = 0; i < size * size; i++) {
        if (i < 9) {
            filter[i] = sobel_y[i];
        } else {
            filter[i] = 0.0f;
        }
    }
}

// Gaussian blur filter
void create_gaussian_blur_filter(float *filter, int size) {
    float sigma = size / 3.0f;
    float sum = 0.0f;
    int center = size / 2;
    
    // Create Gaussian kernel
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - center;
            float dy = y - center;
            float value = exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
            filter[y * size + x] = value;
            sum += value;
        }
    }
    
    // Normalize
    for (int i = 0; i < size * size; i++) {
        filter[i] /= sum;
    }
}

// Sharpen filter
void create_sharpen_filter(float *filter, int size) {
    // 3x3 sharpen filter
    float sharpen[9] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };
    
    for (int i = 0; i < size * size; i++) {
        if (i < 9) {
            filter[i] = sharpen[i];
        } else {
            filter[i] = 0.0f;
        }
    }
}

// Laplacian edge detection filter
void create_laplacian_filter(float *filter, int size) {
    // 3x3 Laplacian filter
    float laplacian[9] = {
         0, -1,  0,
        -1,  4, -1,
         0, -1,  0
    };
    
    for (int i = 0; i < size * size; i++) {
        if (i < 9) {
            filter[i] = laplacian[i];
        } else {
            filter[i] = 0.0f;
        }
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
    
    // Allocate filter
    float *filter = (float*)malloc(N * N * sizeof(float));
    
    // Create filter based on type
    if (strcmp(filter_type, "sobel_x") == 0) {
        create_sobel_x_filter(filter, N);
        printf("Using Sobel X (horizontal edge detection) filter\n");
    } else if (strcmp(filter_type, "sobel_y") == 0) {
        create_sobel_y_filter(filter, N);
        printf("Using Sobel Y (vertical edge detection) filter\n");
    } else if (strcmp(filter_type, "gaussian") == 0) {
        create_gaussian_blur_filter(filter, N);
        printf("Using Gaussian blur filter\n");
    } else if (strcmp(filter_type, "sharpen") == 0) {
        create_sharpen_filter(filter, N);
        printf("Using Sharpen filter\n");
    } else if (strcmp(filter_type, "laplacian") == 0) {
        create_laplacian_filter(filter, N);
        printf("Using Laplacian edge detection filter\n");
    } else {
        fprintf(stderr, "Error: Unknown filter type: %s\n", filter_type);
        free(filter);
        return 1;
    }
    
    // Create or load test image
    Image *input_image = create_test_image(M, M);
    Image *output_image = allocate_image(M, M);
    
    printf("Image size: %d x %d\n", M, M);
    printf("Filter size: %d x %d\n", N, N);
    
    // Measure execution time
    clock_t start = clock();
    convolution(input_image, filter, N, output_image);
    clock_t end = clock();
    
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Convolution completed in %.6f seconds\n", cpu_time_used);
    
    // Save output image
    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "output_%s_M%d_N%d.pgm", 
             filter_type, M, N);
    write_pgm(output_filename, output_image);
    printf("Output saved to: %s\n", output_filename);
    
    // Save input image for reference
    char input_filename[256];
    snprintf(input_filename, sizeof(input_filename), "input_M%d.pgm", M);
    write_pgm(input_filename, input_image);
    printf("Input saved to: %s\n", input_filename);
    
    // Cleanup
    free(filter);
    free_image(input_image);
    free_image(output_image);
    
    return 0;
}
