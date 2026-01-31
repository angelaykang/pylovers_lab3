#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Standalone convolution test runner (links to convolution.c)

extern void convolution_test(int M, int N, const char *filter_type);

int main() {
    printf("=== Convolution Performance Tests ===\n\n");
    
    // Test different image sizes (M) and filter sizes (N)
    int image_sizes[] = {256, 512, 1024};
    int filter_sizes[] = {3, 5, 7};
    const char *filters[] = {"sobel_x", "gaussian", "sharpen"};
    
    printf("| Image Size (M) | Filter Size (N) | Filter Type | Time (seconds) |\n");
    printf("|----------------|-----------------|-------------|----------------|\n");
    
    for (int m_idx = 0; m_idx < 3; m_idx++) {
        int M = image_sizes[m_idx];
        for (int n_idx = 0; n_idx < 3; n_idx++) {
            int N = filter_sizes[n_idx];
            for (int f_idx = 0; f_idx < 3; f_idx++) {
                const char *filter = filters[f_idx];
                
                // Build command
                char cmd[256];
                snprintf(cmd, sizeof(cmd), "convolution.exe %d %d %s", M, N, filter);
                
                printf("Running: %s\n", cmd);
                system(cmd);
                printf("\n");
            }
        }
    }
    
    return 0;
}
