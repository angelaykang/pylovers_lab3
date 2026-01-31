# Performance Comparison Table

**Hardware:** NVIDIA GeForce RTX 3060

## Matrix Multiplication Performance Results

| Implementation | N=512 | N=1024 | N=2048 |
|----------------|-------|--------|--------|
| CPU (C)        | 0.196 sec | 1.374 sec | 57.895 sec |
| Naïve CUDA     | 0.853 ms | 2.707 ms | 21.349 ms |
| Optimized CUDA | 0.405 ms | 2.084 ms | 19.510 ms |
| cuBLAS         | 1.644 ms | 0.538 ms | 2.321 ms |

## Speedup (CPU time / GPU time)

| Matrix Size | Naïve CUDA Speedup | Optimized CUDA Speedup | cuBLAS Speedup |
|-------------|-------------------|------------------------|----------------|
| 512         | 230×              | 484×                   | 119×           |
| 1024        | 507×              | 660×                   | 2554×          |
| 2048        | 2714×             | 2970×                  | 24,941×        |

---

## Convolution Performance Results (filter N=3×3)

Image size M×M, 5 filter types (sobel_x, sobel_y, gaussian, sharpen, laplacian). Times below are per run; GPU times are typical (filter type has little effect).

| Implementation | M=256 | M=512 | M=1024 |
|----------------|-------|-------|--------|
| CPU (C)        | 0.001 sec | 0.004 sec | ~0.018 sec |
| CUDA (GPU)     | ~0.066 ms | ~0.072 ms | ~0.110 ms |

### Convolution speedup (CPU / GPU)

| Image size | Speedup |
|------------|---------|
| 256×256    | ~15×    |
| 512×512    | ~55×    |
| 1024×1024  | ~163×   |

---

## Notes

- CPU time in seconds, GPU time in ms. Speedup: `CPU_time_seconds / (GPU_time_ms / 1000.0)`.
