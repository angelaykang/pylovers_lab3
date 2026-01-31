# GPU-Accelerated Matrix & Image Operations

CPU and CUDA implementations for matrix multiplication and image convolution, with Python bindings via a shared library.

## Project structure

```
├── src/
│   ├── cpu/              # CPU implementations
│   │   ├── matrix_mult_cpu.c
│   │   ├── matrix_mult_interactive.c
│   │   └── convolution_cpu.c
│   ├── cuda/              # CUDA executables
│   │   ├── matrix_mult_naive.cu
│   │   ├── matrix_mult_tiled.cu
│   │   ├── matrix_mult_cublas.cu
│   │   ├── matrix_compare.cu
│   │   ├── convolution_gpu.cu
│   │   └── matrix_gpu_python.cu
│   └── lib/               # Shared library for Python
│       └── matrix_lib.cu
├── py/                    # Python scripts
│   ├── call_matrix_gpu.py
│   ├── call_convolution_gpu.py
│   └── create_test_images.py
├── scripts/               # Build and run scripts
│   ├── build_all.bat
│   ├── build_cpu.bat
│   ├── run_matrix_tests.py
│   ├── run_convolution_tests.py
│   └── run_everything.py
└── samples/               # PGM test images
```

## Requirements

- **Windows:** Visual Studio (e.g. 2022) with x64 Native Tools, [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- **Python:** Python 3 with `numpy` (for `py/` scripts)
- **Linux:** `gcc`, `nvcc`, `numpy`; build commands use `.so` instead of `.dll`

## Build

From the **repo root**, in an **x64 Native Tools Command Prompt** (and with CUDA in `PATH`):

```bat
scripts\build_all.bat
```

This builds:

- `matrix_cpu.exe` — CPU matrix multiply  
- `convolution.exe` — CPU convolution  
- `matrix_gpu.exe` — naïve CUDA matrix multiply  
- `matrix_gpu_tiled.exe` — tiled CUDA matrix multiply  
- `matrix_cublas.exe` — cuBLAS matrix multiply  
- `convolution_gpu.exe` — CUDA convolution  
- `matrix_lib.dll` — shared library for Python  

CPU-only build:

```bat
scripts\build_cpu.bat
```

Executables and `matrix_lib.dll` are created in the repo root.

## Run

From the repo root.

**Matrix multiplication (single size):**

```bat
matrix_cpu.exe 1024
matrix_gpu.exe 1024
matrix_gpu_tiled.exe 1024
matrix_cublas.exe 1024
```

**Matrix benchmarks (512, 1024, 2048):**

```bat
python scripts\run_matrix_tests.py
```

**Convolution (image size M, filter size N, filter type):**

```bat
convolution.exe 512 3 sobel_x
convolution_gpu.exe 512 3 sobel_x
```

Filter types: `sobel_x`, `sobel_y`, `gaussian`, `sharpen`, `laplacian`.

**Convolution benchmarks:**

```bat
python scripts\run_convolution_tests.py
```

**Python + CUDA library:**

```bat
python py\call_matrix_gpu.py
python py\call_convolution_gpu.py 512 3 sobel_x
```

**Run all matrix, convolution, and Python tests:**

```bat
python scripts\run_everything.py
```
