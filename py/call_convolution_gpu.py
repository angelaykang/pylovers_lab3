import ctypes
import numpy as np
import time
import platform
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if platform.system() == "Windows":
    lib_path = os.path.join(root_dir, "matrix_lib.dll")
elif platform.system() == "Darwin":
    lib_path = os.path.join(root_dir, "libmatrix.dylib")
else:
    lib_path = os.path.join(root_dir, "libmatrix.so")

if not os.path.isfile(lib_path):
    print(f"Library not found: {lib_path}")
    print("Build it with: nvcc -Xcompiler /LD -shared matrix_lib.cu -o matrix_lib.dll  (Windows)")
    print("            or: nvcc -Xcompiler -fPIC -shared matrix_lib.cu -o libmatrix.so   (Linux)")
    lib = None
else:
    try:
        lib = ctypes.cdll.LoadLibrary(lib_path)
        lib.gpu_convolution.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int
        ]
    except AttributeError:
        print("Function gpu_convolution not found in the library.")
        print("Rebuild from matrix_lib.cu: nvcc -Xcompiler /LD -shared matrix_lib.cu -o matrix_lib.dll")
        lib = None

def create_sobel_x_filter(filter_size):
    """Create Sobel X (horizontal edge detection) filter."""
    filter_data = np.zeros((filter_size, filter_size), dtype=np.float32)
    if filter_size >= 3:
        sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)
        filter_data[:3, :3] = sobel_x
    return filter_data

def create_sobel_y_filter(filter_size):
    """Create Sobel Y (vertical edge detection) filter."""
    filter_data = np.zeros((filter_size, filter_size), dtype=np.float32)
    if filter_size >= 3:
        sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=np.float32)
        filter_data[:3, :3] = sobel_y
    return filter_data

def create_gaussian_filter(filter_size):
    """Create Gaussian blur filter."""
    filter_data = np.zeros((filter_size, filter_size), dtype=np.float32)
    sigma = filter_size / 3.0
    center = filter_size // 2
    
    for y in range(filter_size):
        for x in range(filter_size):
            dx = x - center
            dy = y - center
            value = np.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
            filter_data[y, x] = value
    
    # Normalize
    filter_data /= np.sum(filter_data)
    return filter_data

def create_sharpen_filter(filter_size):
    """Create sharpen filter."""
    filter_data = np.zeros((filter_size, filter_size), dtype=np.float32)
    if filter_size >= 3:
        sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        filter_data[:3, :3] = sharpen
    return filter_data

def create_laplacian_filter(filter_size):
    """Create Laplacian edge detection filter."""
    filter_data = np.zeros((filter_size, filter_size), dtype=np.float32)
    if filter_size >= 3:
        laplacian = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]], dtype=np.float32)
        filter_data[:3, :3] = laplacian
    return filter_data

def test_convolution(image_size, filter_size, filter_type="sobel_x"):
    """Test convolution function with given parameters."""
    # Create test image (gradient pattern)
    image = np.zeros((image_size, image_size), dtype=np.uint32)
    for y in range(image_size):
        for x in range(image_size):
            image[y, x] = (x + y) % 256
    
    # Create filter based on type
    if filter_type == "sobel_x":
        filter_data = create_sobel_x_filter(filter_size)
    elif filter_type == "sobel_y":
        filter_data = create_sobel_y_filter(filter_size)
    elif filter_type == "gaussian":
        filter_data = create_gaussian_filter(filter_size)
    elif filter_type == "sharpen":
        filter_data = create_sharpen_filter(filter_size)
    elif filter_type == "laplacian":
        filter_data = create_laplacian_filter(filter_size)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Allocate result
    result = np.zeros((image_size, image_size), dtype=np.uint32)
    
    # Call CUDA convolution
    start = time.time()
    lib.gpu_convolution(image.ravel(), filter_data.ravel(), result.ravel(), 
                       image_size, filter_size)
    end = time.time()
    
    elapsed = end - start
    print(f"Convolution completed in {elapsed:.6f} seconds")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Filter size: {filter_size}x{filter_size}")
    print(f"  Filter type: {filter_type}")
    print(f"  Result pixel [0][0]: {result[0, 0]}")
    
    return elapsed, result

if __name__ == "__main__":
    import sys
    
    if lib is None:
        sys.exit(1)
    if len(sys.argv) < 4:
        print("Usage: python call_convolution.py <image_size> <filter_size> <filter_type>")
        print("  filter_type: sobel_x, sobel_y, gaussian, sharpen, laplacian")
        print("\nExample: python call_convolution.py 512 3 sobel_x")
        sys.exit(1)
    
    image_size = int(sys.argv[1])
    filter_size = int(sys.argv[2])
    filter_type = sys.argv[3]
    
    test_convolution(image_size, filter_size, filter_type)
