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
else:
    try:
        lib = ctypes.cdll.LoadLibrary(lib_path)
        lib.gpu_matrix_multiply.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_int
        ]
        N = 1024
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
        C = np.zeros((N, N), dtype=np.float32)
        start = time.time()
        lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
        end = time.time()
        print(f"Completed in {end - start:.4f} seconds")
    except AttributeError as e:
        print("Function gpu_matrix_multiply not found in the library.")
        print("Rebuild the library from matrix_lib.cu:")
        print("  Windows: nvcc -Xcompiler /LD -shared matrix_lib.cu -o matrix_lib.dll")
        print("  Linux:   nvcc -Xcompiler -fPIC -shared matrix_lib.cu -o libmatrix.so")
        raise
