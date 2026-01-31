"""Run all programs: matrix multiplication, convolution, and Python CUDA library tests."""
import os
import subprocess
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
py_dir = os.path.join(root_dir, "py")
os.chdir(root_dir)

def run_py(script_path, *args):
    cmd = [sys.executable, script_path] + list(args)
    subprocess.run(cmd, cwd=root_dir)

# 1. Matrix multiplication (CPU, naïve CUDA, tiled CUDA, cuBLAS)
print("=" * 60)
run_py(os.path.join(script_dir, "run_matrix_tests.py"))
print()

# 2. Convolution (CPU and GPU)
print("=" * 60)
run_py(os.path.join(script_dir, "run_convolution_tests.py"))
print()

# 3. Python + CUDA shared library (matrix multiply and convolution)
print("=" * 60)
print("=== PYTHON + CUDA LIBRARY ===\n")
print("--- Matrix multiply via lib ---")
run_py(os.path.join(py_dir, "call_matrix_gpu.py"))
print("\n--- Convolution via lib (example) ---")
run_py(os.path.join(py_dir, "call_convolution_gpu.py"), "512", "3", "sobel_x")
print()

print("=" * 60)
print("All runs finished.")
