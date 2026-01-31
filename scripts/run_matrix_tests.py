"""Run all matrix multiplication programs (CPU, naïve CUDA, tiled CUDA, cuBLAS)."""
import os
import subprocess
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
is_windows = sys.platform.startswith("win")
ext = ".exe" if is_windows else ""

# Matrix sizes to test
sizes = [512, 1024, 2048]

def exe(name):
    return name + ext

def path_to(name):
    return os.path.join(root_dir, exe(name))

def has_exe(name):
    return os.path.isfile(path_to(name))

def run(cmd):
    exe_path = os.path.join(root_dir, cmd[0])
    if not os.path.isfile(exe_path):
        print(f"(Skipping: {cmd[0]} not found. Build it first.)")
        return
    full_cmd = [exe_path] + list(cmd[1:])
    subprocess.run(full_cmd, cwd=root_dir)

# Use standalone matrix_gpu / matrix_gpu_tiled if present, else matrix_gpu_compare
use_naive_standalone = has_exe("matrix_gpu")
use_tiled_standalone = has_exe("matrix_gpu_tiled")

print("=== MATRIX MULTIPLICATION ===\n")

print("--- CPU ---")
for n in sizes:
    run([exe("matrix_cpu"), str(n)])

print("\n--- Naïve CUDA ---")
for n in sizes:
    if use_naive_standalone:
        run([exe("matrix_gpu"), str(n)])
    else:
        run([exe("matrix_gpu_compare"), str(n), "0"])

print("\n--- Optimized (tiled) CUDA ---")
for n in sizes:
    if use_tiled_standalone:
        run([exe("matrix_gpu_tiled"), str(n)])
    else:
        run([exe("matrix_gpu_compare"), str(n), "1"])

print("\n--- cuBLAS ---")
for n in sizes:
    run([exe("matrix_cublas"), str(n)])

print("\nMatrix tests done.")
