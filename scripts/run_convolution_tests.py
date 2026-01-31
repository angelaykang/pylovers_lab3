"""Run CPU and GPU convolution programs for several image sizes and filter types."""
import os
import subprocess
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
is_windows = sys.platform.startswith("win")
ext = ".exe" if is_windows else ""

image_sizes = [256, 512, 1024]
filter_sizes = [3]
filter_types = ["sobel_x", "sobel_y", "gaussian", "sharpen", "laplacian"]

def exe(name):
    return name + ext

def run(cmd):
    exe_path = os.path.join(root_dir, cmd[0])
    if not os.path.isfile(exe_path):
        print(f"(Skipping: {cmd[0]} not found. Build it first.)")
        return
    full_cmd = [exe_path] + list(cmd[1:])
    subprocess.run(full_cmd, cwd=root_dir)

print("=== CONVOLUTION ===\n")

print("--- CPU Convolution ---")
for m in image_sizes:
    for n in filter_sizes:
        for f in filter_types:
            run([exe("convolution"), str(m), str(n), f])

print("\n--- GPU Convolution ---")
for m in image_sizes:
    for n in filter_sizes:
        for f in filter_types:
            run([exe("convolution_gpu"), str(m), str(n), f])

print("\nConvolution tests done.")
