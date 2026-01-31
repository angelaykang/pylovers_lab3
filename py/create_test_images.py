"""
Python script to create test images in PGM format for convolution testing.
PGM (P5) is a simple grayscale image format that can be read by the C program.
"""

import numpy as np
import struct

def write_pgm(filename, image_data, max_value=255):
    """
    Write a PGM (P5) image file.
    
    Args:
        filename: Output filename
        image_data: 2D numpy array (grayscale, uint8)
        max_value: Maximum pixel value (default 255)
    """
    height, width = image_data.shape
    
    with open(filename, 'wb') as f:
        # Write header
        f.write(b'P5\n')
        f.write(f'{width} {height}\n'.encode())
        f.write(f'{max_value}\n'.encode())
        
        # Write binary data
        f.write(image_data.tobytes())

def create_gradient_image(size, filename):
    """Create a gradient test image."""
    img = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            img[y, x] = ((x + y) % 256)
    write_pgm(filename, img)
    print(f"Created gradient image: {filename} ({size}x{size})")

def create_checkerboard_image(size, filename, square_size=32):
    """Create a checkerboard pattern."""
    img = np.zeros((size, size), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            square_x = x // square_size
            square_y = y // square_size
            if (square_x + square_y) % 2 == 0:
                img[y, x] = 255
            else:
                img[y, x] = 0
    write_pgm(filename, img)
    print(f"Created checkerboard image: {filename} ({size}x{size})")

def create_circle_image(size, filename):
    """Create an image with circles."""
    img = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            dist = int(np.sqrt(dx*dx + dy*dy))
            img[y, x] = (dist % 256)
    write_pgm(filename, img)
    print(f"Created circle image: {filename} ({size}x{size})")

def create_random_image(size, filename):
    """Create a random noise image."""
    img = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    write_pgm(filename, img)
    print(f"Created random image: {filename} ({size}x{size})")

if __name__ == "__main__":
    # Create test images of different sizes
    sizes = [256, 512, 1024]
    
    for size in sizes:
        create_gradient_image(size, f"test_gradient_{size}.pgm")
        create_checkerboard_image(size, f"test_checkerboard_{size}.pgm")
        create_circle_image(size, f"test_circle_{size}.pgm")
        create_random_image(size, f"test_random_{size}.pgm")
    
    print("\nDone.")
