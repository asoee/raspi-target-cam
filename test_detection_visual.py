#!/usr/bin/env python3
"""
Visual test of target detection improvements
Shows the actual binary image after morphological operations
"""

import cv2
import numpy as np
from pathlib import Path

# Test image
image_path = Path("data/captures/capture_20251113_171454.jpg")

if not image_path.exists():
    print(f"Image not found: {image_path}")
    exit(1)

# Load the image
frame = cv2.imread(str(image_path))
h, w = frame.shape[:2]
print(f"Image size: {w}x{h}")

# Replicate the exact detection algorithm
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Scale factor
scale_factor = max(w / 640, h / 480)
print(f"Scale factor: {scale_factor:.2f}")

# Apply median blur
blur_size = max(5, int(5 * scale_factor) | 1)
blurred = cv2.medianBlur(gray, blur_size)
print(f"Blur size: {blur_size}")

# Threshold
_, binary = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

# Morphological operations - OPEN
kernel_size = max(5, int(5 * scale_factor))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
binary_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
print(f"Open kernel size: {kernel_size}")

# Morphological operations - CLOSE (new improved version)
close_kernel_size = max(7, int(7 * scale_factor))
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
binary_closed = cv2.morphologyEx(binary_open, cv2.MORPH_CLOSE, close_kernel, iterations=2)
print(f"Close kernel size: {close_kernel_size}, iterations: 2")

# Save stages
output_dir = Path("test_outputs")
output_dir.mkdir(exist_ok=True)

cv2.imwrite(str(output_dir / "stage1_threshold.jpg"), binary)
cv2.imwrite(str(output_dir / "stage2_open.jpg"), binary_open)
cv2.imwrite(str(output_dir / "stage3_close.jpg"), binary_closed)

# Find contours in final binary
contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nContours found: {len(contours)}")

# Analyze contours with NEW thresholds
min_area = int(1000 * scale_factor * scale_factor)
max_area = int(100000 * scale_factor * scale_factor)
min_radius = int(20 * scale_factor)
max_radius = int(400 * scale_factor)

print(f"\nThresholds:")
print(f"  Area: {min_area} - {max_area}")
print(f"  Radius: {min_radius} - {max_radius}")

# Sort by area
contour_areas = [(cv2.contourArea(c), c) for c in contours]
contour_areas.sort(reverse=True, key=lambda x: x[0])

print(f"\nTop 10 contours:")
for i, (area, contour) in enumerate(contour_areas[:10]):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0

    # Check all filter conditions
    passes_area = min_area <= area <= max_area
    passes_radius = min_radius <= radius <= max_radius
    passes_circularity = circularity > 0.5
    passes_bounds = (x >= radius and y >= radius and x + radius <= w and y + radius <= h)

    passes_all = passes_area and passes_radius and passes_circularity and passes_bounds

    print(f"{i+1}. Area: {area:.0f}, Radius: {radius:.0f}, Circ: {circularity:.2f}, "
          f"Passes: {'YES' if passes_all else 'NO'} "
          f"[A:{passes_area} R:{passes_radius} C:{passes_circularity} B:{passes_bounds}]")

# Visualize the best contours
vis_frame = cv2.cvtColor(binary_closed, cv2.COLOR_GRAY2BGR)

for i, (area, contour) in enumerate(contour_areas[:5]):
    color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)][i]
    cv2.drawContours(vis_frame, [contour], -1, color, 3)

    (x, y), radius = cv2.minEnclosingCircle(contour)
    cv2.circle(vis_frame, (int(x), int(y)), int(radius), color, 2)

cv2.imwrite(str(output_dir / "contours_visualization.jpg"), vis_frame)

print(f"\nSaved visualization to {output_dir}/")
