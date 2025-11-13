#!/usr/bin/env python3
"""
Debug script to test target detection on captured images
"""

import cv2
import numpy as np
from pathlib import Path
from raspi_target_cam.core.target_detection import TargetDetector

# Test with the specific capture that's not detecting
image_path = Path("data/captures/capture_20251113_171454.jpg")

if not image_path.exists():
    print(f"Image not found: {image_path}")
    exit(1)

print(f"Testing detection on: {image_path}")

# Load the image
frame = cv2.imread(str(image_path))
if frame is None:
    print(f"Failed to load image: {image_path}")
    exit(1)

print(f"Image shape: {frame.shape}")
h, w = frame.shape[:2]
print(f"Image size: {w}x{h}")

# Create detector with debug mode enabled
detector = TargetDetector()
detector.set_debug_mode(True)
detector.set_debug_type("circles")

# Force detection (bypass timing checks)
detector.force_detection()

print("\n=== Running target detection ===")
result = detector.detect_target(frame)

print(f"\nDetection Result:")
print(f"  Detected: {result.detected}")
print(f"  Center: {result.center}")
print(f"  Radius: {result.radius}")
print(f"  Target Type: {result.target_type}")
print(f"  Stable: {result.is_stable}")
print(f"  Confidence: {result.confidence:.2f}")
print(f"  Frames detected: {result.frames_detected}")

# Check what the black circle detection found
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f"\nImage statistics:")
print(f"  Mean intensity: {gray.mean():.1f}")
print(f"  Min intensity: {gray.min()}")
print(f"  Max intensity: {gray.max()}")
print(f"  Std dev: {gray.std():.1f}")

# Test the black circle detection directly
print("\n=== Testing black circle detection ===")
black_circle = detector.detect_black_circle_improved(frame)
print(f"Black circle result: {black_circle}")

if black_circle:
    x, y, radius = black_circle
    black_area = np.pi * (radius ** 2)
    total_area = h * w
    percentage = (black_area / total_area) * 100
    print(f"  Black circle area: {black_area:.0f} pixels")
    print(f"  Percentage of image: {percentage:.2f}%")
    print(f"  Target type threshold: 3.0%")

    # Calculate expected target type
    if percentage > 3.0:
        expected_type = "pistol"
    else:
        expected_type = "rifle"
    print(f"  Expected type: {expected_type}")

# Save debug visualization
output_frame = frame.copy()
if result.detected:
    detector.draw_target_overlay(output_frame, target_info=(result.center[0], result.center[1], result.radius))

output_path = Path("test_outputs/detection_test_overlay.jpg")
output_path.parent.mkdir(exist_ok=True)
cv2.imwrite(str(output_path), output_frame)
print(f"\nSaved overlay to: {output_path}")

# Save debug frame if available
debug_frame = detector.get_debug_frame()
if debug_frame is not None:
    debug_path = Path("test_outputs/detection_test_debug.jpg")
    cv2.imwrite(str(debug_path), debug_frame)
    print(f"Saved debug frame to: {debug_path}")

# Analyze the threshold used for detection
blur_size = 5
blurred = cv2.medianBlur(gray, blur_size)
_, binary = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

# Count dark pixels
dark_pixels = cv2.countNonZero(binary)
total_pixels = h * w
dark_percentage = (dark_pixels / total_pixels) * 100

print(f"\nThreshold analysis (threshold=80):")
print(f"  Dark pixels: {dark_pixels} ({dark_percentage:.2f}%)")

# Save binary threshold visualization
binary_path = Path("test_outputs/detection_test_binary.jpg")
cv2.imwrite(str(binary_path), binary)
print(f"Saved binary threshold to: {binary_path}")

# Check if there are any contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nContour analysis:")
print(f"  Total contours found: {len(contours)}")

# Analyze top 10 largest contours
contour_areas = [(cv2.contourArea(c), c) for c in contours]
contour_areas.sort(reverse=True, key=lambda x: x[0])

print(f"\nTop 10 largest contours:")
scale_factor = max(w / 640, h / 480)
min_area = int(1000 * scale_factor * scale_factor)  # Updated threshold
max_area = int(100000 * scale_factor * scale_factor)

for i, (area, contour) in enumerate(contour_areas[:10]):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0

    passes_size = min_area <= area <= max_area
    passes_circularity = circularity > 0.6

    print(f"  {i+1}. Area: {area:.0f} (min: {min_area}, max: {max_area}), "
          f"Radius: {radius:.0f}, Circularity: {circularity:.2f}, "
          f"Passes: {'YES' if (passes_size and passes_circularity) else 'NO'}")

print("\n=== Detection complete ===")
