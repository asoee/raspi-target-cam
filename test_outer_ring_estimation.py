#!/usr/bin/env python3
"""
Test outer ring estimation for rifle and pistol targets
"""

import cv2
import numpy as np
from pathlib import Path
from raspi_target_cam.core.target_detection import TargetDetector

# Test with the rifle target image
image_path = Path("data/captures/capture_20251113_171454.jpg")

if not image_path.exists():
    print(f"Image not found: {image_path}")
    exit(1)

# Load the image
frame = cv2.imread(str(image_path))
h, w = frame.shape[:2]
print(f"Image size: {w}x{h}")

# Create detector
detector = TargetDetector()
detector.set_debug_mode(True)
detector.force_detection()

# Run detection - outer ring should be estimated even on first frame
print("\n=== Running target detection ===")
result = detector.detect_target(frame)
print(f"First frame: Detected={result.detected}, Stable={result.is_stable}")

# Run a few more times to show it becomes stable
print("\nRunning additional frames for stability:")
for i in range(4):
    result = detector.detect_target(frame)
    print(f"Frame {i+2}: Detected={result.detected}, Stable={result.is_stable}")

print(f"\nDetection Result:")
print(f"  Detected: {result.detected}")
print(f"  Center: {result.center}")
print(f"  Inner radius: {result.radius}")
print(f"  Target Type: {result.target_type}")

# Check outer circle
if detector.outer_circle is not None:
    ox, oy, oradius = detector.outer_circle
    print(f"\nOuter Circle:")
    print(f"  Center: ({ox}, {oy})")
    print(f"  Radius: {oradius}")

    # Calculate ratio
    if result.radius:
        ratio = oradius / result.radius
        print(f"  Ratio (outer/inner): {ratio:.3f}")

        if result.target_type == 'rifle':
            expected_ratio = 2.122  # 8.7cm / 4.1cm
            print(f"  Expected ratio (rifle): {expected_ratio:.3f}")
        else:  # pistol
            expected_ratio = 2.0  # 20cm / 10cm
            print(f"  Expected ratio (pistol): {expected_ratio:.3f}")

        print(f"  Difference: {abs(ratio - expected_ratio):.3f}")
else:
    print("\nNo outer circle detected or estimated")

# Visualize the result
output_frame = frame.copy()
if result.detected and detector.outer_circle:
    # Draw inner circle
    cv2.circle(output_frame, result.center, result.radius, (0, 255, 0), 3)
    cv2.circle(output_frame, result.center, 5, (0, 0, 255), -1)

    # Draw outer circle
    ox, oy, oradius = detector.outer_circle
    cv2.circle(output_frame, (ox, oy), oradius, (255, 0, 255), 3)

    # Add labels
    cv2.putText(output_frame, f"Inner: r={result.radius}",
                (result.center[0] - 80, result.center[1] - result.radius - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(output_frame, f"Outer: r={oradius}",
                (ox - 80, oy + oradius + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(output_frame, f"Type: {result.target_type.upper()}",
                (ox - 80, oy - oradius - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)

# Save result
output_dir = Path("test_outputs")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "outer_ring_test.jpg"
cv2.imwrite(str(output_path), output_frame)
print(f"\nSaved visualization to: {output_path}")
