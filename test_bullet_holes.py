#!/usr/bin/env python3
"""
Test script for bullet hole detection
"""

import cv2
import numpy as np
from target_detection import TargetDetector

def test_bullet_hole_detection():
    """Test bullet hole detection on the captured target image"""

    # Load the test image
    image_path = "captures/capture_20250921_163306.jpg"
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return

    print(f"Loaded image: {image_path}")
    print(f"Image dimensions: {frame.shape}")

    # Initialize target detector
    detector = TargetDetector()
    detector.set_detection_enabled(True)
    detector.bullet_hole_detection_enabled = True

    # Detect target and bullet holes
    print("\n=== Running Target Detection ===")
    inner_result = detector.detect_target(frame)
    outer_result = detector.detect_outer_circle(frame)

    if inner_result is not None:
        print(f"Inner target detected: center=({inner_result[0]}, {inner_result[1]}), radius={inner_result[2]}")
        print(f"Inner confidence: {detector.detection_confidence:.3f}")
    else:
        print("No inner target detected")

    if outer_result is not None:
        print(f"Outer target detected: center=({outer_result[0]}, {outer_result[1]}), radius={outer_result[2]}")
        print(f"Outer confidence: {detector.outer_confidence:.3f}")
    else:
        print("No outer target detected")

    # Check bullet holes
    print(f"\n=== Bullet Hole Detection ===")
    if hasattr(detector, 'bullet_holes') and detector.bullet_holes:
        print(f"Found {len(detector.bullet_holes)} bullet holes:")
        for i, (x, y, radius, confidence) in enumerate(detector.bullet_holes):
            print(f"  Hole {i+1}: position=({x}, {y}), radius={radius}, confidence={confidence:.3f}")
    else:
        print("No bullet holes detected")

    # Create visualization
    print(f"\n=== Creating Visualization ===")
    overlay_frame = detector.draw_target_overlay(frame)

    # Save the result
    output_path = "bullet_holes_result.jpg"
    cv2.imwrite(output_path, overlay_frame)
    print(f"Result saved to: {output_path}")

    # Display some detection statistics
    print(f"\n=== Detection Summary ===")
    print(f"Target detection enabled: {detector.detection_enabled}")
    print(f"Bullet hole detection enabled: {detector.bullet_hole_detection_enabled}")
    print(f"Total holes found: {len(detector.bullet_holes) if hasattr(detector, 'bullet_holes') else 0}")

if __name__ == "__main__":
    test_bullet_hole_detection()