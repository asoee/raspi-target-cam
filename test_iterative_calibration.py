#!/usr/bin/env python3
"""
Test script for iterative ellipse calibration
Demonstrates improvement in circularity through iterative refinement
"""
import cv2
import numpy as np
import os
from perspective import Perspective


def create_test_target_with_perspective(width=2592, height=1944, distortion_strength=0.3):
    """
    Create a test image with circular target that has perspective distortion

    Args:
        width: Image width
        height: Image height
        distortion_strength: Amount of perspective distortion (0.0-1.0)

    Returns:
        Distorted image with circular target
    """
    # Create base image with concentric circles
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img.fill(200)  # Light gray background

    center_x, center_y = width // 2, height // 2

    # Draw concentric circles (target rings)
    circles = [
        (500, (50, 50, 50)),      # Outer ring - dark gray
        (400, (100, 100, 100)),   # Second ring
        (300, (150, 150, 150)),   # Third ring
        (200, (80, 80, 80)),      # Fourth ring
        (100, (30, 30, 30)),      # Inner circle - very dark
    ]

    for radius, color in circles:
        cv2.circle(img, (center_x, center_y), radius, color, -1)

    # Add white scoring zones
    cv2.circle(img, (center_x, center_y), 50, (255, 255, 255), -1)

    # Apply perspective distortion
    src_points = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])

    # Create asymmetric distortion
    margin = int(min(width, height) * distortion_strength)
    dst_points = np.float32([
        [margin * 0.5, margin * 0.3],                    # Top-left shifted
        [width - 1 - margin * 0.8, margin * 0.2],        # Top-right shifted more
        [width - 1 - margin * 0.4, height - 1 - margin * 0.6],  # Bottom-right
        [margin * 0.6, height - 1 - margin * 0.5]        # Bottom-left
    ])

    # Calculate and apply perspective transformation
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    distorted = cv2.warpPerspective(img, matrix, (width, height), borderValue=(128, 128, 128))

    return distorted


def test_single_vs_iterative(distortion_strength=0.3):
    """
    Compare single-pass vs iterative calibration

    Args:
        distortion_strength: Amount of perspective distortion to test
    """
    print("=" * 70)
    print(f"TESTING SINGLE-PASS vs ITERATIVE CALIBRATION")
    print(f"Distortion strength: {distortion_strength}")
    print("=" * 70)

    # Create output directory
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Generate test image
    print("\nGenerating test image with perspective distortion...")
    test_image = create_test_target_with_perspective(distortion_strength=distortion_strength)
    cv2.imwrite(os.path.join(output_dir, 'iterative_test_input.png'), test_image)
    print(f"Saved: {output_dir}/iterative_test_input.png")

    # Test 1: Single-pass calibration
    print("\n" + "-" * 70)
    print("TEST 1: SINGLE-PASS ELLIPSE CALIBRATION")
    print("-" * 70)

    perspective1 = Perspective()
    perspective1.set_debug_mode(True)

    success1, message1 = perspective1.calibrate_perspective(
        test_image, method='ellipse', iterative=False
    )

    print(f"\nResult: {success1}")
    print(f"Message: {message1}")

    if success1:
        # Save results
        corrected1 = perspective1.apply_perspective_correction(test_image)
        if corrected1 is not None:
            cv2.imwrite(os.path.join(output_dir, 'iterative_single_pass.png'), corrected1)
            print(f"Saved: {output_dir}/iterative_single_pass.png")

        debug1 = perspective1.get_debug_frame()
        if debug1 is not None:
            cv2.imwrite(os.path.join(output_dir, 'iterative_single_debug.png'), debug1)
            print(f"Saved: {output_dir}/iterative_single_debug.png")

    # Test 2: Iterative calibration
    print("\n" + "-" * 70)
    print("TEST 2: ITERATIVE ELLIPSE CALIBRATION")
    print("-" * 70)

    perspective2 = Perspective()
    perspective2.set_debug_mode(True)

    success2, message2 = perspective2.calibrate_perspective(
        test_image, method='ellipse', iterative=True
    )

    print(f"\nResult: {success2}")
    print(f"Message: {message2}")

    if success2:
        # Save results
        corrected2 = perspective2.apply_perspective_correction(test_image)
        if corrected2 is not None:
            cv2.imwrite(os.path.join(output_dir, 'iterative_refined.png'), corrected2)
            print(f"Saved: {output_dir}/iterative_refined.png")

        debug2 = perspective2.get_debug_frame()
        if debug2 is not None:
            cv2.imwrite(os.path.join(output_dir, 'iterative_refined_debug.png'), debug2)
            print(f"Saved: {output_dir}/iterative_refined_debug.png")

        # Print iteration details
        if perspective2.saved_ellipse_data:
            print(f"\nIteration details:")
            iterations = perspective2.saved_ellipse_data.get('iterations', 0)
            final_circularity = perspective2.saved_ellipse_data.get('final_circularity', 0)
            print(f"  Iterations: {iterations}")
            print(f"  Final circularity: {final_circularity:.4f}")

            iteration_results = perspective2.saved_ellipse_data.get('iteration_results', [])
            for result in iteration_results:
                print(f"    Iteration {result['iteration']}: circularity = {result['circularity']:.4f}")

    # Create comparison image
    if success1 and success2:
        print("\n" + "-" * 70)
        print("CREATING COMPARISON")
        print("-" * 70)

        h, w = test_image.shape[:2]

        # Create side-by-side-by-side comparison
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        comparison[:, :w] = test_image
        comparison[:, w:w*2] = corrected1
        comparison[:, w*2:] = corrected2

        # Add labels
        cv2.putText(comparison, "ORIGINAL (DISTORTED)", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(comparison, "SINGLE-PASS", (w + 10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 128, 0), 3)
        cv2.putText(comparison, "ITERATIVE", (w*2 + 10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Add iteration info for iterative
        if perspective2.saved_ellipse_data:
            iterations = perspective2.saved_ellipse_data.get('iterations', 0)
            final_circularity = perspective2.saved_ellipse_data.get('final_circularity', 0)
            cv2.putText(comparison, f"{iterations} iters, circ={final_circularity:.3f}",
                       (w*2 + 10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Add dividing lines
        cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)
        cv2.line(comparison, (w*2, 0), (w*2, h), (255, 255, 255), 2)

        cv2.imwrite(os.path.join(output_dir, 'iterative_comparison.png'), comparison)
        print(f"Saved: {output_dir}/iterative_comparison.png")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"\nCheck {output_dir}/ directory for output images")


def main():
    """Run tests with different distortion levels"""
    print("\n### Test 1: Moderate distortion (30%) ###\n")
    test_single_vs_iterative(distortion_strength=0.3)

    print("\n\n### Test 2: Heavy distortion (50%) ###\n")
    test_single_vs_iterative(distortion_strength=0.5)

    print("\n\n### Test 3: Light distortion (15%) ###\n")
    test_single_vs_iterative(distortion_strength=0.15)


if __name__ == '__main__':
    main()
