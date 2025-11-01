#!/usr/bin/env python3
"""
Test script for checkerboard calibration
Loads a test image and performs perspective calibration using checkerboard pattern
"""
import cv2
import numpy as np
import os
from perspective import Perspective


def generate_checkerboard_pattern(filename='checkerboard_9x6.png', square_size=50, pattern_size=(9, 6)):
    """
    Generate a checkerboard pattern for testing

    Args:
        filename: Output filename
        square_size: Size of each square in pixels
        pattern_size: Tuple of (columns, rows) of internal corners
    """
    # Calculate total size (add 2 for border squares on each side)
    total_cols = pattern_size[0] + 1
    total_rows = pattern_size[1] + 1

    width = total_cols * square_size
    height = total_rows * square_size

    # Create checkerboard
    board = np.zeros((height, width), dtype=np.uint8)

    for row in range(total_rows):
        for col in range(total_cols):
            # Alternate between black (0) and white (255)
            if (row + col) % 2 == 0:
                y1 = row * square_size
                y2 = (row + 1) * square_size
                x1 = col * square_size
                x2 = (col + 1) * square_size
                board[y1:y2, x1:x2] = 255

    # Convert to BGR for consistency
    board_bgr = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(filename, board_bgr)
    print(f"Generated checkerboard pattern: {filename}")
    print(f"  Size: {width}x{height} pixels")
    print(f"  Pattern: {pattern_size[0]}x{pattern_size[1]} internal corners")
    print(f"  Square size: {square_size}x{square_size} pixels")

    return board_bgr


def apply_perspective_distortion(image, max_angle=30):
    """
    Apply random perspective distortion to an image

    Args:
        image: Input image
        max_angle: Maximum rotation angle in degrees

    Returns:
        Distorted image
    """
    h, w = image.shape[:2]

    # Generate random perspective transformation
    # Define source points (corners of the image)
    src_points = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ])

    # Create random destination points with some distortion
    margin = int(min(w, h) * 0.2)  # 20% margin for distortion
    dst_points = np.float32([
        [np.random.randint(0, margin), np.random.randint(0, margin)],
        [w - 1 - np.random.randint(0, margin), np.random.randint(0, margin)],
        [w - 1 - np.random.randint(0, margin), h - 1 - np.random.randint(0, margin)],
        [np.random.randint(0, margin), h - 1 - np.random.randint(0, margin)]
    ])

    # Calculate perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply transformation
    distorted = cv2.warpPerspective(image, matrix, (w, h), borderValue=(128, 128, 128))

    return distorted


def test_checkerboard_calibration(pattern_size=(9, 6), add_distortion=True):
    """
    Test checkerboard calibration with generated pattern

    Args:
        pattern_size: Tuple of (columns, rows) of internal corners
        add_distortion: Whether to apply perspective distortion for testing
    """
    print("=" * 60)
    print("CHECKERBOARD CALIBRATION TEST")
    print("=" * 60)

    # Create output directory
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Generate checkerboard pattern
    pattern = generate_checkerboard_pattern(
        filename=os.path.join(output_dir, 'checkerboard_reference.png'),
        square_size=60,
        pattern_size=pattern_size
    )

    # Optionally apply perspective distortion
    if add_distortion:
        print("\nApplying random perspective distortion...")
        test_image = apply_perspective_distortion(pattern)
        cv2.imwrite(os.path.join(output_dir, 'checkerboard_distorted.png'), test_image)
        print(f"Saved distorted image: {output_dir}/checkerboard_distorted.png")
    else:
        test_image = pattern

    # Resize to typical camera resolution for realistic testing
    target_height = 1944
    target_width = 2592
    test_image_resized = cv2.resize(test_image, (target_width, target_height))
    cv2.imwrite(os.path.join(output_dir, 'checkerboard_test_input.png'), test_image_resized)
    print(f"Saved test input: {output_dir}/checkerboard_test_input.png ({target_width}x{target_height})")

    # Create Perspective instance
    print("\n" + "-" * 60)
    print("Testing calibration...")
    print("-" * 60)
    perspective = Perspective()
    perspective.set_debug_mode(True)

    # Test checkerboard calibration
    success, message = perspective.calibrate_perspective_checkerboard(test_image_resized, pattern_size)

    print(f"\nCalibration result: {success}")
    print(f"Message: {message}")

    if success:
        print("\nCalibration successful!")

        # Save debug frame
        debug_frame = perspective.get_debug_frame()
        if debug_frame is not None:
            cv2.imwrite(os.path.join(output_dir, 'checkerboard_debug.png'), debug_frame)
            print(f"Saved debug visualization: {output_dir}/checkerboard_debug.png")

        # Test applying the calibration to the original image
        corrected = perspective.apply_perspective_correction(test_image_resized)
        if corrected is not None:
            cv2.imwrite(os.path.join(output_dir, 'checkerboard_corrected.png'), corrected)
            print(f"Saved corrected image: {output_dir}/checkerboard_corrected.png")

        # Test save/load
        print("\nTesting save/load...")
        save_success, save_msg = perspective.save_calibration(camera_resolution=(target_width, target_height))
        print(f"Save result: {save_success}, {save_msg}")

        # Create new instance and load
        perspective2 = Perspective()
        load_success, load_msg = perspective2.load_calibration()
        print(f"Load result: {load_success}, {load_msg}")
        print(f"Loaded method: {perspective2.calibration_method}")
        print(f"Loaded pattern size: {perspective2.saved_checkerboard_data}")

    else:
        print("\nCalibration failed!")
        print("Make sure the checkerboard pattern is clearly visible in the image.")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"\nCheck {output_dir}/ directory for output images")


def main():
    """Run tests"""
    print("\n### Test 1: Checkerboard with perspective distortion ###\n")
    test_checkerboard_calibration(pattern_size=(9, 6), add_distortion=True)

    print("\n\n### Test 2: Perfect checkerboard (no distortion) ###\n")
    test_checkerboard_calibration(pattern_size=(9, 6), add_distortion=False)

    print("\n\n### Test 3: Different pattern size (7x5) ###\n")
    test_checkerboard_calibration(pattern_size=(7, 5), add_distortion=True)


if __name__ == '__main__':
    main()
