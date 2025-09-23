#!/usr/bin/env python3
"""
Simple test script to verify the affine transformation
"""
import cv2
import numpy as np
from target_detection import TargetDetector

def test_transformation():
    # Load a sample image to test
    img_path = "samples/video_20250918_182102.avi"

    # Try to load video first frame or use sample image
    cap = cv2.VideoCapture(img_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not load sample video, exiting")
        return

    # Create target detector
    detector = TargetDetector()
    detector.debug_mode = True
    # Clear any saved calibration to force regeneration
    detector.perspective_matrix = None
    detector.saved_perspective_matrix = None

    # Process frame to get ellipse detection and transformation
    print("Processing frame for ellipse detection...")

    # Get original image dimensions
    original_height, original_width = frame.shape[:2]
    print(f"Original image size: {original_width}x{original_height}")

    # For testing, let's create a custom transformation that targets the original size
    # We'll manually call the ellipse detection and transformation
    ellipse_result = detector.detect_ellipse_perspective(frame)

    # If we got an ellipse result, let's generate a new transformation for original size
    if ellipse_result is not None:
        # Find the best ellipse from the detector's internal state
        # We'll need to access the ellipse detection directly
        best_ellipse = None
        # Let's just use the result from detect_ellipse_perspective for now
        result = ellipse_result

    if result is not None:
        print("Ellipse detected successfully!")

        # Get the transformation matrix
        matrix = result  # detect_ellipse_perspective returns the matrix directly
        if matrix is not None:
            print(f"Got transformation matrix with shape: {matrix.shape}")

            # Apply transformation using 800x800 size (corrected target size)
            corrected = detector.apply_perspective_correction(frame, matrix, (800, 800))

            if corrected is not None:
                # Save both original and corrected images
                cv2.imwrite("test_original.jpg", frame)
                cv2.imwrite("test_corrected.jpg", corrected)
                print("Saved test_original.jpg and test_corrected.jpg")
            else:
                print("Failed to apply transformation")
        else:
            print("No transformation matrix available")
    else:
        print("No ellipse detected")

if __name__ == "__main__":
    test_transformation()