#!/usr/bin/env python3
"""
Test script for ellipse detection and perspective transformation
Uses the perspective.py module for all detection and transformation logic
"""
import cv2
import numpy as np
import os
from perspective import Perspective

def test_transformation():
    # Create Perspective instance
    corrector = Perspective()
    corrector.set_debug_mode(True)

    # Load a sample image to test
    img_path = "samples/video_20250918_182102.avi"

    # Try to load video first frame or use sample image
    cap = cv2.VideoCapture(img_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not load sample video, exiting")
        return

    # Create output directory for test images
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Process frame to get ellipse detection and transformation
    print("Processing frame for ellipse detection...")

    # Get original image dimensions
    original_height, original_width = frame.shape[:2]
    print(f"Original image size: {original_width}x{original_height}")

    # Save the original frame
    original_frame_path = os.path.join(output_dir, "01_original_frame.jpg")
    cv2.imwrite(original_frame_path, frame)
    print(f"Saved: {original_frame_path}")

    # Convert to grayscale and save
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(output_dir, "02_grayscale.jpg")
    cv2.imwrite(gray_path, gray)
    print(f"Saved: {gray_path}")

    # Use edge detection to find the outer ring
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred_path = os.path.join(output_dir, "03_blurred.jpg")
    cv2.imwrite(blurred_path, blurred)
    print(f"Saved: {blurred_path}")

    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    edges_path = os.path.join(output_dir, "04_edges.jpg")
    cv2.imwrite(edges_path, edges)
    print(f"Saved: {edges_path}")

    # Find contours and draw them
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_frame = frame.copy()
    cv2.drawContours(contours_frame, contours, -1, (0, 255, 0), 2)
    contours_path = os.path.join(output_dir, "05_contours.jpg")
    cv2.imwrite(contours_path, contours_frame)
    print(f"Saved: {contours_path}")

    # RING DETECTION ANALYSIS
    print("\n=== ANALYZING RING DETECTION ===")
    corrector.analyze_ring_detection(frame, save_debug=True, output_dir=output_dir)

    # EXPERIMENTAL ELLIPSE DETECTION
    print("\n=== ELLIPSE DETECTION ===")
    # Call experimental detection
    exp_ellipse, exp_contour, exp_reasons, exp_edges, all_ellipses = corrector.experimental_ellipse_detection(frame)

    # Create experimental ellipse visualization
    exp_ellipse_frame = frame.copy()

    if exp_ellipse is not None:
        print("Experimental ellipse detected successfully!")
        print(f"Experimental ellipse: center={exp_ellipse[0]}, axes={exp_ellipse[1]}, angle={exp_ellipse[2]}")

        # Draw the experimental ellipse
        cv2.ellipse(exp_ellipse_frame, exp_ellipse, (0, 255, 0), 3)  # Green ellipse

        # Draw center point
        exp_center = tuple(map(int, exp_ellipse[0]))
        cv2.circle(exp_ellipse_frame, exp_center, 5, (0, 0, 255), -1)  # Red center point

        # Draw axes lines
        center_x, center_y = exp_center
        major_axis, minor_axis = exp_ellipse[1]
        angle = exp_ellipse[2]

        # Calculate endpoints of major and minor axes
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Major axis endpoints
        major_half = major_axis / 2
        major_end1 = (int(center_x + major_half * cos_a), int(center_y + major_half * sin_a))
        major_end2 = (int(center_x - major_half * cos_a), int(center_y - major_half * sin_a))
        cv2.line(exp_ellipse_frame, major_end1, major_end2, (255, 0, 0), 2)  # Blue major axis

        # Minor axis endpoints
        minor_half = minor_axis / 2
        minor_end1 = (int(center_x - minor_half * sin_a), int(center_y + minor_half * cos_a))
        minor_end2 = (int(center_x + minor_half * sin_a), int(center_y - minor_half * cos_a))
        cv2.line(exp_ellipse_frame, minor_end1, minor_end2, (0, 255, 255), 2)  # Cyan minor axis

        # Add text with ellipse parameters
        text_y = 30
        cv2.putText(exp_ellipse_frame, f"EXPERIMENTAL: Center: ({center_x}, {center_y})", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        cv2.putText(exp_ellipse_frame, f"Major axis: {major_axis:.1f}", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        cv2.putText(exp_ellipse_frame, f"Minor axis: {minor_axis:.1f}", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        cv2.putText(exp_ellipse_frame, f"Angle: {angle:.1f}°", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y += 30
        aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)
        cv2.putText(exp_ellipse_frame, f"Aspect ratio: {aspect_ratio:.2f}", (10, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        print("No experimental ellipse detected")
        cv2.putText(exp_ellipse_frame, "No ellipse detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Save the experimental ellipse visualization
    exp_ellipse_path = os.path.join(output_dir, "06_detected_ellipse.jpg")
    cv2.imwrite(exp_ellipse_path, exp_ellipse_frame)
    print(f"Saved: {exp_ellipse_path}")

    # Create visualization showing all detected ellipses
    all_ellipses_frame = corrector.draw_all_ellipses(frame, all_ellipses, exp_ellipse)
    all_ellipses_path = os.path.join(output_dir, "07_all_detected_ellipses.jpg")
    cv2.imwrite(all_ellipses_path, all_ellipses_frame)
    print(f"Saved: {all_ellipses_path}")

    # Apply perspective transformation using experimental detection
    print("\n=== PERSPECTIVE TRANSFORMATION ===")
    transformation_applied = False

    if exp_ellipse is not None:
        print("Applying ellipse-to-circle transformation...")

        # Use the perspective module's high-level function
        transformed_image, transform_matrix, circle_radius = corrector.apply_ellipse_to_circle_transform(
            frame, exp_ellipse, (original_width, original_height))

        if transformed_image is not None:
            # Calculate where the ellipse center ended up
            original_center = exp_ellipse[0]
            transformed_center = corrector.transform_ellipse_center(exp_ellipse, transform_matrix)

            print(f"  Original ellipse center: ({original_center[0]:.1f}, {original_center[1]:.1f})")
            print(f"  Transformed center: ({transformed_center[0]:.1f}, {transformed_center[1]:.1f})")
            print(f"  Center shift: ({transformed_center[0] - original_center[0]:.1f}, {transformed_center[1] - original_center[1]:.1f})")
            print(f"  Resulting circle radius: {circle_radius:.1f} pixels (diameter: {circle_radius*2:.1f}px)")

            # Add visual markers to the transformed image
            annotated_image = corrector.add_transformation_visualization(
                transformed_image, exp_ellipse, transform_matrix, circle_radius, (original_width, original_height))

            corrected_path = os.path.join(output_dir, "08_perspective_corrected.jpg")
            cv2.imwrite(corrected_path, annotated_image)
            print(f"Saved: {corrected_path}")

            # Calculate distance between transformed center and image center
            image_center = (original_width / 2, original_height / 2)
            center_distance = np.sqrt((transformed_center[0] - image_center[0])**2 +
                                    (transformed_center[1] - image_center[1])**2)
            print(f"  Distance from image center: {center_distance:.1f} pixels")
            transformation_applied = True
        else:
            print("Failed to apply transformation")
    else:
        print("No ellipse detected - cannot apply perspective transformation")

    # Create debug frame from ellipse detection results
    if exp_ellipse is not None:
        debug_frame = corrector.create_ellipse_detection_debug_frame(
            exp_edges, all_ellipses, exp_ellipse, exp_reasons, (original_width, original_height)
        )
        debug_path = os.path.join(output_dir, "09_debug_ellipse_detection.jpg")
        cv2.imwrite(debug_path, debug_frame)
        print(f"Saved: {debug_path}")

    print(f"\nAll test images saved to: {output_dir}/")

    # Print summary of results
    print("\n" + "="*60)
    print("EXPERIMENTAL ELLIPSE DETECTION TEST SUMMARY")
    print("="*60)
    print(f"Experimental ellipse detection: {'SUCCESS' if exp_ellipse is not None else 'FAILED'}")

    if exp_ellipse is not None:
        print(f"\nDetected ellipse details:")
        print(f"Center: {exp_ellipse[0]}")
        print(f"Axes: {exp_ellipse[1]}")
        print(f"Angle: {exp_ellipse[2]:.1f}°")
        aspect_ratio = max(exp_ellipse[1]) / min(exp_ellipse[1])
        print(f"Aspect ratio: {aspect_ratio:.3f}")

        if transformation_applied:
            # Calculate the transformation details again for summary
            transform_matrix = corrector.create_ellipse_to_circle_transform(exp_ellipse, (original_width, original_height))
            transformed_center = corrector.transform_ellipse_center(exp_ellipse, transform_matrix)
            circle_radius = corrector.calculate_transformed_circle_size(exp_ellipse, transform_matrix)
            image_center = (original_width / 2, original_height / 2)
            center_distance = np.sqrt((transformed_center[0] - image_center[0])**2 +
                                    (transformed_center[1] - image_center[1])**2)
            print(f"\nTransformation results:")
            print(f"Transformed center: ({transformed_center[0]:.1f}, {transformed_center[1]:.1f})")
            print(f"Distance from image center: {center_distance:.1f} pixels")
            print(f"Resulting circle radius: {circle_radius:.1f} pixels (diameter: {circle_radius*2:.1f}px)")

    print(f"\nDetection statistics:")
    print(f"- Original contours found: {len(contours)}")
    print(f"- Valid ellipses detected: {len(all_ellipses)}")

    if all_ellipses:
        scores = [score for _, score, _ in all_ellipses]
        print(f"- Ellipse scores range: {min(scores):.3f} to {max(scores):.3f}")
        print(f"- Best ellipse score: {max(scores):.3f}")

    print(f"\nKey output files:")
    print(f"- Detected ellipse: 06_detected_ellipse.jpg")
    print(f"- All detected ellipses: 07_all_detected_ellipses.jpg")

    if transformation_applied:
        print(f"- Ellipse-to-circle transformation: 08_perspective_corrected.jpg")
    else:
        print(f"- No ellipse-to-circle transformation could be applied")

    print("="*60)
    print("Test complete!")

if __name__ == "__main__":
    test_transformation()