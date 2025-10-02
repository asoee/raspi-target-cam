#!/usr/bin/env python3
"""
Enhanced test script for bullet hole detection
Forces detection on the black center area even if inner target isn't detected
"""

import cv2
import numpy as np
from target_detection import TargetDetector

def test_enhanced_bullet_hole_detection():
    """Test bullet hole detection with enhanced center area detection"""

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

    # First try normal detection
    print("\n=== Running Normal Target Detection ===")
    inner_result = detector.detect_target(frame)
    outer_result = detector.detect_outer_circle(frame)

    if outer_result is not None:
        print(f"Outer target detected: center=({outer_result[0]}, {outer_result[1]}), radius={outer_result[2]}")

        # Manually set inner target based on the outer detection
        # The black center should be roughly in the center of the outer circle
        outer_x, outer_y, outer_radius = outer_result

        # Estimate inner black circle (typically ~40% of outer radius for this target type)
        inner_radius = int(outer_radius * 0.4)
        inner_x, inner_y = outer_x, outer_y

        print(f"Estimated inner target: center=({inner_x}, {inner_y}), radius={inner_radius}")

        # Manually call bullet hole detection on the black area
        print(f"\n=== Manual Bullet Hole Detection ===")
        holes = detector._detect_bullet_holes_in_area(frame, inner_x, inner_y, inner_radius)

        if holes:
            print(f"Found {len(holes)} bullet holes:")
            for i, (x, y, radius, confidence) in enumerate(holes):
                print(f"  Hole {i+1}: position=({x}, {y}), radius={radius}, confidence={confidence:.3f}")

            # Store holes in detector for visualization
            detector.bullet_holes = holes
        else:
            print("No bullet holes detected in center area")
    else:
        print("No outer target detected - cannot estimate center area")

    # Create visualization
    print(f"\n=== Creating Enhanced Visualization ===")
    overlay_frame = frame.copy()

    # Draw outer circle if detected
    if outer_result is not None:
        cv2.circle(overlay_frame, (outer_result[0], outer_result[1]), outer_result[2], (255, 0, 0), 3)

        # Draw estimated inner area
        inner_x, inner_y = outer_result[0], outer_result[1]
        inner_radius = int(outer_result[2] * 0.4)
        cv2.circle(overlay_frame, (inner_x, inner_y), inner_radius, (0, 255, 0), 3)

    # Draw bullet holes if found
    if hasattr(detector, 'bullet_holes') and detector.bullet_holes:
        for i, (x, y, radius, confidence) in enumerate(detector.bullet_holes):
            # Color based on confidence
            if confidence > 0.7:
                color = (0, 0, 255)  # Red for high confidence
            else:
                color = (0, 165, 255)  # Orange for lower confidence

            # Draw hole
            cv2.circle(overlay_frame, (x, y), max(8, radius), color, 2)
            cv2.circle(overlay_frame, (x, y), 3, color, -1)

            # Draw number
            cv2.putText(overlay_frame, str(i + 1), (x + 12, y - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw hole count
        hole_count = len(detector.bullet_holes)
        cv2.putText(overlay_frame, f"Holes Found: {hole_count}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Save the result
    output_path = "bullet_holes_enhanced_result.jpg"
    cv2.imwrite(output_path, overlay_frame)
    print(f"Enhanced result saved to: {output_path}")

def add_detection_method_to_detector():
    """Add the manual bullet hole detection method to the TargetDetector class"""

    def _detect_bullet_holes_in_area(self, frame, center_x, center_y, search_radius):
        """Detect bullet holes in a specific area"""
        # Create ROI around the center area
        margin = 50
        x1 = max(0, center_x - search_radius - margin)
        y1 = max(0, center_y - search_radius - margin)
        x2 = min(frame.shape[1], center_x + search_radius + margin)
        y2 = min(frame.shape[0], center_y + search_radius + margin)

        roi = frame[y1:y2, x1:x2]

        # Convert to grayscale
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

        # Use multiple detection methods
        all_holes = []

        # Method 1: HoughCircles for round holes
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=15,
            param1=50,
            param2=25,
            minRadius=5,
            maxRadius=25
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Convert back to full frame coordinates
                full_x = x + x1
                full_y = y + y1

                # Check if within search area
                dist = np.sqrt((full_x - center_x)**2 + (full_y - center_y)**2)
                if dist <= search_radius:
                    all_holes.append((full_x, full_y, r, 0.8))

        # Method 2: Contour detection for irregular holes
        # Create binary image focusing on dark areas (holes)
        _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 800:  # Filter by area
                # Get bounding circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                x, y, radius = int(x), int(y), int(radius)

                # Convert to full frame coordinates
                full_x = x + x1
                full_y = y + y1

                # Check if within search area
                dist = np.sqrt((full_x - center_x)**2 + (full_y - center_y)**2)
                if dist <= search_radius and radius > 3:
                    all_holes.append((full_x, full_y, radius, 0.6))

        # Remove duplicates
        merged_holes = []
        used = set()

        for i, (x, y, r, conf) in enumerate(all_holes):
            if i in used:
                continue

            # Find nearby holes
            nearby = []
            for j, (x2, y2, r2, conf2) in enumerate(all_holes):
                if j != i and j not in used:
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if dist < max(15, r):
                        nearby.append(j)

            if nearby:
                # Merge nearby holes
                all_x = [x] + [all_holes[j][0] for j in nearby]
                all_y = [y] + [all_holes[j][1] for j in nearby]
                all_r = [r] + [all_holes[j][2] for j in nearby]
                all_conf = [conf] + [all_holes[j][3] for j in nearby]

                merged_x = int(np.mean(all_x))
                merged_y = int(np.mean(all_y))
                merged_r = int(np.mean(all_r))
                merged_conf = max(all_conf)

                merged_holes.append((merged_x, merged_y, merged_r, merged_conf))

                used.add(i)
                for j in nearby:
                    used.add(j)
            else:
                merged_holes.append((x, y, r, conf))
                used.add(i)

        # Sort by confidence
        merged_holes.sort(key=lambda h: h[3], reverse=True)

        return merged_holes

    # Add method to TargetDetector class
    TargetDetector._detect_bullet_holes_in_area = _detect_bullet_holes_in_area

if __name__ == "__main__":
    # Add the detection method to the class
    add_detection_method_to_detector()

    # Run the test
    test_enhanced_bullet_hole_detection()