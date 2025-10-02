#!/usr/bin/env python3
"""
Final optimized bullet hole detection test
Uses multiple detection methods with sensitive parameters to find all 10 holes
"""

import cv2
import numpy as np
from target_detection import TargetDetector
from scipy import ndimage

def test_final_bullet_hole_detection():
    """Test bullet hole detection with optimized parameters for all 10 holes"""

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

    # Detect outer circle to find center area
    outer_result = detector.detect_outer_circle(frame)

    if outer_result is None:
        print("No outer target detected - cannot proceed")
        return

    outer_x, outer_y, outer_radius = outer_result
    print(f"Outer target: center=({outer_x}, {outer_y}), radius={outer_radius}")

    # Estimate inner black circle area
    inner_radius = int(outer_radius * 0.4)
    inner_x, inner_y = outer_x, outer_y
    print(f"Estimated inner area: center=({inner_x}, {inner_y}), radius={inner_radius}")

    # Enhanced bullet hole detection
    print(f"\n=== Enhanced Multi-Method Detection ===")
    all_holes = detect_all_bullet_holes(frame, inner_x, inner_y, inner_radius)

    print(f"Total holes found: {len(all_holes)}")
    for i, (x, y, radius, confidence, method) in enumerate(all_holes):
        print(f"  Hole {i+1}: pos=({x}, {y}), r={radius}, conf={confidence:.3f}, method={method}")

    # Create final visualization
    overlay_frame = frame.copy()

    # Draw outer circle
    cv2.circle(overlay_frame, (outer_x, outer_y), outer_radius, (255, 0, 0), 3)

    # Draw estimated inner area
    cv2.circle(overlay_frame, (inner_x, inner_y), inner_radius, (0, 255, 0), 3)

    # Draw all detected holes
    for i, (x, y, radius, confidence, method) in enumerate(all_holes):
        # Color by method
        if method == "hough":
            color = (0, 0, 255)  # Red for Hough circles
        elif method == "contour":
            color = (0, 165, 255)  # Orange for contours
        elif method == "distance":
            color = (255, 0, 255)  # Magenta for distance transform
        else:
            color = (0, 255, 255)  # Yellow for other methods

        # Draw hole
        cv2.circle(overlay_frame, (x, y), max(10, radius), color, 2)
        cv2.circle(overlay_frame, (x, y), 3, color, -1)

        # Draw number
        cv2.putText(overlay_frame, str(i + 1), (x + 15, y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Draw summary info
    cv2.putText(overlay_frame, f"Total Holes: {len(all_holes)}", (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Add method legend
    y_pos = 120
    cv2.putText(overlay_frame, "Methods:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay_frame, "Red=Hough, Orange=Contour, Magenta=Distance", (20, y_pos + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save result
    output_path = "bullet_holes_final_result.jpg"
    cv2.imwrite(output_path, overlay_frame)
    print(f"Final result saved to: {output_path}")

def detect_all_bullet_holes(frame, center_x, center_y, search_radius):
    """Comprehensive bullet hole detection using multiple methods"""

    # Create larger ROI to catch edge holes
    margin = 100
    x1 = max(0, center_x - search_radius - margin)
    y1 = max(0, center_y - search_radius - margin)
    x2 = min(frame.shape[1], center_x + search_radius + margin)
    y2 = min(frame.shape[0], center_y + search_radius + margin)

    roi = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    all_holes = []

    # Method 1: Multiple Hough Circle passes with different parameters
    print("Running Hough circle detection...")
    hough_params = [
        {'dp': 1, 'minDist': 10, 'param1': 50, 'param2': 20, 'minRadius': 3, 'maxRadius': 25},
        {'dp': 1, 'minDist': 8, 'param1': 40, 'param2': 15, 'minRadius': 2, 'maxRadius': 20},
        {'dp': 2, 'minDist': 15, 'param1': 60, 'param2': 25, 'minRadius': 4, 'maxRadius': 30}
    ]

    for params in hough_params:
        blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, **params)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                full_x, full_y = x + x1, y + y1
                dist = np.sqrt((full_x - center_x)**2 + (full_y - center_y)**2)
                if dist <= search_radius:
                    all_holes.append((full_x, full_y, r, 0.8, "hough"))

    # Method 2: Contour-based detection with multiple thresholds
    print("Running contour detection...")
    thresholds = [40, 50, 60, 70]

    for thresh in thresholds:
        _, binary = cv2.threshold(gray_roi, thresh, 255, cv2.THRESH_BINARY_INV)

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 < area < 1000:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # More lenient circularity
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        x, y, radius = int(x), int(y), int(radius)

                        full_x, full_y = x + x1, y + y1
                        dist = np.sqrt((full_x - center_x)**2 + (full_y - center_y)**2)
                        if dist <= search_radius and radius > 2:
                            conf = min(0.9, circularity + 0.2)
                            all_holes.append((full_x, full_y, radius, conf, "contour"))

    # Method 3: Distance transform for overlapping holes
    print("Running distance transform detection...")
    _, binary = cv2.threshold(gray_roi, 45, 255, cv2.THRESH_BINARY_INV)

    # Distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Find local maxima
    from scipy import ndimage
    local_maxima = ndimage.maximum_filter(dist_transform, size=10) == dist_transform

    # Filter by minimum distance value
    distance_threshold = 3
    local_maxima = local_maxima & (dist_transform > distance_threshold)

    # Get coordinates of maxima
    maxima_coords = np.where(local_maxima)

    for y, x in zip(maxima_coords[0], maxima_coords[1]):
        full_x, full_y = x + x1, y + y1
        dist = np.sqrt((full_x - center_x)**2 + (full_y - center_y)**2)
        if dist <= search_radius:
            radius = int(dist_transform[y, x])
            if radius > 2:
                all_holes.append((full_x, full_y, radius, 0.7, "distance"))

    # Method 4: Template matching for very small holes
    print("Running template matching...")
    # Create small circle template
    template_size = 15
    template = np.zeros((template_size, template_size), dtype=np.uint8)
    cv2.circle(template, (template_size//2, template_size//2), 3, 255, -1)
    template = cv2.GaussianBlur(template, (3, 3), 1)

    # Apply template matching
    result = cv2.matchTemplate(gray_roi, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= 0.3)

    for pt in zip(*locations[::-1]):
        x, y = pt[0] + template_size//2, pt[1] + template_size//2
        full_x, full_y = x + x1, y + y1
        dist = np.sqrt((full_x - center_x)**2 + (full_y - center_y)**2)
        if dist <= search_radius:
            all_holes.append((full_x, full_y, 4, 0.6, "template"))

    # Remove duplicates and merge nearby detections
    print("Merging nearby detections...")
    merged_holes = merge_nearby_holes(all_holes, merge_distance=20)

    # Sort by confidence
    merged_holes.sort(key=lambda h: h[3], reverse=True)

    return merged_holes

def merge_nearby_holes(holes, merge_distance=15):
    """Merge holes that are very close to each other"""
    if not holes:
        return []

    merged = []
    used = set()

    for i, (x1, y1, r1, conf1, method1) in enumerate(holes):
        if i in used:
            continue

        # Find nearby holes
        nearby = [i]
        for j, (x2, y2, r2, conf2, method2) in enumerate(holes):
            if j != i and j not in used:
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < merge_distance:
                    nearby.append(j)

        if len(nearby) > 1:
            # Merge nearby holes
            all_x = [holes[k][0] for k in nearby]
            all_y = [holes[k][1] for k in nearby]
            all_r = [holes[k][2] for k in nearby]
            all_conf = [holes[k][3] for k in nearby]
            all_methods = [holes[k][4] for k in nearby]

            merged_x = int(np.mean(all_x))
            merged_y = int(np.mean(all_y))
            merged_r = int(np.mean(all_r))
            merged_conf = max(all_conf)
            merged_method = all_methods[np.argmax(all_conf)]

            merged.append((merged_x, merged_y, merged_r, merged_conf, merged_method))

            for k in nearby:
                used.add(k)
        else:
            merged.append((x1, y1, r1, conf1, method1))
            used.add(i)

    return merged

if __name__ == "__main__":
    test_final_bullet_hole_detection()