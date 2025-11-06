#!/usr/bin/env python3
"""
Debug script to visualize the darker_thresh image
"""

import cv2
import numpy as np
from raspi_target_cam.core.target_detection import TargetDetector
import json
import os

# Load ground truth
with open('ground_truth_holes.json', 'r') as f:
    ground_truth = json.load(f)

gt_holes = ground_truth['holes']

# Load images
before_path = "test_frames/frame_0000_clean_target_corrected.jpg"
after_path = "test_frames/frame_0930_all_10_shots_corrected.jpg"

before = cv2.imread(before_path)
after = cv2.imread(after_path)

before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Detect target
target_detector = TargetDetector()
inner_circle = target_detector.detect_black_circle_improved(before)
target_center = (int(inner_circle[0]), int(inner_circle[1]))
inner_radius = int(inner_circle[2])

# Create dark mask
_, dark_mask = cv2.threshold(before_gray, 60, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

# Calculate difference (same as detector)
diff = cv2.absdiff(before_gray, after_gray)
diff_masked = cv2.bitwise_and(diff, diff, mask=dark_mask)

darker_regions = cv2.subtract(before_gray, after_gray)
darker_regions = cv2.bitwise_and(darker_regions, darker_regions, mask=dark_mask)

_, darker_thresh = cv2.threshold(darker_regions, 1.5, 255, cv2.THRESH_BINARY)

# Clean up
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
darker_thresh = cv2.morphologyEx(darker_thresh, cv2.MORPH_OPEN, kernel)
darker_thresh = cv2.morphologyEx(darker_thresh, cv2.MORPH_CLOSE, kernel)

# Create colored visualization
darker_thresh_colored = cv2.cvtColor(darker_thresh, cv2.COLOR_GRAY2BGR)

# Draw GT #7 and #8 positions
for gt_hole in gt_holes:
    if gt_hole['hole_number'] in [7, 8]:
        x, y = gt_hole['x'], gt_hole['y']

        cv2.circle(darker_thresh_colored, (x, y), 30, (0, 0, 255), 3)
        cv2.drawMarker(darker_thresh_colored, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)

        label = f"GT #{gt_hole['hole_number']}"
        cv2.putText(darker_thresh_colored, label, (x + 40, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

# Extract region around GT #7 and #8
roi_x1 = 920
roi_y1 = 920
roi_x2 = 1140
roi_y2 = 1140

roi_thresh = darker_thresh[roi_y1:roi_y2, roi_x1:roi_x2]
roi_after = after_gray[roi_y1:roi_y2, roi_x1:roi_x2]

# Upscale ROI for better visualization
scale = 3
roi_thresh_large = cv2.resize(roi_thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
roi_after_large = cv2.resize(roi_after, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

# Add markers to ROI
roi_thresh_colored = cv2.cvtColor(roi_thresh_large, cv2.COLOR_GRAY2BGR)
roi_after_colored = cv2.cvtColor(roi_after_large, cv2.COLOR_GRAY2BGR)

for gt_hole in gt_holes:
    if gt_hole['hole_number'] in [7, 8]:
        x, y = gt_hole['x'], gt_hole['y']

        # Translate to ROI coordinates and scale
        roi_x = int((x - roi_x1) * scale)
        roi_y = int((y - roi_y1) * scale)

        if 0 <= roi_x < roi_thresh_colored.shape[1] and 0 <= roi_y < roi_thresh_colored.shape[0]:
            cv2.circle(roi_thresh_colored, (roi_x, roi_y), 20, (0, 0, 255), 2)
            cv2.circle(roi_after_colored, (roi_x, roi_y), 20, (0, 0, 255), 2)

            label = f"#{gt_hole['hole_number']}"
            cv2.putText(roi_thresh_colored, label, (roi_x + 25, roi_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(roi_after_colored, label, (roi_x + 25, roi_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Find contours and draw them
contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} contours in ROI")

# Draw contours on zoomed ROI
roi_contours = np.zeros_like(roi_thresh_colored)
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)

    # Scale contour for visualization
    contour_scaled = contour * scale

    cv2.drawContours(roi_contours, [contour_scaled], -1, (0, 255, 0), 2)

    # Get center
    M = cv2.moments(contour)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cx_scaled = cx * scale
        cy_scaled = cy * scale

        cv2.putText(roi_contours, f"{i+1}({int(area)})", (cx_scaled, cy_scaled),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    print(f"  Contour {i+1}: area={area:.0f}px")

# Combine visualizations
combined_roi = np.hstack([roi_after_colored, roi_thresh_colored, roi_contours])

# Save
output_dir = "test_outputs/debug"
os.makedirs(output_dir, exist_ok=True)

cv2.imwrite(f"{output_dir}/darker_thresh_full.jpg", darker_thresh_colored)
cv2.imwrite(f"{output_dir}/darker_thresh_roi.jpg", combined_roi)

print(f"\nSaved to:")
print(f"  {output_dir}/darker_thresh_full.jpg")
print(f"  {output_dir}/darker_thresh_roi.jpg")
print(f"\nROI shows:")
print(f"  Left: After image (raw grayscale)")
print(f"  Middle: Darker threshold (GT #7 and #8 marked)")
print(f"  Right: Detected contours")
