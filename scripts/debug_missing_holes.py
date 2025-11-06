#!/usr/bin/env python3
"""
Debug script to understand why GT #7 and #8 are not detected
"""

import cv2
import numpy as np
from raspi_target_cam.core.target_detection import TargetDetector
import json

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

print(f"Target: center={target_center}, inner_radius={inner_radius}px")

# Create dark mask
_, dark_mask = cv2.threshold(before_gray, 60, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

# Check GT #7 and #8
for gt_hole in gt_holes:
    if gt_hole['hole_number'] in [7, 8]:
        x, y = gt_hole['x'], gt_hole['y']

        print(f"\nGT #{gt_hole['hole_number']}: position=({x}, {y})")

        # Check if in dark mask
        if y < dark_mask.shape[0] and x < dark_mask.shape[1]:
            in_dark = dark_mask[y, x] > 0
            print(f"   In dark mask: {in_dark}")
        else:
            print(f"   Out of bounds!")
            continue

        # Check distance from target center
        dx = x - target_center[0]
        dy = y - target_center[1]
        distance = np.sqrt(dx**2 + dy**2)
        max_distance = inner_radius * 0.9
        print(f"   Distance from center: {distance:.1f}px (max={max_distance:.1f}px)")
        print(f"   Within spatial filter: {distance <= max_distance}")

        # Check intensity values
        before_val = before_gray[y, x]
        after_val = after_gray[y, x]
        print(f"   Intensity: before={before_val}, after={after_val}, diff={before_val - after_val}")

        # Check neighborhood average
        size = 10
        y1 = max(0, y - size)
        y2 = min(before_gray.shape[0], y + size)
        x1 = max(0, x - size)
        x2 = min(before_gray.shape[1], x + size)

        before_neighbor = np.mean(before_gray[y1:y2, x1:x2])
        after_neighbor = np.mean(after_gray[y1:y2, x1:x2])

        print(f"   Neighborhood ({size}px): before={before_neighbor:.1f}, after={after_neighbor:.1f}")

        # Check darkness around hole
        inner_mask = np.zeros(after_gray.shape, np.uint8)
        cv2.circle(inner_mask, (x, y), 15, 255, -1)
        outer_mask = np.zeros(after_gray.shape, np.uint8)
        cv2.circle(outer_mask, (x, y), 20, 255, -1)
        ring_mask = cv2.subtract(outer_mask, inner_mask)

        hole_mean = cv2.mean(after_gray, mask=inner_mask)[0]
        ring_mean = cv2.mean(after_gray, mask=ring_mask)[0]

        print(f"   Hole mean: {hole_mean:.1f}, Ring mean: {ring_mean:.1f}, diff={ring_mean - hole_mean:.1f}")

print("\nCreating visualization...")
viz = after.copy()

# Draw target
cv2.circle(viz, target_center, inner_radius, (100, 100, 100), 2)
cv2.circle(viz, target_center, int(inner_radius * 0.9), (150, 150, 150), 1)
cv2.drawMarker(viz, target_center, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)

# Draw GT #7 and #8 with large markers
for gt_hole in gt_holes:
    if gt_hole['hole_number'] in [7, 8]:
        x, y = gt_hole['x'], gt_hole['y']

        cv2.circle(viz, (x, y), 30, (0, 0, 255), 3)
        cv2.drawMarker(viz, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)

        label = f"GT #{gt_hole['hole_number']}"
        cv2.putText(viz, label, (x + 40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

# Save
import os
output_dir = "test_outputs/debug"
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(f"{output_dir}/missing_holes_debug.jpg", viz)

print(f"\nSaved to: {output_dir}/missing_holes_debug.jpg")
