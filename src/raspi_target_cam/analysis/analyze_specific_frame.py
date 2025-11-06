#!/usr/bin/env python3
"""
Analyze a specific frame with custom reference
"""

import cv2
import numpy as np
from raspi_target_cam.core.perspective import Perspective
from raspi_target_cam.core.target_detection import TargetDetector
import json
import os


def analyze_frame_custom_reference(video_path, reference_frame_num, target_frame_num,
                                   position=None, dark_threshold=80, markers=None):
    """
    Analyze a specific frame with custom reference frame
    """
    print(f"🔍 Analyzing Frame {target_frame_num} vs Reference Frame {reference_frame_num}")
    print("=" * 70)

    cap = cv2.VideoCapture(video_path)
    perspective = Perspective()
    target_detector = TargetDetector()

    # Get reference frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_num)
    ret, ref = cap.read()
    ref = cv2.rotate(ref, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ref = perspective.apply_perspective_correction(ref)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # Detect target
    inner_circle = target_detector.detect_black_circle_improved(ref)
    target_center = (int(inner_circle[0]), int(inner_circle[1]))
    inner_radius = int(inner_circle[2])

    # Get target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
    ret, curr = cap.read()
    curr = cv2.rotate(curr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    curr = perspective.apply_perspective_correction(curr)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    cap.release()

    # Create dark mask (areas that are dark in reference)
    _, dark_mask = cv2.threshold(ref_gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    # If position provided, create ROI around it
    if position:
        roi_size = 150
        shot_x, shot_y = position
        x1 = max(0, shot_x - roi_size)
        y1 = max(0, shot_y - roi_size)
        x2 = min(ref.shape[1], shot_x + roi_size)
        y2 = min(ref.shape[0], shot_y + roi_size)

        # Extract ROIs
        ref_roi = ref[y1:y2, x1:x2]
        curr_roi = curr[y1:y2, x1:x2]
        ref_gray_roi = ref_gray[y1:y2, x1:x2]
        curr_gray_roi = curr_gray[y1:y2, x1:x2]
        dark_mask_roi = dark_mask[y1:y2, x1:x2]

        roi_x = shot_x - x1
        roi_y = shot_y - y1
    else:
        # Use full frame
        ref_roi = ref
        curr_roi = curr
        ref_gray_roi = ref_gray
        curr_gray_roi = curr_gray
        dark_mask_roi = dark_mask
        roi_x, roi_y = None, None

    # Calculate differences
    diff_roi = cv2.absdiff(ref_gray_roi, curr_gray_roi)
    darker_roi = cv2.subtract(ref_gray_roi, curr_gray_roi)

    # Darker in dark areas only
    darker_in_dark_roi = cv2.bitwise_and(darker_roi, darker_roi, mask=dark_mask_roi)

    # Create visualizations
    # 1. Reference ROI
    ref_marked = ref_roi.copy()
    if roi_x is not None:
        cv2.drawMarker(ref_marked, (roi_x, roi_y), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
        cv2.circle(ref_marked, (roi_x, roi_y), 30, (0, 255, 0), 2)

    # Draw additional markers if provided
    if markers:
        for marker in markers:
            mx, my = marker['pos']
            # Adjust to ROI coordinates if using ROI
            if position:
                mx_roi = mx - x1
                my_roi = my - y1
            else:
                mx_roi, my_roi = mx, my

            color = marker.get('color', (255, 0, 255))
            label = marker.get('label', '')
            cv2.drawMarker(ref_marked, (mx_roi, my_roi), color, cv2.MARKER_CROSS, 20, 2)
            cv2.circle(ref_marked, (mx_roi, my_roi), 20, color, 2)
            if label:
                cv2.putText(ref_marked, label, (mx_roi + 25, my_roi - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 2. Current ROI
    curr_marked = curr_roi.copy()
    if roi_x is not None:
        cv2.drawMarker(curr_marked, (roi_x, roi_y), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.circle(curr_marked, (roi_x, roi_y), 30, (0, 0, 255), 2)

    # Draw additional markers
    if markers:
        for marker in markers:
            mx, my = marker['pos']
            if position:
                mx_roi = mx - x1
                my_roi = my - y1
            else:
                mx_roi, my_roi = mx, my

            color = marker.get('color', (255, 0, 255))
            label = marker.get('label', '')
            cv2.drawMarker(curr_marked, (mx_roi, my_roi), color, cv2.MARKER_CROSS, 20, 2)
            cv2.circle(curr_marked, (mx_roi, my_roi), 20, color, 2)
            if label:
                cv2.putText(curr_marked, label, (mx_roi + 25, my_roi - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 3. Dark mask
    dark_mask_colored = cv2.cvtColor(dark_mask_roi, cv2.COLOR_GRAY2BGR)
    dark_mask_colored[dark_mask_roi > 0] = [255, 0, 255]  # Magenta
    if roi_x is not None:
        cv2.drawMarker(dark_mask_colored, (roi_x, roi_y), (255, 255, 255), cv2.MARKER_CROSS, 30, 3)
    if markers:
        for marker in markers:
            mx, my = marker['pos']
            if position:
                mx_roi = mx - x1
                my_roi = my - y1
            else:
                mx_roi, my_roi = mx, my
            color = (255, 255, 255)
            label = marker.get('label', '')
            cv2.drawMarker(dark_mask_colored, (mx_roi, my_roi), color, cv2.MARKER_CROSS, 20, 2)
            if label:
                cv2.putText(dark_mask_colored, label, (mx_roi + 25, my_roi - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4. All darker regions
    darker_enhanced = cv2.normalize(darker_roi, None, 0, 255, cv2.NORM_MINMAX)
    darker_colored = cv2.applyColorMap(darker_enhanced, cv2.COLORMAP_HOT)
    if roi_x is not None:
        cv2.drawMarker(darker_colored, (roi_x, roi_y), (255, 255, 255), cv2.MARKER_CROSS, 30, 3)
    if markers:
        for marker in markers:
            mx, my = marker['pos']
            if position:
                mx_roi = mx - x1
                my_roi = my - y1
            else:
                mx_roi, my_roi = mx, my
            color = (255, 255, 255)
            label = marker.get('label', '')
            cv2.drawMarker(darker_colored, (mx_roi, my_roi), color, cv2.MARKER_CROSS, 20, 2)
            if label:
                cv2.putText(darker_colored, label, (mx_roi + 25, my_roi - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 5. Darker in dark areas only
    darker_dark_enhanced = cv2.normalize(darker_in_dark_roi, None, 0, 255, cv2.NORM_MINMAX)
    darker_dark_colored = cv2.applyColorMap(darker_dark_enhanced, cv2.COLORMAP_HOT)
    if roi_x is not None:
        cv2.drawMarker(darker_dark_colored, (roi_x, roi_y), (255, 255, 255), cv2.MARKER_CROSS, 30, 3)
    if markers:
        for marker in markers:
            mx, my = marker['pos']
            if position:
                mx_roi = mx - x1
                my_roi = my - y1
            else:
                mx_roi, my_roi = mx, my
            color = (255, 255, 255)
            label = marker.get('label', '')
            cv2.drawMarker(darker_dark_colored, (mx_roi, my_roi), color, cv2.MARKER_CROSS, 20, 2)
            if label:
                cv2.putText(darker_dark_colored, label, (mx_roi + 25, my_roi - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 6. Threshold visualization
    _, darker_thresh = cv2.threshold(darker_in_dark_roi, 15, 255, cv2.THRESH_BINARY)
    darker_thresh_colored = cv2.cvtColor(darker_thresh, cv2.COLOR_GRAY2BGR)
    if roi_x is not None:
        cv2.drawMarker(darker_thresh_colored, (roi_x, roi_y), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    if markers:
        for marker in markers:
            mx, my = marker['pos']
            if position:
                mx_roi = mx - x1
                my_roi = my - y1
            else:
                mx_roi, my_roi = mx, my
            color = (0, 255, 255)
            label = marker.get('label', '')
            cv2.drawMarker(darker_thresh_colored, (mx_roi, my_roi), color, cv2.MARKER_CROSS, 20, 2)
            if label:
                cv2.putText(darker_thresh_colored, label, (mx_roi + 25, my_roi - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add labels
    def add_label(img, text, color=(255, 255, 255)):
        labeled = img.copy()
        cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2)
        cv2.rectangle(labeled, (0, 0), (img.shape[1]-1, img.shape[0]-1), (255, 255, 255), 2)
        return labeled

    ref_labeled = add_label(ref_marked, f"REFERENCE (frame {reference_frame_num})")
    curr_labeled = add_label(curr_marked, f"TARGET (frame {target_frame_num})")
    mask_labeled = add_label(dark_mask_colored, f"DARK AREAS (< {dark_threshold})", (255, 0, 255))
    darker_labeled = add_label(darker_colored, "ALL DARKER REGIONS")
    darker_dark_labeled = add_label(darker_dark_colored, "DARKER IN DARK AREAS")
    thresh_labeled = add_label(darker_thresh_colored, "THRESHOLD (>15 units)")

    # Create grid (3x2)
    row1 = np.hstack([ref_labeled, curr_labeled, mask_labeled])
    row2 = np.hstack([darker_labeled, darker_dark_labeled, thresh_labeled])

    grid = np.vstack([row1, row2])

    # Add info panel
    info_height = 200
    info_panel = np.zeros((info_height, grid.shape[1], 3), dtype=np.uint8)
    info_panel[:] = (40, 40, 40)

    # Calculate statistics
    if roi_x is not None:
        sample_size = 10
        sy1 = max(0, roi_y - sample_size)
        sy2 = min(ref_gray_roi.shape[0], roi_y + sample_size)
        sx1 = max(0, roi_x - sample_size)
        sx2 = min(ref_gray_roi.shape[1], roi_x + sample_size)

        ref_val = np.mean(ref_gray_roi[sy1:sy2, sx1:sx2])
        curr_val = np.mean(curr_gray_roi[sy1:sy2, sx1:sx2])
    else:
        ref_val = np.mean(ref_gray_roi)
        curr_val = np.mean(curr_gray_roi)

    diff_val = ref_val - curr_val
    relative_pct = (diff_val / ref_val * 100) if ref_val > 0 else 0

    darker_pixels = np.sum(darker_in_dark_roi > 15)
    total_darker_area = np.sum(darker_in_dark_roi > 0)
    dark_area_pixels = np.sum(dark_mask_roi > 0)

    # Find concentrated darker regions
    _, darker_thresh_find = cv2.threshold(darker_in_dark_roi, 15, 255, cv2.THRESH_BINARY)
    kernel_find = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    darker_closed = cv2.morphologyEx(darker_thresh_find, cv2.MORPH_CLOSE, kernel_find)
    darker_closed = cv2.morphologyEx(darker_closed, cv2.MORPH_OPEN, kernel_find)
    contours, _ = cv2.findContours(darker_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find significant regions
    significant_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 5000:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Measure darkness in this region
                mask = np.zeros(ref_gray_roi.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)

                darker_values = darker_in_dark_roi[mask > 0]
                if len(darker_values) > 0:
                    mean_darkness = np.mean(darker_values)
                    concentration = np.sum(darker_values >= 15) / len(darker_values)

                    if mean_darkness >= 8 and concentration >= 0.3:
                        significant_regions.append({
                            'center': (cx, cy),
                            'area': area,
                            'mean_darkness': mean_darkness,
                            'concentration': concentration
                        })

    y_pos = 30
    pos_str = f" at ({position[0]}, {position[1]})" if position else " (full frame)"
    cv2.putText(info_panel, f"Frame {target_frame_num} vs Reference {reference_frame_num}{pos_str}",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    y_pos += 40
    cv2.putText(info_panel, f"Reference intensity: {ref_val:.1f}  |  Current intensity: {curr_val:.1f}  |  "
               f"Difference: {diff_val:.1f} ({relative_pct:.1f}% darker)",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y_pos += 35
    cv2.putText(info_panel, f"Dark area pixels: {dark_area_pixels}  |  Darker pixels (>15): {darker_pixels} "
               f"({100*darker_pixels/dark_area_pixels:.1f}% of dark area)" if dark_area_pixels > 0 else "N/A",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y_pos += 35
    cv2.putText(info_panel, f"Significant regions found: {len(significant_regions)} "
               f"(area 50-5000px, darkness ≥8, concentration ≥30%)",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

    y_pos += 35
    # Show top 3 regions
    for i, region in enumerate(sorted(significant_regions,
                                     key=lambda x: x['mean_darkness'] * x['concentration'],
                                     reverse=True)[:3]):
        cx, cy = region['center']
        # Adjust coordinates if using ROI
        if position:
            cx += x1
            cy += y1

        cv2.putText(info_panel,
                   f"  #{i+1}: pos=({cx},{cy}), area={region['area']:.0f}, "
                   f"darkness={region['mean_darkness']:.1f}, conc={region['concentration']*100:.0f}%",
                   (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y_pos += 25

    # Final composite
    final = np.vstack([info_panel, grid])

    # Save
    output_dir = "test_outputs/custom_analysis"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/frame_{target_frame_num:04d}_vs_ref_{reference_frame_num:04d}_analysis.jpg"
    cv2.imwrite(filename, final)

    print(f"\n📊 Statistics:")
    print(f"   Reference intensity: {ref_val:.1f}")
    print(f"   Current intensity: {curr_val:.1f}")
    print(f"   Difference: {diff_val:.1f} ({relative_pct:.1f}% darker)")
    print(f"   Darker pixels in dark areas: {darker_pixels} ({100*darker_pixels/dark_area_pixels:.1f}%)")
    print(f"   Significant regions: {len(significant_regions)}")

    for i, region in enumerate(sorted(significant_regions,
                                     key=lambda x: x['mean_darkness'] * x['concentration'],
                                     reverse=True)[:3]):
        cx, cy = region['center']
        if position:
            cx += x1
            cy += y1
        print(f"     #{i+1}: pos=({cx},{cy}), area={region['area']:.0f}, "
              f"darkness={region['mean_darkness']:.1f}, concentration={region['concentration']*100:.0f}%")

    print(f"\n💾 Saved: {filename}")


if __name__ == "__main__":
    video_path = "samples/10-shot-1.mkv"

    # Analyze frame 694 - where GT #2 appears but wasn't detected
    # GT #2 is at (1057, 1180) from ground_truth_holes.json
    print("Analyzing frame 694 (GT #2 location) with frame 50 as reference")
    print("=" * 70)

    # Define markers to show
    markers = [
        {'pos': (1057, 1180), 'color': (255, 128, 0), 'label': 'GT#2'},  # Light Blue - GT #2
        {'pos': (952, 1168), 'color': (0, 255, 255), 'label': 'GT#6'},   # Yellow - GT #6 nearby
    ]

    analyze_frame_custom_reference(video_path,
                                   reference_frame_num=50,  # Use standard reference
                                   target_frame_num=694,
                                   position=(1057, 1180),  # Center ROI on GT #2
                                   markers=markers)

    print(f"\n✅ Analysis complete!")
