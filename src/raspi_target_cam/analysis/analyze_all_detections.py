#!/usr/bin/env python3
"""
Analyze all detections to understand how they were detected
Creates detailed visualizations showing reference, current, difference, and darkness maps
"""

import cv2
import numpy as np
from raspi_target_cam.core.perspective import Perspective
from raspi_target_cam.core.target_detection import TargetDetector
import json
import os


def analyze_detection(video_path, shot_info, reference_frame_num, dark_threshold=80):
    """
    Create detailed visualization showing how a detection occurred
    """
    shot_num = shot_info['shot_number']
    frame_num = shot_info['frame_detected']
    shot_x, shot_y = shot_info['x'], shot_info['y']

    shot_type = shot_info.get('type', 'dark')  # Default to 'dark' for old results
    print(f"🔍 Analyzing Shot #{shot_num} (Frame {frame_num}) - Type: {shot_type}")
    print(f"   Position: ({shot_x}, {shot_y})")
    print(f"   Score: {shot_info['score']:.1f}, Darkness: {shot_info['mean_darkness_increase']:.1f}, "
          f"Relative: {shot_info['relative_change']*100:.1f}%, Concentration: {shot_info['concentration']*100:.1f}%")

    cap = cv2.VideoCapture(video_path)
    perspective = Perspective()

    # Get reference frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_num)
    ret, ref = cap.read()
    ref = cv2.rotate(ref, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ref = perspective.apply_perspective_correction(ref)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # Get detection frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, curr = cap.read()
    curr = cv2.rotate(curr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    curr = perspective.apply_perspective_correction(curr)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    cap.release()

    # Create dark mask (areas that are dark in reference)
    _, dark_mask = cv2.threshold(ref_gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    # Create ROI around detection
    roi_size = 150
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

    # Calculate differences in ROI
    diff_roi = cv2.absdiff(ref_gray_roi, curr_gray_roi)
    darker_roi = cv2.subtract(ref_gray_roi, curr_gray_roi)

    # Darker in dark areas only
    darker_in_dark_roi = cv2.bitwise_and(darker_roi, darker_roi, mask=dark_mask_roi)

    # Mark the detection point in ROI coordinates
    roi_x = shot_x - x1
    roi_y = shot_y - y1

    # Create visualizations
    # 1. Reference ROI with marker
    ref_marked = ref_roi.copy()
    cv2.drawMarker(ref_marked, (roi_x, roi_y), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
    cv2.circle(ref_marked, (roi_x, roi_y), 30, (0, 255, 0), 2)

    # 2. Current ROI with marker
    curr_marked = curr_roi.copy()
    cv2.drawMarker(curr_marked, (roi_x, roi_y), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    cv2.circle(curr_marked, (roi_x, roi_y), 30, (0, 0, 255), 2)

    # 3. Dark mask visualization
    dark_mask_colored = cv2.cvtColor(dark_mask_roi, cv2.COLOR_GRAY2BGR)
    dark_mask_colored[dark_mask_roi > 0] = [255, 0, 255]  # Magenta for dark areas
    cv2.drawMarker(dark_mask_colored, (roi_x, roi_y), (255, 255, 255), cv2.MARKER_CROSS, 30, 3)

    # 4. All darker regions (absolute)
    darker_enhanced = cv2.normalize(darker_roi, None, 0, 255, cv2.NORM_MINMAX)
    darker_colored = cv2.applyColorMap(darker_enhanced, cv2.COLORMAP_HOT)
    cv2.drawMarker(darker_colored, (roi_x, roi_y), (255, 255, 255), cv2.MARKER_CROSS, 30, 3)

    # 5. Darker in dark areas only (the actual detection map)
    darker_dark_enhanced = cv2.normalize(darker_in_dark_roi, None, 0, 255, cv2.NORM_MINMAX)
    darker_dark_colored = cv2.applyColorMap(darker_dark_enhanced, cv2.COLORMAP_HOT)
    cv2.drawMarker(darker_dark_colored, (roi_x, roi_y), (255, 255, 255), cv2.MARKER_CROSS, 30, 3)

    # 6. Threshold visualization (what detector uses)
    _, darker_thresh = cv2.threshold(darker_in_dark_roi, 15, 255, cv2.THRESH_BINARY)
    darker_thresh_colored = cv2.cvtColor(darker_thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawMarker(darker_thresh_colored, (roi_x, roi_y), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)

    # Add labels
    def add_label(img, text, color=(255, 255, 255)):
        labeled = img.copy()
        cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2)
        cv2.rectangle(labeled, (0, 0), (img.shape[1]-1, img.shape[0]-1), (255, 255, 255), 2)
        return labeled

    ref_labeled = add_label(ref_marked, f"REFERENCE (frame {reference_frame_num})")
    curr_labeled = add_label(curr_marked, f"DETECTION (frame {frame_num})")
    mask_labeled = add_label(dark_mask_colored, f"DARK AREAS (< {dark_threshold})", (255, 0, 255))
    darker_labeled = add_label(darker_colored, "ALL DARKER REGIONS")
    darker_dark_labeled = add_label(darker_dark_colored, "DARKER IN DARK AREAS")
    thresh_labeled = add_label(darker_thresh_colored, "THRESHOLD (>15 units)")

    # Create grid layout (3x2)
    row1 = np.hstack([ref_labeled, curr_labeled, mask_labeled])
    row2 = np.hstack([darker_labeled, darker_dark_labeled, thresh_labeled])

    grid = np.vstack([row1, row2])

    # Add analysis info header
    info_height = 200
    info_panel = np.zeros((info_height, grid.shape[1], 3), dtype=np.uint8)
    info_panel[:] = (40, 40, 40)

    # Calculate statistics at detection point
    sample_size = 10
    sy1 = max(0, roi_y - sample_size)
    sy2 = min(ref_gray_roi.shape[0], roi_y + sample_size)
    sx1 = max(0, roi_x - sample_size)
    sx2 = min(ref_gray_roi.shape[1], roi_x + sample_size)

    ref_val = np.mean(ref_gray_roi[sy1:sy2, sx1:sx2])
    curr_val = np.mean(curr_gray_roi[sy1:sy2, sx1:sx2])
    diff_val = ref_val - curr_val
    relative_pct = (diff_val / ref_val * 100) if ref_val > 0 else 0

    darker_pixels = np.sum(darker_in_dark_roi > 15)
    total_darker_pixels = np.sum(darker_in_dark_roi > 0)
    dark_area_pixels = np.sum(dark_mask_roi > 0)

    y_pos = 30
    type_label = f" - Type: {shot_type.upper()}" if shot_type else ""
    cv2.putText(info_panel, f"Shot #{shot_num} - Frame {frame_num} ({frame_num/30:.2f}s){type_label} - Position ({shot_x}, {shot_y})",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    y_pos += 40
    cv2.putText(info_panel, f"Reference intensity: {ref_val:.1f}  |  Current intensity: {curr_val:.1f}  |  "
               f"Difference: {diff_val:.1f} ({relative_pct:.1f}% darker)",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y_pos += 35
    cv2.putText(info_panel, f"Dark area pixels in ROI: {dark_area_pixels}  |  Darker pixels (>15): {darker_pixels} "
               f"({100*darker_pixels/dark_area_pixels:.1f}% of dark area)" if dark_area_pixels > 0 else "N/A",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y_pos += 35
    cv2.putText(info_panel, f"Detection metrics: Score={shot_info['score']:.1f}, "
               f"Mean darkness={shot_info['mean_darkness_increase']:.1f}, "
               f"Relative change={shot_info['relative_change']*100:.1f}%, "
               f"Concentration={shot_info['concentration']*100:.1f}%",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

    y_pos += 35
    # Check against ground truth
    gt_status = "Ground truth: "
    try:
        with open('ground_truth_holes.json', 'r') as f:
            gt = json.load(f)

        min_dist = float('inf')
        closest_gt = None
        for hole in gt['holes']:
            dist = np.sqrt((shot_x - hole['x'])**2 + (shot_y - hole['y'])**2)
            if dist < min_dist:
                min_dist = dist
                closest_gt = hole['hole_number']

        if min_dist < 50:
            gt_status += f"MATCH GT #{closest_gt} (dist={min_dist:.0f}px)"
            gt_color = (0, 255, 0)
        else:
            gt_status += f"FALSE POSITIVE (closest GT #{closest_gt} at {min_dist:.0f}px)"
            gt_color = (0, 0, 255)
    except:
        gt_status += "No ground truth available"
        gt_color = (200, 200, 200)

    cv2.putText(info_panel, gt_status, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gt_color, 2)

    # Final composite
    final = np.vstack([info_panel, grid])

    # Save
    output_dir = "test_outputs/detection_analysis"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/shot_{shot_num:02d}_frame_{frame_num:04d}_analysis.jpg"
    cv2.imwrite(filename, final)

    print(f"   💾 Saved: {filename}")
    print()


def analyze_all_detections(video_path, results_file='improved_detection_results.json'):
    """
    Analyze all detections from the sequential scan
    """
    print("🔬 Analyzing All Improved Shot Detections")
    print("=" * 70)

    # Load detection results
    with open(results_file, 'r') as f:
        results = json.load(f)

    shots = results['shots']
    reference_frame = results['reference_frame']

    print(f"   Loaded {len(shots)} detections from {results_file}")
    print(f"   Reference frame: {reference_frame}")
    print("=" * 70)
    print()

    # Analyze each shot
    for shot in shots:
        analyze_detection(video_path, shot, reference_frame)

    print("=" * 70)
    print(f"✅ Analysis complete!")
    print(f"   {len(shots)} detailed visualizations saved to: test_outputs/detection_analysis/")


if __name__ == "__main__":
    video_path = "samples/10-shot-1.mkv"
    analyze_all_detections(video_path)
