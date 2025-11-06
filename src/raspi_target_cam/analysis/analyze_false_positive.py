#!/usr/bin/env python3
"""
Analyze a specific detection to understand why it's a false positive
"""

import cv2
import numpy as np
from raspi_target_cam.core.perspective import Perspective
from raspi_target_cam.core.target_detection import TargetDetector
import json
import os


def analyze_detection(video_path, shot_info, reference_frame_num=50):
    """
    Create detailed visualization showing why a detection occurred
    """
    shot_num = shot_info['shot_number']
    frame_num = shot_info['frame_detected']
    shot_x, shot_y = shot_info['x'], shot_info['y']

    print(f"🔍 Analyzing Shot #{shot_num} (Frame {frame_num})")
    print(f"   Position: ({shot_x}, {shot_y})")
    print("=" * 70)

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

    # Calculate difference in ROI
    diff_roi = cv2.absdiff(ref_gray_roi, curr_gray_roi)
    darker_roi = cv2.subtract(ref_gray_roi, curr_gray_roi)

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

    # 3. Difference map (enhanced)
    diff_enhanced = cv2.normalize(diff_roi, None, 0, 255, cv2.NORM_MINMAX)
    diff_colored = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)
    cv2.drawMarker(diff_colored, (roi_x, roi_y), (255, 255, 255), cv2.MARKER_CROSS, 30, 3)

    # 4. Darker regions map
    darker_enhanced = cv2.normalize(darker_roi, None, 0, 255, cv2.NORM_MINMAX)
    darker_colored = cv2.applyColorMap(darker_enhanced, cv2.COLORMAP_HOT)
    cv2.drawMarker(darker_colored, (roi_x, roi_y), (255, 255, 255), cv2.MARKER_CROSS, 30, 3)

    # 5. Threshold visualization (what the detector sees)
    _, darker_thresh = cv2.threshold(darker_roi, 10, 255, cv2.THRESH_BINARY)
    darker_thresh_colored = cv2.cvtColor(darker_thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawMarker(darker_thresh_colored, (roi_x, roi_y), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)

    # Add labels
    def add_label(img, text):
        labeled = img.copy()
        cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)
        cv2.rectangle(labeled, (0, 0), (img.shape[1]-1, img.shape[0]-1), (255, 255, 255), 2)
        return labeled

    ref_labeled = add_label(ref_marked, f"REFERENCE (frame {reference_frame_num})")
    curr_labeled = add_label(curr_marked, f"DETECTION (frame {frame_num})")
    diff_labeled = add_label(diff_colored, "DIFFERENCE (absolute)")
    darker_labeled = add_label(darker_colored, "DARKER REGIONS")
    thresh_labeled = add_label(darker_thresh_colored, "THRESHOLD (>10 units)")

    # Create grid layout
    row1 = np.hstack([ref_labeled, curr_labeled])
    row2 = np.hstack([diff_labeled, darker_labeled])
    row3_padding = np.zeros_like(thresh_labeled)
    row3 = np.hstack([thresh_labeled, row3_padding])

    # Ensure all rows have same width
    max_width = max(row1.shape[1], row2.shape[1], row3.shape[1])

    if row3.shape[1] < max_width:
        padding = np.zeros((row3.shape[0], max_width - row3.shape[1], 3), dtype=np.uint8)
        row3 = np.hstack([row3, padding])

    grid = np.vstack([row1, row2, row3])

    # Add analysis info
    info_height = 150
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

    max_darker = np.max(darker_roi)
    mean_darker = np.mean(darker_roi)
    darker_pixels = np.sum(darker_roi > 10)

    y_pos = 30
    cv2.putText(info_panel, f"Shot #{shot_num} Analysis - Position ({shot_x}, {shot_y})",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    y_pos += 35
    cv2.putText(info_panel, f"Reference intensity: {ref_val:.1f}  |  Detection intensity: {curr_val:.1f}  |  Difference: {diff_val:.1f}",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y_pos += 30
    cv2.putText(info_panel, f"ROI darker pixels (>10): {darker_pixels}/{darker_roi.size} ({100*darker_pixels/darker_roi.size:.1f}%)",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y_pos += 30
    cv2.putText(info_panel, f"Max darker: {max_darker:.0f}  |  Mean darker: {mean_darker:.1f}",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Final composite
    final = np.vstack([info_panel, grid])

    # Save
    output_dir = "test_outputs/false_positive_analysis"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/shot_{shot_num:02d}_frame_{frame_num:04d}_analysis.jpg"
    cv2.imwrite(filename, final)

    print(f"\n📊 Statistics at detection point:")
    print(f"   Reference intensity: {ref_val:.1f}")
    print(f"   Current intensity: {curr_val:.1f}")
    print(f"   Difference (ref - curr): {diff_val:.1f}")
    print(f"   Darker pixels in ROI: {darker_pixels} ({100*darker_pixels/darker_roi.size:.1f}%)")
    print(f"   Max darkness in ROI: {max_darker:.0f}")
    print(f"   Mean darkness in ROI: {mean_darker:.1f}")

    print(f"\n💾 Saved detailed analysis to: {filename}")

    return {
        'ref_intensity': ref_val,
        'curr_intensity': curr_val,
        'difference': diff_val,
        'darker_pixels': darker_pixels,
        'max_darker': max_darker,
        'mean_darker': mean_darker
    }


if __name__ == "__main__":
    # Load detection results
    with open('sequential_detection_results.json', 'r') as f:
        results = json.load(f)

    shots = results['shots']
    reference_frame = results['reference_frame']

    video_path = "samples/10-shot-1.mkv"

    print("🔬 Analyzing detections for false positives")
    print("=" * 70)

    # Analyze shots 2-7 (suspected false positives)
    for shot_num in [2, 3, 4, 5, 6, 7]:
        if shot_num <= len(shots):
            shot = shots[shot_num - 1]
            print(f"\n{'='*70}")
            stats = analyze_detection(video_path, shot, reference_frame)

    print(f"\n✅ Analysis complete!")
    print(f"   Check test_outputs/false_positive_analysis/ for detailed visualizations")
