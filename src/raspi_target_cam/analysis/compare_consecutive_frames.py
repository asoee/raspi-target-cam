#!/usr/bin/env python3
"""
Compare consecutive frames to understand shot detection
"""

import cv2
import numpy as np
from raspi_target_cam.core.perspective import Perspective
from raspi_target_cam.core.target_detection import TargetDetector
import json
import os


def compare_frames(video_path, frame1_num, frame2_num, reference_frame_num=50):
    """
    Create detailed comparison between two frames
    """
    print(f"🔍 Comparing Frame {frame1_num} -> Frame {frame2_num}")
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

    # Get frame 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame1_num)
    ret, frame1 = cap.read()
    frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame1 = perspective.apply_perspective_correction(frame1)
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Get frame 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame2_num)
    ret, frame2 = cap.read()
    frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame2 = perspective.apply_perspective_correction(frame2)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    cap.release()

    # Calculate differences
    # 1. Frame 1 vs Reference
    diff1_ref = cv2.absdiff(ref_gray, frame1_gray)
    darker1_ref = cv2.subtract(ref_gray, frame1_gray)

    # 2. Frame 2 vs Reference
    diff2_ref = cv2.absdiff(ref_gray, frame2_gray)
    darker2_ref = cv2.subtract(ref_gray, frame2_gray)

    # 3. Frame 2 vs Frame 1 (what changed)
    diff_frames = cv2.absdiff(frame1_gray, frame2_gray)
    darker_frames = cv2.subtract(frame1_gray, frame2_gray)

    # Create dark mask for reference
    _, dark_mask = cv2.threshold(ref_gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Darker in dark areas only
    darker1_dark = cv2.bitwise_and(darker1_ref, darker1_ref, mask=dark_mask)
    darker2_dark = cv2.bitwise_and(darker2_ref, darker2_ref, mask=dark_mask)

    # Threshold for visualization
    _, darker1_thresh = cv2.threshold(darker1_dark, 15, 255, cv2.THRESH_BINARY)
    _, darker2_thresh = cv2.threshold(darker2_dark, 15, 255, cv2.THRESH_BINARY)
    _, diff_frames_thresh = cv2.threshold(diff_frames, 10, 255, cv2.THRESH_BINARY)

    # Create visualizations
    def add_label(img, text, color=(255, 255, 255)):
        labeled = img.copy()
        cv2.putText(labeled, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2)
        cv2.rectangle(labeled, (0, 0), (img.shape[1]-1, img.shape[0]-1), (255, 255, 255), 2)
        return labeled

    # Row 1: Original frames
    frame1_labeled = add_label(frame1, f"Frame {frame1_num} ({frame1_num/30:.2f}s)")
    frame2_labeled = add_label(frame2, f"Frame {frame2_num} ({frame2_num/30:.2f}s)")

    # Row 2: Difference from reference (darker in dark areas)
    darker1_colored = cv2.applyColorMap(darker1_dark, cv2.COLORMAP_HOT)
    darker1_labeled = add_label(darker1_colored, f"Frame {frame1_num} vs Ref")

    darker2_colored = cv2.applyColorMap(darker2_dark, cv2.COLORMAP_HOT)
    darker2_labeled = add_label(darker2_colored, f"Frame {frame2_num} vs Ref")

    # Row 3: Thresholded views
    darker1_thresh_colored = cv2.cvtColor(darker1_thresh, cv2.COLOR_GRAY2BGR)
    darker1_thresh_labeled = add_label(darker1_thresh_colored, f"Frame {frame1_num} Threshold")

    darker2_thresh_colored = cv2.cvtColor(darker2_thresh, cv2.COLOR_GRAY2BGR)
    darker2_thresh_labeled = add_label(darker2_thresh_colored, f"Frame {frame2_num} Threshold")

    # Row 4: Frame-to-frame difference
    diff_frames_colored = cv2.applyColorMap(diff_frames, cv2.COLORMAP_JET)
    diff_frames_labeled = add_label(diff_frames_colored, f"Frame {frame1_num} -> {frame2_num}")

    diff_frames_thresh_colored = cv2.cvtColor(diff_frames_thresh, cv2.COLOR_GRAY2BGR)
    diff_frames_thresh_labeled = add_label(diff_frames_thresh_colored, f"Frame Diff Threshold")

    # Create grid
    row1 = np.hstack([frame1_labeled, frame2_labeled])
    row2 = np.hstack([darker1_labeled, darker2_labeled])
    row3 = np.hstack([darker1_thresh_labeled, darker2_thresh_labeled])
    row4 = np.hstack([diff_frames_labeled, diff_frames_thresh_labeled])

    grid = np.vstack([row1, row2, row3, row4])

    # Add info panel
    info_height = 150
    info_panel = np.zeros((info_height, grid.shape[1], 3), dtype=np.uint8)
    info_panel[:] = (40, 40, 40)

    y_pos = 30
    cv2.putText(info_panel, f"Frame Comparison: {frame1_num} ({frame1_num/30:.2f}s) -> {frame2_num} ({frame2_num/30:.2f}s)",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    y_pos += 40
    # Calculate statistics
    mean_diff_frames = np.mean(diff_frames)
    max_diff_frames = np.max(diff_frames)
    pixels_changed = np.sum(diff_frames > 10)
    total_pixels = diff_frames.size

    cv2.putText(info_panel, f"Frame-to-frame change: mean={mean_diff_frames:.1f}, max={max_diff_frames:.0f}, "
               f"pixels changed={pixels_changed} ({100*pixels_changed/total_pixels:.2f}%)",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y_pos += 35
    # Dark area statistics
    darker1_pixels = np.sum(darker1_thresh > 0)
    darker2_pixels = np.sum(darker2_thresh > 0)
    new_darker_pixels = darker2_pixels - darker1_pixels

    cv2.putText(info_panel, f"Darker pixels (dark areas): Frame {frame1_num}={darker1_pixels}, "
               f"Frame {frame2_num}={darker2_pixels}, New={new_darker_pixels}",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y_pos += 35
    cv2.putText(info_panel, f"Target: center={target_center}, inner_radius={inner_radius}px",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

    # Final composite
    final = np.vstack([info_panel, grid])

    # Save
    output_dir = "test_outputs/frame_comparison"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/comparison_{frame1_num:04d}_to_{frame2_num:04d}.jpg"
    cv2.imwrite(filename, final)

    print(f"\n📊 Statistics:")
    print(f"   Frame-to-frame change: mean={mean_diff_frames:.1f}, max={max_diff_frames:.0f}")
    print(f"   Pixels changed: {pixels_changed} ({100*pixels_changed/total_pixels:.2f}%)")
    print(f"   Darker pixels in dark areas:")
    print(f"     Frame {frame1_num}: {darker1_pixels}")
    print(f"     Frame {frame2_num}: {darker2_pixels}")
    print(f"     New darker pixels: {new_darker_pixels}")
    print(f"\n💾 Saved: {filename}")


if __name__ == "__main__":
    video_path = "samples/10-shot-1.mkv"

    # Load improved results to find the shot before frame 694
    try:
        with open('improved_detection_results.json', 'r') as f:
            results = json.load(f)

        shots = results['shots']

        # Find shots around frame 694
        print("📋 Detected shots:")
        for shot in shots:
            print(f"   Shot #{shot['shot_number']}: frame {shot['frame_detected']} ({shot['frame_detected']/30:.2f}s)")

        # Find the shot just before frame 694
        before_694 = [s for s in shots if s['frame_detected'] < 694]
        if before_694:
            last_shot_frame = before_694[-1]['frame_detected']
            print(f"\n🔍 Last shot before frame 694: Shot #{before_694[-1]['shot_number']} at frame {last_shot_frame}")
        else:
            last_shot_frame = 50  # Use reference if no shots before 694

        print(f"\nComparing frame {last_shot_frame} (last shot) to frame 694 (GT #6 location)")
        print("=" * 70)

        compare_frames(video_path, last_shot_frame, 694, reference_frame_num=50)

    except FileNotFoundError:
        print("No results file found, comparing frame 344 to 694")
        compare_frames(video_path, 344, 694, reference_frame_num=50)

    print(f"\n✅ Comparison complete!")
