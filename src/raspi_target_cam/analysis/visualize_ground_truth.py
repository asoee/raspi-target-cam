#!/usr/bin/env python3
"""
Visualize all ground truth holes on a frame
"""

import cv2
import numpy as np
from raspi_target_cam.core.perspective import Perspective
from raspi_target_cam.core.target_detection import TargetDetector
import json


def visualize_ground_truth(video_path, gt_file='ground_truth_holes.json', frame_num=930):
    """
    Create visualization showing all ground truth hole positions
    """
    print(f"🎯 Visualizing Ground Truth Holes")
    print("=" * 70)

    # Load ground truth
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    holes = gt_data['holes']
    print(f"   Loaded {len(holes)} ground truth holes from {gt_file}")

    cap = cv2.VideoCapture(video_path)
    perspective = Perspective()
    target_detector = TargetDetector()

    # Get frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Could not read frame {frame_num}")
        return

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = perspective.apply_perspective_correction(frame)
    cap.release()

    # Detect target for reference
    inner_circle = target_detector.detect_black_circle_improved(frame)
    target_center = (int(inner_circle[0]), int(inner_circle[1]))
    inner_radius = int(inner_circle[2])

    # Create visualization
    result = frame.copy()

    # Draw target circle
    cv2.circle(result, target_center, inner_radius, (100, 100, 100), 3)

    # Define colors for each hole (rainbow-like progression)
    colors = [
        (255, 0, 0),      # GT #1 - Blue
        (255, 128, 0),    # GT #2 - Light Blue
        (255, 255, 0),    # GT #3 - Cyan
        (0, 255, 0),      # GT #4 - Green
        (0, 255, 128),    # GT #5 - Light Green
        (0, 255, 255),    # GT #6 - Yellow
        (0, 128, 255),    # GT #7 - Orange
        (0, 0, 255),      # GT #8 - Red
        (128, 0, 255),    # GT #9 - Purple
        (255, 0, 255),    # GT #10 - Magenta
    ]

    # Draw all ground truth holes
    print(f"\n📍 Ground Truth Positions:")
    for hole in holes:
        hole_num = hole['hole_number']
        x, y = hole['x'], hole['y']

        # Get color for this hole
        color = colors[(hole_num - 1) % len(colors)]

        # Draw marker
        cv2.drawMarker(result, (x, y), color, cv2.MARKER_CROSS, 30, 3)
        cv2.circle(result, (x, y), 25, color, 2)

        # Add label with hole number
        label = f"#{hole_num}"
        # Position label to avoid overlapping with marker
        label_x = x + 35
        label_y = y - 10

        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(result, (label_x - 2, label_y - label_h - 2),
                     (label_x + label_w + 2, label_y + 2), (0, 0, 0), -1)

        # Draw label text
        cv2.putText(result, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        print(f"   GT #{hole_num}: ({x}, {y}) - Color: {color}")

    # Add header info
    info_height = 150
    info_panel = np.zeros((info_height, result.shape[1], 3), dtype=np.uint8)
    info_panel[:] = (40, 40, 40)

    y_pos = 35
    cv2.putText(info_panel, f"Ground Truth Holes - Frame {frame_num} ({frame_num/30:.2f}s)",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    y_pos += 45
    cv2.putText(info_panel, f"Total holes: {len(holes)} | Target center: {target_center} | Inner radius: {inner_radius}px",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

    y_pos += 40
    cv2.putText(info_panel, f"Source: {gt_file}",
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

    # Combine
    final = np.vstack([info_panel, result])

    # Save
    output_dir = "test_outputs/ground_truth"
    import os
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{output_dir}/ground_truth_all_holes_frame_{frame_num:04d}.jpg"
    cv2.imwrite(filename, final)

    print(f"\n💾 Saved: {filename}")
    print(f"✅ Visualization complete!")


if __name__ == "__main__":
    video_path = "samples/10-shot-1.mkv"

    # Use frame 930 which should have all 10 shots
    visualize_ground_truth(video_path, frame_num=930)
