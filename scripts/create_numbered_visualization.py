#!/usr/bin/env python3
"""
Create a clear numbered visualization for manual inspection
User can identify which numbered detections are false positives
"""

import cv2
import numpy as np
from raspi_target_cam.detection.hybrid_bullet_detector import HybridBulletDetector
from raspi_target_cam.core.target_detection import TargetDetector
import os


def create_numbered_viz():
    """Create easy-to-inspect visualization with large numbers"""
    print("🎯 Creating Numbered Visualization for Manual Inspection")
    print("=" * 60)

    before_path = "test_frames/frame_0000_clean_target_corrected.jpg"
    after_path = "test_frames/frame_0930_all_10_shots_corrected.jpg"

    before = cv2.imread(before_path)
    after = cv2.imread(after_path)

    # Detect target
    target_detector = TargetDetector()
    inner_circle = target_detector.detect_black_circle_improved(before)
    target_center = (int(inner_circle[0]), int(inner_circle[1]))
    inner_radius = int(inner_circle[2])

    # Detect holes
    detector = HybridBulletDetector()
    holes = detector.detect_bullet_holes(before, after, target_center)

    # Create large visualization
    result = after.copy()

    # Make it bigger for easier inspection
    scale = 1.5
    result = cv2.resize(result, None, fx=scale, fy=scale)
    target_center_scaled = (int(target_center[0] * scale), int(target_center[1] * scale))
    inner_radius_scaled = int(inner_radius * scale)

    # Draw target zones
    cv2.circle(result, target_center_scaled, inner_radius_scaled, (100, 100, 100), 3)
    cv2.drawMarker(result, target_center_scaled, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)

    # Draw each detection with LARGE number
    for i, hole in enumerate(holes):
        x, y, radius, confidence, area, circularity = hole

        # Scale coordinates
        x_scaled = int(x * scale)
        y_scaled = int(y * scale)
        radius_scaled = int(radius * scale)

        # Calculate distance
        dx = x - target_center[0]
        dy = y - target_center[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Color based on confidence
        if confidence > 0.6:
            color = (0, 255, 0)  # Green - high conf
        elif confidence > 0.4:
            color = (0, 255, 255)  # Yellow - medium
        else:
            color = (0, 165, 255)  # Orange - low

        # Draw circle
        cv2.circle(result, (x_scaled, y_scaled), radius_scaled + 10, color, 4)
        cv2.circle(result, (x_scaled, y_scaled), 6, color, -1)

        # Draw LARGE number label
        label = f"{i+1}"
        label_pos = (x_scaled - 25, y_scaled - radius_scaled - 25)

        # Large background rectangle for number
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4
        )
        cv2.rectangle(result,
                     (label_pos[0] - 10, label_pos[1] - text_height - 10),
                     (label_pos[0] + text_width + 10, label_pos[1] + 10),
                     (0, 0, 0), -1)

        # Large number
        cv2.putText(result, label, label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

    # Create info panel
    info_width = 800
    info_height = result.shape[0]
    info_panel = np.zeros((info_height, info_width, 3), dtype=np.uint8)

    y = 40
    cv2.putText(info_panel, "Detection Details", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    y += 50
    cv2.putText(info_panel, f"Total: {len(holes)} detections", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y += 40
    cv2.putText(info_panel, "Expected: ~10 real holes", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    y += 60
    cv2.putText(info_panel, "# | Conf | Area | Circ | Dist", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y += 5
    cv2.line(info_panel, (20, y), (info_width - 20, y), (100, 100, 100), 1)

    for i, hole in enumerate(holes):
        x, y_hole, radius, confidence, area, circularity = hole

        dx = x - target_center[0]
        dy_dist = y_hole - target_center[1]
        distance = np.sqrt(dx**2 + dy_dist**2)

        y += 30

        # Color based on confidence
        if confidence > 0.6:
            color = (0, 255, 0)
        elif confidence > 0.4:
            color = (0, 255, 255)
        else:
            color = (0, 165, 255)

        text = f"{i+1:2d}  {confidence:.2f}  {int(area):4d}  {circularity:.2f}  {int(distance):3d}px"
        cv2.putText(info_panel, text, (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    y += 60
    cv2.putText(info_panel, "Please identify which numbered", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 1)
    y += 30
    cv2.putText(info_panel, "detections are FALSE POSITIVES", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 1)

    # Combine image and info panel
    combined = np.hstack([result, info_panel])

    # Save
    output_dir = "test_outputs/manual_inspection"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/numbered_detections.jpg"
    cv2.imwrite(output_path, combined)

    print(f"\n💾 Saved to: {output_path}")
    print(f"\n✅ Please inspect the image and identify false positives!")
    print(f"   Image shows all {len(holes)} detections with large numbers")
    print(f"   Expected: ~10 real bullet holes")
    print(f"   Please report which numbers are FALSE POSITIVES")

    return output_path


if __name__ == "__main__":
    create_numbered_viz()
