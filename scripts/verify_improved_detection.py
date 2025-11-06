#!/usr/bin/env python3
"""
Create detailed numbered visualization of improved detector results
For user verification and iterative tuning
"""

import cv2
import numpy as np
from raspi_target_cam.detection.improved_dark_detector import ImprovedDarkDetector
from raspi_target_cam.detection.bullet_hole_detection import BulletHoleDetector
from raspi_target_cam.core.target_detection import TargetDetector
import os


def create_verification_visualization():
    """Create large, clear visualization for verification"""
    print("🎯 Creating Verification Visualization (Improved Detector)")
    print("=" * 60)

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

    print(f"✅ Target: center={target_center}, inner_radius={inner_radius}px")

    # Create dark mask
    _, dark_mask = cv2.threshold(before_gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    # Light mask
    light_mask = cv2.bitwise_not(dark_mask)

    # Detect with improved detector
    print(f"\n🔍 Running improved detection...")
    detector = ImprovedDarkDetector()

    # Detect text zones
    text_zones = detector.detect_text_zones(before_gray, dark_mask)
    print(f"   Text zones excluded: {len(text_zones)}")

    # Detect dark holes
    dark_holes, merged_candidates = detector.detect_darker_holes(before_gray, after_gray, dark_mask,
                                                                  target_center, inner_radius)
    print(f"   Dark area holes: {len(dark_holes)}")
    if merged_candidates:
        print(f"   ⚠️  Merged hole candidates: {len(merged_candidates)}")
        for mc in merged_candidates:
            print(f"      {mc['reason']}, estimated {mc['estimated_count']} holes")

    # Detect light holes
    standard_detector = BulletHoleDetector()
    all_light_holes = standard_detector.detect_bullet_holes(before, after)

    light_holes = []
    for hole in all_light_holes:
        x, y = int(hole[0]), int(hole[1])
        if y < light_mask.shape[0] and x < light_mask.shape[1]:
            if light_mask[y, x] > 0:
                light_holes.append(hole)

    print(f"   Light area holes: {len(light_holes)}")

    # Combine all holes
    all_holes = []

    # Add dark holes with 'D' prefix
    for hole in dark_holes:
        all_holes.append({
            'type': 'dark',
            'data': hole,
            'x': int(hole[0]),
            'y': int(hole[1]),
            'radius': int(hole[2]),
            'confidence': hole[3],
            'area': hole[4],
            'darkness': hole[6] if len(hole) > 6 else 0
        })

    # Add light holes with 'L' prefix
    for hole in light_holes:
        all_holes.append({
            'type': 'light',
            'data': hole,
            'x': int(hole[0]),
            'y': int(hole[1]),
            'radius': int(hole[2]),
            'confidence': hole[3],
            'area': hole[4] if len(hole) > 4 else 0,
            'darkness': 0
        })

    print(f"\n📊 TOTAL DETECTIONS: {len(all_holes)}")
    print(f"   Expected: ~10 real bullet holes")

    # Create large visualization
    scale = 1.5
    result = cv2.resize(after.copy(), None, fx=scale, fy=scale)
    target_center_scaled = (int(target_center[0] * scale), int(target_center[1] * scale))
    inner_radius_scaled = int(inner_radius * scale)

    # Draw target zones
    cv2.circle(result, target_center_scaled, inner_radius_scaled, (100, 100, 100), 3)
    cv2.circle(result, target_center_scaled, int(inner_radius_scaled * 0.9), (150, 150, 150), 1)
    cv2.drawMarker(result, target_center_scaled, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)

    # Draw text exclusion zones
    for zone in text_zones:
        x1_scaled = int(zone['x1'] * scale)
        y1_scaled = int(zone['y1'] * scale)
        x2_scaled = int(zone['x2'] * scale)
        y2_scaled = int(zone['y2'] * scale)
        cv2.rectangle(result, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled),
                     (0, 0, 200), 2)

    # Draw each detection with large number
    for i, hole_info in enumerate(all_holes):
        x_scaled = int(hole_info['x'] * scale)
        y_scaled = int(hole_info['y'] * scale)
        radius_scaled = int(hole_info['radius'] * scale)

        # Calculate distance from center
        dx = hole_info['x'] - target_center[0]
        dy = hole_info['y'] - target_center[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Color based on type and confidence
        if hole_info['type'] == 'dark':
            if hole_info['confidence'] > 0.5:
                color = (0, 255, 0)  # Green - high conf dark
            else:
                color = (0, 255, 255)  # Yellow - lower conf dark
            prefix = "D"
        else:
            color = (0, 165, 255)  # Orange - light area
            prefix = "L"

        # Draw detection
        cv2.circle(result, (x_scaled, y_scaled), radius_scaled + 10, color, 4)
        cv2.circle(result, (x_scaled, y_scaled), 6, color, -1)

        # Draw line to center
        cv2.line(result, (x_scaled, y_scaled), target_center_scaled, color, 1, cv2.LINE_AA)

        # Large number label
        label = f"{i+1}"
        label_pos = (x_scaled - 25, y_scaled - radius_scaled - 25)

        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4
        )
        cv2.rectangle(result,
                     (label_pos[0] - 10, label_pos[1] - text_height - 10),
                     (label_pos[0] + text_width + 10, label_pos[1] + 10),
                     (0, 0, 0), -1)

        cv2.putText(result, label, label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

    # Create info panel
    info_width = 900
    info_height = result.shape[0]
    info_panel = np.zeros((info_height, info_width, 3), dtype=np.uint8)

    y = 40
    cv2.putText(info_panel, "IMPROVED DETECTOR - Verification", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    y += 50
    cv2.putText(info_panel, f"Total: {len(all_holes)} detections (expect ~10)", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y += 40
    cv2.putText(info_panel, f"Dark area: {len(dark_holes)} | Light area: {len(light_holes)}", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    y += 60
    cv2.putText(info_panel, "#  | Type | Conf | Dark | Area | Dist", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y += 5
    cv2.line(info_panel, (20, y), (info_width - 20, y), (100, 100, 100), 1)

    for i, hole_info in enumerate(all_holes):
        dx = hole_info['x'] - target_center[0]
        dy = hole_info['y'] - target_center[1]
        distance = np.sqrt(dx**2 + dy**2)

        y += 30

        # Color
        if hole_info['type'] == 'dark':
            if hole_info['confidence'] > 0.5:
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)
        else:
            color = (0, 165, 255)

        type_char = 'D' if hole_info['type'] == 'dark' else 'L'
        dark_val = hole_info['darkness']

        text = f"{i+1:2d}   {type_char}    {hole_info['confidence']:.2f}  {dark_val:4.1f}  {int(hole_info['area']):4d}  {int(distance):3d}px"
        cv2.putText(info_panel, text, (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    y += 60
    cv2.putText(info_panel, "Color Code:", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 30
    cv2.putText(info_panel, "Green: Dark area (high conf)", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += 25
    cv2.putText(info_panel, "Yellow: Dark area (low conf)", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y += 25
    cv2.putText(info_panel, "Orange: Light area", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    y += 25
    cv2.putText(info_panel, "Red boxes: Text zones (excluded)", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

    y += 50
    cv2.putText(info_panel, "Please identify:", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 1)
    y += 30
    cv2.putText(info_panel, "- Which # are FALSE POSITIVES?", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)
    y += 25
    cv2.putText(info_panel, "- Any MISSED real holes?", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)

    # Combine
    combined = np.hstack([result, info_panel])

    # Save
    output_dir = "test_outputs/verification"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/improved_detector_numbered.jpg"
    cv2.imwrite(output_path, combined)

    print(f"\n💾 Saved to: {output_path}")
    print(f"\n✅ Please inspect and report:")
    print(f"   1. Which numbered detections are FALSE POSITIVES?")
    print(f"   2. Are any REAL HOLES missed?")
    print(f"   3. Overall quality assessment")

    # Print summary table
    print(f"\n📋 Detection Summary:")
    print(f"{'#':<4} {'Type':<6} {'Conf':<6} {'Dark':<6} {'Area':<6} {'Dist':<6} {'Status'}")
    print("-" * 60)
    for i, hole_info in enumerate(all_holes):
        dx = hole_info['x'] - target_center[0]
        dy = hole_info['y'] - target_center[1]
        distance = np.sqrt(dx**2 + dy**2)

        type_str = 'Dark' if hole_info['type'] == 'dark' else 'Light'

        # Flag suspicious ones
        status = ""
        if hole_info['confidence'] < 0.25:
            status = "⚠️ Low conf"
        elif distance > inner_radius * 0.85:
            status = "⚠️ Edge"
        elif hole_info['darkness'] < 2.0 and hole_info['type'] == 'dark':
            status = "⚠️ Barely darker"

        print(f"{i+1:<4} {type_str:<6} {hole_info['confidence']:<6.2f} "
              f"{hole_info['darkness']:<6.1f} {int(hole_info['area']):<6} "
              f"{int(distance):<6} {status}")

    return all_holes


if __name__ == "__main__":
    create_verification_visualization()
