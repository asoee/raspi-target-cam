#!/usr/bin/env python3
"""
Analyze False Positives in Detection
Helps identify what's being detected incorrectly (e.g., ring numbers)
"""

import cv2
import numpy as np
from hybrid_bullet_detector import HybridBulletDetector
from target_detection import TargetDetector
import os


def analyze_detections_on_final_frame():
    """Analyze detections on the final frame with all 10 holes"""
    print("🎯 Analyzing Detection on Final Frame (All 10 Holes)")
    print("=" * 60)

    # Use frame 930 (31 seconds - all 10 holes)
    before_path = "test_frames/frame_0000_clean_target_corrected.jpg"
    after_path = "test_frames/frame_0930_all_10_shots_corrected.jpg"

    before = cv2.imread(before_path)
    after = cv2.imread(after_path)

    if before is None or after is None:
        print("❌ Could not load frames")
        print(f"   Before exists: {before is not None}")
        print(f"   After exists: {after is not None}")
        return

    print(f"✅ Loaded frames:")
    print(f"   Before: {before_path}")
    print(f"   After:  {after_path}")

    # Detect target
    print(f"\n📍 Detecting target...")
    target_detector = TargetDetector()
    inner_circle = target_detector.detect_black_circle_improved(before)

    if inner_circle:
        target_center = (int(inner_circle[0]), int(inner_circle[1]))
        inner_radius = int(inner_circle[2])
        print(f"   Target center: {target_center}")
        print(f"   Inner radius: {inner_radius}px")
    else:
        target_center = (before.shape[1] // 2, before.shape[0] // 2)
        inner_radius = 300

    # Detect holes
    print(f"\n🔍 Running hybrid detection...")
    detector = HybridBulletDetector()
    holes = detector.detect_bullet_holes(before, after, target_center)

    print(f"\n📊 Detection Analysis:")
    print(f"   Total detections: {len(holes)}")

    # Categorize detections by location
    inner_black_holes = []  # Inside black circle
    outer_ring_holes = []   # Outside black circle
    far_outside_holes = []  # Very far from center

    for i, hole in enumerate(holes):
        x, y, radius, confidence, area, circularity = hole

        # Calculate distance from center
        dx = x - target_center[0]
        dy = y - target_center[1]
        distance = np.sqrt(dx**2 + dy**2)

        hole_info = {
            'index': i + 1,
            'x': x,
            'y': y,
            'radius': radius,
            'confidence': confidence,
            'area': area,
            'circularity': circularity,
            'distance': distance
        }

        if distance <= inner_radius:
            inner_black_holes.append(hole_info)
        elif distance <= inner_radius * 2:
            outer_ring_holes.append(hole_info)
        else:
            far_outside_holes.append(hole_info)

    print(f"\n📍 Location Analysis:")
    print(f"   Inside black circle (10-9-8-7 rings): {len(inner_black_holes)} holes")
    print(f"   Outer rings (6-5-4-3): {len(outer_ring_holes)} holes")
    print(f"   Far outside target: {len(far_outside_holes)} holes")

    # Show details of each category
    print(f"\n🎯 Holes Inside Black Circle (Expected: ~7-8):")
    for h in inner_black_holes:
        print(f"      #{h['index']}: pos=({h['x']}, {h['y']}), "
              f"dist={h['distance']:.0f}px, conf={h['confidence']:.2f}, "
              f"area={h['area']:.0f}px, circ={h['circularity']:.2f}")

    print(f"\n🎯 Holes in Outer Rings (Expected: ~2-3):")
    for h in outer_ring_holes:
        print(f"      #{h['index']}: pos=({h['x']}, {h['y']}), "
              f"dist={h['distance']:.0f}px, conf={h['confidence']:.2f}, "
              f"area={h['area']:.0f}px, circ={h['circularity']:.2f}")

    if far_outside_holes:
        print(f"\n⚠️  Suspicious: Holes FAR Outside Target (Likely False Positives):")
        for h in far_outside_holes:
            print(f"      #{h['index']}: pos=({h['x']}, {h['y']}), "
                  f"dist={h['distance']:.0f}px, conf={h['confidence']:.2f}, "
                  f"area={h['area']:.0f}px, circ={h['circularity']:.2f}")

    # Create detailed visualization with zones marked
    result = after.copy()

    # Draw zones
    # Black circle
    cv2.circle(result, target_center, inner_radius, (0, 0, 255), 2)
    cv2.putText(result, "Black Circle", (target_center[0] - 80, target_center[1]),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Outer ring boundary
    cv2.circle(result, target_center, inner_radius * 2, (0, 255, 255), 2)
    cv2.putText(result, "Outer Rings", (target_center[0] - 80, target_center[1] - inner_radius - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw target center
    cv2.drawMarker(result, target_center, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)

    # Draw all detections with color coding
    for i, hole in enumerate(holes):
        x, y, radius, confidence = int(hole[0]), int(hole[1]), int(hole[2]), hole[3]

        dx = x - target_center[0]
        dy = y - target_center[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Color based on location
        if distance <= inner_radius:
            color = (0, 255, 0)  # Green - in black circle (expected)
            label_suffix = "IN"
        elif distance <= inner_radius * 2:
            color = (0, 255, 255)  # Yellow - in outer rings (expected)
            label_suffix = "OUT"
        else:
            color = (0, 0, 255)  # Red - far outside (suspicious!)
            label_suffix = "FAR!"

        # Draw detection
        cv2.circle(result, (x, y), radius + 5, color, 3)
        cv2.circle(result, (x, y), 3, color, -1)

        # Label
        label = f"#{i+1} {label_suffix}"
        cv2.putText(result, label, (x - 30, y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw line to center
        cv2.line(result, (x, y), target_center, color, 1, cv2.LINE_AA)

    # Add legend
    legend_height = 150
    legend = np.zeros((legend_height, result.shape[1], 3), dtype=np.uint8)

    y_off = 30
    cv2.putText(legend, "Detection Analysis - Final Frame (All 10 Holes)",
               (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    y_off += 35
    cv2.putText(legend, f"Green (IN): Inside black circle - {len(inner_black_holes)} detections",
               (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    y_off += 30
    cv2.putText(legend, f"Yellow (OUT): Outer rings - {len(outer_ring_holes)} detections",
               (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    y_off += 30
    cv2.putText(legend, f"Red (FAR): Far outside target - {len(far_outside_holes)} detections (FALSE POSITIVES!)",
               (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Combine
    final_viz = np.vstack([legend, result])

    # Save
    output_dir = "test_outputs/false_positive_analysis"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/final_frame_analysis.jpg"
    cv2.imwrite(output_path, final_viz)

    print(f"\n💾 Saved visualization to: {output_path}")

    # Also save a crop of far outside detections for detailed inspection
    if far_outside_holes:
        print(f"\n🔍 Creating detail crops of suspicious detections...")
        for h in far_outside_holes:
            x, y = h['x'], h['y']

            # Crop 200x200 region around detection
            crop_size = 200
            x1 = max(0, x - crop_size // 2)
            y1 = max(0, y - crop_size // 2)
            x2 = min(after.shape[1], x + crop_size // 2)
            y2 = min(after.shape[0], y + crop_size // 2)

            crop = after[y1:y2, x1:x2].copy()

            # Draw marker at detection point
            local_x = x - x1
            local_y = y - y1
            cv2.drawMarker(crop, (local_x, local_y), (0, 0, 255),
                          cv2.MARKER_CROSS, 20, 2)

            crop_path = f"{output_dir}/suspicious_detection_{h['index']:02d}.jpg"
            cv2.imwrite(crop_path, crop)
            print(f"      Saved crop #{h['index']}: {crop_path}")

    print(f"\n✅ Analysis complete!")

    # Return summary
    return {
        'total': len(holes),
        'inner_black': len(inner_black_holes),
        'outer_rings': len(outer_ring_holes),
        'far_outside': len(far_outside_holes)
    }


if __name__ == "__main__":
    summary = analyze_detections_on_final_frame()

    if summary:
        print(f"\n📊 Summary:")
        print(f"   Expected: ~10 total holes (7-8 in black circle, 2-3 in outer rings)")
        print(f"   Detected: {summary['total']} total")
        print(f"      - {summary['inner_black']} in black circle")
        print(f"      - {summary['outer_rings']} in outer rings")
        print(f"      - {summary['far_outside']} far outside (FALSE POSITIVES)")

        if summary['far_outside'] > 0:
            print(f"\n⚠️  Action needed: Filter out {summary['far_outside']} false positives")
            print(f"   Check test_outputs/false_positive_analysis/ for detail crops")
