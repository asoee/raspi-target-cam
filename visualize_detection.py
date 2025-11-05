#!/usr/bin/env python3
"""
Visualize Bullet Hole Detection Results
Creates detailed visualization of detected holes with annotations
"""

import cv2
import numpy as np
from hybrid_bullet_detector import HybridBulletDetector
from target_detection import TargetDetector
from target_scoring import TargetScoringSystem
import os


def create_comprehensive_visualization():
    """Create comprehensive visualization of detection results"""
    print("🎯 Creating Comprehensive Detection Visualization")
    print("=" * 60)

    # Load test frames
    before_path = "test_frames/frame_0000_clean_target_corrected.jpg"
    after_path = "test_frames/frame_0900_near_end_corrected.jpg"

    before = cv2.imread(before_path)
    after = cv2.imread(after_path)

    if before is None or after is None:
        print("❌ Could not load test frames")
        return

    print(f"✅ Loaded test frames:")
    print(f"   Before: {before_path}")
    print(f"   After:  {after_path}")

    # Detect target
    print(f"\n📍 Detecting target...")
    target_detector = TargetDetector()
    inner_circle = target_detector.detect_black_circle_improved(before)
    outer_circle = target_detector.detect_outer_circle(before)

    if inner_circle:
        target_center = (int(inner_circle[0]), int(inner_circle[1]))
        inner_radius = int(inner_circle[2])
        print(f"   Inner circle: center={target_center}, radius={inner_radius}px")
    else:
        print(f"   ⚠️  No inner circle detected")
        target_center = (before.shape[1] // 2, before.shape[0] // 2)
        inner_radius = 300

    if outer_circle:
        outer_radius = int(outer_circle[2])
        print(f"   Outer circle: radius={outer_radius}px")
    else:
        outer_radius = inner_radius * 2

    # Setup scoring system
    print(f"\n🎯 Setting up scoring system...")
    scoring = TargetScoringSystem()
    scoring.calibrate_profile('large', outer_radius, inner_radius)

    # Detect holes using hybrid detector
    print(f"\n🔍 Detecting bullet holes...")
    detector = HybridBulletDetector()
    holes = detector.detect_bullet_holes(before, after, target_center)

    print(f"   Found {len(holes)} holes")

    # Score the holes
    print(f"\n📊 Scoring holes...")
    session = scoring.start_session('large', target_center, before)

    scored_holes = []
    for hole in holes:
        x, y, radius, confidence, area, circularity = hole

        # Add to scoring system
        shot_data = scoring.add_shot_to_current_session(x, y, radius, confidence)

        if shot_data:
            scored_holes.append({
                'hole': hole,
                'shot_data': shot_data
            })

    # Get session status
    status = scoring.get_session_status()
    print(f"   Total score: {status['total_score']}")
    print(f"   Shots scored: {status['shot_count']}")

    # Create visualizations
    print(f"\n🎨 Creating visualizations...")

    # 1. Annotated detection result
    result_annotated = after.copy()

    # Draw scoring rings
    ring_profile = scoring.profiles['large']
    for ring_num in [10, 9, 8, 7, 6, 5, 4, 3]:
        ring_radius = int(ring_profile.ring_radii.get(ring_num, 0))
        if ring_radius > 0:
            if ring_num >= 8:
                color = (0, 0, 200)  # Dark red for high-value rings
                thickness = 2
            elif ring_num >= 6:
                color = (0, 150, 200)  # Orange
                thickness = 1
            else:
                color = (200, 200, 0)  # Cyan
                thickness = 1

            cv2.circle(result_annotated, target_center, ring_radius, color, thickness)

    # Draw target center
    cv2.drawMarker(result_annotated, target_center, (0, 255, 0),
                   cv2.MARKER_CROSS, 30, 3)

    # Draw each detected hole with score
    for i, scored in enumerate(scored_holes):
        hole = scored['hole']
        shot = scored['shot_data']

        x, y, radius, confidence = int(hole[0]), int(hole[1]), int(hole[2]), hole[3]
        score = shot['score']

        # Color based on score
        if score >= 9:
            color = (0, 255, 0)  # Green - excellent
        elif score >= 7:
            color = (0, 255, 255)  # Yellow - good
        elif score >= 5:
            color = (0, 165, 255)  # Orange - okay
        else:
            color = (0, 0, 255)  # Red - low score

        # Draw circles around hole
        cv2.circle(result_annotated, (x, y), radius + 8, color, 3)
        cv2.circle(result_annotated, (x, y), 4, color, -1)

        # Draw line to center
        cv2.line(result_annotated, (x, y), target_center, color, 1, cv2.LINE_AA)

        # Add label with shot number and score
        label = f"#{i+1}: {score}"
        label_pos = (x - 30, y - radius - 15)

        # Background for label
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(result_annotated,
                     (label_pos[0] - 5, label_pos[1] - text_height - 5),
                     (label_pos[0] + text_width + 5, label_pos[1] + 5),
                     (0, 0, 0), -1)

        # Label text
        cv2.putText(result_annotated, label, label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Add session info
    info_panel_height = 200
    info_panel = np.zeros((info_panel_height, result_annotated.shape[1], 3), dtype=np.uint8)

    y_offset = 30
    cv2.putText(info_panel, f"Bullet Hole Detection Results",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    y_offset += 40
    cv2.putText(info_panel, f"Total Holes Detected: {len(holes)}",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y_offset += 35
    cv2.putText(info_panel, f"Total Score: {status['total_score']}",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    y_offset += 35
    cv2.putText(info_panel, f"Shots Scored: {status['shot_count']}",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    y_offset += 40
    cv2.putText(info_panel, f"Color Code: Green=9-10 | Yellow=7-8 | Orange=5-6 | Red=0-4",
               (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Combine info panel with result
    result_with_info = np.vstack([info_panel, result_annotated])

    # 2. Create side-by-side comparison
    comparison_width = before.shape[1]
    comparison_height = before.shape[0]

    # Resize for side-by-side (make them fit better)
    scale = 0.5  # Scale down to 50%
    before_small = cv2.resize(before, None, fx=scale, fy=scale)
    after_small = cv2.resize(result_annotated, None, fx=scale, fy=scale)

    # Add labels
    label_height = 50
    before_label = np.zeros((label_height, before_small.shape[1], 3), dtype=np.uint8)
    after_label = np.zeros((label_height, after_small.shape[1], 3), dtype=np.uint8)

    cv2.putText(before_label, "BEFORE (Clean Target)", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(after_label, f"AFTER ({len(holes)} Holes Detected)", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    before_with_label = np.vstack([before_label, before_small])
    after_with_label = np.vstack([after_label, after_small])

    comparison = np.hstack([before_with_label, after_with_label])

    # 3. Create detailed shot list visualization
    shot_list_width = 800
    shot_list_height = max(600, len(scored_holes) * 35 + 100)
    shot_list = np.zeros((shot_list_height, shot_list_width, 3), dtype=np.uint8)

    y = 40
    cv2.putText(shot_list, "Shot-by-Shot Breakdown", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    y += 50
    cv2.putText(shot_list, "Shot# | Score | Distance | Confidence | Radius", (20, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    y += 10
    cv2.line(shot_list, (20, y), (shot_list_width - 20, y), (100, 100, 100), 1)

    for i, scored in enumerate(scored_holes):
        shot = scored['shot_data']
        hole = scored['hole']

        y += 35

        score = shot['score']
        distance = shot['distance_from_center']
        confidence = hole[3]
        radius = hole[2]

        # Color based on score
        if score >= 9:
            color = (0, 255, 0)
        elif score >= 7:
            color = (0, 255, 255)
        elif score >= 5:
            color = (0, 165, 255)
        else:
            color = (128, 128, 128)

        text = f"  #{i+1:2d}     {score:2d}      {distance:6.1f}px      {confidence:.2f}       {radius}px"
        cv2.putText(shot_list, text, (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Save all visualizations
    output_dir = "test_outputs/final_visualization"
    os.makedirs(output_dir, exist_ok=True)

    # Save individual images
    cv2.imwrite(f"{output_dir}/1_annotated_result.jpg", result_with_info)
    print(f"   ✅ Saved: {output_dir}/1_annotated_result.jpg")

    cv2.imwrite(f"{output_dir}/2_before_after_comparison.jpg", comparison)
    print(f"   ✅ Saved: {output_dir}/2_before_after_comparison.jpg")

    cv2.imwrite(f"{output_dir}/3_shot_breakdown.jpg", shot_list)
    print(f"   ✅ Saved: {output_dir}/3_shot_breakdown.jpg")

    # Create combined master visualization
    # Stack: comparison on top, annotated result in middle, shot list on bottom
    # Need to match widths
    target_width = max(comparison.shape[1], result_with_info.shape[1], shot_list.shape[1])

    def resize_to_width(img, target_width):
        if img.shape[1] < target_width:
            pad_width = target_width - img.shape[1]
            padding = np.zeros((img.shape[0], pad_width, 3), dtype=np.uint8)
            return np.hstack([img, padding])
        elif img.shape[1] > target_width:
            scale = target_width / img.shape[1]
            return cv2.resize(img, None, fx=scale, fy=scale)
        return img

    comparison_resized = resize_to_width(comparison, target_width)
    result_resized = resize_to_width(result_with_info, target_width)
    shot_list_resized = resize_to_width(shot_list, target_width)

    # Add separator lines
    separator = np.ones((5, target_width, 3), dtype=np.uint8) * 50

    master_viz = np.vstack([
        comparison_resized,
        separator,
        result_resized,
        separator,
        shot_list_resized
    ])

    cv2.imwrite(f"{output_dir}/0_master_visualization.jpg", master_viz)
    print(f"   ✅ Saved: {output_dir}/0_master_visualization.jpg")

    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print(f"\n📊 Summary:")
    print(f"   Holes detected: {len(holes)}")
    print(f"   Total score: {status['total_score']}")
    print(f"   Average score per shot: {status['total_score'] / len(holes):.1f}")

    scoring.end_session()

    return master_viz


if __name__ == "__main__":
    create_comprehensive_visualization()
