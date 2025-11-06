#!/usr/bin/env python3
"""
Compare Detector Output with Ground Truth
Analyzes how well the detector performs against manually marked holes
"""

import cv2
import numpy as np
import json
from raspi_target_cam.detection.improved_dark_detector import ImprovedDarkDetector
from raspi_target_cam.detection.bullet_hole_detection import BulletHoleDetector
from raspi_target_cam.core.target_detection import TargetDetector
import os


def compare_with_ground_truth():
    """Compare detector output with ground truth"""
    print("🎯 Comparing Detector vs Ground Truth")
    print("=" * 60)

    # Load ground truth
    with open('ground_truth_holes.json', 'r') as f:
        ground_truth = json.load(f)

    gt_holes = ground_truth['holes']
    print(f"✅ Loaded ground truth: {len(gt_holes)} holes")

    # Load image
    image_path = ground_truth['image_path']
    before_path = "test_frames/frame_0000_clean_target_corrected.jpg"

    before = cv2.imread(before_path)
    after = cv2.imread(image_path)

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Detect target
    target_detector = TargetDetector()
    inner_circle = target_detector.detect_black_circle_improved(before)
    target_center = (int(inner_circle[0]), int(inner_circle[1]))
    inner_radius = int(inner_circle[2])

    print(f"📍 Target: center={target_center}, inner_radius={inner_radius}px")

    # Create masks
    _, dark_mask = cv2.threshold(before_gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
    light_mask = cv2.bitwise_not(dark_mask)

    # Run detector
    print(f"\n🔍 Running detector...")
    dark_detector = ImprovedDarkDetector()
    standard_detector = BulletHoleDetector()

    # Detect text zones
    text_zones = dark_detector.detect_text_zones(before_gray, dark_mask)

    # Detect dark holes
    dark_holes, merged = dark_detector.detect_darker_holes(
        before_gray, after_gray, dark_mask, target_center, inner_radius
    )

    # Detect light holes
    all_light = standard_detector.detect_bullet_holes(before, after)
    light_holes = []
    for hole in all_light:
        x, y = int(hole[0]), int(hole[1])
        if y < light_mask.shape[0] and x < light_mask.shape[1]:
            if light_mask[y, x] > 0:
                light_holes.append(hole)

    # Combine all detections
    all_detections = []
    for hole in dark_holes:
        all_detections.append({
            'type': 'dark',
            'x': int(hole[0]),
            'y': int(hole[1]),
            'radius': int(hole[2]),
            'confidence': hole[3]
        })

    for hole in light_holes:
        all_detections.append({
            'type': 'light',
            'x': int(hole[0]),
            'y': int(hole[1]),
            'radius': int(hole[2]),
            'confidence': hole[3]
        })

    print(f"   Detected: {len(all_detections)} holes ({len(dark_holes)} dark + {len(light_holes)} light)")
    if merged:
        print(f"   Merged: {len(merged)} candidates")

    # Match detections to ground truth
    print(f"\n📊 Matching detections to ground truth...")
    match_distance = 50  # pixels

    matches = []
    unmatched_gt = []
    unmatched_detections = []

    # For each ground truth hole, find closest detection
    for gt_hole in gt_holes:
        gt_x, gt_y = gt_hole['x'], gt_hole['y']
        gt_num = gt_hole['hole_number']

        # Find closest detection
        best_match = None
        best_distance = float('inf')

        for det in all_detections:
            dx = det['x'] - gt_x
            dy = det['y'] - gt_y
            distance = np.sqrt(dx**2 + dy**2)

            if distance < best_distance and distance < match_distance:
                best_distance = distance
                best_match = det

        if best_match:
            matches.append({
                'gt_hole': gt_hole,
                'detection': best_match,
                'distance': best_distance
            })
            # Remove from detection list to avoid duplicate matching
            all_detections.remove(best_match)
        else:
            unmatched_gt.append(gt_hole)

    # Remaining detections are false positives
    unmatched_detections = all_detections

    # Print results
    print(f"\n✅ MATCHED: {len(matches)}/{len(gt_holes)} ground truth holes")
    print(f"❌ MISSED: {len(unmatched_gt)} ground truth holes")
    print(f"⚠️  FALSE POSITIVES: {len(unmatched_detections)} extra detections")

    if matches:
        print(f"\n📋 Matched Holes:")
        for match in matches:
            gt = match['gt_hole']
            det = match['detection']
            dist = match['distance']
            print(f"   GT #{gt['hole_number']}: ({gt['x']}, {gt['y']}) → "
                  f"Detected at ({det['x']}, {det['y']}) [{det['type']}] "
                  f"distance={dist:.1f}px")

    if unmatched_gt:
        print(f"\n❌ MISSED Ground Truth Holes:")
        for gt in unmatched_gt:
            print(f"   GT #{gt['hole_number']}: ({gt['x']}, {gt['y']})")

    if unmatched_detections:
        print(f"\n⚠️  FALSE POSITIVE Detections:")
        for det in unmatched_detections:
            print(f"   Detected at ({det['x']}, {det['y']}) [{det['type']}, conf={det['confidence']:.2f}]")

    # Create visualization
    print(f"\n🎨 Creating comparison visualization...")
    create_comparison_viz(after, gt_holes, matches, unmatched_gt,
                         unmatched_detections, target_center, inner_radius)

    # Calculate metrics
    print(f"\n📈 Performance Metrics:")
    precision = len(matches) / (len(matches) + len(unmatched_detections)) if (len(matches) + len(unmatched_detections)) > 0 else 0
    recall = len(matches) / len(gt_holes) if len(gt_holes) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"   Precision: {precision:.1%} ({len(matches)} correct / {len(matches) + len(unmatched_detections)} detected)")
    print(f"   Recall:    {recall:.1%} ({len(matches)} found / {len(gt_holes)} total)")
    print(f"   F1 Score:  {f1:.1%}")

    return {
        'matches': matches,
        'unmatched_gt': unmatched_gt,
        'unmatched_detections': unmatched_detections,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def create_comparison_viz(image, gt_holes, matches, missed, false_positives,
                         target_center, inner_radius):
    """Create visualization comparing ground truth and detections"""
    viz = image.copy()

    # Draw target
    cv2.circle(viz, target_center, inner_radius, (100, 100, 100), 2)
    cv2.drawMarker(viz, target_center, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)

    # Draw all ground truth holes (blue circles)
    for gt in gt_holes:
        x, y = gt['x'], gt['y']
        cv2.circle(viz, (x, y), 20, (255, 0, 0), 2)  # Blue for ground truth

    # Draw matched detections (green)
    for match in matches:
        gt = match['gt_hole']
        det = match['detection']

        # Green circle for correct detection
        cv2.circle(viz, (det['x'], det['y']), 15, (0, 255, 0), 3)
        cv2.circle(viz, (det['x'], det['y']), 3, (0, 255, 0), -1)

        # Line connecting GT to detection
        cv2.line(viz, (gt['x'], gt['y']), (det['x'], det['y']),
                (0, 255, 0), 1, cv2.LINE_AA)

        # Label
        label = f"GT{gt['hole_number']}"
        cv2.putText(viz, label, (det['x'] + 20, det['y'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw missed holes (red X)
    for gt in missed:
        x, y = gt['x'], gt['y']
        cv2.drawMarker(viz, (x, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 3)

        label = f"MISS {gt['hole_number']}"
        cv2.putText(viz, label, (x + 20, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw false positives (orange)
    for det in false_positives:
        cv2.circle(viz, (det['x'], det['y']), 15, (0, 165, 255), 3)
        cv2.drawMarker(viz, (det['x'], det['y']), (0, 165, 255),
                      cv2.MARKER_CROSS, 20, 2)

        label = "FP"
        cv2.putText(viz, label, (det['x'] + 20, det['y'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Add legend
    legend_height = 200
    legend = np.zeros((legend_height, viz.shape[1], 3), dtype=np.uint8)

    y = 30
    cv2.putText(legend, "Detector vs Ground Truth Comparison",
               (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    y += 45
    cv2.circle(legend, (40, y), 15, (255, 0, 0), 2)
    cv2.putText(legend, f"Ground Truth: {len(gt_holes)} holes (blue circles)",
               (70, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    y += 35
    cv2.circle(legend, (40, y), 10, (0, 255, 0), 3)
    cv2.putText(legend, f"Matched: {len(matches)} holes (green, connected to GT)",
               (70, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    y += 35
    cv2.drawMarker(legend, (40, y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 3)
    cv2.putText(legend, f"Missed: {len(missed)} holes (red X)",
               (70, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    y += 35
    cv2.circle(legend, (40, y), 10, (0, 165, 255), 3)
    cv2.putText(legend, f"False Positives: {len(false_positives)} detections (orange)",
               (70, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Combine
    final = np.vstack([legend, viz])

    # Save
    output_dir = "test_outputs/ground_truth"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "detector_vs_ground_truth.jpg")
    cv2.imwrite(output_path, final)

    print(f"   ✅ Saved to: {output_path}")


if __name__ == "__main__":
    results = compare_with_ground_truth()

    print(f"\n{'='*60}")
    print(f"🎯 SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Correct:  {len(results['matches'])}/10")
    print(f"❌ Missed:   {len(results['unmatched_gt'])}/10")
    print(f"⚠️  False:   {len(results['unmatched_detections'])}")
    print(f"\n📊 F1 Score: {results['f1']:.1%}")
