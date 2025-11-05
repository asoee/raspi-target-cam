#!/usr/bin/env python3
"""
Test detection of a single shot by comparing clean target to after first shot
"""

import cv2
import numpy as np
from target_detection import TargetDetector
from improved_dark_detector import ImprovedDarkDetector


def find_first_shot_frame(video_path, start_frame=0):
    """
    Find the frame where the first shot appears
    """
    cap = cv2.VideoCapture(video_path)

    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    from perspective import Perspective

    # Initialize perspective correction
    perspective = Perspective()

    # Get reference (clean) frame
    ret, ref_frame = cap.read()
    if not ret:
        return None, None

    ref_frame = cv2.rotate(ref_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Apply perspective correction if calibration exists
    ref_corrected = perspective.apply_perspective_correction(ref_frame)
    transform_matrix = perspective.saved_perspective_matrix

    ref_gray = cv2.cvtColor(ref_corrected, cv2.COLOR_BGR2GRAY)

    print("Scanning for first shot...")

    frame_num = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Apply transformations
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = perspective.apply_perspective_correction(frame)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate difference
        diff = cv2.absdiff(ref_gray, curr_gray)
        mean_diff = np.mean(diff)

        if frame_num % 30 == 0:
            print(f"  Frame {frame_num}: mean_diff={mean_diff:.2f}")

        # Significant change detected
        if mean_diff > 2.0:
            print(f"  ✓ First shot detected at frame {frame_num}")
            cap.release()
            return frame_num, mean_diff

    cap.release()
    return None, None


def test_clean_vs_first_shot():
    """
    Compare clean target frame with frame after first shot
    """
    print("🎯 Testing Clean Target vs First Shot")
    print("=" * 60)

    video_path = "samples/10-shot-1.mkv"

    # Find first shot frame
    first_shot_frame, mean_diff = find_first_shot_frame(video_path, start_frame=0)

    if first_shot_frame is None:
        print("❌ Could not find first shot")
        return

    print(f"\n📹 Extracting frames...")
    print(f"   Clean target: frame 0")
    print(f"   After shot 1: frame {first_shot_frame}")

    # Extract the two frames
    cap = cv2.VideoCapture(video_path)

    # Get clean frame (frame 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, before_frame = cap.read()

    # Get after-shot frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_shot_frame)
    ret, after_frame = cap.read()

    cap.release()

    # Apply transformations
    from perspective import Perspective

    perspective = Perspective()

    before_frame = cv2.rotate(before_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    before_corrected = perspective.apply_perspective_correction(before_frame)

    after_frame = cv2.rotate(after_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    after_corrected = perspective.apply_perspective_correction(after_frame)

    # Convert to grayscale
    before_gray = cv2.cvtColor(before_corrected, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_corrected, cv2.COLOR_BGR2GRAY)

    print(f"\n🔍 Detecting target...")

    # Detect target
    target_detector = TargetDetector()
    inner_circle = target_detector.detect_black_circle_improved(before_corrected)

    if inner_circle:
        target_center = (int(inner_circle[0]), int(inner_circle[1]))
        inner_radius = int(inner_circle[2])
        print(f"   Target: center={target_center}, radius={inner_radius}px")
    else:
        print("   ⚠️  Could not detect target")
        target_center = (before_corrected.shape[1] // 2, before_corrected.shape[0] // 2)
        inner_radius = 300

    # Create dark mask
    _, dark_mask = cv2.threshold(before_gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    # Detect holes
    print(f"\n🔍 Detecting holes...")

    dark_detector = ImprovedDarkDetector()
    dark_detector.detect_text_zones(before_gray, dark_mask)

    holes, merged = dark_detector.detect_darker_holes(
        before_gray,
        after_gray,
        dark_mask,
        target_center,
        inner_radius
    )

    print(f"\n📊 Detection Results:")
    print(f"   Holes detected: {len(holes)}")

    if holes:
        print(f"\n   Detected holes:")
        for i, hole in enumerate(holes):
            x, y, radius, conf, area, circ, darkness = hole
            dist_from_center = np.sqrt((x - target_center[0])**2 + (y - target_center[1])**2)
            print(f"      #{i+1}: pos=({x},{y}), dist={dist_from_center:.0f}px, conf={conf:.2f}, darkness={darkness:.1f}")

    # Create visualization
    print(f"\n🎨 Creating visualization...")

    result = after_corrected.copy()

    # Draw target
    cv2.circle(result, target_center, inner_radius, (100, 100, 100), 2)
    cv2.drawMarker(result, target_center, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)

    # Draw detected holes
    for i, hole in enumerate(holes):
        x, y, radius, conf = int(hole[0]), int(hole[1]), int(hole[2]), hole[3]

        color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)

        cv2.circle(result, (x, y), radius + 5, color, 3)
        cv2.circle(result, (x, y), 3, color, -1)

        label = f"#{i+1}"
        cv2.putText(result, label, (x - 15, y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Add info overlay
    info_text = [
        f"Clean target (frame 0) vs After shot 1 (frame {first_shot_frame})",
        f"Detected: {len(holes)} hole(s)",
        f"Expected: 1 hole"
    ]

    y_offset = 30
    for text in info_text:
        cv2.putText(result, text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 35

    # Save outputs
    import os
    output_dir = "test_outputs/single_shot"
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(f"{output_dir}/clean_target.jpg", before_corrected)
    cv2.imwrite(f"{output_dir}/after_shot_1.jpg", after_corrected)
    cv2.imwrite(f"{output_dir}/detection_result.jpg", result)

    # Save difference image for visualization
    diff = cv2.absdiff(before_gray, after_gray)
    diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    cv2.imwrite(f"{output_dir}/difference_map.jpg", diff_colored)

    print(f"\n💾 Saved outputs to: {output_dir}/")
    print(f"   - clean_target.jpg")
    print(f"   - after_shot_1.jpg")
    print(f"   - detection_result.jpg")
    print(f"   - difference_map.jpg")

    print(f"\n✅ Test complete!")

    return holes


if __name__ == "__main__":
    test_clean_vs_first_shot()
