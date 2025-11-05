#!/usr/bin/env python3
"""
Frame-by-Frame Bullet Hole Detection
Detects shots by comparing consecutive frames to find NEW holes as they appear
This should be cleaner than comparing against a reference frame with all accumulated damage
"""

import cv2
import numpy as np
import yaml
from improved_dark_detector import ImprovedDarkDetector
from bullet_hole_detection import BulletHoleDetector
from target_detection import TargetDetector
import os


def frame_by_frame_detection():
    """
    Process video frame by frame, detecting new holes as they appear
    """
    print("🎯 Frame-by-Frame Bullet Hole Detection")
    print("=" * 60)

    video_path = "samples/10-shot-1.mkv"

    # Load perspective matrix
    with open('perspective_calibration.yaml', 'r') as f:
        calib = yaml.safe_load(f)
    perspective_matrix = np.array(calib['perspective_matrix'], dtype=np.float32)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"📹 Video: {video_path}")
    print(f"   Frames: {total_frames}, FPS: {fps}, Duration: {total_frames/fps:.1f}s")

    # Read and process first frame
    ret, first_frame_raw = cap.read()
    if not ret:
        print("❌ Cannot read first frame")
        return

    # Apply rotation and perspective correction
    first_frame_rotated = cv2.rotate(first_frame_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
    first_frame = cv2.warpPerspective(first_frame_rotated, perspective_matrix,
                                      (first_frame_rotated.shape[1], first_frame_rotated.shape[0]))

    # Detect target in first frame
    target_detector = TargetDetector()
    inner_circle = target_detector.detect_black_circle_improved(first_frame)

    if inner_circle:
        target_center = (int(inner_circle[0]), int(inner_circle[1]))
        inner_radius = int(inner_circle[2])
    else:
        target_center = (first_frame.shape[1] // 2, first_frame.shape[0] // 2)
        inner_radius = 300

    print(f"\n📍 Target: center={target_center}, inner_radius={inner_radius}px")

    # Setup detectors
    dark_detector = ImprovedDarkDetector()
    standard_detector = BulletHoleDetector()

    # Create dark mask from first frame
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(first_gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    # Detect text zones
    text_zones = dark_detector.detect_text_zones(first_gray, dark_mask)
    print(f"   Text zones: {len(text_zones)}")

    # Process video frame by frame
    print(f"\n🔍 Processing frames to detect new holes...")
    print(f"   Checking every 30 frames (1 second intervals)")

    prev_frame = first_frame.copy()
    prev_frame_num = 0

    all_shots = []
    shot_count = 0

    # Check at 1-second intervals (every 30 frames)
    check_interval = 30

    for frame_num in range(check_interval, min(total_frames, 950), check_interval):
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame_raw = cap.read()

        if not ret:
            break

        # Apply rotation and perspective
        frame_rotated = cv2.rotate(frame_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        current_frame = cv2.warpPerspective(frame_rotated, perspective_matrix,
                                           (frame_rotated.shape[1], frame_rotated.shape[0]))

        # Compare current frame to previous frame (not to first frame!)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Calculate frame difference
        diff = cv2.absdiff(prev_gray, current_gray)
        changed_pixels = np.count_nonzero(diff > 10)
        change_percentage = changed_pixels / (diff.shape[0] * diff.shape[1])

        # If significant change, detect new holes
        if change_percentage > 0.001:  # 0.1% change
            # Detect holes in dark area
            dark_holes, merged = dark_detector.detect_darker_holes(
                prev_gray, current_gray, dark_mask,
                target_center, inner_radius
            )

            # Detect holes in light area
            light_mask = cv2.bitwise_not(dark_mask)
            all_light = standard_detector.detect_bullet_holes(prev_frame, current_frame)
            light_holes = []
            for hole in all_light:
                x, y = int(hole[0]), int(hole[1])
                if y < light_mask.shape[0] and x < light_mask.shape[1]:
                    if light_mask[y, x] > 0:
                        light_holes.append(hole)

            new_holes_count = len(dark_holes) + len(light_holes)

            if new_holes_count > 0:
                shot_count += 1

                shot_info = {
                    'shot_num': shot_count,
                    'frame': frame_num,
                    'time': frame_num / fps,
                    'prev_frame': prev_frame_num,
                    'change_pct': change_percentage,
                    'dark_holes': dark_holes,
                    'light_holes': light_holes,
                    'merged': merged,
                    'total_new_holes': new_holes_count
                }

                all_shots.append(shot_info)

                print(f"\n   Shot #{shot_count} at frame {frame_num} ({frame_num/fps:.1f}s):")
                print(f"      Change: {change_percentage:.3%}")
                print(f"      New holes: {len(dark_holes)} dark + {len(light_holes)} light = {new_holes_count}")

                if merged:
                    print(f"      ⚠️  Merged holes detected: {len(merged)}")
                    for m in merged:
                        print(f"         {m['reason']}")

                # Update prev_frame to current after detecting shot
                prev_frame = current_frame.copy()
                prev_frame_num = frame_num

    cap.release()

    # Summary
    print(f"\n" + "=" * 60)
    print(f"📊 DETECTION SUMMARY")
    print(f"=" * 60)
    print(f"   Total shots detected: {len(all_shots)}")
    print(f"   Expected shots: 10")

    total_holes = sum(shot['total_new_holes'] for shot in all_shots)
    total_merged_estimate = sum(
        sum(m['estimated_count'] for m in shot['merged'])
        for shot in all_shots if shot['merged']
    )

    print(f"   Total new hole detections: {total_holes}")
    print(f"   Merged hole estimates: {total_merged_estimate}")

    print(f"\n📋 Shot-by-Shot Breakdown:")
    print(f"{'Shot':<6} {'Frame':<8} {'Time':<8} {'Dark':<6} {'Light':<6} {'Merged':<8} {'Total'}")
    print("-" * 70)

    for shot in all_shots:
        merged_str = f"{len(shot['merged'])}" if shot['merged'] else "-"
        print(f"{shot['shot_num']:<6} {shot['frame']:<8} {shot['time']:<8.1f} "
              f"{len(shot['dark_holes']):<6} {len(shot['light_holes']):<6} "
              f"{merged_str:<8} {shot['total_new_holes']}")

    # Create visualization of all detected shots
    print(f"\n🎨 Creating visualization...")
    create_shot_visualization(all_shots, first_frame, target_center, inner_radius)

    return all_shots


def create_shot_visualization(all_shots, reference_frame, target_center, inner_radius):
    """Create visualization showing all detected shots"""
    result = reference_frame.copy()

    # Draw target
    cv2.circle(result, target_center, inner_radius, (100, 100, 100), 2)
    cv2.drawMarker(result, target_center, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)

    shot_num = 0
    for shot in all_shots:
        shot_num += 1

        # Draw dark holes
        for hole in shot['dark_holes']:
            x, y, radius = int(hole[0]), int(hole[1]), int(hole[2])
            color = (0, 255, 0)  # Green for dark

            cv2.circle(result, (x, y), radius + 5, color, 2)
            cv2.circle(result, (x, y), 3, color, -1)

            label = f"S{shot_num}"
            cv2.putText(result, label, (x - 15, y - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw light holes
        for hole in shot['light_holes']:
            x, y, radius = int(hole[0]), int(hole[1]), int(hole[2])
            color = (0, 165, 255)  # Orange for light

            cv2.circle(result, (x, y), radius + 5, color, 2)
            cv2.circle(result, (x, y), 3, color, -1)

            label = f"S{shot_num}"
            cv2.putText(result, label, (x - 15, y - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Add info
    info_height = 100
    info_panel = np.zeros((info_height, result.shape[1], 3), dtype=np.uint8)

    cv2.putText(info_panel, f"Frame-by-Frame Detection: {len(all_shots)} shots detected",
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    total_holes = sum(shot['total_new_holes'] for shot in all_shots)
    cv2.putText(info_panel, f"Total holes: {total_holes}",
               (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    final = np.vstack([info_panel, result])

    # Save
    output_dir = "test_outputs/frame_by_frame"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/frame_by_frame_detection.jpg"
    cv2.imwrite(output_path, final)

    print(f"   ✅ Saved to: {output_path}")


if __name__ == "__main__":
    frame_by_frame_detection()
