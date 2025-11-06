#!/usr/bin/env python3
"""
Visualize dark-to-darker detections
"""

import cv2
import numpy as np
from raspi_target_cam.core.perspective import Perspective
from raspi_target_cam.core.target_detection import TargetDetector
import json
import os


def visualize_detections(video_path, results_file='dark_sequential_results.json'):
    """
    Create visualizations for dark-to-darker detections
    """
    print("🎨 Creating visualizations for dark-to-darker detection")
    print("=" * 70)

    # Load detection results
    with open(results_file, 'r') as f:
        results = json.load(f)

    shots = results['shots']
    reference_frame_num = results['reference_frame']

    print(f"   Loaded {len(shots)} shots from {results_file}")

    cap = cv2.VideoCapture(video_path)
    perspective = Perspective()

    # Load reference frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_num)
    ret, ref = cap.read()
    ref = cv2.rotate(ref, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ref = perspective.apply_perspective_correction(ref)

    # Detect target
    target_detector = TargetDetector()
    inner_circle = target_detector.detect_black_circle_improved(ref)
    target_center = (int(inner_circle[0]), int(inner_circle[1]))
    inner_radius = int(inner_circle[2])

    # Create output directory
    output_dir = "test_outputs/dark_sequential"
    os.makedirs(output_dir, exist_ok=True)

    # Group shots by frame
    frames_with_shots = {}
    for shot in shots:
        frame_num = shot['frame_detected']
        if frame_num not in frames_with_shots:
            frames_with_shots[frame_num] = []
        frames_with_shots[frame_num].append(shot)

    # Process key frames
    print(f"\n📸 Creating visualizations...")

    for frame_num in sorted(frames_with_shots.keys()):
        frame_shots = frames_with_shots[frame_num]

        # Load frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = perspective.apply_perspective_correction(frame)

        # Create annotated frame
        result = frame.copy()

        # Draw target
        cv2.circle(result, target_center, inner_radius, (100, 100, 100), 2)

        # Draw shots detected at this frame
        for shot in frame_shots:
            cx, cy = shot['x'], shot['y']

            # Color based on score
            if shot['score'] > 70:
                color = (0, 255, 0)  # Green - high confidence
            elif shot['score'] > 50:
                color = (0, 255, 255)  # Yellow - medium
            else:
                color = (0, 165, 255)  # Orange - low

            cv2.circle(result, (cx, cy), 25, color, 3)
            cv2.drawMarker(result, (cx, cy), color, cv2.MARKER_CROSS, 25, 2)

            # Label
            label = f"#{shot['shot_number']}"
            cv2.putText(result, label, (cx + 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Info
            info = f"score={shot['score']:.0f}"
            cv2.putText(result, info, (cx + 30, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Add header info
        info_bg = np.zeros((100, result.shape[1], 3), dtype=np.uint8)
        info_bg[:] = (40, 40, 40)

        cv2.putText(info_bg, f"Frame {frame_num} ({frame_num/30:.2f}s)", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(info_bg, f"{len(frame_shots)} shot(s) detected at this frame", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Combine
        annotated = np.vstack([info_bg, result])

        # Save
        filename = f"{output_dir}/frame_{frame_num:04d}_{len(frame_shots)}_shots.jpg"
        cv2.imwrite(filename, annotated)

        print(f"   ✅ Frame {frame_num}: {len(frame_shots)} shot(s) -> {filename}")

    cap.release()

    # Create summary with all shots
    print(f"\n📊 Creating final summary...")

    # Use last frame
    last_frame_num = max(frames_with_shots.keys())
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_num)
    ret, final = cap.read()
    final = cv2.rotate(final, cv2.ROTATE_90_COUNTERCLOCKWISE)
    final = perspective.apply_perspective_correction(final)
    cap.release()

    result = final.copy()
    cv2.circle(result, target_center, inner_radius, (100, 100, 100), 2)

    # Draw all shots with numbers
    for shot in shots:
        cx, cy = shot['x'], shot['y']
        cv2.circle(result, (cx, cy), 20, (0, 255, 255), 2)

        label = f"#{shot['shot_number']}"
        cv2.putText(result, label, (cx - 15, cy - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Add info
    info_bg = np.zeros((100, result.shape[1], 3), dtype=np.uint8)
    info_bg[:] = (40, 40, 40)

    cv2.putText(info_bg, f"All {len(shots)} detections", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    annotated = np.vstack([info_bg, result])
    cv2.imwrite(f"{output_dir}/summary_all_shots.jpg", annotated)

    print(f"\n💾 All visualizations saved to: {output_dir}/")
    print(f"✅ Visualization complete!")


if __name__ == "__main__":
    video_path = "samples/10-shot-1.mkv"
    visualize_detections(video_path)
