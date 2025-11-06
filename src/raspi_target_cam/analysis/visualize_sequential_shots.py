#!/usr/bin/env python3
"""
Visualize sequential shot detection - save annotated frames
"""

import cv2
import numpy as np
from raspi_target_cam.core.perspective import Perspective
from raspi_target_cam.core.target_detection import TargetDetector
import json
import os


def create_annotated_frame(frame, shots, target_center, inner_radius, frame_num, new_shot=None):
    """
    Create annotated frame showing all detected shots and highlighting new one
    """
    result = frame.copy()

    # Draw target
    cv2.circle(result, target_center, inner_radius, (100, 100, 100), 2)
    cv2.drawMarker(result, target_center, (0, 255, 0), cv2.MARKER_CROSS, 30, 2)

    # Draw all previous shots in white
    for shot in shots:
        if new_shot and shot['shot_number'] == new_shot['shot_number']:
            continue  # Skip the new one, we'll highlight it

        x, y = shot['x'], shot['y']
        cv2.circle(result, (x, y), 20, (200, 200, 200), 2)
        cv2.putText(result, f"#{shot['shot_number']}", (x - 15, y - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Highlight new shot in bright color
    if new_shot:
        x, y = new_shot['x'], new_shot['y']
        cv2.circle(result, (x, y), 30, (0, 255, 0), 4)
        cv2.drawMarker(result, (x, y), (0, 255, 0), cv2.MARKER_STAR, 40, 3)

        label = f"NEW SHOT #{new_shot['shot_number']}"
        cv2.putText(result, label, (x - 80, y - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # Add frame info overlay
    info_bg = np.zeros((100, result.shape[1], 3), dtype=np.uint8)
    info_bg[:] = (40, 40, 40)

    cv2.putText(info_bg, f"Frame {frame_num}", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.putText(info_bg, f"Total Shots: {len(shots)}", (20, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if new_shot:
        cv2.putText(info_bg, f"NEW!", (400, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Combine
    annotated = np.vstack([info_bg, result])

    return annotated


def visualize_detection_sequence(video_path, shots_file='sequential_detection_results.json'):
    """
    Create visualizations for each detected shot
    """
    print("🎨 Creating visualizations for sequential detection")
    print("=" * 70)

    # Load detection results
    with open(shots_file, 'r') as f:
        results = json.load(f)

    shots = results['shots']
    reference_frame_num = results['reference_frame']

    print(f"   Loaded {len(shots)} shots from {shots_file}")

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
    output_dir = "test_outputs/sequential_shots"
    os.makedirs(output_dir, exist_ok=True)

    # Save reference frame
    ref_annotated = create_annotated_frame(ref, [], target_center, inner_radius, reference_frame_num)
    cv2.imwrite(f"{output_dir}/frame_{reference_frame_num:04d}_reference.jpg", ref_annotated)
    print(f"   ✅ Saved reference frame {reference_frame_num}")

    # Process each shot detection
    for i, shot in enumerate(shots):
        frame_num = shot['frame_detected']

        # Load frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = perspective.apply_perspective_correction(frame)

        # Create annotated frame showing all shots up to this point
        shots_so_far = shots[:i+1]
        annotated = create_annotated_frame(frame, shots_so_far, target_center, inner_radius,
                                          frame_num, new_shot=shot)

        # Save
        filename = f"{output_dir}/shot_{shot['shot_number']:02d}_frame_{frame_num:04d}.jpg"
        cv2.imwrite(filename, annotated)

        print(f"   ✅ Shot #{shot['shot_number']} at frame {frame_num} -> {filename}")

    # Create a final summary frame with all shots
    if shots:
        final_frame_num = shots[-1]['frame_detected']
        cap.set(cv2.CAP_PROP_POS_FRAMES, final_frame_num)
        ret, final = cap.read()
        final = cv2.rotate(final, cv2.ROTATE_90_COUNTERCLOCKWISE)
        final = perspective.apply_perspective_correction(final)

        final_annotated = create_annotated_frame(final, shots, target_center, inner_radius, final_frame_num)
        cv2.imwrite(f"{output_dir}/final_all_shots.jpg", final_annotated)
        print(f"   ✅ Final summary -> {output_dir}/final_all_shots.jpg")

    cap.release()

    print(f"\n💾 All visualizations saved to: {output_dir}/")
    print(f"✅ Visualization complete!")


if __name__ == "__main__":
    video_path = "samples/10-shot-1.mkv"
    visualize_detection_sequence(video_path)
