#!/usr/bin/env python3
"""
Extract key frames from the 10-shot video for testing
Creates a small test set with:
- Frame 0: Clean target (no holes)
- Frame 300: After a few shots
- Frame 600: After more shots
- Frame 900: Near end (most holes)
"""

import cv2
import numpy as np
import yaml
import os

def load_perspective_matrix(calibration_file="perspective_calibration.yaml"):
    """Load perspective correction matrix from file"""
    with open(calibration_file, 'r') as f:
        calibration = yaml.safe_load(f)

    matrix_data = calibration.get('perspective_matrix')
    if matrix_data:
        return np.array(matrix_data, dtype=np.float32)
    return None

def rotate_frame(frame, rotation_code=cv2.ROTATE_90_COUNTERCLOCKWISE):
    """Rotate frame 90 degrees anti-clockwise"""
    return cv2.rotate(frame, rotation_code)

def apply_perspective_correction(frame, matrix):
    """Apply perspective correction to frame"""
    if matrix is None:
        return frame
    return cv2.warpPerspective(frame, matrix, (frame.shape[1], frame.shape[0]))

def extract_test_frames():
    """Extract key frames from video for testing"""
    print("🎯 Extracting Test Frames from Video")
    print("=" * 50)

    video_path = "samples/10-shot-1.mkv"
    output_dir = "test_frames"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load perspective matrix
    perspective_matrix = load_perspective_matrix()
    if perspective_matrix is not None:
        print(f"✅ Loaded perspective correction matrix")
    else:
        print(f"⚠️  No perspective correction available")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"📹 Video: {video_path}")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   Duration: {total_frames/fps:.1f}s")

    # Frames to extract
    # Based on the video: 10 shots in 34 seconds ≈ 1 shot every 3.4 seconds
    frames_to_extract = {
        0: "clean_target",
        100: "after_shot_1_approx",  # ~3.3s
        300: "after_shot_3_approx",  # ~10s
        500: "mid_session",           # ~16.7s
        700: "late_session",          # ~23.3s
        900: "near_end",              # ~30s
        1018: "final_frame"           # Last frame before target removal
    }

    print(f"\n📸 Extracting {len(frames_to_extract)} key frames...")

    frame_num = 0
    extracted = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num in frames_to_extract:
            frame_name = frames_to_extract[frame_num]

            # Save raw frame
            raw_path = os.path.join(output_dir, f"frame_{frame_num:04d}_{frame_name}_raw.jpg")
            cv2.imwrite(raw_path, frame)

            # Apply rotation (90 degrees anti-clockwise)
            frame_rotated = rotate_frame(frame)

            # Save rotated frame
            rotated_path = os.path.join(output_dir, f"frame_{frame_num:04d}_{frame_name}_rotated.jpg")
            cv2.imwrite(rotated_path, frame_rotated)

            # Apply perspective correction to rotated frame and save
            if perspective_matrix is not None:
                frame_corrected = apply_perspective_correction(frame_rotated, perspective_matrix)
                corrected_path = os.path.join(output_dir, f"frame_{frame_num:04d}_{frame_name}_corrected.jpg")
                cv2.imwrite(corrected_path, frame_corrected)

                extracted[frame_name] = {
                    'frame_num': frame_num,
                    'time': frame_num / fps,
                    'raw_path': raw_path,
                    'corrected_path': corrected_path
                }

                print(f"   ✅ Frame {frame_num:4d} ({frame_num/fps:5.1f}s): {frame_name}")
            else:
                extracted[frame_name] = {
                    'frame_num': frame_num,
                    'time': frame_num / fps,
                    'raw_path': raw_path,
                    'corrected_path': None
                }

                print(f"   ✅ Frame {frame_num:4d} ({frame_num/fps:5.1f}s): {frame_name} (raw only)")

        frame_num += 1

    cap.release()

    print(f"\n✅ Extracted {len(extracted)} frames to {output_dir}/")
    print(f"\n📋 Test Frame Summary:")
    for name, info in extracted.items():
        print(f"   {name:20s}: Frame {info['frame_num']:4d} ({info['time']:5.1f}s)")

    # Create a simple test script suggestion
    print(f"\n💡 Suggested Test Approach:")
    print(f"   1. Manually inspect frames in {output_dir}/")
    print(f"   2. Identify which frames have new bullet holes")
    print(f"   3. Mark approximate hole locations")
    print(f"   4. Test detection on frame pairs:")
    print(f"      - clean_target vs after_shot_1")
    print(f"      - after_shot_1 vs mid_session")
    print(f"      - mid_session vs near_end")

    return extracted

if __name__ == "__main__":
    extract_test_frames()
