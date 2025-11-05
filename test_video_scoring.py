#!/usr/bin/env python3
"""
Test Video Scoring
Processes the 10-shot video and attempts to detect and score bullet holes
"""

import cv2
import numpy as np
import yaml
from enhanced_bullet_detector import EnhancedBulletDetector

def load_perspective_matrix(calibration_file="perspective_calibration.yaml"):
    """Load perspective correction matrix from file"""
    with open(calibration_file, 'r') as f:
        calibration = yaml.safe_load(f)

    matrix_data = calibration.get('perspective_matrix')
    if matrix_data:
        return np.array(matrix_data, dtype=np.float32)
    return None

def apply_perspective_correction(frame, matrix):
    """Apply perspective correction to frame"""
    if matrix is None:
        return frame
    return cv2.warpPerspective(frame, matrix, (frame.shape[1], frame.shape[0]))

def test_video_with_manual_detection():
    """
    Test the video by manually stepping through and detecting bullet holes
    Uses frame-by-frame comparison instead of reference comparison
    """
    print("🎯 Testing Video Scoring with Frame-by-Frame Detection")
    print("=" * 60)

    video_path = "samples/10-shot-1.mkv"

    # Load video
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

    # Load perspective matrix
    perspective_matrix = load_perspective_matrix()
    if perspective_matrix is not None:
        print(f"✅ Loaded perspective correction matrix")
    else:
        print(f"⚠️  No perspective correction available")

    # Create detector
    detector = EnhancedBulletDetector()

    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Cannot read first frame")
        return

    # Apply perspective correction to first frame
    first_frame_corrected = apply_perspective_correction(first_frame, perspective_matrix)

    # Start session with first frame
    print("\n🎯 Starting scoring session...")
    detector.start_session(first_frame_corrected, profile_name='large', auto_calibrate=True)

    # Disable auto-detection (we'll manually trigger)
    detector.enable_auto_detection(False)

    print(f"\n📊 Processing frames...")
    print(f"   Looking for bullet holes appearing in the video")
    print()

    # Process frames and look for changes
    frame_num = 0
    prev_frame_corrected = first_frame_corrected.copy()

    # Key frames where we expect shots (approximate based on video analysis)
    # We'll sample at intervals and manually detect
    check_intervals = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300,
                      330, 360, 390, 420, 450, 480, 510, 540, 570, 600,
                      630, 660, 690, 720, 750, 780, 810, 840, 870, 900]

    detected_holes_by_frame = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Apply perspective correction
        frame_corrected = apply_perspective_correction(frame, perspective_matrix)

        # Check at intervals
        if frame_num in check_intervals:
            # Calculate frame difference
            diff_ratio = detector.calculate_frame_difference(prev_frame_corrected, frame_corrected)

            if diff_ratio > 0.001:  # 0.1% change
                # Detect holes in this frame vs first frame
                holes = detector.bullet_detector.detect_bullet_holes(
                    first_frame_corrected,
                    frame_corrected
                )

                if holes:
                    detected_holes_by_frame[frame_num] = holes
                    print(f"   Frame {frame_num:4d} ({frame_num/fps:5.1f}s): "
                          f"{len(holes)} holes detected, "
                          f"frame diff: {diff_ratio:.3%}")

        prev_frame_corrected = frame_corrected.copy()

    cap.release()

    # Now analyze all detected holes and determine which are real bullet holes
    print(f"\n📈 Analysis Results:")
    print(f"   Total frames checked: {len(check_intervals)}")
    print(f"   Frames with detected holes: {len(detected_holes_by_frame)}")

    if detected_holes_by_frame:
        # Get the frame with most holes (likely the final state)
        max_holes_frame = max(detected_holes_by_frame.keys(),
                             key=lambda f: len(detected_holes_by_frame[f]))
        final_holes = detected_holes_by_frame[max_holes_frame]

        print(f"\n   Frame with most holes: {max_holes_frame} ({max_holes_frame/fps:.1f}s)")
        print(f"   Total holes in that frame: {len(final_holes)}")

        # Score these holes
        print(f"\n🎯 Scoring {len(final_holes)} detected holes:")
        for i, hole_data in enumerate(final_holes):
            x, y, radius, confidence = hole_data[:4]

            # Add to scoring system
            shot_data = detector.scoring_system.add_shot_to_current_session(
                x, y, radius, confidence
            )

            if shot_data:
                print(f"      Hole #{shot_data['shot_number']}: "
                      f"Score {shot_data['score']}, "
                      f"Distance {shot_data['distance_from_center']:.1f}px, "
                      f"Confidence {confidence:.2f}")

    # Get final session status
    print(f"\n📊 Final Session Status:")
    status = detector.get_session_status()
    if status:
        print(f"   Total shots scored: {status['shot_count']}")
        print(f"   Total score: {status['total_score']}")
        print(f"   Session duration: {status['duration_seconds']:.1f}s")

    # Save session
    session_file = detector.stop_session()

    print(f"\n✅ Test complete!")
    return status

def test_video_with_auto_detect():
    """
    Simpler test: just use the enhanced detector's auto-detection
    but feed it pre-corrected frames
    """
    print("🎯 Testing Video with Auto-Detection (Pre-Corrected Frames)")
    print("=" * 60)

    video_path = "samples/10-shot-1.mkv"

    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"📹 Video: {video_path}")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps}")

    # Load perspective matrix
    perspective_matrix = load_perspective_matrix()
    if perspective_matrix is not None:
        print(f"✅ Loaded perspective correction matrix")
    else:
        print(f"⚠️  No perspective correction - using raw frames")

    # Create detector (disable internal perspective correction since we're doing it here)
    detector = EnhancedBulletDetector()
    detector.perspective_enabled = False  # We're providing pre-corrected frames

    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Cannot read first frame")
        return

    # Apply perspective correction
    first_frame_corrected = apply_perspective_correction(first_frame, perspective_matrix)

    # Start session
    detector.start_session(first_frame_corrected, profile_name='large')
    detector.enable_auto_detection(True)

    # Lower the frame difference threshold for more sensitivity
    detector.frame_diff_threshold = 0.003  # 0.3% change
    detector.min_frames_between_shots = 10  # Allow more frequent detection

    print(f"\n📊 Processing {total_frames} frames...")

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Apply perspective correction
        frame_corrected = apply_perspective_correction(frame, perspective_matrix)

        # Process with auto-detection
        new_shots, _ = detector.process_frame(frame_corrected)

        if frame_num % 100 == 0:
            print(f"   Processed {frame_num}/{total_frames} frames...")

    cap.release()

    # Get results
    print(f"\n📊 Final Results:")
    status = detector.get_session_status()
    if status:
        print(f"   Total shots detected: {status['shot_count']}")
        print(f"   Total score: {status['total_score']}")

        if status['shots']:
            print(f"\n   Shot details:")
            for shot in status['shots']:
                print(f"      Shot #{shot['shot_number']}: "
                      f"Score {shot['score']}, "
                      f"Position ({shot['x']}, {shot['y']}), "
                      f"Distance {shot['distance_from_center']:.1f}px")

    detector.stop_session()

    print(f"\n✅ Test complete!")

if __name__ == "__main__":
    print("Select test mode:")
    print("1. Manual frame-by-frame detection")
    print("2. Auto-detection with pre-corrected frames")

    # For now, run both
    print("\n" + "=" * 60)
    print("Running Test 1: Manual Detection")
    print("=" * 60 + "\n")
    test_video_with_manual_detection()

    print("\n\n" + "=" * 60)
    print("Running Test 2: Auto-Detection")
    print("=" * 60 + "\n")
    test_video_with_auto_detect()
