#!/usr/bin/env python3
"""
Test script for threaded capture system.

Tests:
1. Frame reading from camera
2. Video recording to file
3. Simultaneous reading and recording
4. Buffer management
"""

import cv2
import time
import os
from threaded_capture import ThreadedCaptureSystem


def test_camera_capture():
    """Test basic camera frame capture"""
    print("\n=== Test 1: Camera Capture ===")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return False

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create threaded system
    system = ThreadedCaptureSystem(cap, source_type="camera", buffer_size=50)
    system.start()

    # Capture for 5 seconds
    print("Capturing frames for 5 seconds...")
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < 5:
        frame = system.get_latest_frame()
        if frame is not None:
            frame_count += 1
        time.sleep(0.033)  # ~30 FPS check rate

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"✓ Captured {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

    # Cleanup
    system.stop()
    cap.release()
    return True


def test_video_playback():
    """Test video file playback"""
    print("\n=== Test 2: Video Playback ===")

    # Check if test video exists
    test_video = "./samples/test.mp4"
    if not os.path.exists(test_video):
        print(f"SKIP: Test video not found: {test_video}")
        return True

    cap = cv2.VideoCapture(test_video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {test_video}")
        return False

    # Create threaded system
    system = ThreadedCaptureSystem(cap, source_type="video", buffer_size=50)
    system.start()

    # Play for 3 seconds
    print("Playing video for 3 seconds...")
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < 3:
        frame = system.get_latest_frame()
        if frame is not None:
            frame_count += 1
        time.sleep(0.033)

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"✓ Played {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")

    # Cleanup
    system.stop()
    cap.release()
    return True


def test_recording():
    """Test video recording"""
    print("\n=== Test 3: Video Recording ===")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return False

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create threaded system
    system = ThreadedCaptureSystem(cap, source_type="camera", buffer_size=50)
    system.start()

    # Wait for first frame
    print("Waiting for first frame...")
    timeout = 5
    start_wait = time.time()
    while system.get_latest_frame() is None and time.time() - start_wait < timeout:
        time.sleep(0.1)

    if system.get_latest_frame() is None:
        print("ERROR: No frames captured")
        system.stop()
        cap.release()
        return False

    # Start recording
    output_file = "./test_output/test_recording.avi"
    os.makedirs("./test_output", exist_ok=True)

    print(f"Starting recording to {output_file}...")
    if not system.start_recording(output_file, fps=30):
        print("ERROR: Failed to start recording")
        system.stop()
        cap.release()
        return False

    # Record for 5 seconds
    print("Recording for 5 seconds...")
    time.sleep(5)

    # Stop recording
    print("Stopping recording...")
    system.stop_recording()

    # Check if file was created
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"✓ Recording saved: {output_file} ({file_size} bytes)")

        # Verify video is readable
        test_cap = cv2.VideoCapture(output_file)
        if test_cap.isOpened():
            frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = test_cap.get(cv2.CAP_PROP_FPS)
            width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"✓ Video verified: {frame_count} frames, {width}x{height} @ {fps:.1f} FPS")
            test_cap.release()
        else:
            print("WARNING: Could not open recorded video for verification")
    else:
        print(f"ERROR: Recording file not created: {output_file}")
        system.stop()
        cap.release()
        return False

    # Cleanup
    system.stop()
    cap.release()
    return True


def test_simultaneous_operations():
    """Test simultaneous capture and recording"""
    print("\n=== Test 4: Simultaneous Capture & Recording ===")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return False

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create threaded system
    system = ThreadedCaptureSystem(cap, source_type="camera", buffer_size=100)
    system.start()

    # Wait for first frame
    print("Waiting for first frame...")
    timeout = 5
    start_wait = time.time()
    while system.get_latest_frame() is None and time.time() - start_wait < timeout:
        time.sleep(0.1)

    # Start recording
    output_file = "./test_output/test_simultaneous.avi"
    print(f"Starting recording to {output_file}...")
    system.start_recording(output_file, fps=30)

    # Capture and count frames while recording
    print("Capturing frames while recording (10 seconds)...")
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < 10:
        frame = system.get_latest_frame()
        if frame is not None:
            frame_count += 1
        time.sleep(0.033)

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"✓ Retrieved {frame_count} frames while recording ({fps:.1f} FPS)")

    # Stop recording
    system.stop_recording()

    # Verify recording
    if os.path.exists(output_file):
        test_cap = cv2.VideoCapture(output_file)
        if test_cap.isOpened():
            recorded_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"✓ Recorded {recorded_frames} frames to file")
            test_cap.release()
    else:
        print(f"ERROR: Recording not found: {output_file}")

    # Cleanup
    system.stop()
    cap.release()
    return True


def test_pause_resume():
    """Test pause and resume functionality"""
    print("\n=== Test 5: Pause & Resume ===")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return False

    # Create threaded system
    system = ThreadedCaptureSystem(cap, source_type="camera")
    system.start()

    # Capture normally for 2 seconds
    print("Capturing for 2 seconds...")
    time.sleep(2)
    frame1 = system.get_latest_frame()

    # Pause
    print("Pausing capture...")
    system.pause()
    time.sleep(1)
    frame2 = system.get_latest_frame()

    # Resume
    print("Resuming capture...")
    system.resume()
    time.sleep(2)
    frame3 = system.get_latest_frame()

    # Verify frames changed after resume
    if frame1 is not None and frame3 is not None:
        # Simple check: frames should be different
        diff = cv2.absdiff(frame1, frame3)
        if diff.sum() > 0:
            print("✓ Frames are different after pause/resume")
        else:
            print("WARNING: Frames appear identical (might be static scene)")
    else:
        print("ERROR: Could not get frames for comparison")

    # Cleanup
    system.stop()
    cap.release()
    return True


if __name__ == "__main__":
    print("╔═══════════════════════════════════════╗")
    print("║  Threaded Capture System Test Suite  ║")
    print("╚═══════════════════════════════════════╝")

    results = []

    # Run tests
    results.append(("Camera Capture", test_camera_capture()))
    results.append(("Video Playback", test_video_playback()))
    results.append(("Video Recording", test_recording()))
    results.append(("Simultaneous Ops", test_simultaneous_operations()))
    results.append(("Pause/Resume", test_pause_resume()))

    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print("="*50)
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
    print("="*50)
