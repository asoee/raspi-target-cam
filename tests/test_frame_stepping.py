#!/usr/bin/env python3
"""
Test script for frame stepping functionality.
Tests that frame position updates correctly when stepping forward/backward while paused.
"""

import cv2
import time
from raspi_target_cam.camera.threaded_capture import ThreadedCaptureSystem

def test_frame_stepping():
    """Test frame stepping with a video file"""
    video_path = "./samples/recording_20251009_202417-3fps.mkv"

    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR: Could not open video file")
        return False

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps} FPS")

    # Create threaded capture system
    print("\nCreating threaded capture system...")
    system = ThreadedCaptureSystem(cap, source_type="video", camera_index=0, buffer_size=50)
    system.start()

    # Wait for initial frame
    time.sleep(1.0)

    # Get initial position
    pos = system.get_playback_position()
    print(f"Initial position: Frame {pos['current_frame']}/{pos['total_frames']}")

    # Pause playback
    print("\nPausing playback...")
    system.pause()
    time.sleep(0.5)

    # Test stepping forward
    print("\n=== Testing Step Forward ===")
    for i in range(5):
        pos_before = system.get_playback_position()
        print(f"Before step {i+1}: Frame {pos_before['current_frame']}")

        success = system.step_forward()
        time.sleep(0.3)  # Give time for command to process

        pos_after = system.get_playback_position()
        print(f"After step {i+1}: Frame {pos_after['current_frame']} (success={success})")

        if pos_after['current_frame'] != pos_before['current_frame'] + 1:
            print(f"  ❌ FAIL: Expected frame {pos_before['current_frame'] + 1}, got {pos_after['current_frame']}")
        else:
            print(f"  ✅ PASS: Frame incremented correctly")

    # Test stepping backward
    print("\n=== Testing Step Backward ===")
    for i in range(5):
        pos_before = system.get_playback_position()
        print(f"Before step {i+1}: Frame {pos_before['current_frame']}")

        success = system.step_backward()
        time.sleep(0.3)  # Give time for command to process

        pos_after = system.get_playback_position()
        print(f"After step {i+1}: Frame {pos_after['current_frame']} (success={success})")

        if pos_after['current_frame'] != pos_before['current_frame'] - 1:
            print(f"  ❌ FAIL: Expected frame {pos_before['current_frame'] - 1}, got {pos_after['current_frame']}")
        else:
            print(f"  ✅ PASS: Frame decremented correctly")

    # Cleanup
    print("\nCleaning up...")
    system.stop(timeout=2.0)
    cap.release()

    print("\n=== Test Complete ===")
    return True

if __name__ == '__main__':
    try:
        test_frame_stepping()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
