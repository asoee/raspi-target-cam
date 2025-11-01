#!/usr/bin/env python3
"""
Test script for time-based stepping functionality.
Tests that stepping by seconds works correctly and respects frame rate.
"""

import cv2
import time
from threaded_capture import ThreadedCaptureSystem

def test_step_seconds():
    """Test stepping by seconds with a video file"""
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
    print(f"Expected frames per second: {fps}")

    # Create threaded capture system
    print("\nCreating threaded capture system...")
    system = ThreadedCaptureSystem(cap, source_type="video", camera_index=0, buffer_size=50)
    system.start()

    # Wait for initial frame
    time.sleep(1.0)

    # Pause playback
    print("\nPausing playback...")
    system.pause()
    time.sleep(0.5)

    # Get initial position
    pos = system.get_playback_position()
    start_frame = pos['current_frame']
    print(f"Starting from frame {start_frame}")

    # Test stepping forward 1 second
    print(f"\n=== Test: Step forward 1 second (should be ~{int(fps)} frames) ===")
    success = system.step_seconds(1.0, forward=True)
    time.sleep(0.5)
    pos = system.get_playback_position()
    expected_frame = start_frame + int(fps)
    print(f"Expected: frame {expected_frame}, Actual: frame {pos['current_frame']}")
    if abs(pos['current_frame'] - expected_frame) <= 1:  # Allow 1 frame tolerance
        print("✅ PASS")
    else:
        print("❌ FAIL")

    # Test stepping backward 1 second
    print(f"\n=== Test: Step backward 1 second (should be ~{int(fps)} frames) ===")
    before = pos['current_frame']
    success = system.step_seconds(1.0, forward=False)
    time.sleep(0.5)
    pos = system.get_playback_position()
    expected_frame = before - int(fps)
    print(f"Expected: frame {expected_frame}, Actual: frame {pos['current_frame']}")
    if abs(pos['current_frame'] - expected_frame) <= 1:
        print("✅ PASS")
    else:
        print("❌ FAIL")

    # Test stepping forward 10 seconds
    print(f"\n=== Test: Step forward 10 seconds (should be ~{int(fps * 10)} frames) ===")
    before = pos['current_frame']
    success = system.step_seconds(10.0, forward=True)
    time.sleep(0.5)
    pos = system.get_playback_position()
    expected_frame = before + int(fps * 10)
    print(f"Expected: frame {expected_frame}, Actual: frame {pos['current_frame']}")
    if abs(pos['current_frame'] - expected_frame) <= 1:
        print("✅ PASS")
    else:
        print("❌ FAIL")

    # Test stepping backward 10 seconds
    print(f"\n=== Test: Step backward 10 seconds (should be ~{int(fps * 10)} frames) ===")
    before = pos['current_frame']
    success = system.step_seconds(10.0, forward=False)
    time.sleep(0.5)
    pos = system.get_playback_position()
    expected_frame = before - int(fps * 10)
    print(f"Expected: frame {expected_frame}, Actual: frame {pos['current_frame']}")
    if abs(pos['current_frame'] - expected_frame) <= 1:
        print("✅ PASS")
    else:
        print("❌ FAIL")

    # Test boundary: step beyond end
    print(f"\n=== Test: Step beyond end (should clamp to last frame) ===")
    success = system.step_seconds(10000.0, forward=True)
    time.sleep(0.5)
    pos = system.get_playback_position()
    expected_frame = total_frames - 1
    print(f"Expected: frame {expected_frame}, Actual: frame {pos['current_frame']}")
    if pos['current_frame'] == expected_frame:
        print("✅ PASS")
    else:
        print("❌ FAIL")

    # Test boundary: step beyond beginning
    print(f"\n=== Test: Step beyond beginning (should clamp to frame 0) ===")
    success = system.step_seconds(10000.0, forward=False)
    time.sleep(0.5)
    pos = system.get_playback_position()
    print(f"Expected: frame 0, Actual: frame {pos['current_frame']}")
    if pos['current_frame'] == 0:
        print("✅ PASS")
    else:
        print("❌ FAIL")

    # Cleanup
    print("\nCleaning up...")
    system.stop(timeout=2.0)
    cap.release()

    print("\n=== Test Complete ===")
    return True

if __name__ == '__main__':
    try:
        test_step_seconds()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
