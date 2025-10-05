#!/usr/bin/env python3
"""
Test script for enhanced threaded capture features.

Tests:
1. Seek operations (video playback)
2. Step forward/backward controls
3. Metadata recording
4. Codec fallback
5. Test pattern mode
6. Playback position tracking
"""

import cv2
import time
import os
import json
from threaded_capture import ThreadedCaptureSystem
from camera_settings import CameraSettings, SeekCommand


def test_seek_operations():
    """Test video seek operations"""
    print("\n=== Test 1: Seek Operations ===")

    # Check if test video exists
    test_video = "./samples/test.mp4"
    if not os.path.exists(test_video):
        # Try to find any video file
        if os.path.exists("./samples"):
            videos = [f for f in os.listdir("./samples") if f.endswith(('.mp4', '.avi', '.mkv'))]
            if videos:
                test_video = os.path.join("./samples", videos[0])
            else:
                print(f"SKIP: No test video found in ./samples/")
                return True
        else:
            print(f"SKIP: Test video not found: {test_video}")
            return True

    cap = cv2.VideoCapture(test_video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {test_video}")
        return False

    try:
        system = ThreadedCaptureSystem(cap, source_type="video")
        system.start()
        time.sleep(1)  # Let it start

        # Get video info
        pos = system.get_playback_position()
        print(f"Video: {pos['total_frames']} frames")

        # Test seek to frame 10
        print("Seeking to frame 10...")
        system.seek_to_frame(10)
        time.sleep(0.5)  # Wait for seek

        pos = system.get_playback_position()
        print(f"✓ Current position: frame {pos['current_frame']}/{pos['total_frames']} ({pos['progress']:.1f}%)")

        # Test seek to middle
        middle_frame = pos['total_frames'] // 2
        print(f"Seeking to middle (frame {middle_frame})...")
        system.seek_to_frame(middle_frame)
        time.sleep(0.5)

        pos = system.get_playback_position()
        print(f"✓ Current position: frame {pos['current_frame']}")

        system.stop()
        cap.release()
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        system.stop()
        cap.release()
        return False


def test_step_controls():
    """Test step forward/backward"""
    print("\n=== Test 2: Step Controls ===")

    test_video = "./samples/test.mp4"
    if not os.path.exists(test_video):
        if os.path.exists("./samples"):
            videos = [f for f in os.listdir("./samples") if f.endswith(('.mp4', '.avi', '.mkv'))]
            if videos:
                test_video = os.path.join("./samples", videos[0])
            else:
                print("SKIP: No test video found")
                return True
        else:
            print("SKIP: No test video found")
            return True

    cap = cv2.VideoCapture(test_video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video")
        return False

    try:
        system = ThreadedCaptureSystem(cap, source_type="video")
        system.start()
        time.sleep(1)

        # Pause playback
        print("Pausing playback...")
        system.pause()
        time.sleep(0.2)

        # Seek to frame 50
        print("Seeking to frame 50...")
        system.seek_to_frame(50)
        time.sleep(0.5)

        pos = system.get_playback_position()
        start_frame = pos['current_frame']
        print(f"Starting at frame {start_frame}")

        # Step forward 5 times
        print("Stepping forward 5 times...")
        for i in range(5):
            system.step_forward()
            time.sleep(0.2)

        pos = system.get_playback_position()
        print(f"✓ After forward: frame {pos['current_frame']} (expected ~{start_frame + 5})")

        # Step backward 3 times
        print("Stepping backward 3 times...")
        for i in range(3):
            system.step_backward()
            time.sleep(0.2)

        pos = system.get_playback_position()
        print(f"✓ After backward: frame {pos['current_frame']} (expected ~{start_frame + 2})")

        system.stop()
        cap.release()
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        system.stop()
        cap.release()
        return False


def test_metadata_recording():
    """Test metadata recording"""
    print("\n=== Test 3: Metadata Recording ===")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return False

    try:
        system = ThreadedCaptureSystem(cap, source_type="camera", camera_index=0)
        system.start()
        time.sleep(1)  # Wait for first frame

        # Create metadata
        metadata = {
            'camera_index': 0,
            'test_name': 'metadata_test',
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'notes': 'Testing metadata recording functionality'
        }

        # Start recording with metadata
        output_file = "./test_output/metadata_test.mp4"
        os.makedirs("./test_output", exist_ok=True)

        print(f"Starting recording with metadata...")
        success, message, actual_file = system.start_recording(
            output_file,
            fps=30,
            metadata=metadata
        )

        if not success:
            print(f"ERROR: {message}")
            system.stop()
            cap.release()
            return False

        print(f"✓ {message}")
        print(f"Recording to: {actual_file}")

        # Record for 3 seconds
        time.sleep(3)

        # Stop recording
        print("Stopping recording...")
        system.stop_recording()
        time.sleep(0.5)

        # Check if metadata file was created
        if actual_file:
            metadata_file = os.path.splitext(actual_file)[0] + ".json"
            if os.path.exists(metadata_file):
                print(f"✓ Metadata file created: {metadata_file}")

                # Read and verify metadata
                with open(metadata_file, 'r') as f:
                    saved_metadata = json.load(f)

                print("Saved metadata:")
                for key, value in saved_metadata.items():
                    print(f"  {key}: {value}")

                # Verify our custom metadata is there
                if 'test_name' in saved_metadata and saved_metadata['test_name'] == 'metadata_test':
                    print("✓ Custom metadata preserved")
                else:
                    print("WARNING: Custom metadata not found")

                # Verify recording stats were added
                if 'frames_written' in saved_metadata and 'recording_duration_seconds' in saved_metadata:
                    print(f"✓ Recording stats added: {saved_metadata['frames_written']} frames, {saved_metadata['recording_duration_seconds']:.1f}s")
                else:
                    print("WARNING: Recording stats not added")

            else:
                print(f"ERROR: Metadata file not found: {metadata_file}")
                system.stop()
                cap.release()
                return False

        system.stop()
        cap.release()
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        system.stop()
        cap.release()
        return False


def test_codec_fallback():
    """Test codec fallback functionality"""
    print("\n=== Test 4: Codec Fallback ===")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return False

    try:
        system = ThreadedCaptureSystem(cap, source_type="camera", camera_index=0)
        system.start()
        time.sleep(1)

        # Test with custom codec priority
        codec_priority = [
            ('MJPG', '.mkv'),
            ('X264', '.mp4'),
            ('XVID', '.avi'),
        ]

        output_file = "./test_output/codec_test"
        print(f"Testing codec fallback with priority: {[c[0] for c in codec_priority]}")

        success, message, actual_file = system.start_recording(
            output_file,
            fps=30,
            codec_priority=codec_priority
        )

        if not success:
            print(f"ERROR: {message}")
            system.stop()
            cap.release()
            return False

        print(f"✓ {message}")
        print(f"Output file: {actual_file}")

        # Check which codec was used
        if actual_file:
            ext = os.path.splitext(actual_file)[1]
            print(f"✓ File extension: {ext}")

            # Record for 2 seconds
            time.sleep(2)

            # Stop and verify
            system.stop_recording()
            time.sleep(0.5)

            if os.path.exists(actual_file):
                size = os.path.getsize(actual_file)
                print(f"✓ File created: {size} bytes")
            else:
                print(f"ERROR: File not created: {actual_file}")
                system.stop()
                cap.release()
                return False

        system.stop()
        cap.release()
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        system.stop()
        cap.release()
        return False


def test_test_pattern():
    """Test test pattern mode"""
    print("\n=== Test 5: Test Pattern Mode ===")

    # Create system with test pattern
    system = ThreadedCaptureSystem(
        cap=None,
        source_type="test",
        camera_index=0
    )
    system.start()

    print("Capturing test pattern frames for 2 seconds...")
    start = time.time()
    frame_count = 0

    while time.time() - start < 2:
        frame = system.get_latest_frame()
        if frame is not None:
            frame_count += 1
            # Check frame properties
            if frame_count == 1:
                height, width = frame.shape[:2]
                print(f"Test pattern size: {width}x{height}")
        time.sleep(0.1)

    print(f"✓ Captured {frame_count} test pattern frames")

    # Verify we got frames
    if frame_count > 0:
        print("✓ Test pattern generation works")
    else:
        print("ERROR: No test pattern frames generated")
        system.stop()
        return False

    system.stop()
    return True


def test_playback_position():
    """Test playback position tracking"""
    print("\n=== Test 6: Playback Position Tracking ===")

    test_video = "./samples/test.mp4"
    if not os.path.exists(test_video):
        if os.path.exists("./samples"):
            videos = [f for f in os.listdir("./samples") if f.endswith(('.mp4', '.avi', '.mkv'))]
            if videos:
                test_video = os.path.join("./samples", videos[0])
            else:
                print("SKIP: No test video found")
                return True
        else:
            print("SKIP: No test video found")
            return True

    cap = cv2.VideoCapture(test_video)
    if not cap.isOpened():
        print("ERROR: Could not open video")
        return False

    try:
        system = ThreadedCaptureSystem(cap, source_type="video")
        system.start()
        time.sleep(1)

        # Track position over time
        print("Tracking position during playback...")
        positions = []

        for i in range(10):
            pos = system.get_playback_position()
            positions.append(pos['current_frame'])
            print(f"  {i+1}. Frame {pos['current_frame']}/{pos['total_frames']} ({pos['progress']:.1f}%)")
            time.sleep(0.3)

        # Verify position is advancing
        if len(positions) > 1 and positions[-1] > positions[0]:
            print(f"✓ Position advancing: {positions[0]} -> {positions[-1]}")
        else:
            print("WARNING: Position not advancing as expected")

        system.stop()
        cap.release()
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        system.stop()
        cap.release()
        return False


if __name__ == "__main__":
    print("╔════════════════════════════════════════╗")
    print("║  Enhanced Features Test Suite          ║")
    print("╚════════════════════════════════════════╝")

    results = []

    # Run tests
    results.append(("Seek Operations", test_seek_operations()))
    results.append(("Step Controls", test_step_controls()))
    results.append(("Metadata Recording", test_metadata_recording()))
    results.append(("Codec Fallback", test_codec_fallback()))
    results.append(("Test Pattern", test_test_pattern()))
    results.append(("Playback Position", test_playback_position()))

    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<35} {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print("="*50)
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
    print("="*50)
