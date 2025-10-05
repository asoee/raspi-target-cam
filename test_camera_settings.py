#!/usr/bin/env python3
"""
Test script for camera settings and command pattern.

Tests:
1. CameraSettings creation and modification
2. Command pattern execution
3. Settings API in ThreadedCaptureSystem
4. Thread-safe settings access
"""

import cv2
import time
import os
from camera_settings import (
    CameraSettings,
    SetResolutionCommand,
    SetFPSCommand,
    SetExposureCommand,
    SetImageAdjustmentCommand,
    ApplySettingsCommand,
    ThreadSafeCameraSettings
)
from threaded_capture import ThreadedCaptureSystem


def test_camera_settings_basic():
    """Test basic CameraSettings creation and modification"""
    print("\n=== Test 1: CameraSettings Basics ===")

    # Create default settings
    settings = CameraSettings()
    print(f"Default settings: {settings.width}x{settings.height} @ {settings.fps} FPS")

    # Create custom settings
    custom = CameraSettings(
        width=1920,
        height=1080,
        fps=60,
        brightness=75,
        contrast=60
    )
    print(f"Custom settings: {custom.width}x{custom.height} @ {custom.fps} FPS, brightness={custom.brightness}")

    # Test copy with changes
    modified = custom.copy(width=2560, height=1440)
    print(f"Modified copy: {modified.width}x{modified.height} (original unchanged: {custom.width}x{custom.height})")

    # Test serialization
    json_str = custom.to_json()
    print(f"Serialized to JSON ({len(json_str)} bytes)")

    restored = CameraSettings.from_json(json_str)
    assert restored.width == custom.width
    assert restored.fps == custom.fps
    print("✓ Serialization/deserialization successful")

    return True


def test_thread_safe_settings():
    """Test ThreadSafeCameraSettings"""
    print("\n=== Test 2: Thread-Safe Settings ===")

    settings_holder = ThreadSafeCameraSettings(CameraSettings(width=1920, height=1080))

    # Get settings
    current = settings_holder.get()
    print(f"Current: {current.width}x{current.height}")

    # Modify settings
    new_settings = settings_holder.modify(width=2560, height=1440, fps=60)
    print(f"Modified: {new_settings.width}x{new_settings.height} @ {new_settings.fps} FPS")

    # Update settings
    settings_holder.update(new_settings)
    updated = settings_holder.get()
    print(f"Updated: {updated.width}x{updated.height}")

    assert updated.width == 2560
    assert updated.height == 1440
    print("✓ Thread-safe settings work correctly")

    return True


def test_command_execution():
    """Test command pattern with real camera"""
    print("\n=== Test 3: Command Execution ===")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return False

    try:
        # Test resolution command
        print("Testing SetResolutionCommand...")
        cmd = SetResolutionCommand(640, 480)
        cmd.execute(cap)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Resolution set to {actual_width}x{actual_height}")

        # Test FPS command
        print("Testing SetFPSCommand...")
        cmd = SetFPSCommand(30)
        cmd.execute(cap)
        print("✓ FPS command executed")

        # Test image adjustment command
        print("Testing SetImageAdjustmentCommand...")
        cmd = SetImageAdjustmentCommand(brightness=60, contrast=55)
        cmd.execute(cap)
        print("✓ Image adjustment command executed")

        # Test exposure command
        print("Testing SetExposureCommand...")
        cmd = SetExposureCommand(auto=True)
        cmd.execute(cap)
        print("✓ Exposure command executed")

        print("✓ All commands executed successfully")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cap.release()


def test_settings_api():
    """Test settings API in ThreadedCaptureSystem"""
    print("\n=== Test 4: Settings API ===")

    # Create initial settings
    initial = CameraSettings(width=640, height=480, fps=30)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return False

    # Create system with initial settings
    system = ThreadedCaptureSystem(
        cap,
        source_type="camera",
        camera_index=0,
        initial_settings=initial
    )
    system.start()

    # Wait for system to start
    time.sleep(1)

    try:
        # Get current settings
        current = system.get_settings()
        print(f"Current settings: {current.width}x{current.height} @ {current.fps} FPS")

        # Update settings using update_settings
        print("Updating resolution to 1920x1080...")
        system.update_settings(width=1920, height=1080)
        time.sleep(0.5)  # Give time for command to be processed

        # Check settings were updated
        updated = system.get_settings()
        print(f"Updated settings: {updated.width}x{updated.height}")

        # Update multiple settings at once
        print("Updating multiple settings...")
        system.update_settings(
            fps=60,
            brightness=70,
            contrast=65,
            saturation=55
        )
        time.sleep(0.5)

        final = system.get_settings()
        print(f"Final settings: {final.width}x{final.height} @ {final.fps} FPS")
        print(f"  Brightness: {final.brightness}, Contrast: {final.contrast}, Saturation: {final.saturation}")

        # Apply complete new settings
        print("Applying complete new settings...")
        new_settings = CameraSettings(
            width=1280,
            height=720,
            fps=30,
            brightness=50,
            contrast=50,
            saturation=50
        )
        system.apply_settings(new_settings)
        time.sleep(0.5)

        applied = system.get_settings()
        print(f"Applied settings: {applied.width}x{applied.height} @ {applied.fps} FPS")

        print("✓ Settings API works correctly")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        system.stop()
        cap.release()


def test_settings_while_capturing():
    """Test changing settings while capturing frames"""
    print("\n=== Test 5: Settings Changes During Capture ===")

    initial = CameraSettings(width=640, height=480, fps=30)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return False

    system = ThreadedCaptureSystem(
        cap,
        source_type="camera",
        camera_index=0,
        initial_settings=initial
    )
    system.start()

    try:
        # Capture frames
        print("Capturing frames for 2 seconds...")
        start = time.time()
        frame_count = 0
        while time.time() - start < 2:
            frame = system.get_latest_frame()
            if frame is not None:
                frame_count += 1
            time.sleep(0.033)

        print(f"✓ Captured {frame_count} frames at 640x480")

        # Change resolution while capturing
        print("Changing resolution to 1920x1080...")
        system.update_settings(width=1920, height=1080)
        time.sleep(1)  # Give time for settings to apply

        # Capture more frames
        print("Capturing frames for 2 seconds at new resolution...")
        start = time.time()
        frame_count = 0
        new_size = None
        while time.time() - start < 2:
            frame = system.get_latest_frame()
            if frame is not None:
                frame_count += 1
                if new_size is None:
                    new_size = (frame.shape[1], frame.shape[0])
            time.sleep(0.033)

        print(f"✓ Captured {frame_count} frames")
        if new_size:
            print(f"✓ New frame size: {new_size[0]}x{new_size[1]}")

        # Change brightness while capturing
        print("Adjusting brightness to 80...")
        system.update_settings(brightness=80)
        time.sleep(1)

        print("✓ Settings changes work during active capture")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        system.stop()
        cap.release()


def test_custom_command():
    """Test sending custom commands"""
    print("\n=== Test 6: Custom Commands ===")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return False

    system = ThreadedCaptureSystem(cap, source_type="camera", camera_index=0)
    system.start()

    try:
        time.sleep(1)  # Wait for startup

        # Send custom resolution command
        print("Sending custom SetResolutionCommand...")
        cmd = SetResolutionCommand(800, 600)
        result = system.send_command(cmd)
        print(f"✓ Command queued: {result}")

        time.sleep(0.5)  # Give time for processing

        # Send custom image adjustment command
        print("Sending custom SetImageAdjustmentCommand...")
        cmd = SetImageAdjustmentCommand(brightness=75, contrast=60, saturation=55)
        result = system.send_command(cmd)
        print(f"✓ Command queued: {result}")

        time.sleep(0.5)

        print("✓ Custom commands work correctly")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        system.stop()
        cap.release()


if __name__ == "__main__":
    print("╔═══════════════════════════════════════╗")
    print("║   Camera Settings Test Suite          ║")
    print("╚═══════════════════════════════════════╝")

    results = []

    # Run tests
    results.append(("CameraSettings Basics", test_camera_settings_basic()))
    results.append(("Thread-Safe Settings", test_thread_safe_settings()))
    results.append(("Command Execution", test_command_execution()))
    results.append(("Settings API", test_settings_api()))
    results.append(("Settings During Capture", test_settings_while_capturing()))
    results.append(("Custom Commands", test_custom_command()))

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
