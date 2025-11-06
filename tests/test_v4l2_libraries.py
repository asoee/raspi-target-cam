#!/usr/bin/env python3
"""
Test various Python libraries that can use V4L2
"""

import sys
import subprocess

def check_library_availability():
    """Check which V4L2 Python libraries are available"""
    print("=== V4L2 PYTHON LIBRARY AVAILABILITY ===")

    libraries = [
        ("opencv-python", "cv2", "OpenCV with V4L2 backend"),
        ("v4l2py", "v4l2py", "Pure Python V4L2 wrapper"),
        ("python-v4l2capture", "v4l2capture", "Simple V4L2 capture library"),
        ("linuxpy", "linuxpy.video.device", "Linux Video/V4L2 wrapper"),
        ("pyv4l2", "v4l2", "Another V4L2 Python wrapper"),
        ("imageio", "imageio", "ImageIO with V4L2 support"),
    ]

    available_libs = []

    for package, import_name, description in libraries:
        try:
            __import__(import_name)
            print(f"✓ {package}: {description}")
            available_libs.append((package, import_name, description))
        except ImportError:
            print(f"✗ {package}: Not installed")

    print()
    return available_libs

def test_opencv_v4l2():
    """Test OpenCV with V4L2 backend"""
    print("=== OPENCV V4L2 TEST ===")
    try:
        import cv2

        # Check OpenCV build info for V4L2 support
        build_info = cv2.getBuildInformation()
        v4l2_support = "Video I/O" in build_info and "v4l2" in build_info.lower()
        print(f"OpenCV V4L2 support detected: {'Yes' if v4l2_support else 'Unknown'}")

        # Test opening camera with V4L2
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cap.isOpened():
            print("✓ Successfully opened camera with V4L2 backend")

            # Get current format
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)

            print(f"Current format: {int(width)}x{int(height)} @ {fps}fps")
            print(f"FourCC: {fourcc} ({cv2.VideoWriter.fourcc(*'MJPG')} = MJPG)")

            # Try to set YUYV format
            print("\nTesting YUYV format:")
            yuyv_fourcc = cv2.VideoWriter.fourcc(*'YUYV')
            success = cap.set(cv2.CAP_PROP_FOURCC, yuyv_fourcc)
            if success:
                new_fourcc = cap.get(cv2.CAP_PROP_FOURCC)
                print(f"✓ YUYV format set successfully: {new_fourcc}")

                # Try to capture a frame
                ret, frame = cap.read()
                if ret:
                    print(f"✓ YUYV frame captured: {frame.shape}")
                else:
                    print("✗ Failed to capture YUYV frame")
            else:
                print("✗ Failed to set YUYV format")

            cap.release()
        else:
            print("✗ Failed to open camera with V4L2 backend")

    except Exception as e:
        print(f"OpenCV V4L2 test error: {e}")
    print()

def test_linuxpy_v4l2():
    """Test linuxpy library"""
    print("=== LINUXPY V4L2 TEST ===")
    try:
        import linuxpy.video.device as video

        # Open device
        device = video.Device("/dev/video0")
        print(f"✓ Opened {device.filename}")

        # Get device info
        info = device.info
        print(f"Driver: {info.driver}")
        print(f"Card: {info.card}")
        print(f"Bus: {info.bus_info}")

        # Get available formats
        print("\nAvailable formats:")
        for fmt in device.formats:
            print(f"  {fmt}")

        # Get controls
        print(f"\nAvailable controls: {len(device.controls)}")
        for ctrl in list(device.controls)[:5]:  # Show first 5
            print(f"  {ctrl}")

        device.close()

    except Exception as e:
        print(f"linuxpy test error: {e}")
    print()

def test_v4l2py():
    """Test v4l2py library if available"""
    print("=== V4L2PY TEST ===")
    try:
        import v4l2py

        device = v4l2py.Device("/dev/video0")
        print(f"✓ Opened device with v4l2py")

        # Get device info
        info = device.info
        print(f"Driver: {info.driver}")
        print(f"Card: {info.card}")

        # List formats
        formats = list(device.formats)
        print(f"Available formats: {len(formats)}")
        for fmt in formats[:3]:  # Show first 3
            print(f"  {fmt}")

        device.close()

    except ImportError:
        print("v4l2py not available")
    except Exception as e:
        print(f"v4l2py test error: {e}")
    print()

def test_v4l2capture():
    """Test v4l2capture library if available"""
    print("=== V4L2CAPTURE TEST ===")
    try:
        import v4l2capture

        # Open device
        video = v4l2capture.Video_device("/dev/video0")
        print("✓ Opened device with v4l2capture")

        # Get device info
        print(f"Device name: {video.name}")

        # Try to start capture
        size_x, size_y = video.set_format(640, 480)
        print(f"Set format: {size_x}x{size_y}")

        video.start()
        print("✓ Started capture")

        # Get a frame
        image_data = video.read()
        print(f"✓ Captured frame: {len(image_data)} bytes")

        video.stop()
        video.close()

    except ImportError:
        print("v4l2capture not available")
    except Exception as e:
        print(f"v4l2capture test error: {e}")
    print()

def main():
    print("V4L2 Python Libraries Test")
    print("=" * 50)

    available_libs = check_library_availability()

    # Test available libraries
    test_opencv_v4l2()
    test_linuxpy_v4l2()
    test_v4l2py()
    test_v4l2capture()

    print("=" * 50)
    print("RECOMMENDATIONS:")
    print("- OpenCV with CAP_V4L2: Best for computer vision with format control")
    print("- linuxpy: Good for device enumeration and control access")
    print("- v4l2py: Modern pure Python V4L2 wrapper")
    print("- v4l2capture: Simple frame capture")

if __name__ == "__main__":
    main()