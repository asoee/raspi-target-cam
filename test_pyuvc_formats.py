#!/usr/bin/env python3
"""
Test pyuvc image format capabilities - MJPEG, YUYV, etc.
"""

import uvc

def test_image_formats():
    """Test available image formats and their properties"""
    print("=== PYUVC IMAGE FORMATS TEST ===")

    try:
        # Get device list
        devices = uvc.device_list()
        if not devices:
            print("No UVC devices found")
            return

        device = devices[0]
        print(f"Testing formats for: {device.get('name', 'Unknown')}")
        print(f"UID: {device.get('uid', 'Unknown')}")
        print()

        # Open the camera
        cap = uvc.Capture(device['uid'])

        # Get available modes/formats
        print("=== AVAILABLE MODES AND FORMATS ===")
        try:
            modes = cap.available_modes
            print(f"Total available modes: {len(modes)}")

            # Group by format
            formats = {}
            for mode in modes:
                format_name = mode.format_name
                if format_name not in formats:
                    formats[format_name] = []
                formats[format_name].append(mode)

            for format_name, format_modes in formats.items():
                print(f"\n--- FORMAT: {format_name} ---")
                print(f"Format Native ID: {format_modes[0].format_native}")
                print(f"Available resolutions: {len(format_modes)}")

                for mode in format_modes:
                    supported_status = "✓" if mode.supported else "✗"
                    print(f"  {supported_status} {mode.width}x{mode.height} @ {mode.fps}fps")

        except Exception as e:
            print(f"Modes enumeration error: {e}")

        # Test format switching capabilities
        print("\n=== FORMAT SWITCHING TEST ===")

        # Try different formats if available
        test_modes = [
            (640, 480, 30),  # Common resolution
            (1280, 960, 15), # Higher resolution
            (320, 240, 30)   # Lower resolution
        ]

        for width, height, fps in test_modes:
            try:
                print(f"\nTesting {width}x{height} @ {fps}fps:")

                # Find matching mode
                matching_modes = [m for m in modes if m.width == width and m.height == height and m.fps == fps]
                if matching_modes:
                    mode = matching_modes[0]
                    print(f"  Format: {mode.format_name}")
                    print(f"  Native ID: {mode.format_native}")
                    print(f"  Supported: {'Yes' if mode.supported else 'No'}")

                    if mode.supported:
                        try:
                            # Try to set this mode
                            cap.frame_mode = mode
                            print(f"  ✓ Successfully set mode")

                            # Get a frame to verify it works
                            frame = cap.get_frame(timeout=1.0)
                            if frame:
                                print(f"  ✓ Successfully captured frame")
                                print(f"    Frame shape: {frame.bgr.shape if hasattr(frame, 'bgr') else 'N/A'}")
                                print(f"    Frame timestamp: {frame.timestamp if hasattr(frame, 'timestamp') else 'N/A'}")
                            else:
                                print(f"  ✗ Failed to capture frame")
                        except Exception as e:
                            print(f"  ✗ Failed to set mode: {e}")
                else:
                    print(f"  No matching mode found")

            except Exception as e:
                print(f"  Error testing mode: {e}")

        # Test format conversion capabilities
        print("\n=== FORMAT CONVERSION TEST ===")
        try:
            # Get current frame in different formats
            frame = cap.get_frame(timeout=1.0)
            if frame:
                print("Available frame data formats:")

                # Check what frame attributes are available
                frame_attrs = [attr for attr in dir(frame) if not attr.startswith('_')]
                for attr in frame_attrs:
                    try:
                        value = getattr(frame, attr)
                        if hasattr(value, 'shape'):  # It's likely a numpy array
                            print(f"  {attr}: shape {value.shape}, dtype {value.dtype}")
                        elif callable(value):
                            print(f"  {attr}: <method>")
                        else:
                            print(f"  {attr}: {type(value).__name__}")
                    except Exception as e:
                        print(f"  {attr}: Error accessing - {e}")
            else:
                print("No frame captured for format testing")

        except Exception as e:
            print(f"Format conversion test error: {e}")

        cap.close()

    except Exception as e:
        print(f"Main error: {e}")

if __name__ == "__main__":
    test_image_formats()