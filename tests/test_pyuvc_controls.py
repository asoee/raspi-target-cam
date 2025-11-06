#!/usr/bin/env python3
"""
Test pyuvc camera control capabilities - brightness, white balance, etc.
"""

import uvc

def test_camera_controls():
    """Test camera control information and ranges"""
    print("=== PYUVC CAMERA CONTROLS TEST ===")

    try:
        # Get device list
        devices = uvc.device_list()
        if not devices:
            print("No UVC devices found")
            return

        device = devices[0]
        print(f"Testing controls for: {device.get('name', 'Unknown')}")
        print(f"UID: {device.get('uid', 'Unknown')}")
        print()

        # Open the camera
        cap = uvc.Capture(device['uid'])

        # Get available controls
        print("=== AVAILABLE CONTROLS ===")
        try:
            controls = cap.controls
            print(f"Found {len(controls)} controls:")

            for control in controls:
                print(f"\nControl: {control}")

                # Get control info
                try:
                    info = cap.get_control_info(control)
                    print(f"  Info: {info}")
                except Exception as e:
                    print(f"  Info error: {e}")

                # Get current value
                try:
                    current_value = getattr(cap, control)
                    print(f"  Current value: {current_value}")
                except Exception as e:
                    print(f"  Current value error: {e}")

        except Exception as e:
            print(f"Controls enumeration error: {e}")

        # Test specific common controls
        common_controls = [
            'brightness', 'contrast', 'saturation', 'hue',
            'white_balance_temperature', 'white_balance_temperature_auto',
            'gamma', 'gain', 'exposure_time', 'exposure_auto',
            'focus_absolute', 'focus_auto', 'zoom_absolute',
            'backlight_compensation', 'power_line_frequency'
        ]

        print("\n=== TESTING COMMON CONTROLS ===")
        for control_name in common_controls:
            try:
                # Check if control exists
                if hasattr(cap, control_name):
                    current_value = getattr(cap, control_name)
                    print(f"{control_name}: {current_value}")

                    # Try to get control info/range
                    try:
                        info = cap.get_control_info(control_name)
                        print(f"  Range/Info: {info}")
                    except:
                        print(f"  Range: Not available")
                else:
                    print(f"{control_name}: Not available")
            except Exception as e:
                print(f"{control_name}: Error - {e}")

        # Test frame formats and controls
        print("\n=== FRAME FORMATS ===")
        try:
            modes = cap.available_modes
            print(f"Available modes: {len(modes)}")
            for i, mode in enumerate(modes):
                print(f"  Mode {i}: {mode}")
        except Exception as e:
            print(f"Modes error: {e}")

        cap.close()

    except Exception as e:
        print(f"Main error: {e}")

if __name__ == "__main__":
    test_camera_controls()