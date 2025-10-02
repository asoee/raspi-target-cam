#!/usr/bin/env python3
"""
Camera detection and settings query using pyuvc
Based on: https://github.com/pupil-labs/pyuvc
"""

import uvc
import sys
import traceback

def find_uvc_devices():
    """Find all UVC (USB Video Class) devices"""
    try:
        devices = uvc.device_list()
        return devices
    except Exception as e:
        print(f"Error finding UVC devices: {e}")
        return []

def get_device_info(device_info):
    """Get basic device information from device_info dict"""
    try:
        return {
            'name': device_info.get('name', 'Unknown'),
            'uid': device_info.get('uid', 'Unknown'),
            'vendor_id': device_info.get('vendor_id', 'Unknown'),
            'product_id': device_info.get('product_id', 'Unknown'),
            'serial_number': device_info.get('serial_number', 'Unknown')
        }
    except Exception as e:
        return f"Error getting device info: {e}"

def get_supported_formats(cap):
    """Get supported formats from an opened capture device"""
    try:
        formats = []
        # Get supported frame modes (format + resolution combinations)
        available_modes = cap.available_modes

        if available_modes:
            for mode in available_modes:
                formats.append({
                    'format': mode[0],  # Format (e.g., 'MJPG', 'YUYV')
                    'width': mode[1],   # Width
                    'height': mode[2],  # Height
                    'fps': mode[3]      # FPS
                })

        return formats
    except Exception as e:
        return f"Error getting formats: {e}"

def get_controls(cap):
    """Get available camera controls"""
    controls = []
    try:
        # Get available controls
        available_controls = cap.controls

        for control_name, control in available_controls.items():
            try:
                control_info = {
                    'name': control_name,
                    'value': None,
                    'range': None,
                    'type': 'Unknown'
                }

                # Try to get current value
                try:
                    control_info['value'] = control.value
                except Exception as e:
                    control_info['value'] = f"Could not read: {e}"

                # Try to get range information
                try:
                    if hasattr(control, 'range'):
                        control_info['range'] = control.range
                    elif hasattr(control, 'min_val') and hasattr(control, 'max_val'):
                        control_info['range'] = (control.min_val, control.max_val)
                except Exception:
                    pass

                # Try to determine control type
                try:
                    if hasattr(control, 'type'):
                        control_info['type'] = str(control.type)
                    elif hasattr(control, '__class__'):
                        control_info['type'] = control.__class__.__name__
                except Exception:
                    pass

                controls.append(control_info)
            except Exception as e:
                controls.append({
                    'name': control_name,
                    'error': f"Error reading control: {e}"
                })

    except Exception as e:
        return f"Error getting controls: {e}"

    return controls

def capture_single_frame(cap):
    """Try to capture a single frame to test if camera is working"""
    try:
        frame = cap.get_frame_robust()
        if frame is not None:
            return {
                'success': True,
                'width': frame.width,
                'height': frame.height,
                'format': frame.data_format if hasattr(frame, 'data_format') else 'Unknown'
            }
        else:
            return {'success': False, 'error': 'No frame captured'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    print("Detecting UVC cameras using pyuvc...")
    print("=" * 60)

    devices = find_uvc_devices()

    if not devices:
        print("No UVC devices found!")
        return

    print(f"Found {len(devices)} UVC device(s):")

    for i, device_info in enumerate(devices):
        print(f"\nDevice {i}: {device_info.get('name', 'Unknown Device')}")
        print("-" * 40)

        # Basic device information
        info = get_device_info(device_info)
        if isinstance(info, dict):
            print("Device Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"Device info: {info}")

        # Try to open the device for detailed inspection
        cap = None
        try:
            print(f"\nTrying to open device...")
            cap = uvc.Capture(device_info['uid'])
            print("✓ Device opened successfully")

            # Get supported formats
            print("\nSupported Formats:")
            formats = get_supported_formats(cap)
            if isinstance(formats, list) and formats:
                # Group formats by format type
                format_groups = {}
                for fmt in formats:
                    format_type = fmt['format']
                    if format_type not in format_groups:
                        format_groups[format_type] = []
                    format_groups[format_type].append(fmt)

                for format_type, format_list in format_groups.items():
                    print(f"  {format_type}:")
                    for fmt in format_list[:10]:  # Limit to first 10 per format
                        print(f"    {fmt['width']}x{fmt['height']} @ {fmt['fps']}fps")
                    if len(format_list) > 10:
                        print(f"    ... and {len(format_list) - 10} more resolutions")
            else:
                print(f"  {formats}")

            # Get camera controls
            print("\nCamera Controls:")
            controls = get_controls(cap)
            if isinstance(controls, list) and controls:
                for control in controls:
                    if 'error' in control:
                        print(f"  {control['name']}: {control['error']}")
                    else:
                        value_str = f" = {control['value']}" if control['value'] is not None else ""
                        range_str = f" (range: {control['range']})" if control['range'] else ""
                        print(f"  {control['name']}: {control['type']}{range_str}{value_str}")
            else:
                print(f"  {controls}")

            # Try to capture a test frame
            print("\nTest Frame Capture:")
            try:
                # Set a common format for testing
                available_modes = cap.available_modes
                if available_modes:
                    # Try to find a small MJPG format for quick test
                    test_mode = None
                    for mode in available_modes:
                        if mode[0] == 'MJPG' and mode[1] <= 640:  # Small MJPG format
                            test_mode = mode
                            break

                    if not test_mode:
                        # Fall back to any available mode
                        test_mode = available_modes[0]

                    print(f"  Setting mode: {test_mode[0]} {test_mode[1]}x{test_mode[2]} @ {test_mode[3]}fps")
                    cap.frame_mode = test_mode

                    result = capture_single_frame(cap)
                    if result['success']:
                        print(f"  ✓ Frame captured: {result['width']}x{result['height']} ({result['format']})")
                    else:
                        print(f"  ✗ Frame capture failed: {result['error']}")
                else:
                    print("  No available modes for testing")

            except Exception as e:
                print(f"  ✗ Frame capture test failed: {e}")

        except Exception as e:
            print(f"✗ Could not open device: {e}")
            print(f"  Full error: {traceback.format_exc()}")

        finally:
            if cap:
                try:
                    cap.close()
                except:
                    pass

        print("\n" + "=" * 60)

    # Comparison with other tools
    print("\nComparison with other tools:")
    print("-" * 60)
    try:
        import subprocess

        print("v4l2-ctl --list-devices:")
        result = subprocess.run(['v4l2-ctl', '--list-devices'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("v4l2-ctl not available or failed")

    except Exception as e:
        print(f"Could not run comparison tools: {e}")

if __name__ == "__main__":
    main()