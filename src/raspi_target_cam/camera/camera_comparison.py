#!/usr/bin/env python3
"""
Comprehensive camera detection comparison between different libraries:
- pyuvc (UVC specific)
- linuxpy (V4L2 wrapper)
- v4l2-ctl (system tool)
- Direct device enumeration
"""

import os
import subprocess
import sys

def test_direct_enumeration():
    """Test direct /dev/video* enumeration"""
    print("=== DIRECT DEVICE ENUMERATION ===")
    devices = []
    for i in range(20):
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            try:
                stat_info = os.stat(device_path)
                devices.append({
                    'path': device_path,
                    'major': os.major(stat_info.st_rdev),
                    'minor': os.minor(stat_info.st_rdev)
                })
            except Exception as e:
                devices.append({'path': device_path, 'error': str(e)})

    print(f"Found {len(devices)} video devices:")
    for device in devices:
        if 'error' in device:
            print(f"  {device['path']}: {device['error']}")
        else:
            print(f"  {device['path']}: major={device['major']}, minor={device['minor']}")
    print()

def test_pyuvc():
    """Test pyuvc library"""
    print("=== PYUVC (UVC-specific) ===")
    try:
        import uvc
        devices = uvc.device_list()

        print(f"Found {len(devices)} UVC devices:")
        for i, device in enumerate(devices):
            print(f"  Device {i}:")
            print(f"    Name: {device.get('name', 'Unknown')}")
            print(f"    Manufacturer: {device.get('manufacturer', 'Unknown')}")
            print(f"    Product ID: 0x{device.get('idProduct', 0):04x}")
            print(f"    Vendor ID: 0x{device.get('idVendor', 0):04x}")
            print(f"    Bus: {device.get('bus_number', 'Unknown')}")
            print(f"    Address: {device.get('device_address', 'Unknown')}")
            print(f"    UID: {device.get('uid', 'Unknown')}")

            # Try to open device
            try:
                cap = uvc.Capture(device['uid'])
                print(f"    Status: ✓ Can open device")

                # Try to get basic info if it opens
                try:
                    modes = cap.available_modes
                    print(f"    Available modes: {len(modes) if modes else 0}")
                    if modes:
                        print(f"    Sample modes:")
                        for mode in modes[:3]:  # Show first 3 modes
                            print(f"      {mode[0]} {mode[1]}x{mode[2]} @ {mode[3]}fps")
                        if len(modes) > 3:
                            print(f"      ... and {len(modes) - 3} more")
                except Exception as e:
                    print(f"    Modes: Error getting modes: {e}")

                cap.close()
            except Exception as e:
                print(f"    Status: ✗ Cannot open device: {e}")
    except ImportError:
        print("pyuvc not available")
    except Exception as e:
        print(f"pyuvc error: {e}")
    print()

def test_linuxpy():
    """Test linuxpy library"""
    print("=== LINUXPY (V4L2 wrapper) ===")
    try:
        import linuxpy.video.device as video

        # Try the iterator
        devices = []
        try:
            for device in video.iter_video_capture_devices():
                devices.append(device)
        except Exception as e:
            print(f"iter_video_capture_devices failed: {e}")

        print(f"Found {len(devices)} devices via linuxpy:")
        for i, device in enumerate(devices):
            try:
                device_path = getattr(device, 'filename', f'Device {i}')
                print(f"  Device {i}: {device_path}")

                # Try to get device info
                try:
                    info = device.info
                    if info:
                        print(f"    Driver: {getattr(info, 'driver', 'Unknown')}")
                        print(f"    Card: {getattr(info, 'card', 'Unknown')}")
                        print(f"    Bus: {getattr(info, 'bus_info', 'Unknown')}")
                    else:
                        print(f"    Info: Not available")
                except Exception as e:
                    print(f"    Info error: {e}")

                device.close()
            except Exception as e:
                print(f"    Device {i} error: {e}")
    except ImportError:
        print("linuxpy not available")
    except Exception as e:
        print(f"linuxpy error: {e}")
    print()

def test_v4l2_ctl():
    """Test v4l2-ctl system tool"""
    print("=== V4L2-CTL (System tool) ===")
    try:
        # List devices
        result = subprocess.run(['v4l2-ctl', '--list-devices'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("Device list:")
            print(result.stdout)
        else:
            print(f"v4l2-ctl --list-devices failed: {result.stderr}")

        # Get detailed info for /dev/video0 if it exists
        if os.path.exists('/dev/video0'):
            print("Detailed info for /dev/video0:")
            result = subprocess.run(['v4l2-ctl', '--device=/dev/video0', '--all'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Show first 20 lines of output (it can be very long)
                lines = result.stdout.split('\n')[:20]
                print('\n'.join(lines))
                if len(result.stdout.split('\n')) > 20:
                    print(f"... (output truncated, {len(result.stdout.split('\n'))} total lines)")
            else:
                print(f"v4l2-ctl --all failed: {result.stderr}")
    except Exception as e:
        print(f"v4l2-ctl error: {e}")
    print()

def test_usb_info():
    """Get USB device information"""
    print("=== USB DEVICE INFO ===")
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("USB devices:")
            for line in result.stdout.split('\n'):
                if 'camera' in line.lower() or 'video' in line.lower() or '32e4:8830' in line:
                    print(f"  {line}")

        # Try to get detailed USB info for the camera
        result = subprocess.run(['lsusb', '-d', '32e4:8830', '-v'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\nDetailed USB info for camera (32e4:8830):")
            lines = result.stdout.split('\n')[:30]  # First 30 lines
            for line in lines:
                if line.strip():
                    print(f"  {line}")
            if len(result.stdout.split('\n')) > 30:
                print("  ... (output truncated)")
    except Exception as e:
        print(f"USB info error: {e}")

def main():
    print("Camera Detection Comparison Tool")
    print("=" * 60)

    test_direct_enumeration()
    test_pyuvc()
    test_linuxpy()
    test_v4l2_ctl()
    test_usb_info()

    print("=" * 60)
    print("Summary:")
    print("- Direct enumeration: Shows which /dev/video* devices exist")
    print("- pyuvc: UVC-specific, may have access restrictions")
    print("- linuxpy: V4L2 wrapper, broader device support")
    print("- v4l2-ctl: System tool, most comprehensive information")

if __name__ == "__main__":
    main()