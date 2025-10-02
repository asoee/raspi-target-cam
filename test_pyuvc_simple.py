#!/usr/bin/env python3
"""
Simple pyuvc test to debug access issues
"""

import uvc
import sys

def main():
    print("Testing pyuvc access...")

    # List devices
    devices = uvc.device_list()
    print(f"Found {len(devices)} devices: {devices}")

    if not devices:
        print("No devices found")
        return

    for device_info in devices:
        print(f"\nTrying device: {device_info}")

        try:
            # Try different approaches to open the device
            print("Attempt 1: Standard open")
            cap = uvc.Capture(device_info['uid'])
            print("✓ Successfully opened device")
            cap.close()
            break
        except Exception as e:
            print(f"✗ Standard open failed: {e}")

        try:
            print("Attempt 2: Direct device path")
            # If we know it's /dev/video0, try that directly
            cap = uvc.Capture("/dev/video0")
            print("✓ Successfully opened via /dev/video0")
            cap.close()
            break
        except Exception as e:
            print(f"✗ Direct path failed: {e}")

        try:
            print("Attempt 3: By bus/device numbers")
            cap = uvc.Capture(f"{device_info['bus_number']}:{device_info['device_address']}")
            print("✓ Successfully opened via bus:device")
            cap.close()
            break
        except Exception as e:
            print(f"✗ Bus:device failed: {e}")

if __name__ == "__main__":
    main()