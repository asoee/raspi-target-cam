#!/usr/bin/env python3
"""
Camera detection and settings query using linuxpy
Based on: https://tiagocoutinho.github.io/linuxpy/user_guide/video/
"""

import linuxpy.video.device as video
import os
import sys

def find_video_devices():
    """Find all available video devices"""
    devices = []
    # Use linuxpy's built-in iterator
    try:
        for device in video.iter_video_capture_devices():
            devices.append(device)
    except Exception as e:
        print(f"Error iterating devices: {e}")

        # Fallback: manually check /dev/video* devices
        for i in range(20):
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                try:
                    device = video.Device(device_path)
                    devices.append(device)
                except Exception as e:
                    print(f"Could not open {device_path}: {e}")
    return devices

def get_device_info(device):
    """Get basic device information"""
    try:
        info = device.info
        if info is None:
            return "No device info available"

        return {
            'driver': getattr(info, 'driver', 'Unknown'),
            'card': getattr(info, 'card', 'Unknown'),
            'bus_info': getattr(info, 'bus_info', 'Unknown'),
            'version': getattr(info, 'version', 'Unknown'),
            'capabilities': getattr(info, 'capabilities', 'Unknown')
        }
    except Exception as e:
        return f"Error getting device info: {e}"

def get_supported_formats(device):
    """Get supported pixel formats"""
    formats = []
    try:
        # Try different buffer types for format enumeration
        buffer_types = [
            video.BufferType.VIDEO_CAPTURE,
            video.BufferType.VIDEO_OUTPUT,
            video.BufferType.VIDEO_CAPTURE_MPLANE,
            video.BufferType.VIDEO_OUTPUT_MPLANE
        ]

        for buf_type in buffer_types:
            try:
                for fmt in video.iter_read_formats(device, buf_type):
                    formats.append({
                        'description': getattr(fmt, 'description', 'Unknown'),
                        'pixel_format': getattr(fmt, 'pixel_format', 'Unknown'),
                        'type': str(buf_type)
                    })
                if formats:  # If we found formats with this type, we're done
                    break
            except Exception:
                continue

    except Exception as e:
        return f"Error getting formats: {e}"
    return formats

def get_frame_sizes(device, pixel_format):
    """Get supported frame sizes for a given pixel format"""
    sizes = []
    try:
        for size in video.iter_read_discrete_frame_sizes(device, pixel_format):
            sizes.append({
                'type': 'discrete',
                'width': getattr(size, 'width', 'Unknown'),
                'height': getattr(size, 'height', 'Unknown')
            })
    except Exception as e:
        return f"Error getting frame sizes: {e}"
    return sizes

def get_frame_intervals(device, pixel_format, width, height):
    """Get supported frame intervals (frame rates) for given format and size"""
    intervals = []
    try:
        size = video.Size(width=width, height=height)
        for interval in video.iter_read_frame_intervals(device, pixel_format, size):
            # Try to calculate FPS from the interval
            try:
                if hasattr(interval, 'numerator') and hasattr(interval, 'denominator'):
                    fps = interval.denominator / interval.numerator
                    intervals.append({
                        'fps': round(fps, 2),
                        'numerator': interval.numerator,
                        'denominator': interval.denominator
                    })
                else:
                    intervals.append({'interval': str(interval)})
            except:
                intervals.append({'interval': str(interval)})
    except Exception as e:
        return f"Error getting frame intervals: {e}"
    return intervals

def get_controls(device):
    """Get available camera controls"""
    controls = []
    try:
        device_controls = device.controls
        if device_controls is None:
            return "No controls available"

        for control in device_controls.values():
            try:
                control_info = {
                    'name': getattr(control, 'config_name', 'Unknown'),
                    'id': getattr(getattr(control, 'config', None), 'id', 'Unknown'),
                    'type': str(getattr(getattr(control, 'config', None), 'type', 'Unknown')),
                    'minimum': getattr(getattr(control, 'config', None), 'minimum', None),
                    'maximum': getattr(getattr(control, 'config', None), 'maximum', None),
                    'step': getattr(getattr(control, 'config', None), 'step', None),
                    'default': getattr(getattr(control, 'config', None), 'default', None),
                    'value': None
                }

                # Try to get current value
                try:
                    control_info['value'] = control.value
                except:
                    control_info['value'] = "Could not read"

                controls.append(control_info)
            except Exception as e:
                controls.append({'error': f"Error reading control: {e}"})

    except Exception as e:
        return f"Error getting controls: {e}"
    return controls

def main():
    print("Detecting video devices using linuxpy...")
    print("=" * 60)

    # First show what devices are available via direct enumeration
    print("\nDirect device enumeration:")
    for i in range(10):
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            print(f"  {device_path} - exists")

    # Try linuxpy's device enumeration
    print("\nLinuxpy device enumeration:")
    try:
        count = 0
        for device in video.iter_video_capture_devices():
            print(f"  Device {count}: {getattr(device, 'filename', 'Unknown path')}")
            count += 1
        if count == 0:
            print("  No capture devices found via iter_video_capture_devices()")
    except Exception as e:
        print(f"  Error with iter_video_capture_devices(): {e}")

    print("\nDetailed device analysis:")
    print("-" * 60)

    devices = find_video_devices()

    if not devices:
        print("No video devices found!")
        return

    for i, device in enumerate(devices):
        try:
            device_path = getattr(device, 'filename', f'/dev/video{i}')
            print(f"\nDevice {i}: {device_path}")
            print("-" * 40)

            # Device info
            info = get_device_info(device)
            if isinstance(info, dict):
                print("Device Information:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
            else:
                print(f"Device info: {info}")

            # Supported formats
            print("\nSupported Formats:")
            formats = get_supported_formats(device)
            if isinstance(formats, list) and formats:
                for fmt in formats:
                    print(f"  {fmt['description']} ({fmt['pixel_format']})")

                    # Get frame sizes for this format
                    pixel_format = fmt['pixel_format']
                    sizes = get_frame_sizes(device, pixel_format)
                    if isinstance(sizes, list) and sizes:
                        print(f"    Frame sizes:")
                        for size in sizes[:5]:  # Limit to first 5 sizes
                            print(f"      {size['width']}x{size['height']}")

                            # Get frame rates for this size
                            intervals = get_frame_intervals(device, pixel_format,
                                                          size['width'], size['height'])
                            if isinstance(intervals, list) and intervals:
                                fps_list = []
                                for interval in intervals[:3]:
                                    if 'fps' in interval:
                                        fps_list.append(f"{interval['fps']}fps")
                                    else:
                                        fps_list.append(str(interval))
                                if fps_list:
                                    print(f"        Frame rates: {', '.join(fps_list)}")
                        if len(sizes) > 5:
                            print(f"    ... and {len(sizes) - 5} more sizes")
            else:
                print(f"  {formats}")

            # Camera controls
            print("\nCamera Controls:")
            controls = get_controls(device)
            if isinstance(controls, list) and controls:
                for control in controls:
                    if 'error' in control:
                        print(f"  {control['error']}")
                    else:
                        value_str = f"= {control['value']}" if control['value'] is not None else ""
                        range_str = ""
                        if control['minimum'] is not None and control['maximum'] is not None:
                            range_str = f" (range: {control['minimum']}-{control['maximum']})"
                        print(f"  {control['name']}: {control['type']}{range_str} {value_str}")
            else:
                print(f"  {controls}")

        except Exception as e:
            print(f"Error processing device {i}: {e}")
        finally:
            try:
                if hasattr(device, 'close'):
                    device.close()
            except:
                pass

        print("\n" + "=" * 60)

    # Comparison with v4l2-ctl
    print("\nComparison with v4l2-ctl:")
    print("-" * 60)
    try:
        import subprocess
        result = subprocess.run(['v4l2-ctl', '--list-devices'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("v4l2-ctl --list-devices output:")
            print(result.stdout)
        else:
            print("v4l2-ctl not available or failed")

        # Show detailed format info for first USB camera found
        if result.returncode == 0 and '/dev/video0' in result.stdout:
            print("Detailed format info for /dev/video0:")
            format_result = subprocess.run(['v4l2-ctl', '--device=/dev/video0', '--list-formats-ext'],
                                         capture_output=True, text=True, timeout=5)
            if format_result.returncode == 0:
                print(format_result.stdout)

    except Exception as e:
        print(f"Could not run v4l2-ctl comparison: {e}")

if __name__ == "__main__":
    main()