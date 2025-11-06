#!/usr/bin/env python3
"""
OpenCV V4L2 Camera Control Auto-Detection
Automatically detects and tests available camera controls
"""

import cv2
import subprocess
import re
import json
from typing import Dict, List, Any, Optional

class OpenCVControlDetector:
    """Auto-detect and manage OpenCV camera controls"""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.device_path = f"/dev/video{device_id}"
        self.cap = None
        self.controls = {}
        self.v4l2_controls = {}

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open camera with V4L2 backend"""
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.device_id}")
        print(f"✓ Opened camera {self.device_id} with V4L2 backend")

    def close(self):
        """Close camera"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def detect_v4l2_controls(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed control info from v4l2-ctl"""
        print(f"=== DETECTING V4L2 CONTROLS for {self.device_path} ===")

        try:
            # Get control list
            result = subprocess.run(
                ['v4l2-ctl', f'--device={self.device_path}', '--list-ctrls'],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                print(f"v4l2-ctl failed: {result.stderr}")
                return {}

            controls = {}

            # Parse v4l2-ctl output
            for line in result.stdout.split('\n'):
                line = line.strip()
                if not line or line.startswith(('User Controls', 'Camera Controls')):
                    continue

                # Parse control line
                # Format: "brightness 0x00980900 (int) : min=-64 max=64 step=1 default=0 value=0"
                match = re.match(r'(\w+)\s+0x([0-9a-f]+)\s+\((\w+)\)\s*:\s*(.+)', line)
                if match:
                    name, control_id, control_type, params = match.groups()

                    # Parse parameters
                    param_dict = {}
                    for param in params.split():
                        if '=' in param:
                            key, value = param.split('=', 1)
                            # Convert numeric values
                            try:
                                if key in ['min', 'max', 'step', 'default', 'value']:
                                    param_dict[key] = int(value)
                                else:
                                    param_dict[key] = value
                            except ValueError:
                                param_dict[key] = value

                    controls[name] = {
                        'id': f"0x{control_id}",
                        'type': control_type,
                        'params': param_dict
                    }

            self.v4l2_controls = controls
            print(f"Detected {len(controls)} V4L2 controls")
            return controls

        except Exception as e:
            print(f"Error detecting V4L2 controls: {e}")
            return {}

    def detect_opencv_controls(self) -> Dict[str, Dict[str, Any]]:
        """Detect available OpenCV camera properties"""
        print(f"=== DETECTING OPENCV CONTROLS ===")

        if not self.cap:
            raise RuntimeError("Camera not opened")

        # Common OpenCV camera properties to test
        opencv_properties = {
            # Basic properties
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'saturation': cv2.CAP_PROP_SATURATION,
            'hue': cv2.CAP_PROP_HUE,
            'gain': cv2.CAP_PROP_GAIN,
            'gamma': cv2.CAP_PROP_GAMMA,

            # Exposure controls
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'auto_exposure': cv2.CAP_PROP_AUTO_EXPOSURE,

            # White balance
            'white_balance_blue_u': cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,
            'white_balance_red_v': cv2.CAP_PROP_WHITE_BALANCE_RED_V,
            'auto_wb': cv2.CAP_PROP_AUTO_WB,

            # Focus
            'focus': cv2.CAP_PROP_FOCUS,
            'autofocus': cv2.CAP_PROP_AUTOFOCUS,

            # Format controls
            'fourcc': cv2.CAP_PROP_FOURCC,
            'frame_width': cv2.CAP_PROP_FRAME_WIDTH,
            'frame_height': cv2.CAP_PROP_FRAME_HEIGHT,
            'fps': cv2.CAP_PROP_FPS,

            # Advanced controls
            'zoom': cv2.CAP_PROP_ZOOM,
            'pan': cv2.CAP_PROP_PAN,
            'tilt': cv2.CAP_PROP_TILT,
            'backlight': cv2.CAP_PROP_BACKLIGHT,
            'sharpness': cv2.CAP_PROP_SHARPNESS,
            'temperature': cv2.CAP_PROP_TEMPERATURE,
            'iris': cv2.CAP_PROP_IRIS,
            'trigger': cv2.CAP_PROP_TRIGGER,
            'trigger_delay': cv2.CAP_PROP_TRIGGER_DELAY,
            'wb_temperature': cv2.CAP_PROP_WB_TEMPERATURE,
        }

        detected_controls = {}

        for name, prop_id in opencv_properties.items():
            try:
                # Try to get current value
                current_value = self.cap.get(prop_id)

                # Check if property is supported (usually returns -1 if not supported)
                if current_value != -1:
                    control_info = {
                        'opencv_id': prop_id,
                        'current_value': current_value,
                        'supported': True
                    }

                    # Try to determine range by testing
                    range_info = self._test_property_range(prop_id, current_value)
                    control_info.update(range_info)

                    detected_controls[name] = control_info
                    print(f"✓ {name}: {current_value} (range: {range_info.get('min', '?')}-{range_info.get('max', '?')})")
                else:
                    print(f"✗ {name}: Not supported")

            except Exception as e:
                print(f"✗ {name}: Error - {e}")

        self.controls = detected_controls
        print(f"Detected {len(detected_controls)} OpenCV controls")
        return detected_controls

    def _test_property_range(self, prop_id: int, current_value: float) -> Dict[str, Any]:
        """Test property range by attempting to set different values"""
        range_info = {}

        try:
            # Save current value
            original_value = current_value

            # Test for minimum value
            test_values = [0, -100, -1000, 1, 0.1]
            for test_val in test_values:
                if self.cap.set(prop_id, test_val):
                    actual_val = self.cap.get(prop_id)
                    if actual_val != original_value:
                        range_info['min'] = actual_val
                        break

            # Test for maximum value
            test_values = [100, 1000, 10000, 255, 1.0]
            for test_val in test_values:
                if self.cap.set(prop_id, test_val):
                    actual_val = self.cap.get(prop_id)
                    if actual_val != original_value:
                        range_info['max'] = actual_val
                        break

            # Restore original value
            self.cap.set(prop_id, original_value)

        except Exception as e:
            range_info['range_error'] = str(e)

        return range_info

    def map_controls(self) -> Dict[str, Dict[str, Any]]:
        """Map V4L2 controls to OpenCV properties"""
        print(f"=== MAPPING V4L2 TO OPENCV CONTROLS ===")

        # Common mappings
        mapping = {
            'brightness': 'brightness',
            'contrast': 'contrast',
            'saturation': 'saturation',
            'hue': 'hue',
            'gain': 'gain',
            'gamma': 'gamma',
            'white_balance_temperature': 'wb_temperature',
            'sharpness': 'sharpness',
            'exposure_time_absolute': 'exposure',
            'auto_exposure': 'auto_exposure',
        }

        mapped_controls = {}

        for v4l2_name, opencv_name in mapping.items():
            if v4l2_name in self.v4l2_controls and opencv_name in self.controls:
                v4l2_info = self.v4l2_controls[v4l2_name]
                opencv_info = self.controls[opencv_name]

                mapped_controls[v4l2_name] = {
                    'v4l2': v4l2_info,
                    'opencv': opencv_info,
                    'mapped': True
                }
                print(f"✓ Mapped {v4l2_name} -> {opencv_name}")
            else:
                if v4l2_name in self.v4l2_controls:
                    mapped_controls[v4l2_name] = {
                        'v4l2': self.v4l2_controls[v4l2_name],
                        'opencv': None,
                        'mapped': False
                    }
                    print(f"✗ {v4l2_name}: V4L2 only")

        return mapped_controls

    def set_control(self, name: str, value: Any) -> bool:
        """Set a control value using OpenCV"""
        if name not in self.controls:
            print(f"Control '{name}' not available")
            return False

        try:
            prop_id = self.controls[name]['opencv_id']
            success = self.cap.set(prop_id, value)
            if success:
                actual_value = self.cap.get(prop_id)
                print(f"✓ Set {name} = {value} (actual: {actual_value})")
                return True
            else:
                print(f"✗ Failed to set {name} = {value}")
                return False
        except Exception as e:
            print(f"✗ Error setting {name}: {e}")
            return False

    def get_control(self, name: str) -> Optional[float]:
        """Get current control value"""
        if name not in self.controls:
            return None

        try:
            prop_id = self.controls[name]['opencv_id']
            return self.cap.get(prop_id)
        except Exception as e:
            print(f"Error getting {name}: {e}")
            return None

    def print_summary(self):
        """Print a summary of detected controls"""
        print(f"\n=== CONTROL DETECTION SUMMARY ===")
        print(f"V4L2 controls detected: {len(self.v4l2_controls)}")
        print(f"OpenCV controls detected: {len(self.controls)}")

        print(f"\nAvailable OpenCV controls:")
        for name, info in self.controls.items():
            current = info['current_value']
            min_val = info.get('min', '?')
            max_val = info.get('max', '?')
            print(f"  {name}: {current} (range: {min_val} to {max_val})")


def main():
    """Test the control detector"""
    print("OpenCV V4L2 Control Auto-Detection")
    print("=" * 50)

    try:
        with OpenCVControlDetector(0) as detector:
            # Detect controls
            detector.detect_v4l2_controls()
            detector.detect_opencv_controls()

            # Map controls
            mapped = detector.map_controls()

            # Print summary
            detector.print_summary()

            # Test setting a control
            print(f"\n=== TESTING CONTROL CHANGES ===")
            if 'brightness' in detector.controls:
                original = detector.get_control('brightness')
                print(f"Original brightness: {original}")

                # Test changing brightness
                detector.set_control('brightness', 10)
                new_value = detector.get_control('brightness')
                print(f"New brightness: {new_value}")

                # Restore original
                detector.set_control('brightness', original)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()