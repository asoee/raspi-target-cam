#!/usr/bin/env python3
"""
Reusable Camera Control Management for OpenCV V4L2
Auto-detects and manages camera controls with range validation
"""

import cv2
import subprocess
import re
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

@dataclass
class ControlInfo:
    """Information about a camera control"""
    name: str
    opencv_id: int
    current_value: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    default_value: Optional[float] = None
    control_type: str = "float"
    supported: bool = True

class CameraControlManager:
    """
    Auto-detecting camera control manager for OpenCV V4L2

    Usage:
        with CameraControlManager(0) as controls:
            # Auto-detect controls
            controls.detect_controls()

            # List available controls
            print(controls.list_controls())

            # Set control with validation
            controls.set('brightness', 10)

            # Get control value
            brightness = controls.get('brightness')
    """

    def __init__(self, device_id: int = 0, auto_detect: bool = True):
        self.device_id = device_id
        self.device_path = f"/dev/video{device_id}"
        self.cap = None
        self.controls: Dict[str, ControlInfo] = {}
        self._original_values: Dict[str, float] = {}

        if auto_detect:
            self.open()
            self.detect_controls()

    def __enter__(self):
        if not self.cap:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore_original_values()
        self.close()

    def open(self):
        """Open camera with V4L2 backend"""
        if self.cap is not None:
            return

        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.device_id}")

    def close(self):
        """Close camera"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def detect_controls(self) -> Dict[str, ControlInfo]:
        """Auto-detect available camera controls"""
        if not self.cap:
            raise RuntimeError("Camera not opened")

        # Get V4L2 control ranges
        v4l2_ranges = self._get_v4l2_ranges()

        # OpenCV properties to test
        opencv_properties = {
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'saturation': cv2.CAP_PROP_SATURATION,
            'hue': cv2.CAP_PROP_HUE,
            'gain': cv2.CAP_PROP_GAIN,
            'gamma': cv2.CAP_PROP_GAMMA,
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'auto_exposure': cv2.CAP_PROP_AUTO_EXPOSURE,
            'auto_wb': cv2.CAP_PROP_AUTO_WB,
            'backlight': cv2.CAP_PROP_BACKLIGHT,
            'sharpness': cv2.CAP_PROP_SHARPNESS,
            'temperature': cv2.CAP_PROP_TEMPERATURE,
            'wb_temperature': cv2.CAP_PROP_WB_TEMPERATURE,
            'fourcc': cv2.CAP_PROP_FOURCC,
            'frame_width': cv2.CAP_PROP_FRAME_WIDTH,
            'frame_height': cv2.CAP_PROP_FRAME_HEIGHT,
            'fps': cv2.CAP_PROP_FPS,
        }

        detected_controls = {}

        for name, prop_id in opencv_properties.items():
            try:
                current_value = self.cap.get(prop_id)

                if current_value != -1:  # -1 indicates unsupported
                    # Save original value
                    self._original_values[name] = current_value

                    # Get range information
                    min_val, max_val, step, default = self._get_control_range(name, prop_id, v4l2_ranges)

                    control_info = ControlInfo(
                        name=name,
                        opencv_id=prop_id,
                        current_value=current_value,
                        min_value=min_val,
                        max_value=max_val,
                        step=step,
                        default_value=default,
                        supported=True
                    )

                    detected_controls[name] = control_info

            except Exception as e:
                continue

        self.controls = detected_controls
        return detected_controls

    def _get_v4l2_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get control ranges from v4l2-ctl"""
        try:
            result = subprocess.run(
                ['v4l2-ctl', f'--device={self.device_path}', '--list-ctrls'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                return {}

            ranges = {}
            for line in result.stdout.split('\n'):
                match = re.match(r'(\w+)\s+0x[0-9a-f]+\s+\(\w+\)\s*:\s*(.+)', line.strip())
                if match:
                    name, params = match.groups()
                    param_dict = {}

                    for param in params.split():
                        if '=' in param:
                            key, value = param.split('=', 1)
                            try:
                                if key in ['min', 'max', 'step', 'default', 'value']:
                                    param_dict[key] = int(value)
                                else:
                                    param_dict[key] = value
                            except ValueError:
                                param_dict[key] = value

                    ranges[name] = param_dict

            return ranges

        except Exception:
            return {}

    def _get_control_range(self, name: str, prop_id: int, v4l2_ranges: Dict) -> tuple:
        """Get control range from V4L2 data or by testing"""

        # Map OpenCV names to V4L2 names
        v4l2_mapping = {
            'brightness': 'brightness',
            'contrast': 'contrast',
            'saturation': 'saturation',
            'hue': 'hue',
            'gain': 'gain',
            'gamma': 'gamma',
            'exposure': 'exposure_time_absolute',
            'auto_exposure': 'auto_exposure',
            'backlight': 'backlight_compensation',
            'sharpness': 'sharpness',
            'temperature': 'white_balance_temperature',
            'wb_temperature': 'white_balance_temperature',
        }

        v4l2_name = v4l2_mapping.get(name)
        if v4l2_name and v4l2_name in v4l2_ranges:
            range_info = v4l2_ranges[v4l2_name]
            return (
                range_info.get('min'),
                range_info.get('max'),
                range_info.get('step'),
                range_info.get('default')
            )

        return None, None, None, None

    def list_controls(self) -> List[str]:
        """Get list of available control names"""
        return list(self.controls.keys())

    def get_control_info(self, name: str) -> Optional[ControlInfo]:
        """Get detailed information about a control"""
        return self.controls.get(name)

    def get(self, name: str) -> Optional[float]:
        """Get current value of a control"""
        if name not in self.controls:
            raise ValueError(f"Control '{name}' not available")

        try:
            return self.cap.get(self.controls[name].opencv_id)
        except Exception as e:
            raise RuntimeError(f"Failed to get {name}: {e}")

    def set(self, name: str, value: Union[int, float], validate: bool = True) -> bool:
        """
        Set control value with optional validation

        Args:
            name: Control name
            value: Value to set
            validate: Whether to validate against min/max range

        Returns:
            True if successful, False otherwise
        """
        if name not in self.controls:
            raise ValueError(f"Control '{name}' not available")

        control = self.controls[name]

        # Validate range if requested and range is known
        if validate and control.min_value is not None and control.max_value is not None:
            if value < control.min_value or value > control.max_value:
                raise ValueError(
                    f"Value {value} out of range for {name} "
                    f"(valid range: {control.min_value} to {control.max_value})"
                )

        try:
            success = self.cap.set(control.opencv_id, float(value))
            if success:
                # Update cached current value
                control.current_value = self.cap.get(control.opencv_id)
                return True
            else:
                return False
        except Exception as e:
            raise RuntimeError(f"Failed to set {name}: {e}")

    def reset(self, name: str) -> bool:
        """Reset control to its default value"""
        if name not in self.controls:
            raise ValueError(f"Control '{name}' not available")

        control = self.controls[name]
        if control.default_value is not None:
            return self.set(name, control.default_value, validate=False)
        else:
            raise ValueError(f"No default value known for {name}")

    def reset_all(self) -> Dict[str, bool]:
        """Reset all controls to their default values"""
        results = {}
        for name in self.controls:
            try:
                results[name] = self.reset(name)
            except Exception:
                results[name] = False
        return results

    def restore_original_values(self) -> Dict[str, bool]:
        """Restore all controls to their original values when first detected"""
        results = {}
        for name, original_value in self._original_values.items():
            try:
                results[name] = self.set(name, original_value, validate=False)
            except Exception:
                results[name] = False
        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all controls"""
        status = {}
        for name, control in self.controls.items():
            current_value = self.get(name)
            status[name] = {
                'current': current_value,
                'min': control.min_value,
                'max': control.max_value,
                'default': control.default_value,
                'original': self._original_values.get(name)
            }
        return status

    def save_preset(self, name: str) -> Dict[str, float]:
        """Save current control values as a preset"""
        preset = {}
        for control_name in self.controls:
            preset[control_name] = self.get(control_name)
        return preset

    def load_preset(self, preset: Dict[str, float]) -> Dict[str, bool]:
        """Load control values from a preset"""
        results = {}
        for name, value in preset.items():
            try:
                results[name] = self.set(name, value)
            except Exception:
                results[name] = False
        return results

    def print_status(self):
        """Print current status of all controls"""
        print(f"Camera {self.device_id} Controls:")
        print("-" * 50)

        for name, control in self.controls.items():
            current = self.get(name)
            min_val = control.min_value if control.min_value is not None else "?"
            max_val = control.max_value if control.max_value is not None else "?"
            default = control.default_value if control.default_value is not None else "?"

            print(f"{name:20} {current:8.1f} (range: {min_val} to {max_val}, default: {default})")


def main():
    """Example usage of CameraControlManager"""
    print("Camera Control Manager Example")
    print("=" * 40)

    try:
        with CameraControlManager(0) as controls:
            # Print detected controls
            controls.print_status()

            # Save original state as preset
            original_preset = controls.save_preset("original")

            # Test changing some controls
            print(f"\nTesting control changes:")
            if 'brightness' in controls.list_controls():
                print(f"Setting brightness to 20")
                controls.set('brightness', 20)
                print(f"New brightness: {controls.get('brightness')}")

            if 'contrast' in controls.list_controls():
                print(f"Setting contrast to 50")
                controls.set('contrast', 50)
                print(f"New contrast: {controls.get('contrast')}")

            print(f"\nRestoring original values...")
            controls.load_preset(original_preset)

            print(f"Brightness restored to: {controls.get('brightness')}")
            print(f"Contrast restored to: {controls.get('contrast')}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()