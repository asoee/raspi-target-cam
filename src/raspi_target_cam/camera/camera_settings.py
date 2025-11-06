#!/usr/bin/env python3
"""
Camera settings management with thread-safe access.

Provides:
- CameraSettings dataclass for all camera parameters
- Command pattern for thread-safe camera control
- Settings validation and constraints
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import threading
import json


@dataclass
class CameraSettings:
    """
    Immutable camera settings container.

    All settings are stored here and can be read safely from any thread.
    To change settings, create a new CameraSettings instance with updated values.
    """
    # Resolution
    width: int = 1920
    height: int = 1080

    # Frame rate
    fps: float = 30.0

    # Exposure
    auto_exposure: bool = True
    exposure: Optional[int] = None  # Exposure time in microseconds

    # White balance
    auto_white_balance: bool = True
    white_balance_temp: Optional[int] = None  # Color temperature in Kelvin

    # Focus
    auto_focus: bool = True
    focus: Optional[int] = None  # Focus position

    # Gain/ISO
    auto_gain: bool = True
    gain: Optional[float] = None
    iso: Optional[int] = None

    # Brightness, contrast, saturation
    brightness: int = 50  # 0-100
    contrast: int = 50    # 0-100
    saturation: int = 50  # 0-100
    sharpness: int = 50   # 0-100

    # Other
    buffer_size: int = 1
    backend: Optional[int] = None  # cv2.CAP_* backend

    # v4l2 specific controls
    power_line_frequency: int = 1  # 0=disabled, 1=50Hz, 2=60Hz
    backlight_compensation: bool = False

    def copy(self, **changes) -> 'CameraSettings':
        """Create a copy of settings with specified changes"""
        current = asdict(self)
        current.update(changes)
        return CameraSettings(**current)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraSettings':
        """Create from dictionary"""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'CameraSettings':
        """Deserialize from JSON"""
        return cls.from_dict(json.loads(json_str))


class CameraCommand:
    """Base class for camera control commands"""

    def execute(self, cap) -> bool:
        """
        Execute command on capture device.

        Args:
            cap: cv2.VideoCapture instance

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError


class SetResolutionCommand(CameraCommand):
    """Change camera resolution"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def execute(self, cap) -> bool:
        import cv2
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # Verify the change
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Resolution set to {actual_width}x{actual_height} (requested {self.width}x{self.height})")
            return True
        except Exception as e:
            print(f"Failed to set resolution: {e}")
            return False


class SetFPSCommand(CameraCommand):
    """Change camera frame rate"""

    def __init__(self, fps: float):
        self.fps = fps

    def execute(self, cap) -> bool:
        import cv2
        try:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"FPS set to {actual_fps} (requested {self.fps})")
            return True
        except Exception as e:
            print(f"Failed to set FPS: {e}")
            return False


class SetExposureCommand(CameraCommand):
    """Change exposure settings"""

    def __init__(self, auto: bool, value: Optional[int] = None):
        self.auto = auto
        self.value = value

    def execute(self, cap) -> bool:
        import cv2
        try:
            # Set auto exposure
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0 if self.auto else 0.0)

            # Set manual exposure if specified
            if not self.auto and self.value is not None:
                cap.set(cv2.CAP_PROP_EXPOSURE, self.value)

            print(f"Exposure: auto={self.auto}, value={self.value}")
            return True
        except Exception as e:
            print(f"Failed to set exposure: {e}")
            return False


class SetWhiteBalanceCommand(CameraCommand):
    """Change white balance settings"""

    def __init__(self, auto: bool, temp: Optional[int] = None):
        self.auto = auto
        self.temp = temp

    def execute(self, cap) -> bool:
        import cv2
        try:
            cap.set(cv2.CAP_PROP_AUTO_WB, 1.0 if self.auto else 0.0)

            if not self.auto and self.temp is not None:
                cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.temp)

            print(f"White balance: auto={self.auto}, temp={self.temp}")
            return True
        except Exception as e:
            print(f"Failed to set white balance: {e}")
            return False


class SetFocusCommand(CameraCommand):
    """Change focus settings"""

    def __init__(self, auto: bool, value: Optional[int] = None):
        self.auto = auto
        self.value = value

    def execute(self, cap) -> bool:
        import cv2
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if self.auto else 0.0)

            if not self.auto and self.value is not None:
                cap.set(cv2.CAP_PROP_FOCUS, self.value)

            print(f"Focus: auto={self.auto}, value={self.value}")
            return True
        except Exception as e:
            print(f"Failed to set focus: {e}")
            return False


class SetGainCommand(CameraCommand):
    """Change gain/ISO settings"""

    def __init__(self, auto: bool, gain: Optional[float] = None, iso: Optional[int] = None):
        self.auto = auto
        self.gain = gain
        self.iso = iso

    def execute(self, cap) -> bool:
        import cv2
        try:
            # Some cameras use gain, others use ISO
            if self.gain is not None:
                cap.set(cv2.CAP_PROP_GAIN, self.gain)

            if self.iso is not None:
                cap.set(cv2.CAP_PROP_ISO_SPEED, self.iso)

            print(f"Gain: auto={self.auto}, gain={self.gain}, iso={self.iso}")
            return True
        except Exception as e:
            print(f"Failed to set gain: {e}")
            return False


class SetImageAdjustmentCommand(CameraCommand):
    """Change brightness, contrast, saturation, sharpness"""

    def __init__(self, brightness: Optional[int] = None, contrast: Optional[int] = None,
                 saturation: Optional[int] = None, sharpness: Optional[int] = None):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.sharpness = sharpness

    def execute(self, cap) -> bool:
        import cv2
        try:
            if self.brightness is not None:
                cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness / 100.0)

            if self.contrast is not None:
                cap.set(cv2.CAP_PROP_CONTRAST, self.contrast / 100.0)

            if self.saturation is not None:
                cap.set(cv2.CAP_PROP_SATURATION, self.saturation / 100.0)

            if self.sharpness is not None:
                cap.set(cv2.CAP_PROP_SHARPNESS, self.sharpness / 100.0)

            print(f"Image adjustment: brightness={self.brightness}, contrast={self.contrast}, "
                  f"saturation={self.saturation}, sharpness={self.sharpness}")
            return True
        except Exception as e:
            print(f"Failed to set image adjustments: {e}")
            return False


class SetBufferSizeCommand(CameraCommand):
    """Change camera buffer size"""

    def __init__(self, size: int):
        self.size = size

    def execute(self, cap) -> bool:
        import cv2
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.size)
            print(f"Buffer size set to {self.size}")
            return True
        except Exception as e:
            print(f"Failed to set buffer size: {e}")
            return False


class V4L2Command(CameraCommand):
    """Execute v4l2-ctl command for Linux cameras"""

    def __init__(self, device: str, control: str, value: Any):
        self.device = device
        self.control = control
        self.value = value

    def execute(self, cap) -> bool:
        import subprocess
        try:
            result = subprocess.run(
                ['v4l2-ctl', '-d', self.device, '--set-ctrl', f'{self.control}={self.value}'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                print(f"v4l2: {self.control}={self.value}")
                return True
            else:
                print(f"v4l2 failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"Failed to execute v4l2 command: {e}")
            return False


class SeekCommand(CameraCommand):
    """Seek to specific frame in video file"""

    def __init__(self, frame_number: int):
        self.frame_number = frame_number

    def execute(self, cap) -> bool:
        import cv2
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"Seeked to frame {actual} (requested {self.frame_number})")
            return True
        except Exception as e:
            print(f"Failed to seek to frame {self.frame_number}: {e}")
            return False


class SeekToTimeCommand(CameraCommand):
    """Seek to specific time in video file (milliseconds)"""

    def __init__(self, time_ms: int):
        self.time_ms = time_ms

    def execute(self, cap) -> bool:
        import cv2
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.time_ms)
            actual = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            print(f"Seeked to {actual}ms (requested {self.time_ms}ms)")
            return True
        except Exception as e:
            print(f"Failed to seek to time {self.time_ms}ms: {e}")
            return False


class GetFramePositionCommand(CameraCommand):
    """Get current frame position (stores result in command)"""

    def __init__(self):
        self.frame_number = None
        self.total_frames = None

    def execute(self, cap) -> bool:
        import cv2
        try:
            self.frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return True
        except Exception as e:
            print(f"Failed to get frame position: {e}")
            return False


class ApplySettingsCommand(CameraCommand):
    """Apply a complete CameraSettings object"""

    def __init__(self, settings: CameraSettings, device_path: Optional[str] = None):
        self.settings = settings
        self.device_path = device_path

    def execute(self, cap) -> bool:
        import cv2
        success = True

        try:
            # Resolution
            SetResolutionCommand(self.settings.width, self.settings.height).execute(cap)

            # FPS
            SetFPSCommand(self.settings.fps).execute(cap)

            # Exposure
            SetExposureCommand(self.settings.auto_exposure, self.settings.exposure).execute(cap)

            # White balance
            SetWhiteBalanceCommand(self.settings.auto_white_balance,
                                  self.settings.white_balance_temp).execute(cap)

            # Focus
            SetFocusCommand(self.settings.auto_focus, self.settings.focus).execute(cap)

            # Gain
            SetGainCommand(self.settings.auto_gain, self.settings.gain, self.settings.iso).execute(cap)

            # Image adjustments
            SetImageAdjustmentCommand(
                self.settings.brightness,
                self.settings.contrast,
                self.settings.saturation,
                self.settings.sharpness
            ).execute(cap)

            # Buffer size
            SetBufferSizeCommand(self.settings.buffer_size).execute(cap)

            # v4l2 controls (Linux only)
            if self.device_path:
                V4L2Command(self.device_path, 'power_line_frequency',
                           self.settings.power_line_frequency).execute(cap)
                V4L2Command(self.device_path, 'backlight_compensation',
                           1 if self.settings.backlight_compensation else 0).execute(cap)

            print("✓ All settings applied")
            return True

        except Exception as e:
            print(f"Failed to apply settings: {e}")
            return False


class ThreadSafeCameraSettings:
    """
    Thread-safe wrapper for CameraSettings.

    Allows reading current settings from any thread,
    and updating settings through the command queue.
    """

    def __init__(self, initial_settings: Optional[CameraSettings] = None):
        self._settings = initial_settings or CameraSettings()
        self._lock = threading.RLock()

    def get(self) -> CameraSettings:
        """Get current settings (thread-safe)"""
        with self._lock:
            return self._settings

    def update(self, new_settings: CameraSettings):
        """Update settings (thread-safe)"""
        with self._lock:
            self._settings = new_settings

    def modify(self, **changes) -> CameraSettings:
        """
        Modify specific settings and return new settings object.

        Example:
            new_settings = settings.modify(width=1920, height=1080)
        """
        with self._lock:
            return self._settings.copy(**changes)
