# Threaded Capture System - Usage Guide

## Overview

The threaded capture system provides a robust, thread-safe architecture for camera/video capture and recording with the following key components:

1. **Frame Reader Thread** - Continuously reads frames from camera/video
2. **Video Writer Thread** - Asynchronously writes frames to video file
3. **Frame Buffer** - Thread-safe shared buffer for frame distribution
4. **Command Pattern** - Thread-safe camera control via command queue
5. **Settings Management** - Centralized, thread-safe camera settings

## Key Design Principles

### ✅ Thread Safety
- **Only the FrameReader thread accesses the capture device** (cv2.VideoCapture)
- All camera control commands go through a command queue
- Settings are protected by thread-safe wrappers
- Frame buffer uses locks for concurrent access

### ✅ Separation of Concerns
- Frame capture runs independently of processing
- Video recording runs independently of capture
- Settings changes don't block frame capture
- Multiple consumers can read from frame buffer

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  ThreadedCaptureSystem                   │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────────┐      ┌─────────────────┐           │
│  │  FrameReader    │─────▶│  FrameBuffer    │           │
│  │    Thread       │      │  (deque)        │           │
│  │                 │      │                 │           │
│  │ • Reads frames  │      │ • Latest frame  │           │
│  │ • Processes     │      │ • Rolling buffer│           │
│  │   commands      │      │ • Thread-safe   │           │
│  │ • Updates       │      └────────┬────────┘           │
│  │   settings      │               │                    │
│  └────────▲────────┘               │                    │
│           │                        │                    │
│           │                        ▼                    │
│  ┌────────┴─────────┐     ┌───────────────┐            │
│  │  Command Queue   │     │ VideoWriter   │            │
│  │                  │     │   Thread      │            │
│  │ • SetResolution  │     │               │            │
│  │ • SetFPS         │     │ • Reads from  │            │
│  │ • SetExposure    │     │   buffer      │            │
│  │ • ApplySettings  │     │ • Writes to   │            │
│  │ • Custom commands│     │   video file  │            │
│  └──────────────────┘     └───────────────┘            │
│                                                          │
│  ┌──────────────────────────────────────────┐           │
│  │     ThreadSafeCameraSettings             │           │
│  │  • Current resolution, FPS, exposure     │           │
│  │  • Brightness, contrast, saturation      │           │
│  │  • Read from any thread                  │           │
│  │  • Updated via command queue             │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

## Basic Usage

### 1. Simple Capture

```python
from threaded_capture import ThreadedCaptureSystem
import cv2
import time

# Open camera
cap = cv2.VideoCapture(0)

# Create system
system = ThreadedCaptureSystem(cap, source_type="camera", camera_index=0)
system.start()

# Get frames
while True:
    frame = system.get_latest_frame()
    if frame is not None:
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
system.stop()
cap.release()
cv2.destroyAllWindows()
```

### 2. Capture with Recording

```python
from threaded_capture import ThreadedCaptureSystem
import cv2
import time

cap = cv2.VideoCapture(0)
system = ThreadedCaptureSystem(cap, source_type="camera", camera_index=0)
system.start()

# Wait for first frame
time.sleep(1)

# Start recording (runs in separate thread)
system.start_recording("output.avi", fps=30)

# Capture for 10 seconds
time.sleep(10)

# Stop recording
system.stop_recording()

# Continue capturing without recording
time.sleep(3)

# Cleanup
system.stop()
cap.release()
```

## Camera Settings

### Using CameraSettings Object

```python
from camera_settings import CameraSettings
from threaded_capture import ThreadedCaptureSystem

# Create initial settings
settings = CameraSettings(
    width=1920,
    height=1080,
    fps=60,
    brightness=75,
    contrast=65,
    saturation=60,
    auto_exposure=True,
    auto_white_balance=True
)

# Create system with initial settings
cap = cv2.VideoCapture(0)
system = ThreadedCaptureSystem(
    cap,
    source_type="camera",
    camera_index=0,
    initial_settings=settings
)
system.start()
```

### Reading Current Settings

```python
# Get current settings (thread-safe)
current = system.get_settings()
print(f"Resolution: {current.width}x{current.height}")
print(f"FPS: {current.fps}")
print(f"Brightness: {current.brightness}")
```

### Updating Settings

```python
# Method 1: Update specific settings
system.update_settings(width=2560, height=1440, fps=30)

# Method 2: Create and apply new settings
new_settings = CameraSettings(
    width=1280,
    height=720,
    fps=60,
    brightness=80,
    contrast=70
)
system.apply_settings(new_settings)

# Method 3: Copy and modify existing settings
current = system.get_settings()
modified = current.copy(brightness=90, contrast=80)
system.apply_settings(modified)
```

### Settings Are Applied Asynchronously

```python
# Settings changes are queued and processed by FrameReader thread
system.update_settings(width=1920, height=1080)

# Give time for command to be processed
time.sleep(0.5)

# Verify settings
new_settings = system.get_settings()
print(f"New resolution: {new_settings.width}x{new_settings.height}")
```

## Command Pattern

### Built-in Commands

```python
from camera_settings import (
    SetResolutionCommand,
    SetFPSCommand,
    SetExposureCommand,
    SetWhiteBalanceCommand,
    SetFocusCommand,
    SetGainCommand,
    SetImageAdjustmentCommand,
    SetBufferSizeCommand,
    V4L2Command
)

# Resolution
system.send_command(SetResolutionCommand(1920, 1080))

# FPS
system.send_command(SetFPSCommand(60))

# Exposure
system.send_command(SetExposureCommand(auto=False, value=-5))

# White balance
system.send_command(SetWhiteBalanceCommand(auto=False, temp=5500))

# Image adjustments
system.send_command(SetImageAdjustmentCommand(
    brightness=75,
    contrast=65,
    saturation=60,
    sharpness=55
))

# V4L2 controls (Linux)
system.send_command(V4L2Command('/dev/video0', 'power_line_frequency', 1))
```

### Custom Commands

```python
from camera_settings import CameraCommand
import cv2

class SetZoomCommand(CameraCommand):
    def __init__(self, zoom_level):
        self.zoom_level = zoom_level

    def execute(self, cap):
        try:
            cap.set(cv2.CAP_PROP_ZOOM, self.zoom_level)
            print(f"Zoom set to {self.zoom_level}")
            return True
        except Exception as e:
            print(f"Failed to set zoom: {e}")
            return False

# Use custom command
system.send_command(SetZoomCommand(2.0))
```

## Advanced Features

### Pause/Resume

```python
# Pause frame reading (thread stays alive)
system.pause()

# Resume frame reading
system.resume()
```

### Frame Buffer Access

```python
# Get latest frame
frame = system.get_latest_frame()

# Get copy of entire buffer (for playback controls)
buffer_copy = system.frame_buffer.get_buffer_copy()

# Clear buffer
system.frame_buffer.clear()
```

### Video File Playback

```python
# Play video file instead of camera
cap = cv2.VideoCapture("video.mp4")
system = ThreadedCaptureSystem(cap, source_type="video")
system.start()

# Frames are read at native video FPS
while True:
    frame = system.get_latest_frame()
    if frame is not None:
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

system.stop()
```

### Recording with Custom FPS and Size

```python
# Start recording with specific settings
system.start_recording(
    "output.avi",
    fps=60,
    frame_size=(1920, 1080)  # Or None to auto-detect
)
```

## Integration with Existing Code

### Replacing Current CameraController

```python
# Old approach (in _capture_loop)
ret, frame = self.cap.read()  # Direct access to cap
if self.recording:
    self.video_writer.write(frame)  # Blocking write

# New approach
system = ThreadedCaptureSystem(self.cap, ...)
system.start()
system.start_recording("output.avi")  # Non-blocking

# In processing loop
frame = system.get_latest_frame()  # Get latest frame
# Apply transformations, detection, etc.
```

### Settings from HTTP API

```python
def handle_set_resolution(self, width, height):
    """HTTP endpoint to change resolution"""
    # Thread-safe settings update
    self.capture_system.update_settings(width=width, height=height)

    # Read back actual settings
    current = self.capture_system.get_settings()
    return {"width": current.width, "height": current.height}

def handle_get_settings(self):
    """HTTP endpoint to get current settings"""
    settings = self.capture_system.get_settings()
    return settings.to_dict()  # Convert to dictionary for JSON response
```

## Best Practices

### 1. Always Use Settings API

```python
# ❌ WRONG - Don't access cap directly from other threads
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

# ✅ CORRECT - Use settings API
system.update_settings(width=1920, height=1080)
```

### 2. Handle Async Settings Changes

```python
# Settings are applied asynchronously
system.update_settings(width=1920, height=1080)

# Option 1: Wait for processing
time.sleep(0.5)

# Option 2: Poll until changed
target_width = 1920
timeout = 2.0
start = time.time()
while time.time() - start < timeout:
    if system.get_settings().width == target_width:
        break
    time.sleep(0.1)
```

### 3. Initialize with Settings

```python
# Define settings before starting
settings = CameraSettings(width=1920, height=1080, fps=30)

# Pass to system
system = ThreadedCaptureSystem(
    cap,
    initial_settings=settings
)
system.start()
```

### 4. Cleanup Properly

```python
try:
    system.start()
    # ... use system ...
finally:
    system.stop()  # Stops all threads
    cap.release()  # Release capture device
```

## Testing

Run the test suites:

```bash
# Test basic threaded capture
python test_threaded_capture.py

# Test camera settings and commands
python test_camera_settings.py
```

## Troubleshooting

### Command Not Applied
- Commands are processed at most 10 per frame cycle
- Check if command queue is full
- Add delay after sending command

### Settings Not Changing
- Some cameras don't support all properties
- Check camera capabilities with v4l2-ctl (Linux)
- Verify actual values with get_settings()

### Frame Drops During Recording
- Recording runs in separate thread
- Check disk write speed
- Reduce resolution or FPS

### Thread Safety Issues
- Never access `cap` directly outside FrameReader
- Always use settings API and commands
- Use locks when accessing shared data

## Performance Tips

1. **Buffer Size**: Adjust based on memory and latency needs
   ```python
   system = ThreadedCaptureSystem(cap, buffer_size=50)  # 50 frames
   ```

2. **Command Batching**: Send multiple related commands at once
   ```python
   settings = CameraSettings(width=1920, height=1080, fps=60, brightness=75)
   system.apply_settings(settings)  # Single command vs multiple
   ```

3. **Recording Performance**: Video writing runs independently
   - No blocking in capture thread
   - Use appropriate codec (XVID, H264, etc.)

4. **Frame Access**: Getting frames is non-blocking
   ```python
   frame = system.get_latest_frame()  # Returns immediately
   ```
