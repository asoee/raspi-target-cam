# Threaded Capture System - Implementation Summary

## Overview

A complete threaded architecture for camera capture and video recording with thread-safe camera control via command pattern.

## Created Files

### Core Implementation

1. **[camera_settings.py](camera_settings.py)** - Settings and Command Pattern
   - `CameraSettings` - Dataclass for all camera parameters
   - `ThreadSafeCameraSettings` - Thread-safe settings wrapper
   - Command classes:
     - `SetResolutionCommand`
     - `SetFPSCommand`
     - `SetExposureCommand`
     - `SetWhiteBalanceCommand`
     - `SetFocusCommand`
     - `SetGainCommand`
     - `SetImageAdjustmentCommand`
     - `SetBufferSizeCommand`
     - `V4L2Command`
     - `ApplySettingsCommand`

2. **[threaded_capture.py](threaded_capture.py)** - Threading System
   - `FrameBuffer` - Thread-safe frame buffer with deque
   - `FrameReader` - Dedicated thread for reading frames
     - Command queue for camera control
     - Settings management
     - Pause/resume support
   - `VideoWriter` - Dedicated thread for video recording
   - `ThreadedCaptureSystem` - Orchestrator class

### Examples and Tests

3. **[camera_integration_example.py](camera_integration_example.py)**
   - Example CameraController using threaded system
   - Shows integration pattern
   - Runnable demo

4. **[test_threaded_capture.py](test_threaded_capture.py)**
   - Camera capture test
   - Video playback test
   - Recording test
   - Simultaneous operations test
   - Pause/resume test

5. **[test_camera_settings.py](test_camera_settings.py)**
   - Settings creation/modification test
   - Thread-safe settings test
   - Command execution test
   - Settings API test
   - Dynamic settings changes test
   - Custom commands test

### Documentation

6. **[THREADED_CAPTURE_USAGE.md](THREADED_CAPTURE_USAGE.md)**
   - Complete usage guide
   - Code examples
   - Best practices
   - Troubleshooting

## Key Features

### ✅ Thread Safety
- **Only FrameReader thread accesses capture device**
- All camera control via command queue
- Thread-safe settings access
- No race conditions or blocking

### ✅ Command Pattern
- Queue-based command processing
- Built-in commands for common operations
- Extensible for custom commands
- Non-blocking execution

### ✅ Settings Management
- `CameraSettings` dataclass with all parameters
- Thread-safe read/write access
- Immutable settings objects
- JSON serialization support

### ✅ Async Operations
- Frame reading in dedicated thread
- Video writing in dedicated thread
- Settings changes are non-blocking
- Independent frame buffer

## Architecture Benefits

### Separation of Concerns
```
┌────────────────┐     ┌──────────────┐     ┌─────────────┐
│  FrameReader   │────▶│ FrameBuffer  │────▶│  Processing │
│    Thread      │     │  (shared)    │     │    Thread   │
└────────────────┘     └──────┬───────┘     └─────────────┘
        ▲                     │
        │                     ▼
┌───────┴────────┐     ┌──────────────┐
│ Command Queue  │     │ VideoWriter  │
│  (settings)    │     │   Thread     │
└────────────────┘     └──────────────┘
```

### No Blocking
- Camera read doesn't block video write
- Video write doesn't block frame capture
- Settings changes don't block anything
- UI/API remains responsive

### Resource Efficiency
- Circular frame buffer (bounded memory)
- Configurable buffer size
- Automatic frame dropping on slow consumers
- Clean thread lifecycle management

## Usage Examples

### Basic Setup
```python
from threaded_capture import ThreadedCaptureSystem
from camera_settings import CameraSettings

# Configure initial settings
settings = CameraSettings(
    width=1920,
    height=1080,
    fps=60,
    brightness=75
)

# Create and start system
cap = cv2.VideoCapture(0)
system = ThreadedCaptureSystem(
    cap,
    source_type="camera",
    camera_index=0,
    initial_settings=settings
)
system.start()
```

### Change Settings (Thread-Safe)
```python
# Method 1: Update specific settings
system.update_settings(width=2560, height=1440)

# Method 2: Apply complete settings
new_settings = CameraSettings(width=1280, height=720, fps=30)
system.apply_settings(new_settings)

# Method 3: Send custom command
from camera_settings import SetResolutionCommand
system.send_command(SetResolutionCommand(1920, 1080))
```

### Read Settings (Thread-Safe)
```python
# Get current settings from any thread
current = system.get_settings()
print(f"{current.width}x{current.height} @ {current.fps} FPS")

# Convert to dict for JSON API
settings_dict = current.to_dict()
```

### Recording
```python
# Start recording (non-blocking)
system.start_recording("output.avi", fps=30)

# Continue capturing and processing
frame = system.get_latest_frame()

# Stop recording
system.stop_recording()
```

## Integration with camera_web_stream.py

### Current Approach (Problematic)
```python
# In _capture_loop:
ret, frame = self.cap.read()  # ❌ Multiple threads access cap
if self.recording:
    self.video_writer.write(frame)  # ❌ Blocking write

# Elsewhere:
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # ❌ Race condition
```

### New Approach (Thread-Safe)
```python
# In __init__:
self.capture_system = ThreadedCaptureSystem(
    self.cap,
    source_type=self.source_type,
    camera_index=self.camera_index
)

# Start capture:
self.capture_system.start()

# In processing loop:
frame = self.capture_system.get_latest_frame()  # ✅ Non-blocking
# Apply transformations, detection, etc.

# Change settings:
self.capture_system.update_settings(width=1920, height=1080)  # ✅ Thread-safe

# Start/stop recording:
self.capture_system.start_recording("output.avi")  # ✅ Non-blocking
self.capture_system.stop_recording()
```

### HTTP API Integration
```python
def handle_set_resolution(self, width, height):
    """HTTP endpoint"""
    self.capture_system.update_settings(width=width, height=height)
    return self.capture_system.get_settings().to_dict()

def handle_start_recording(self, filename):
    """HTTP endpoint"""
    return self.capture_system.start_recording(filename)

def handle_get_settings(self):
    """HTTP endpoint"""
    return self.capture_system.get_settings().to_dict()
```

## Migration Steps

### Step 1: Add to Imports
```python
from threaded_capture import ThreadedCaptureSystem
from camera_settings import CameraSettings
```

### Step 2: Replace Capture Initialization
```python
# Old:
self.cap = cv2.VideoCapture(self.camera_index)
threading.Thread(target=self._capture_loop, daemon=True).start()

# New:
self.cap = cv2.VideoCapture(self.camera_index)
self.capture_system = ThreadedCaptureSystem(
    self.cap,
    source_type=self.source_type,
    camera_index=self.camera_index,
    initial_settings=self._create_initial_settings()
)
self.capture_system.start()
```

### Step 3: Replace Frame Access
```python
# Old:
with self.lock:
    frame = self.frame.copy()

# New:
frame = self.capture_system.get_latest_frame()
```

### Step 4: Replace Settings Changes
```python
# Old:
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# New:
self.capture_system.update_settings(width=width, height=height)
```

### Step 5: Replace Recording
```python
# Old:
self.video_writer = cv2.VideoWriter(...)
# ... write in capture loop ...

# New:
self.capture_system.start_recording(filename, fps=30)
# ... recording happens automatically in background ...
self.capture_system.stop_recording()
```

## Testing

### Run Tests
```bash
# Test threaded capture
python test_threaded_capture.py

# Test settings and commands
python test_camera_settings.py

# Test integration example
python camera_integration_example.py
```

### Expected Results
- All frame reading/writing happens in background
- Settings changes don't block or crash
- Recording works simultaneously with capture
- Multiple threads can read settings safely

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| Thread Safety | ❌ Multiple threads access cap | ✅ Only FrameReader accesses cap |
| Settings Changes | ❌ Direct set() calls (unsafe) | ✅ Command queue (safe) |
| Video Recording | ❌ Blocks capture thread | ✅ Independent thread |
| Settings Access | ❌ No centralized storage | ✅ CameraSettings object |
| API Response | ❌ May block on camera operations | ✅ Always responsive |
| Code Organization | ❌ Mixed concerns | ✅ Clear separation |

## Next Steps

1. **Test with actual hardware**
   ```bash
   python test_camera_settings.py
   ```

2. **Integrate into camera_web_stream.py**
   - Replace direct cap access
   - Use ThreadedCaptureSystem
   - Expose settings via HTTP API

3. **Add settings persistence**
   - Save/load CameraSettings to JSON
   - Auto-apply on startup
   - Per-camera profiles

4. **Extend commands**
   - Add camera-specific controls
   - Region of interest (ROI)
   - Advanced v4l2 controls

## Files Reference

- **Core**: [camera_settings.py](camera_settings.py), [threaded_capture.py](threaded_capture.py)
- **Examples**: [camera_integration_example.py](camera_integration_example.py)
- **Tests**: [test_threaded_capture.py](test_threaded_capture.py), [test_camera_settings.py](test_camera_settings.py)
- **Docs**: [THREADED_CAPTURE_USAGE.md](THREADED_CAPTURE_USAGE.md), [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
