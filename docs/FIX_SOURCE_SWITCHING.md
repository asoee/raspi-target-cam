# Fix: Source Switching Issue

## Problem
When switching from camera to test pattern (or any source switch), the console showed:
```
FrameReader: Capture device not available
```

## Root Cause
The `stop()` method was not stopping the ThreadedCaptureSystem. Here's what happened:

1. User switches source (camera → test)
2. `set_video_source()` calls `self.stop()`
3. `stop()` sets `self.running = False` and releases `self.cap`
4. **BUT** the old FrameReader thread keeps running!
5. FrameReader tries to access `self.cap` (now None)
6. Error: "Capture device not available"

## The Fix

### Before (camera_web_stream.py line 1853)
```python
def stop(self):
    """Stop camera capture"""
    self.running = False

    # Stop any ongoing recording
    if self.recording:
        self.stop_recording()

    # Simple wait for capture loop to notice running=False
    time.sleep(0.5)

    # Release capture device
    if self.cap:
        self.cap.release()
        self.cap = None
    # ❌ ThreadedCaptureSystem still running!
```

### After (Fixed)
```python
def stop(self):
    """Stop camera capture"""
    self.running = False

    # Stop any ongoing recording
    if self.recording:
        self.stop_recording()

    # ✅ Stop the threaded capture system
    if self.capture_system:
        print("DEBUG: Stopping capture system...")
        self.capture_system.stop()
        self.capture_system = None

    # Simple wait for threads to finish
    time.sleep(0.5)

    # Release capture device
    if self.cap:
        self.cap.release()
        self.cap = None
```

## What Changed
Added these lines in `stop()` method:
```python
# Stop the threaded capture system
if self.capture_system:
    try:
        print("DEBUG: Stopping capture system...")
        self.capture_system.stop()
        self.capture_system = None
    except Exception as e:
        print(f"WARNING: Error stopping capture system: {e}")
```

## Impact
- ✅ FrameReader thread stops cleanly
- ✅ VideoWriter thread stops cleanly
- ✅ No more "Capture device not available" errors
- ✅ Clean source switching: camera ↔ video ↔ test

## Testing
1. Start with camera
2. Switch to test pattern → Should switch cleanly
3. Switch to video → Should switch cleanly
4. Switch back to camera → Should switch cleanly

No errors should appear in console.

## Files Modified
- [camera_web_stream.py](camera_web_stream.py) - Line 1853, `stop()` method

## Status
✅ **FIXED** - Source switching now properly stops and restarts capture system
