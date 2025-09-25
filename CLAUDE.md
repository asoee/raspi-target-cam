# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Raspberry Pi camera targeting system that uses OpenCV for computer vision and image processing. The project includes:

- **camera_web_stream.py**: Main web-based streaming application with HTTP API for camera/video playback, perspective correction, and target detection
- **target_detection.py**: Core target detection class with circle detection and perspective transformation
- **perspective.py**: Perspective correction and ellipse detection utilities (use this module directly, not through wrappers)
- **test_transformation.py**: Test script for ellipse detection and perspective transformation
- **cam_detect.py**: Legacy webcam processing application with GUI (tkinter)
- **target.py**: Legacy video analysis tool
- **cam_props.py**: Camera property testing and configuration utilities
- **list_cameras.py**: Camera port detection and enumeration
- **cam_resolutions.py**: Camera resolution testing capabilities
- **cam_test.py**: Simple camera testing script

## Architecture

### Core Components

1. **camera_web_stream.py**: Web-based streaming server with:
   - HTTP API endpoints for camera/video control
   - Real-time MJPEG streaming
   - Video playback controls (pause/play/step forward/backward)
   - Frame buffering system (50 frames: 25 before + 25 after pause point)
   - Perspective correction integration
   - Target detection with real-time visualization

2. **target_detection.py**: TargetDetector class for:
   - Inner black circle detection with improved algorithms
   - Perspective transformation coordinate mapping
   - Integration with perspective.py functions (use direct imports, not wrapper methods)
   - Calibration save/load functionality

3. **perspective.py**: Core perspective correction module:
   - Ellipse detection and perspective transformation
   - Always use functions from this module directly
   - Works with original frame dimensions (no fixed 800x800 sizing)
   - Comprehensive debug visualization functions

4. **Legacy Components**:
   - `cam_detect.py`: GUI-based camera application (tkinter)
   - `target.py`: Original video analysis tool

## Dependencies

The project uses Python with these key libraries:
- `opencv-python` (cv2) - Computer vision operations
- `numpy` - Numerical operations
- `tkinter` - GUI interface
- `PIL` - Image processing
- `scikit-image` - Image metrics (structural similarity)

## Running the Application

**Main Web Streaming Application:**
```bash
source venv/bin/activate
python camera_web_stream.py
```
Then open http://localhost:8088 in a browser

**Test Perspective Transformation:**
```bash
source venv/bin/activate
python test_transformation.py
```

**Test Playback Controls:**
```bash
source venv/bin/activate
python test_playback_controls.py
```

**Legacy Applications:**
```bash
python cam_detect.py      # GUI application
python list_cameras.py    # List cameras
python cam_props.py       # Test camera properties
python cam_test.py        # Simple camera test
python target.py          # Legacy video analysis
```

## Architecture Guidelines

### Code Organization Principles

1. **Use perspective.py directly**: Always import and use functions from `perspective.py` directly rather than through wrapper methods in `target_detection.py`. This avoids code duplication and maintains a single source of truth.

2. **Work with original frame dimensions**: The codebase has been refactored to work with original frame dimensions instead of fixed 800x800 sizing. This provides better image quality and precision.

3. **Frame buffering for video playback**: The system implements a rolling buffer of 50 frames (25 before + 25 after pause point) to enable smooth step forward/backward controls.

### Key Implementation Details

- **Virtual environment**: Always use `source venv/bin/activate` before running Python scripts
- **Web interface**: Main application runs on localhost:8088 with HTTP API endpoints
- **Perspective correction**: Uses ellipse detection to automatically correct camera perspective
- **Target detection**: Focuses on detecting inner black circles within perspective-corrected frames
- **Playback controls**: Supports pause/play/step for video files with frame-accurate positioning
- **Debug visualization**: Comprehensive debug output in `test_outputs/` directory

### Video Playback System

- **Frame buffering**: Maintains circular buffer of recent frames for step controls
- **Pause functionality**: Freezes video at current position while maintaining buffer
- **Step controls**: Single-frame forward/backward navigation
- **API endpoints**: RESTful API for all playback operations
- **Status tracking**: Real-time frame position and playback state

### Perspective Correction Workflow

1. **Ellipse detection**: Automatic detection of target ring ellipse in camera view
2. **Transform calculation**: Generate perspective transform matrix to correct ellipse to circle
3. **Frame correction**: Apply transformation to entire frame (not just target area)
4. **Coordinate mapping**: Transform detection coordinates between original and corrected frames
5. **Calibration persistence**: Save/load transformation matrices for consistent operation