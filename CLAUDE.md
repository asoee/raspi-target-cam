# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Raspberry Pi camera targeting system that uses OpenCV for computer vision and image processing. The project includes:

- **cam_detect.py**: Main webcam processing application with GUI using tkinter for camera control, image capture, and video recording
- **target.py**: Video analysis tool for target detection using perspective transformation and structural similarity comparison
- **cam_props.py**: Camera property testing and configuration utilities
- **list_cameras.py**: Camera port detection and enumeration
- **cam_resolutions.py**: Camera resolution testing capabilities
- **cam_test.py**: Simple camera testing script

## Architecture

### Core Components

1. **WebcamProcessor class** (`cam_detect.py`): Central camera handling with GUI interface for:
   - Camera connection and disconnection
   - Image capture to `./captures/` directory
   - Video recording capabilities
   - Real-time camera feed display

2. **Video Analysis** (`target.py`): Computer vision pipeline for:
   - Perspective transformation using `cv.getPerspectiveTransform()`
   - Frame-to-frame comparison using structural similarity
   - Thresholding and filtering operations
   - Motion/change detection

3. **Camera Utilities**:
   - Port scanning and camera enumeration
   - Property testing and configuration
   - Resolution capability detection

## Dependencies

The project uses Python with these key libraries:
- `opencv-python` (cv2) - Computer vision operations
- `numpy` - Numerical operations
- `tkinter` - GUI interface
- `PIL` - Image processing
- `scikit-image` - Image metrics (structural similarity)

## Running the Application

**Main GUI Application:**
```bash
python cam_detect.py
```

**List Available Cameras:**
```bash
python list_cameras.py
```

**Test Camera Properties:**
```bash
python cam_props.py
```

**Simple Camera Test:**
```bash
python cam_test.py
```

**Video Analysis:**
```bash
python target.py
```

## Key Implementation Details

- Images are saved to `./captures/` directory (auto-created)
- Default camera resolution testing at 3264x2448
- GUI uses threading for non-blocking camera operations
- Video analysis uses perspective transformation with predefined points
- Frame comparison threshold set at 0.97 for motion detection