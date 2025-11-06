# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Raspberry Pi camera targeting system that uses OpenCV for computer vision and image processing. The project is organized as a Python package using modern tooling (uv, pyproject.toml).

### Package Structure

The project follows a standard Python package layout:

- **src/raspi_target_cam/**: Main package
  - **core/**: Core modules (perspective.py, target_detection.py, streaming_handler.py)
  - **camera/**: Camera utilities, controls, and device detection
  - **detection/**: Various bullet hole detection algorithms
  - **analysis/**: Analysis and visualization tools
  - **utils/**: Utility modules (scoring, metadata handling)
  - **web/**: Web streaming application (camera_web_stream.py)
- **tests/**: Complete test suite (40+ tests)
- **scripts/**: Utility and debugging scripts
- **legacy/**: Deprecated code (cam_detect.py, target.py, web_server.py)
- **docs/**: Documentation files
- **data/**: Runtime data directories (captures, recordings, samples, etc.)

## Architecture

### Core Components

1. **src/raspi_target_cam/web/camera_web_stream.py**: Web-based streaming server with:
   - HTTP API endpoints for camera/video control
   - Real-time MJPEG streaming
   - Video playback controls (pause/play/step forward/backward)
   - Frame buffering system (50 frames: 25 before + 25 after pause point)
   - Perspective correction integration
   - Target detection with real-time visualization

2. **src/raspi_target_cam/core/target_detection.py**: TargetDetector class for:
   - Inner black circle detection with improved algorithms
   - Perspective transformation coordinate mapping
   - Integration with perspective.py functions (use direct imports, not wrapper methods)
   - Calibration save/load functionality

3. **src/raspi_target_cam/core/perspective.py**: Core perspective correction module:
   - Ellipse detection and perspective transformation
   - Always use functions from this module directly
   - Works with original frame dimensions (no fixed 800x800 sizing)
   - Comprehensive debug visualization functions

4. **Legacy Components** (in legacy/ directory):
   - `cam_detect.py`: GUI-based camera application (tkinter)
   - `target.py`: Original video analysis tool
   - `web_server.py`: Old web server implementation

## Dependencies

The project uses **uv** for dependency management (defined in pyproject.toml):

**Core dependencies:**
- `opencv-python` (cv2) - Computer vision operations
- `numpy` - Numerical operations
- `PyYAML` - Configuration file support
- `linuxpy` - Linux device access
- `Pillow` (PIL) - Image processing
- `piexif` - EXIF metadata handling
- `pupil-labs-uvc` - USB video class camera support (git dependency)

**Development dependencies:**
- `pytest` - Testing framework
- `pytest-cov` - Test coverage
- `scipy` - Scientific computing (for some detection algorithms)
- `requests` - HTTP library (for API tests)

**System dependencies** (see docs/SYSTEM_DEPS.md):
- `libusb-1.0-0-dev` - USB device access
- `libturbojpeg-dev` - JPEG encoding/decoding
- `libudev-dev` - Device enumeration

## Running the Application

### Setup

Install uv (first time only):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:
```bash
uv sync                    # Install core dependencies
uv sync --extra dev        # Install with dev dependencies
```

### Main Applications

**Web Streaming Application:**
```bash
uv run raspi-cam-stream
```
Then open http://localhost:8088 in a browser

**GUI Application (Legacy):**
```bash
uv run raspi-cam-detect
```

**List Cameras:**
```bash
uv run raspi-list-cameras
```

### Running Tests

**All tests:**
```bash
uv run pytest
```

**Specific test:**
```bash
uv run pytest tests/test_transformation.py
uv run pytest tests/test_playback_controls.py
```

**With coverage:**
```bash
uv run pytest --cov=raspi_target_cam
```

## Architecture Guidelines

### Code Organization Principles

1. **Package imports**: Always use package-qualified imports:
   ```python
   from raspi_target_cam.core import perspective
   from raspi_target_cam.core import target_detection
   from raspi_target_cam.utils import target_scoring
   ```

2. **Use perspective.py directly**: Always import and use functions from `perspective.py` directly rather than through wrapper methods in `target_detection.py`. This avoids code duplication and maintains a single source of truth.

3. **Work with original frame dimensions**: The codebase has been refactored to work with original frame dimensions instead of fixed 800x800 sizing. This provides better image quality and precision.

4. **Frame buffering for video playback**: The system implements a rolling buffer of 50 frames (25 before + 25 after pause point) to enable smooth step forward/backward controls.

### Key Implementation Details

- **Dependency management**: Use `uv` for all dependency operations (add, remove, sync, run)
- **Entry points**: Use `uv run raspi-cam-stream` instead of direct Python execution
- **Web interface**: Main application runs on localhost:8088 with HTTP API endpoints
- **Perspective correction**: Uses ellipse detection to automatically correct camera perspective
- **Target detection**: Focuses on detecting inner black circles within perspective-corrected frames
- **Playback controls**: Supports pause/play/step for video files with frame-accurate positioning
- **Debug visualization**: Comprehensive debug output in `test_outputs/` directory
- **Testing**: Use `uv run pytest` for all test execution

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