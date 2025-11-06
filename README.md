# Raspi Target Cam

A Raspberry Pi camera targeting system with OpenCV computer vision for target detection, perspective correction, and real-time video streaming.

## Features

- Real-time MJPEG video streaming with web interface
- Automatic perspective correction using ellipse detection
- Target detection with bullet hole identification
- Video playback controls (pause/play/step forward/backward)
- Frame buffering system for precise shot analysis
- REST API for camera and playback control
- Comprehensive test suite

## System Requirements

### Linux Packages

Install required system dependencies:

```bash
apt-get update -y
apt-get install -y libusb-1.0-0-dev libturbojpeg-dev libudev-dev
```

See [docs/SYSTEM_DEPS.md](docs/SYSTEM_DEPS.md) for more details.

## Installation

This project uses [uv](https://astral.sh/uv) for fast, reliable Python dependency management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Project Dependencies

```bash
uv sync
```

This will create a virtual environment and install all dependencies automatically.

### Development Dependencies

To install additional development dependencies (pytest, scipy, requests):

```bash
uv sync --extra dev
```

## Running the Application

### Main Web Streaming Application

```bash
uv run raspi-cam-stream
```

Then open http://localhost:8088 in your browser.

### GUI Camera Application (Legacy)

```bash
uv run raspi-cam-detect
```

### List Available Cameras

```bash
uv run raspi-list-cameras
```

### Direct Python Execution

You can also run the modules directly:

```bash
uv run python -m raspi_target_cam.web.camera_web_stream
```

## Running Tests

Run all tests with pytest:

```bash
uv run pytest
```

Run with coverage:

```bash
uv run pytest --cov=raspi_target_cam
```

Run specific test file:

```bash
uv run pytest tests/test_transformation.py
```

## Project Structure

```
raspi-target-cam/
├── src/raspi_target_cam/      # Main package
│   ├── core/                  # Core modules (perspective, detection)
│   ├── camera/                # Camera utilities and controls
│   ├── detection/             # Detection algorithms
│   ├── analysis/              # Analysis and visualization tools
│   ├── utils/                 # Utility modules
│   └── web/                   # Web streaming application
├── tests/                     # Test suite
├── scripts/                   # Utility and debug scripts
├── legacy/                    # Deprecated code
├── docs/                      # Documentation
├── data/                      # Runtime data (captures, recordings, etc.)
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Development Workflow

### Adding New Dependencies

```bash
uv add <package-name>
```

### Removing Dependencies

```bash
uv remove <package-name>
```

### Updating All Dependencies

```bash
uv sync --upgrade
```

### Running Code Linting

```bash
uv run ruff check src/
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Project guidelines for AI assistance
- [docs/SYSTEM_DEPS.md](docs/SYSTEM_DEPS.md) - System dependencies
- [docs/CHECKERBOARD_CALIBRATION.md](docs/CHECKERBOARD_CALIBRATION.md) - Calibration guide
- [docs/THREADED_CAPTURE_USAGE.md](docs/THREADED_CAPTURE_USAGE.md) - Threading usage

## Architecture

See [CLAUDE.md](CLAUDE.md) for detailed architecture information, including:

- Component descriptions
- Workflow documentation
- Code organization principles
- Key implementation details

## License

See [LICENSE](LICENSE) for details.
