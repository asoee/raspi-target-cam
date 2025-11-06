# System Dependencies

This project requires certain system packages to be installed before the Python requirements.

## Required System Packages

For pyuvc support:
```bash
sudo apt update
sudo apt install -y libusb-1.0-0-dev libturbojpeg0-dev libudev-dev
```

## Installation Order

1. Install system dependencies first:
   ```bash
   sudo apt update
   sudo apt install -y libusb-1.0-0-dev libturbojpeg0-dev libudev-dev
   ```

2. Then install Python requirements:
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Notes

- `libusb-1.0-0-dev`: Required for USB device access (pyuvc)
- `libturbojpeg0-dev`: Required for JPEG encoding/decoding (pyuvc)
- `libudev-dev`: Required for device enumeration (pyuvc)