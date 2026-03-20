# Systemd Service Setup

The Raspberry Pi Camera Streaming System can be configured to start automatically on boot using systemd.

## Default Settings and Camera Presets on Startup

### Automatic Settings Loading

When the service starts, it automatically loads settings from [defaults.json](defaults.json), which includes:
- Video source configuration (camera/video/test)
- Camera resolution and FPS
- Zoom, pan, and rotation settings
- Target detection settings
- **Camera controls (V4L2)** - brightness, contrast, exposure, white balance, etc.

The system will apply these settings automatically on each startup.

### Camera Preset Loading

If you're using a physical camera (not test pattern or video file), the system will also attempt to load camera control settings from [data/presets/default.json](data/presets/default.json) after startup. This allows you to have consistent camera settings across reboots.

**How it works:**
1. The service starts and loads [defaults.json](defaults.json)
2. The camera initializes (this may take several seconds)
3. A background thread attempts to load the "default" preset for up to 20 seconds
4. If successful, camera controls from the preset are applied
5. If unsuccessful, the camera controls from [defaults.json](defaults.json) are still in effect

**To save your camera settings as the default preset:**
1. Open the web interface at http://localhost:8088
2. Adjust camera controls to your preference
3. Save the preset with the name "default"

**Note:** Camera controls are stored in two places:
- [defaults.json](defaults.json) - Contains a `camera_controls` section that's applied when switching to camera mode
- [data/presets/default.json](data/presets/default.json) - Dedicated preset file loaded on startup

For best results, save your camera settings in both locations using the web interface.

### Camera Controls in Web Interface

The web interface automatically detects when camera controls become available:
- When you first load the page, controls may show as "not available" if the camera is still initializing
- The interface polls the camera status every second
- Once the camera finishes initializing, controls will automatically become enabled
- No page refresh is needed - the controls will appear automatically

If camera controls don't appear after 30 seconds:
1. Check that you're using a camera source (not test pattern or video file)
2. Verify the camera is connected and accessible: `ls -la /dev/video*`
3. Check the service logs: `sudo journalctl -u raspi-cam-stream.service -n 50`

## Quick Start

### Install the Service

```bash
./install_service.sh
```

This will:
- Copy the service file to `/etc/systemd/system/`
- Enable the service to start on boot
- Start the service immediately

### Uninstall the Service

```bash
./uninstall_service.sh
```

## Manual Installation

If you prefer to install manually:

```bash
# Copy service file
sudo cp raspi-cam-stream.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable raspi-cam-stream.service
sudo systemctl start raspi-cam-stream.service
```

## Service Management Commands

### Check Service Status
```bash
sudo systemctl status raspi-cam-stream.service
```

### View Live Logs
```bash
sudo journalctl -u raspi-cam-stream.service -f
```

### View Recent Logs
```bash
sudo journalctl -u raspi-cam-stream.service -n 50
```

### Stop the Service
```bash
sudo systemctl stop raspi-cam-stream.service
```

### Start the Service
```bash
sudo systemctl start raspi-cam-stream.service
```

### Restart the Service
```bash
sudo systemctl restart raspi-cam-stream.service
```

### Disable Auto-Start on Boot
```bash
sudo systemctl disable raspi-cam-stream.service
```

### Enable Auto-Start on Boot
```bash
sudo systemctl enable raspi-cam-stream.service
```

## Service Configuration

The service file is located at [raspi-cam-stream.service](raspi-cam-stream.service).

Key configuration:
- **User**: `pi` (runs as the pi user)
- **Working Directory**: `/home/pi/raspi-target-cam`
- **Command**: `/home/pi/.local/bin/uv run raspi-cam-stream`
- **Auto-restart**: The service will automatically restart if it crashes (after 10 second delay)
- **Logging**: All output goes to systemd journal (view with `journalctl`)

## Accessing the Web Interface

Once the service is running, access the web interface at:

```
http://localhost:8088
```

Or from another device on the same network:

```
http://<raspberry-pi-ip>:8088
```

## Troubleshooting

### Service Fails to Start

1. Check if port 8088 is already in use:
   ```bash
   sudo lsof -i :8088
   ```

2. Check the logs for errors:
   ```bash
   sudo journalctl -u raspi-cam-stream.service -n 50
   ```

3. Try running manually to see detailed error messages:
   ```bash
   uv run raspi-cam-stream
   ```

### Service Keeps Restarting

Check the logs to identify the issue:
```bash
sudo journalctl -u raspi-cam-stream.service -f
```

Common issues:
- Camera device not available
- Port 8088 already in use
- Missing dependencies (run `uv sync` to install)

### Camera Not Accessible

The service runs as the `pi` user. Ensure the user has access to video devices:
```bash
ls -l /dev/video*
groups pi  # Should include 'video' group
```

If needed, add the user to the video group:
```bash
sudo usermod -a -G video pi
```

## System Requirements

- Raspberry Pi OS (or similar Linux distribution with systemd)
- uv package manager installed at `/home/pi/.local/bin/uv`
- All project dependencies installed (`uv sync`)
- Camera hardware or video files for playback

## Auto-Start Behavior

When enabled, the service will:
- Start automatically on system boot
- Wait for network to be available before starting
- Restart automatically if it crashes (with 10 second delay)
- Run in the background as a system service
- Log all output to systemd journal

To verify the service is enabled for auto-start:
```bash
sudo systemctl is-enabled raspi-cam-stream.service
```

Should output: `enabled`
