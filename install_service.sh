#!/bin/bash
# Script to install the raspi-cam-stream systemd service

set -e

SERVICE_NAME="raspi-cam-stream.service"
SERVICE_FILE="/home/pi/raspi-target-cam/${SERVICE_NAME}"
SYSTEMD_DIR="/etc/systemd/system"

echo "Installing Raspberry Pi Camera Streaming Service..."

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: Service file not found at $SERVICE_FILE"
    exit 1
fi

# Copy service file to systemd directory (requires sudo)
echo "Copying service file to $SYSTEMD_DIR..."
sudo cp "$SERVICE_FILE" "$SYSTEMD_DIR/"

# Reload systemd daemon
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable the service to start on boot
echo "Enabling service to start on boot..."
sudo systemctl enable "$SERVICE_NAME"

# Start the service now
echo "Starting service..."
sudo systemctl start "$SERVICE_NAME"

echo ""
echo "Service installed successfully!"
echo ""
echo "Useful commands:"
echo "  Check status:    sudo systemctl status $SERVICE_NAME"
echo "  View logs:       sudo journalctl -u $SERVICE_NAME -f"
echo "  Stop service:    sudo systemctl stop $SERVICE_NAME"
echo "  Start service:   sudo systemctl start $SERVICE_NAME"
echo "  Restart service: sudo systemctl restart $SERVICE_NAME"
echo "  Disable service: sudo systemctl disable $SERVICE_NAME"
echo ""
