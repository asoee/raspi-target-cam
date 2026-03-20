#!/bin/bash
# Script to uninstall the raspi-cam-stream systemd service

set -e

SERVICE_NAME="raspi-cam-stream.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "Uninstalling Raspberry Pi Camera Streaming Service..."

# Stop the service if running
echo "Stopping service..."
sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true

# Disable the service
echo "Disabling service..."
sudo systemctl disable "$SERVICE_NAME" 2>/dev/null || true

# Remove service file
echo "Removing service file..."
sudo rm -f "$SYSTEMD_DIR/$SERVICE_NAME"

# Reload systemd daemon
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo ""
echo "Service uninstalled successfully!"
echo ""
