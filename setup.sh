#!/bin/bash
# Setup script for the Raspberry Pi Target Camera project

echo "Setting up Raspberry Pi Target Camera environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete! 🎯"
echo ""
echo "To activate the environment in the future, run:"
echo "  ./activate.sh"
echo ""
echo "To start the camera system, run:"
echo "  python camera_web_stream.py"
echo ""