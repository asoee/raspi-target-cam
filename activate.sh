#!/bin/bash
# Easy activation script for the virtual environment

echo "Activating virtual environment..."
source venv/bin/activate

echo "Virtual environment activated!"
echo "Python path: $(which python)"
echo "To deactivate, run: deactivate"

# Keep the shell open in the activated environment
#exec bash