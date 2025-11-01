#!/bin/bash
# fix_framerate.sh - Fix MKV framerate metadata without re-encoding
# Usage: ./fix_framerate.sh <input_file> <target_fps>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_file> <target_fps>"
    echo "Example: $0 recording.mkv 3"
    exit 1
fi

INPUT_FILE="$1"
TARGET_FPS="$2"

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found"
    exit 1
fi

# Check if mkvpropedit is installed
if ! command -v mkvpropedit &> /dev/null; then
    echo "Error: mkvpropedit not found. Install with: sudo apt-get install mkvtoolnix"
    exit 1
fi

# Calculate default duration in nanoseconds
# Formula: 1000000000 / fps
DEFAULT_DURATION=$(echo "1000000000 / $TARGET_FPS" | bc)

echo "Processing: $INPUT_FILE"
echo "Current framerate info:"
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,avg_frame_rate -of default=noprint_wrappers=1 "$INPUT_FILE"

echo ""
echo "Setting framerate to ${TARGET_FPS} fps (default-duration=${DEFAULT_DURATION}ns)..."

# Fix the framerate
mkvpropedit "$INPUT_FILE" --edit track:v1 --set default-duration="$DEFAULT_DURATION"

if [ $? -eq 0 ]; then
    echo ""
    echo "Success! New framerate info:"
    ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,avg_frame_rate -of default=noprint_wrappers=1 "$INPUT_FILE"
else
    echo "Error: Failed to update framerate"
    exit 1
fi
