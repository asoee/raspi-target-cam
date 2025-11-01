#!/bin/bash
# validate_video.sh - Validate all frames in a video file
# Usage: ./validate_video.sh <input_file>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file>"
    echo "Example: $0 recording.mkv"
    exit 1
fi

INPUT_FILE="$1"

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found"
    exit 1
fi

echo "=========================================="
echo "Validating: $INPUT_FILE"
echo "=========================================="
echo ""

# Show file info
echo "File information:"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,r_frame_rate,avg_frame_rate,width,height -of default=noprint_wrappers=1 "$INPUT_FILE"
echo ""

# Count total frames
echo "Counting frames..."
TOTAL_FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 "$INPUT_FILE")
echo "Total frames: $TOTAL_FRAMES"
echo ""

# Validate all frames by decoding them
echo "Validating all frames (this may take a while)..."
echo "Decoding frames to null output to check for errors..."
echo ""

ffmpeg -v error -i "$INPUT_FILE" -f null - 2>&1 | tee /tmp/validate_errors.txt

# Check if there were any errors
if [ -s /tmp/validate_errors.txt ]; then
    echo ""
    echo "=========================================="
    echo "ERRORS DETECTED:"
    echo "=========================================="
    cat /tmp/validate_errors.txt
    echo ""
    echo "Result: FAILED - Video has corrupted or invalid frames"
    exit 1
else
    echo ""
    echo "=========================================="
    echo "Result: PASSED - All frames are valid"
    echo "=========================================="
    exit 0
fi
