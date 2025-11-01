# Checkerboard Pattern Calibration

This document explains how to use the checkerboard pattern calibration feature as an alternative to ellipse-based calibration for perspective correction.

## Overview

The system now supports two calibration methods:
1. **Ellipse calibration** - Detects circular target rings that appear as ellipses due to perspective distortion
2. **Checkerboard calibration** - Uses a standard calibration checkerboard pattern (new feature)

## Checkerboard Pattern Requirements

### Standard Patterns

The system works with standard checkerboard calibration patterns:
- Default: **9x6 pattern** (9 columns × 6 rows of internal corners = 10x7 squares)
- Also supports: 7x5, 8x6, and other common patterns
- Pattern should be **printed on flat, rigid surface** (cardboard, foam board, etc.)

### Printing Your Pattern

You can:
1. **Generate a pattern** using OpenCV tools or online generators
2. **Download** pre-made patterns from calibration resources
3. **Print** at actual size (no scaling) on standard paper
4. **Mount** on a rigid backing to keep it flat

**Important**: The pattern must be flat and undistorted for accurate calibration!

## How to Use

### Option 1: Using the Web Interface

1. Open the web interface at `http://localhost:8088`
2. Go to the Perspective Calibration section
3. Select **Checkerboard** as the calibration method
4. Specify pattern size (default: 9x6)
5. Hold the checkerboard pattern in front of the camera
6. Ensure the pattern is:
   - Well-lit
   - Fully visible
   - Not occluded
   - Held relatively steady
7. Click **Calibrate**
8. If successful, click **Save Calibration**

### Option 2: Using the API

```bash
# Calibrate using checkerboard (9x6 pattern)
curl -X POST http://localhost:8088/api/calibrate_perspective \
  -H "Content-Type: application/json" \
  -d '{"method": "checkerboard", "pattern_size": [9, 6]}'

# Save the calibration
curl -X POST http://localhost:8088/api/save_calibration
```

### Option 3: Auto Mode (Tries Both Methods)

```bash
# Try ellipse first, fallback to checkerboard
curl -X POST http://localhost:8088/api/calibrate_perspective \
  -H "Content-Type: application/json" \
  -d '{"method": "auto"}'
```

## API Parameters

### Calibration Method
- `method`: `"ellipse"` | `"checkerboard"` | `"auto"`
  - `ellipse`: Use ellipse detection (for circular targets)
  - `checkerboard`: Use checkerboard detection
  - `auto`: Try ellipse first, fallback to checkerboard

### Pattern Size (Checkerboard Only)
- `pattern_size`: `[columns, rows]` - Number of **internal corners**
  - `[9, 6]`: Default (10x7 squares)
  - `[7, 5]`: 8x6 squares
  - `[8, 6]`: 9x7 squares

**Note**: Pattern size refers to the number of *internal corners*, not squares!
- A 10x7 square board has 9x6 internal corners

## Testing

Run the included test script to verify checkerboard calibration:

```bash
source venv/bin/activate
python3 test_checkerboard_calibration.py
```

This will:
1. Generate synthetic checkerboard patterns
2. Apply perspective distortion
3. Test detection and calibration
4. Save debug visualizations to `test_outputs/`

## Calibration File Format

The calibration is saved to `perspective_calibration.yaml`:

```yaml
calibration_method: checkerboard
camera_resolution: [2592, 1944]
checkerboard_data:
  calibration_timestamp: 1234567890.123
  pattern_size: [9, 6]
perspective_matrix:
  - [1.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0]
  - [0.0, 0.0, 1.0]
timestamp: 1234567890.123
```

## When to Use Each Method

### Use Ellipse Calibration When:
- You have a circular target (like shooting targets)
- The target is permanently mounted
- You want automatic detection during normal operation

### Use Checkerboard Calibration When:
- You don't have a circular target available
- You need precise calibration for measurement applications
- You're setting up a new camera position
- The circular target is difficult to detect (lighting, occlusion, etc.)

## Troubleshooting

### "Checkerboard pattern not detected"
**Solutions**:
- Ensure the pattern is **fully visible** in the camera frame
- Improve **lighting** - pattern should be evenly lit
- Check pattern is **flat** and not warped
- Verify correct **pattern size** parameter
- Try holding pattern at different **angles/distances**
- Use a **larger** checkerboard if detection fails

### "Calibration fails after detection"
**Solutions**:
- Pattern might be too **small** in the frame
- Try moving checkerboard **closer** to camera
- Ensure pattern has good **contrast** (clean print)

### Wrong pattern size?
Count the **internal corners**, not the squares:
- 10×7 squares = 9×6 corners ✓
- If you have 8×6 squares, use pattern_size: [7, 5]

## Advantages of Checkerboard Calibration

1. **Precise**: More corner points = better accuracy
2. **Standard**: Uses well-established computer vision methods
3. **Flexible**: Works without specific target requirements
4. **Repeatable**: Same pattern can be used anywhere
5. **Diagnostic**: Easy to see if calibration is working

## Example Use Cases

### Initial Camera Setup
Use checkerboard for initial rough calibration, then fine-tune with ellipse detection on actual target.

### Quality Control
Periodically verify calibration accuracy using checkerboard pattern.

### Multi-Camera Systems
Use same checkerboard to calibrate multiple cameras for consistency.

### Documentation
Include checkerboard calibration photo in setup documentation for reproducibility.

## Technical Details

### Detection Algorithm
- Uses OpenCV's `findChessboardCorners()` with adaptive thresholding
- Corner refinement to sub-pixel accuracy using `cornerSubPix()`
- Perspective transformation calculated from outer corners
- Result centered in frame for consistency

### Coordinate System
- Top-left corner = origin
- Pattern aligned to image edges after correction
- Preserves aspect ratio of original pattern

## See Also

- [perspective.py](perspective.py) - Implementation details
- [test_checkerboard_calibration.py](test_checkerboard_calibration.py) - Test script
- [CLAUDE.md](CLAUDE.md) - General project documentation
