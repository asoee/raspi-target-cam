# Perspective UI Updates

The perspective adjustment UI has been enhanced with comprehensive controls for all calibration features.

## New Features in the UI

### 1. Calibration Method Selection

A dropdown selector allows choosing between:
- **Auto** - Tries ellipse first, falls back to checkerboard
- **Ellipse Detection** (default) - Detects circular target rings
- **Checkerboard Pattern** - Uses calibration checkerboard

### 2. Iterative Refinement Toggle

New toggle switch for iterative ellipse calibration:
- **Enabled by default** for better results
- Achieves 95-98% circularity vs 85-93% single-pass
- Shows as "Calibrating perspective (iterative)..." when running
- Only visible for ellipse/auto methods (hidden for checkerboard)

### 3. Checkerboard Pattern Settings

When checkerboard method is selected, shows pattern size options:
- 9×6 corners (10×7 squares) - default
- 7×5 corners (8×6 squares)
- 8×6 corners (9×7 squares)

### 4. Dynamic UI Updates

The interface automatically adapts based on selected method:
- Checkerboard: Shows pattern size selector, hides iterative toggle
- Ellipse/Auto: Shows iterative toggle, hides pattern size selector

### 5. Enhanced Status Messages

Calibration status now shows:
- Method being used (iterative/single-pass/checkerboard)
- Pattern size for checkerboard
- Iteration results in success message
- Checkmark (✓) when iterations are successful

## UI Layout

```
┌─────────────────────────────────────────────┐
│ 📐 Perspective Adjustment                   │
├─────────────────────────────────────────────┤
│ ┌─────────────┐  ┌─────────────────────┐   │
│ │ Perspective │  │ Corrected View      │   │
│ │ Debug View  │  │                     │   │
│ │             │  │                     │   │
│ └─────────────┘  └─────────────────────┘   │
├─────────────────────────────────────────────┤
│ CONTROLS:                                   │
│                                             │
│ ⚙️ Method: [Ellipse Detection      ▼]      │
│                                             │
│ [✓] Iterative Refinement                   │
│                                             │
│ ┌──────────────────────────────────┐       │
│ │ 🔄 Calibrate                      │       │
│ │ 🔍 Force Re-Detection             │       │
│ │ 💾 Save Calibration               │       │
│ └──────────────────────────────────┘       │
│                                             │
│ ℹ️ Instructions                             │
│ • Iterative Refinement (NEW!)              │
│   Improves circularity through multiple    │
│   corrections (95-98% vs 85-93%)           │
└─────────────────────────────────────────────┘
```

## Example Workflows

### Ellipse Calibration (Iterative - Default)
1. Select "Ellipse Detection" method
2. Ensure "Iterative Refinement" is checked ✓
3. Click "Calibrate"
4. Watch debug view for multi-iteration progress
5. Status shows: "Iterative ellipse calibration successful: 2 iterations, circularity 0.9612 ✓"
6. Click "Save Calibration"

### Ellipse Calibration (Single-Pass)
1. Select "Ellipse Detection" method
2. Uncheck "Iterative Refinement"
3. Click "Calibrate"
4. Status shows: "Ellipse calibration successful"
5. Click "Save Calibration"

### Checkerboard Calibration
1. Select "Checkerboard Pattern" method
2. Choose pattern size (default 9×6)
3. Hold checkerboard in front of camera
4. Click "Calibrate"
5. Status shows: "Checkerboard calibration successful (9x6)"
6. Click "Save Calibration"

### Auto Calibration
1. Select "Auto" method
2. Keep "Iterative Refinement" checked for best results
3. Click "Calibrate"
4. System tries ellipse first, falls back to checkerboard if needed
5. Status shows which method succeeded
6. Click "Save Calibration"

## Status Messages

The UI now provides detailed feedback:

| Scenario | Status Message |
|----------|---------------|
| **Iterative Success** | "Iterative ellipse calibration successful: 2 iterations, circularity 0.9612 ✓" |
| **Single-pass Success** | "Ellipse calibration successful" |
| **Checkerboard Success** | "Checkerboard calibration successful (9x6)" |
| **Auto Success** | "Iterative ellipse calibration successful: 1 iterations, circularity 0.9594 (auto) ✓" |
| **Failure** | "No suitable ellipse found for calibration" (red background) |

## Technical Details

### API Payload
```javascript
{
  "method": "ellipse",      // or "checkerboard", "auto"
  "iterative": true,        // for ellipse method
  "pattern_size": [9, 6]    // for checkerboard method
}
```

### API Response
```javascript
{
  "success": true,
  "message": "Iterative ellipse calibration successful: 2 iterations, circularity 0.9612",
  "method": "ellipse",
  "iterative": true
}
```

## Accessing the UI

Navigate to: `http://localhost:8088/perspective`

## Files Modified

- [perspective_adjustment.html](perspective_adjustment.html) - Enhanced UI with new controls

## Benefits

1. **Complete Control** - All calibration options accessible from UI
2. **Visual Feedback** - See exactly what method and settings are being used
3. **Smart Defaults** - Iterative enabled by default for best quality
4. **Adaptive Interface** - Only shows relevant controls for selected method
5. **Detailed Status** - Know exactly what happened during calibration

## Screenshots

The UI features:
- Modern gradient design
- Toggle switches for boolean options
- Dropdown selectors for methods
- Clear status messages with color coding
- Responsive layout for mobile/desktop
- Real-time video streams showing before/after
