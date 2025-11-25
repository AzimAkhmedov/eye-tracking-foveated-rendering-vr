# VR Foveated Rendering Integration Guide

## Overview

This project implements a complete VR foveated rendering system with two integrated modules:

1. **Eye Tracking Module** (Python) - Tracks user gaze and broadcasts data
2. **Rendering Module** (JavaScript/Three.js) - Receives gaze data and applies foveated rendering

## Changes Made

### 1. Y-Axis Sensitivity Fixes

**Problem:** Y-axis gaze tracking was not sensitive/accurate enough.

**Solutions Implemented:**
- Increased base Y-axis sensitivity from `20.5` to `25.0`
- Added Y-axis gain multiplier (`1.2x`) for better vertical tracking
- Fixed `sensitivity_multiplier` to actually apply to calculations (was defined but never used)
- Added debug visualization showing Y-axis raw values and sensitivity

**Key Changes in `main.py`:**
```python
# Line 113: Increased base Y sensitivity
base_sensitivity_y = 25.0  # Increased from 20.5

# Line 120: Added Y-axis gain
y_gain = 1.2  # Additional gain specifically for Y-axis

# Line 121: Applied to calculation
screen_y = screen_h / 2 + gaze_world[1] * screen_h * sensitivity_y * y_gain * scale_factor
```

### 2. Module Integration

**WebSocket Communication:**
- Python module now runs a WebSocket server on `ws://localhost:8765`
- Broadcasts gaze data at ~100 FPS
- JavaScript module connects automatically and receives real-time gaze coordinates

**Data Format:**
```json
{
  "x": 0.5,           // Normalized X (0-1)
  "y": 0.5,           // Normalized Y (0-1)
  "screenX": 960,     // Screen pixel X
  "screenY": 540,     // Screen pixel Y
  "confidence": 0.85, // Tracking confidence (0-1)
  "distance": 0.6,    // Distance from camera (meters)
  "timestamp": 1234.56
}
```

### 3. Foveated Rendering Implementation

**Features:**
- Dynamic point size adjustment based on gaze position
- Opacity reduction in peripheral areas
- Confidence-based fallback to uniform rendering
- Real-time quality adjustment

**Configuration:**
```javascript
const FOV_CONFIG = {
  highQualityRadius: 0.15,      // 15% radius for high quality
  mediumQualityRadius: 0.35,    // 35% for medium quality
  lowQualityRadius: 0.6,        // 60% for low quality
  minPointSize: 0.0005,         // Peripheral point size
  maxPointSize: 0.002,          // Foveal point size
  minOpacity: 0.3,              // Peripheral opacity
  maxOpacity: 1.0               // Foveal opacity
};
```

### 4. Debug Visualization

**Python Module:**
- On-screen display of gaze coordinates (normalized and screen)
- Y-axis raw values and sensitivity multipliers
- Confidence and distance metrics
- Real-time sensitivity adjustment with `+/-` keys

**JavaScript Module:**
- Debug panel showing connection status, gaze position, confidence
- Visual gaze indicator (green circle) showing where user is looking
- Real-time foveated rendering status

## Installation

### Step 0: Configure environment variables

Copy the sample file and adjust the URLs/ports to match your setup:

```bash
cp env.example .env.local
# Optionally copy into the render module for IDE tooling / per-module overrides
cp env.example render-module/.env.local
```

Important keys:

- `GAZE_WS_HOST` / `GAZE_WS_PORT` — how the Python WebSocket server binds
- `VITE_GAZE_WS_URL` (plus host/port overrides) — URL the browser should call

### Prerequisites

1. **Python Module:**
```bash
cd eye-tracking-module
pip install -r dependencies.txt  # now includes python-dotenv
# Or manually: pip install opencv-python mediapipe numpy pyautogui websockets python-dotenv
```

2. **JavaScript Module:**
```bash
cd render-module
npm install
```

## Usage

### Step 1: Start Eye Tracking Module

```bash
cd eye-tracking-module
python main.py
```

**Controls:**
- `q` - Quit
- `c` - Calibrate center (resets gaze to screen center)
- `+` / `=` - Increase sensitivity
- `-` / `_` - Decrease sensitivity

**What to expect:**
- Camera window showing face detection
- Fullscreen red dot showing gaze position
- WebSocket server starts automatically on port 8765

### Step 2: Start Rendering Module

```bash
cd render-module
npm run dev
```

Open browser to the URL shown (typically `http://localhost:5173`)

**What to expect:**
- Three.js scene with point cloud
- Debug panel in top-left corner
- Green circle indicator showing gaze position (when connected)
- Automatic quality adjustment based on gaze

## Troubleshooting

### Y-Axis Still Not Sensitive Enough

1. **Increase Y-axis gain:**
   - Edit `main.py` line 119: Change `y_gain = 1.2` to higher value (e.g., `1.5`)

2. **Increase base Y sensitivity:**
   - Edit `main.py` line 113: Change `base_sensitivity_y = 25.0` to higher value (e.g., `30.0`)

3. **Use sensitivity multiplier:**
   - Press `+` key multiple times in the eye tracking window to increase overall sensitivity

### WebSocket Connection Issues

1. **Check if Python module is running:**
   - Look for message: `[✓] WebSocket server started on ws://localhost:8765`

2. **Check browser console:**
   - Open DevTools (F12) and check for WebSocket connection errors
   - Should see: `✓ Connected to gaze tracking server`

3. **Firewall/Port issues:**
   - Ensure port 8765 is not blocked
   - Try changing port in both files if needed

### Foveated Rendering Not Working

1. **Check confidence threshold:**
   - Foveated rendering only activates when `confidence > 0.3`
   - Ensure good lighting and face detection

2. **Verify gaze data:**
   - Check debug panel shows valid gaze coordinates
   - Green indicator should move when you look around

3. **Calibrate center:**
   - Press `c` in eye tracking window to reset center point

## Performance Optimization

### Current Implementation

The current foveated rendering uses a simplified approach:
- Adjusts overall point size and opacity based on gaze distance from center
- Works well for demonstration but can be optimized further

### Advanced Implementation (Future)

For true per-point foveated rendering:
1. Use custom shader material with per-point attributes
2. Calculate distance from gaze for each point in vertex shader
3. Adjust size/opacity per-point in GPU
4. This requires more complex shader code but provides better performance

## Technical Details

### Coordinate Systems

- **Normalized (0-1):** Used for VR rendering, sent via WebSocket
- **Screen pixels:** Used for display and calibration
- **World space:** Used internally for 3D calculations

### Gaze Calculation Pipeline

1. MediaPipe detects face landmarks
2. Calculate iris position relative to eye center
3. Estimate head pose using 3D model points
4. Project gaze vector to world space
5. Apply sensitivity multipliers and gain
6. Convert to screen coordinates
7. Normalize for VR rendering
8. Broadcast via WebSocket

### Confidence Calculation

```python
confidence = min(1.0, max(0.0, 1.0 - abs(distance - 0.6) / 0.3)) * min(1.0, gaze_magnitude * 10)
```

- Based on distance from optimal camera distance (0.6m)
- Weighted by gaze vector magnitude
- Used to determine if gaze data is reliable enough for foveated rendering

## Next Steps

1. **Improve foveated rendering:**
   - Implement true per-point shader-based rendering
   - Add multiple LOD levels with separate point clouds
   - Optimize for VR headsets (stereo rendering)

2. **Enhanced calibration:**
   - Multi-point calibration routine
   - Save/load calibration profiles
   - Per-user calibration storage

3. **Performance monitoring:**
   - FPS counter
   - Render time metrics
   - Quality vs performance tradeoff visualization

4. **VR Headset Integration:**
   - Support for VR headset eye tracking APIs
   - Stereo rendering with independent gaze per eye
   - Headset-specific optimizations

## Files Modified

- `eye-tracking-module/main.py` - Added WebSocket server, fixed Y-axis sensitivity
- `eye-tracking-module/dependencies.txt` - Added `websockets` package
- `render-module/src/main.js` - Added WebSocket client, foveated rendering, debug UI

## Support

For issues or questions:
1. Check debug output in both modules
2. Verify all dependencies are installed
3. Ensure camera permissions are granted
4. Check browser console for JavaScript errors

