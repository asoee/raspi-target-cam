#!/usr/bin/env python3
"""
Unified Camera HTTP Streaming and Web Interface Server
Combines MJPEG camera stream and web interface in a single threaded application
"""

import faulthandler
import signal
import sys
import os

# Enable faulthandler to catch segmentation faults
faulthandler.enable()

# Also register faulthandler for specific signals
faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

print("🔍 Faulthandler enabled - will show traceback on segmentation fault")
print("💡 Send SIGUSR1 signal to dump all thread stacks: kill -USR1 <pid>")
print(f"🆔 Process ID: {os.getpid()}")

def crash_handler(signum, frame):
    """Handle crashes with detailed information"""
    # Prevent recursive crashes
    signal.signal(signum, signal.SIG_DFL)  # Restore default handler

    # Simple crash reporting
    try:
        with open('/tmp/camera_crash.log', 'w') as f:
            f.write(f"CRASH: Signal {signum}\n")
            faulthandler.dump_traceback(file=f, all_threads=True)
        print(f"\n💥 SEGFAULT DETECTED - Signal {signum}")
        print("📋 Full crash details saved to /tmp/camera_crash.log")
    except:
        pass  # Don't let crash handler crash

    os._exit(1)  # Force exit without cleanup

# Register crash handler only for segfault
signal.signal(signal.SIGSEGV, crash_handler)

import cv2
import threading
import time
import json
import subprocess
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from target_detection import TargetDetector
from perspective import Perspective
from bullet_hole_detection import BulletHoleDetector
from camera_controls import CameraControlManager
from PIL import Image
from PIL.ExifTags import TAGS
import piexif


class CameraController:
    """Centralized camera control and streaming"""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.video_file = None
        self.source_type = "test"  # Start with test pattern to avoid camera hang
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

        # Camera settings
        self.resolution = (2592, 1944)
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.rotation = 270  # degrees clockwise (270 = 90 anti-clockwise)

        # Capture settings
        self.captures_dir = "./captures"
        os.makedirs(self.captures_dir, exist_ok=True)

        # Video recording settings
        self.recordings_dir = "./recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
        self.recording = False
        self.video_writer = None
        self.recording_filename = None
        self.recording_start_time = None
        self.recording_width = 0
        self.recording_height = 0

        # Video file settings
        self.samples_dir = "./samples"
        self.video_fps = 30  # Default FPS for video files
        self.video_frame_time = 1.0 / self.video_fps  # Time between frames
        self.native_video_resolution = None  # Native resolution of video file
        self.native_video_fps = None  # Native FPS of video file

        # Available sources (detected once at startup)
        self.available_sources = {'cameras': [], 'videos': []}

        # Detect available sources
        self._detect_available_sources()

        # Playback controls
        self.paused = False
        self.step_frame = False  # Flag to advance one frame when paused
        self.step_direction = 1  # 1 for forward, -1 for backward
        self.current_frame_number = 0
        self.display_frame_number = 0  # Frame number to display (doesn't advance when stepping back)
        self.total_frames = 0

        # Frame buffer for pause functionality (50 frames total: 25 before + 25 after pause point)
        self.frame_buffer = []
        self.buffer_size = 50
        self.pause_buffer_index = 25  # Index in buffer where we paused (25 frames back from pause point)
        self.pause_frame_number = 0  # Frame number when we paused

        # Target detection
        self.target_detector = TargetDetector()

        # Perspective correction for main stream
        self.perspective = Perspective()  # Own perspective correction instance
        self.perspective_correction_enabled = True
        
        # Test frame generation
        self.test_frame_counter = 0
        
        # Bullet hole detection
        self.bullet_hole_detector = BulletHoleDetector()
        self.reference_frame = None  # Store reference frame for bullet hole detection
        self.bullet_holes = []  # Detected bullet holes

        # Camera controls
        self.camera_controls = None  # Will be initialized when camera is opened
        self.cached_camera_controls = {'available': False, 'controls': {}}  # Cached controls from detection

    def generate_test_frame(self):
        """Generate a static test frame with useful information"""
        import numpy as np
        
        # Create frame with current resolution
        height, width = self.resolution[1], self.resolution[0]
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            for x in range(width):
                frame[y, x] = [
                    int(50 + (x / width) * 100),      # Red gradient
                    int(30 + (y / height) * 80),      # Green gradient
                    100                                # Blue constant
                ]
        
        # Add title
        cv2.putText(frame, "🎯 Raspberry Pi Target Camera", (50, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Add digital clock in top-right corner
        from datetime import datetime
        current_time = datetime.now()
        time_str = current_time.strftime("%H:%M:%S")
        date_str = current_time.strftime("%Y-%m-%d")
        
        # Clock background
        clock_bg_x = width - 300
        clock_bg_y = 30
        clock_bg_w = 250
        clock_bg_h = 80
        cv2.rectangle(frame, (clock_bg_x, clock_bg_y), 
                     (clock_bg_x + clock_bg_w, clock_bg_y + clock_bg_h), 
                     (0, 0, 0), -1)  # Black background
        cv2.rectangle(frame, (clock_bg_x, clock_bg_y), 
                     (clock_bg_x + clock_bg_w, clock_bg_y + clock_bg_h), 
                     (100, 100, 100), 2)  # Gray border
        
        # Digital time display
        cv2.putText(frame, time_str, (clock_bg_x + 10, clock_bg_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)  # Green digits
        cv2.putText(frame, date_str, (clock_bg_x + 10, clock_bg_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)  # Gray date
        
        # Add status information
        y_pos = 150
        info_lines = [
            f"Resolution: {width} x {height}",
            f"Mode: Test Pattern",
            f"Status: Waiting for source selection",
            "",
            "Select a camera or video source",
            "from the web interface controls",
            "",
            f"Available sources:",
            f"• {len(self.available_sources.get('cameras', []))} camera(s) detected",
            f"• {len(self.available_sources.get('videos', []))} video file(s) available"
        ]
        
        for line in info_lines:
            cv2.putText(frame, line, (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 40
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Generated: {timestamp}", (50, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Add frame counter (to show it's "live")
        self.test_frame_counter += 1
        cv2.putText(frame, f"Frame: {self.test_frame_counter}", (width - 200, height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Add some visual elements
        # Corner markers
        corner_size = 50
        cv2.rectangle(frame, (0, 0), (corner_size, corner_size), (0, 255, 0), 3)
        cv2.rectangle(frame, (width - corner_size, 0), (width, corner_size), (0, 255, 0), 3)
        cv2.rectangle(frame, (0, height - corner_size), (corner_size, height), (0, 255, 0), 3)
        cv2.rectangle(frame, (width - corner_size, height - corner_size), (width, height), (0, 255, 0), 3)
        
        # Center crosshair
        center_x, center_y = width // 2, height // 2
        cv2.line(frame, (center_x - 50, center_y), (center_x + 50, center_y), (255, 0, 0), 2)
        cv2.line(frame, (center_x, center_y - 50), (center_x, center_y + 50), (255, 0, 0), 2)
        
        return frame

    def start_capture(self):
        """Initialize and start camera or video file capture - SAFE VERSION"""
        try:
            print(f"DEBUG: Initializing {self.source_type} capture...")
            
            if self.source_type == "test":
                # Test mode - no actual capture device needed
                print("DEBUG: Test mode - generating static test frame")
                self.running = True
                print("DEBUG: Starting capture thread...")
                threading.Thread(target=self._capture_loop, daemon=True).start()
                return True
                
            elif self.source_type == "camera":
                print(f"DEBUG: Opening camera {self.camera_index}...")
                
                # Use timeout mechanism to prevent hanging
                try:
                    self.cap = cv2.VideoCapture(self.camera_index)
                    
                    # Check if opened within a reasonable time
                    if not self.cap or not self.cap.isOpened():
                        if self.cap:
                            self.cap.release()
                        raise RuntimeError(f"Could not open camera {self.camera_index}")
                    
                    print(f"DEBUG: Camera {self.camera_index} opened successfully")

                    # Set anti-flicker to 50Hz using v4l2-ctl
                    try:
                        result = subprocess.run(
                            ['v4l2-ctl', '-d', f'/dev/video{self.camera_index}', '--set-ctrl', 'power_line_frequency=1'],
                            capture_output=True, text=True, timeout=2
                        )
                        if result.returncode == 0:
                            print(f"DEBUG: Anti-flicker set to 50Hz")
                        else:
                            print(f"WARNING: Could not set anti-flicker: {result.stderr}")
                    except Exception as e:
                        print(f"WARNING: Could not set anti-flicker: {e}")

                    # Set camera properties safely
                    try:
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        print(f"DEBUG: Camera properties set")
                    except Exception as e:
                        print(f"WARNING: Could not set camera properties: {e}")
                        # Continue anyway, camera might still work with default settings

                    # Use cached camera controls from source detection
                    try:
                        print(f"DEBUG: Loading cached camera controls for camera {self.camera_index}...")

                        # Find the camera in available sources to get cached controls
                        camera_source = None
                        for camera in self.available_sources.get('cameras', []):
                            if camera['index'] == self.camera_index:
                                camera_source = camera
                                break

                        if camera_source and camera_source.get('controls', {}).get('available'):
                            # Use cached control information
                            self.cached_camera_controls = camera_source['controls']
                            print(f"DEBUG: Using cached controls - {len(self.cached_camera_controls['controls'])} controls available")
                        else:
                            self.cached_camera_controls = {'available': False, 'controls': {}}
                            print(f"DEBUG: No cached controls available for camera {self.camera_index}")

                    except Exception as e:
                        print(f"WARNING: Could not load cached camera controls: {e}")
                        self.cached_camera_controls = {'available': False, 'controls': {}}
                        
                except Exception as e:
                    print(f"ERROR: Failed to initialize camera {self.camera_index}: {e}")
                    if self.cap:
                        try:
                            self.cap.release()
                        except:
                            pass
                        self.cap = None
                    raise RuntimeError(f"Camera initialization failed: {str(e)}")
            else:  # video file
                if not self.video_file or not os.path.exists(self.video_file):
                    raise RuntimeError(f"Video file not found: {self.video_file}")

                self.cap = cv2.VideoCapture(self.video_file)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Could not open video file: {self.video_file}")

                # Get video properties and override settings
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if fps > 0:
                    self.video_fps = fps
                    self.native_video_fps = fps
                    self.video_frame_time = 1.0 / fps

                if width > 0 and height > 0:
                    self.native_video_resolution = (width, height)
                    # Override resolution for video files to match native resolution
                    self.resolution = (width, height)

                if frame_count > 0:
                    self.total_frames = frame_count
                    self.current_frame_number = 0
                    self.display_frame_number = 0

                print(f"Video properties: {width}x{height} @ {fps:.1f} FPS, {frame_count} frames")

            self.running = True
            # Load calibration in perspective instance
            try:
                print("DEBUG: Loading perspective calibration...")
                self.perspective.load_calibration()
                print("DEBUG: Perspective calibration loaded successfully")
            except Exception as e:
                print(f"WARNING: Failed to load perspective calibration: {e}")

            print("DEBUG: Starting capture thread...")
            threading.Thread(target=self._capture_loop, daemon=True).start()
            return True
        except Exception as e:
            print(f"Failed to initialize capture: {e}")
            return False

    def _capture_loop(self):
        """Continuous frame capture and processing loop"""
        while self.running:
            try:
                # Check if we should stop before processing
                if not self.running:
                    break
                    
                # Handle test mode differently
                if self.source_type == "test":
                    # Generate test frame
                    test_frame = self.generate_test_frame()
                    with self.lock:
                        self.frame = test_frame.copy()
                    time.sleep(0.2)  # 10 FPS for test frame updates
                    continue
                
                # Safely check if capture device is available
                if not self.cap or not self.cap.isOpened():
                    print("WARNING: Capture device not available")
                    time.sleep(0.1)
                    continue
                # Handle pause/play logic
                if self.paused and not self.step_frame:
                    # When paused and not stepping, serve frames from buffer
                    if self.frame_buffer and self.pause_buffer_index < len(self.frame_buffer):
                        buffered_frame = self.frame_buffer[self.pause_buffer_index]
                        with self.lock:
                            self.frame = buffered_frame.copy()
                    time.sleep(0.1)  # Sleep while paused
                    continue

                # Read new frame from source
                ret, frame = self.cap.read()
                if ret:
                    # Validate the frame from the source
                    if not self._is_valid_frame(frame):
                        print(f"WARNING: Invalid frame from source, skipping")
                        continue

                    # Write raw frame to video if recording (before any processing)
                    if self.recording and self.video_writer is not None:
                        try:
                            # Ensure frame is in correct format for video writing
                            if self._is_valid_frame(frame):
                                # Check frame dimensions match recording dimensions
                                frame_height, frame_width = frame.shape[:2]
                                if frame_width != self.recording_width or frame_height != self.recording_height:
                                    # Resize frame to match recording dimensions
                                    write_frame = cv2.resize(frame, (self.recording_width, self.recording_height))
                                else:
                                    write_frame = frame

                                # Make sure frame is contiguous in memory
                                if not write_frame.flags['C_CONTIGUOUS']:
                                    write_frame = write_frame.copy()

                                # Ensure proper color format (BGR, 3 channels)
                                if len(write_frame.shape) == 2:
                                    # Convert grayscale to BGR
                                    write_frame = cv2.cvtColor(write_frame, cv2.COLOR_GRAY2BGR)
                                elif write_frame.shape[2] == 4:
                                    # Convert BGRA to BGR
                                    write_frame = cv2.cvtColor(write_frame, cv2.COLOR_BGRA2BGR)
                                elif write_frame.shape[2] != 3:
                                    print(f"WARNING: Unexpected channel count {write_frame.shape[2]}, skipping frame")
                                    continue

                                # Ensure uint8 data type
                                if write_frame.dtype != 'uint8':
                                    write_frame = write_frame.astype('uint8')

                                # Validate final frame before writing
                                if write_frame.shape == (self.recording_height, self.recording_width, 3):
                                    self.video_writer.write(write_frame)
                                else:
                                    print(f"WARNING: Frame shape mismatch {write_frame.shape} vs expected ({self.recording_height}, {self.recording_width}, 3)")
                        except Exception as e:
                            print(f"ERROR writing frame to video: {e}")
                            import traceback
                            traceback.print_exc()

                    # Update current frame number for video files
                    if self.source_type == "video":
                        self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                    # Apply zoom and pan transformations
                    try:
                        processed_frame = self._apply_transformations(frame)

                        # Validate processed frame
                        if not self._is_valid_frame(processed_frame):
                            print(f"WARNING: Transformation resulted in invalid frame")
                            processed_frame = frame  # Use original frame

                    except Exception as e:
                        print(f"ERROR in _apply_transformations: {e}")
                        processed_frame = frame  # Use original frame if transformation fails

                    # Add to frame buffer (maintain rolling buffer)
                    # Ensure we're adding a valid frame
                    if self._is_valid_frame(processed_frame):
                        self.frame_buffer.append(processed_frame.copy())
                        if len(self.frame_buffer) > self.buffer_size:
                            self.frame_buffer.pop(0)
                    else:
                        print(f"WARNING: Skipping invalid processed frame")

                    # Note: Frame stepping is now handled directly via seek_to_frame() 
                    # in step_frame_forward/backward methods, not through the capture loop

                    # Update display frame number for normal playback
                    if not self.paused and self.source_type == "video":
                        self.display_frame_number = self.current_frame_number

                    with self.lock:
                        self.frame = processed_frame.copy()
                else:
                    # If video file reached end, loop back to beginning
                    if self.source_type == "video":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_number = 0
                        self.display_frame_number = 0
                        continue

                # Use appropriate sleep timing
                if self.source_type == "video" and not self.paused:
                    time.sleep(self.video_frame_time)
                else:
                    time.sleep(0.03)  # ~30 FPS for camera

            except Exception as e:
                print(f"ERROR in _capture_loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)  # Brief pause before retrying

    def _handle_frame_step(self):
        """
        Handle frame stepping when paused.
        Returns True if a frame was stepped and should continue, False otherwise.
        """
        if not self.frame_buffer:
            return False

        old_index = self.pause_buffer_index
        new_index = self.pause_buffer_index + self.step_direction

        # Check boundaries
        if new_index < 0 or new_index >= len(self.frame_buffer):
            return False

        # Update buffer index
        self.pause_buffer_index = new_index

        # Update display frame number based on actual movement in buffer
        if self.pause_buffer_index != old_index:
            self.display_frame_number += self.step_direction
            # Ensure we don't go below 0 or above total frames
            self.display_frame_number = max(0, min(self.total_frames, self.display_frame_number))

        # Get the buffered frame and update display
        buffered_frame = self.frame_buffer[self.pause_buffer_index]
        with self.lock:
            self.frame = buffered_frame.copy()

        return True

    def _apply_transformations(self, frame):
        """Apply rotation, zoom, pan, and perspective transformations to frame"""
        original_size = (frame.shape[1], frame.shape[0])  # (width, height)

        # Apply rotation first
        if self.rotation != 0:
            frame = self._rotate_frame(frame, self.rotation)

        # Handle perspective correction and target detection together
        if self.perspective_correction_enabled:
            corrected_frame = self.perspective.apply_perspective_correction(frame)
            if corrected_frame is not None:
                # Run target detection on the corrected frame for better accuracy
                # This ensures detection works on the geometrically corrected image
                frame = self.target_detector.draw_target_overlay(corrected_frame, 
                                                               target_info=self.target_detector.detect_target(corrected_frame), 
                                                               frame_is_corrected=True)
            else:
                # Perspective correction failed, run detection on original frame
                frame = self.target_detector.draw_target_overlay(frame, 
                                                               target_info=self.target_detector.detect_target(frame), 
                                                               frame_is_corrected=False)
        else:
            # No perspective correction, run target detection on original frame
            frame = self.target_detector.draw_target_overlay(frame, 
                                                           target_info=self.target_detector.detect_target(frame), 
                                                           frame_is_corrected=False)
        
        # Add bullet hole overlays if any have been detected
        if self.bullet_holes:
            frame = self.bullet_hole_detector.draw_bullet_hole_overlays(frame, self.bullet_holes)

        # Skip other transformations if no zoom or pan
        if self.zoom == 1.0 and self.pan_x == 0 and self.pan_y == 0:
            return frame

        h, w = frame.shape[:2]

        # Apply zoom
        if self.zoom > 1.0:
            # Calculate crop dimensions
            crop_w = int(w / self.zoom)
            crop_h = int(h / self.zoom)

            # Calculate center with pan offset
            center_x = w // 2 + int(self.pan_x * w / 200)  # pan_x is -100 to 100
            center_y = h // 2 + int(self.pan_y * h / 200)  # pan_y is -100 to 100

            # Ensure crop stays within bounds
            x1 = max(0, center_x - crop_w // 2)
            y1 = max(0, center_y - crop_h // 2)
            x2 = min(w, x1 + crop_w)
            y2 = min(h, y1 + crop_h)

            # Crop and resize back to original resolution
            cropped = frame[y1:y2, x1:x2]

            # Validate cropped region
            if not self._is_valid_frame(cropped):
                print(f"WARNING: Invalid cropped region, skipping resize")
                return frame

            if cropped.shape[0] <= 0 or cropped.shape[1] <= 0:
                print(f"WARNING: Empty cropped region {cropped.shape}, skipping resize")
                return frame

            if w <= 0 or h <= 0:
                print(f"WARNING: Invalid resize dimensions {w}x{h}")
                return frame

            try:
                frame = cv2.resize(cropped, (w, h))
            except Exception as e:
                print(f"ERROR in cv2.resize: {e}")
                return frame  # Return original if resize fails

        return frame

    def _rotate_frame(self, frame, degrees):
        """Rotate frame by specified degrees clockwise"""
        # Validate input frame
        if not self._is_valid_frame(frame):
            print(f"WARNING: Invalid frame for rotation, returning original")
            return frame

        try:
            if degrees == 0:
                return frame
            elif degrees == 90:
                return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif degrees == 180:
                return cv2.rotate(frame, cv2.ROTATE_180)
            elif degrees == 270:
                return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                # For arbitrary angles, use affine transformation
                h, w = frame.shape[:2]
                if h <= 0 or w <= 0:
                    print(f"WARNING: Invalid frame dimensions {w}x{h} for rotation")
                    return frame

                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, -degrees, 1.0)

                # Validate rotation matrix
                if rotation_matrix is None or rotation_matrix.shape != (2, 3):
                    print(f"WARNING: Invalid rotation matrix generated")
                    return frame

                return cv2.warpAffine(frame, rotation_matrix, (w, h))
        except Exception as e:
            print(f"ERROR in _rotate_frame: {e}")
            return frame  # Return original frame on error

    def _is_valid_frame(self, frame):
        """Validate frame for OpenCV operations"""
        if frame is None:
            return False
        if not hasattr(frame, 'shape') or not hasattr(frame, 'dtype'):
            return False
        if len(frame.shape) not in [2, 3]:  # Grayscale or color
            return False
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            return False
        if len(frame.shape) == 3 and frame.shape[2] not in [1, 3, 4]:  # Valid channel counts
            return False
        if frame.size == 0:
            return False
        return True

    def get_frame_jpeg(self):
        """Get the latest frame as JPEG bytes"""
        try:
            with self.lock:
                if self.frame is not None:
                    # Validate frame before OpenCV operations
                    if not self._is_valid_frame(self.frame):
                        print(f"WARNING: Invalid frame detected, skipping encode")
                        return None

                    # Make a copy to avoid memory issues
                    frame_copy = self.frame.copy()

                    # Ensure proper data type and layout
                    if frame_copy.dtype != 'uint8':
                        frame_copy = frame_copy.astype('uint8')

                    # Ensure contiguous memory layout
                    if not frame_copy.flags['C_CONTIGUOUS']:
                        frame_copy = frame_copy.copy()

                    _, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    return buffer.tobytes()
            return None
        except Exception as e:
            print(f"ERROR in get_frame_jpeg: {e}")
            import traceback
            traceback.print_exc()
            return None

    def set_resolution(self, width, height):
        """Change camera resolution (only works for cameras, not video files)"""
        if self.source_type == "video":
            # Cannot change resolution for video files
            return False

        self.resolution = (width, height)
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return True

    def set_zoom(self, zoom_level):
        """Set zoom level (1.0 to 5.0)"""
        self.zoom = max(1.0, min(5.0, float(zoom_level)))
        return True

    def set_pan(self, pan_x, pan_y):
        """Set pan position (-100 to 100 for both axes)"""
        self.pan_x = max(-100, min(100, int(pan_x)))
        self.pan_y = max(-100, min(100, int(pan_y)))
        return True

    def set_rotation(self, degrees):
        """Set rotation in degrees (0, 90, 180, 270)"""
        valid_rotations = [0, 90, 180, 270]
        if degrees in valid_rotations:
            self.rotation = degrees
            return True
        return False

    def set_target_detection(self, enabled):
        """Enable or disable target detection overlay"""
        self.target_detector.set_detection_enabled(enabled)
        return True

    def set_debug_mode(self, enabled):
        """Enable or disable debug visualization mode"""
        self.target_detector.set_debug_mode(enabled)
        return True

    def set_perspective_correction(self, enabled):
        """Enable or disable perspective correction for main stream"""
        self.perspective_correction_enabled = enabled
        return True

    def pause_playback(self):
        """Pause video playback"""
        if self.source_type == "video":
            self.paused = True
            # Set pause buffer index to current position in buffer
            self.pause_buffer_index = len(self.frame_buffer) - 1 if self.frame_buffer else 0
            # Record the frame number when we paused
            self.pause_frame_number = self.display_frame_number
            return True
        return False

    def resume_playback(self):
        """Resume video playback"""
        if self.source_type == "video":
            self.paused = False
            return True
        return False

    def step_frame_forward(self):
        """Step one frame forward when paused - works throughout entire video"""
        if self.source_type != "video" or not self.paused:
            return False
            
        target_frame = self.display_frame_number + 1
        if target_frame >= self.total_frames:
            return False  # Already at end
            
        try:
            success, message = self.seek_to_frame(target_frame)
            if success:
                print(f"DEBUG: Step forward successful: {message}")
            else:
                print(f"DEBUG: Step forward failed: {message}")
            return success
        except Exception as e:
            print(f"ERROR in step_frame_forward: {e}")
            return False

    def step_frame_backward(self):
        """Step one frame backward when paused - works throughout entire video"""
        if self.source_type != "video" or not self.paused:
            return False
            
        target_frame = self.display_frame_number - 1
        if target_frame < 0:
            return False  # Already at start
            
        try:
            success, message = self.seek_to_frame(target_frame)
            if success:
                print(f"DEBUG: Step backward successful: {message}")
            else:
                print(f"DEBUG: Step backward failed: {message}")
            return success
        except Exception as e:
            print(f"ERROR in step_frame_backward: {e}")
            return False

    def seek_to_frame(self, target_frame):
        """Seek to a specific frame number in video - SAFE VERSION"""
        if self.source_type != "video":
            return False, "Seek only works with video files"
        
        # Thread safety: acquire lock for video operations
        with self.lock:
            if not self.cap or not self.cap.isOpened():
                return False, "Video capture not available"
            
            # Validate frame number
            original_target = target_frame
            if target_frame < 0:
                target_frame = 0
            elif self.total_frames > 0 and target_frame >= self.total_frames:
                target_frame = self.total_frames - 1
            
            # Store current position for potential rollback
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            try:
                print(f"DEBUG: Seeking from frame {current_pos} to frame {target_frame}")
                
                # Attempt to set video position
                success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                if not success:
                    print(f"WARNING: Failed to set frame position to {target_frame}")
                    return False, f"Failed to seek to frame {target_frame}"
                
                # Verify the seek worked
                actual_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                print(f"DEBUG: Actually seeked to frame {actual_frame}")
                
                # Update frame counters
                self.current_frame_number = actual_frame
                self.display_frame_number = actual_frame
                
                # Clear frame buffer since we've jumped to a new position
                self.frame_buffer.clear()
                self.pause_buffer_index = 0
                
                # Read and process the frame at the new position to update the display
                try:
                    ret, new_frame = self.cap.read()
                    if ret and self._is_valid_frame(new_frame):
                        # Apply transformations (rotation, zoom, pan, target detection, bullet holes)
                        processed_frame = self._apply_transformations(new_frame)
                        if self._is_valid_frame(processed_frame):
                            # Update the display frame
                            self.frame = processed_frame.copy()
                            
                            # Add to frame buffer for potential stepping
                            self.frame_buffer.append(processed_frame.copy())
                            
                            print(f"DEBUG: Display frame updated for position {actual_frame}")
                        else:
                            print(f"WARNING: Frame transformation failed at position {actual_frame}")
                            # Still update with original frame
                            self.frame = new_frame.copy()
                    else:
                        print(f"WARNING: Cannot read frame at position {actual_frame}")
                        # Try to restore original position
                        try:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                            return False, f"No valid frame at position {target_frame}"
                        except:
                            return False, f"Seek failed and cannot restore position"
                        
                except Exception as e:
                    print(f"WARNING: Failed to read frame at new position: {e}")
                    # Try to restore original position
                    try:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                        return False, f"Frame read failed after seek: {str(e)}"
                    except:
                        return False, f"Seek failed and cannot restore position"
                
                if actual_frame != original_target:
                    return True, f"Seeked to frame {actual_frame} (requested {original_target})"
                else:
                    return True, f"Seeked to frame {actual_frame}"
                
            except Exception as e:
                print(f"ERROR in seek_to_frame: {e}")
                # Attempt to restore original position
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
                    print(f"DEBUG: Restored to original position {current_pos}")
                except:
                    print(f"ERROR: Cannot restore original position")
                    
                return False, f"Seek failed: {str(e)}"

    def is_paused(self):
        """Check if playback is paused"""
        return self.paused

    def get_playback_info(self):
        """Get current playback information"""
        return {
            'paused': self.paused,
            'current_frame': self.display_frame_number,  # Use display frame number instead
            'total_frames': self.total_frames,
            'buffer_size': len(self.frame_buffer),
            'pause_buffer_index': self.pause_buffer_index if self.paused else None,
            'can_step_backward': self.paused and self.display_frame_number > 0,
            'can_step_forward': self.paused and self.display_frame_number < self.total_frames - 1,
            'supports_playback_controls': self.source_type == "video"
        }

    def get_debug_frame_jpeg(self, debug_type='combined'):
        """Get debug frame as JPEG bytes for specific debug type

        Args:
            debug_type: Type of debug frame ('combined', 'perspective', 'circles', 'corrected')
        """
        # Generate fresh debug frame from current frame if debug mode is on
        with self.lock:
            if self.frame is not None and self.target_detector.debug_mode:
                self.target_detector.generate_debug_frame(self.frame)

                # For corrected debug type, provide the perspective-corrected frame
                if debug_type == 'corrected' and self.perspective_correction_enabled:
                    # Fallback to generic perspective correction
                    corrected_frame = self.perspective.apply_perspective_correction(self.frame)
                    if corrected_frame is not None:
                        # Temporarily store corrected frame for debug access
                        self.target_detector.corrected_debug_frame = corrected_frame

        return self.target_detector.get_debug_frame_jpeg(debug_type)

    def _get_camera_controls_metadata(self):
        """Get current camera controls as metadata dictionary"""
        metadata = {}

        # Add timestamp
        metadata['capture_timestamp'] = datetime.now().isoformat()

        # Add source information
        metadata['source_type'] = self.source_type
        if self.source_type == 'video':
            metadata['video_file'] = self.video_file or 'unknown'
        elif self.source_type == 'camera':
            metadata['camera_index'] = self.camera_index

        # Add camera settings
        metadata['resolution'] = f"{self.resolution[0]}x{self.resolution[1]}"
        metadata['zoom'] = self.zoom
        metadata['rotation'] = self.rotation

        # Add camera controls if available
        if self.cached_camera_controls.get('available'):
            controls = self.cached_camera_controls.get('controls', {})
            for name, info in controls.items():
                current_value = info.get('current')
                if current_value is not None:
                    metadata[f'camera_{name}'] = current_value

        # Add detection settings
        metadata['perspective_correction'] = self.perspective_correction_enabled

        return metadata

    def _embed_video_metadata(self, video_filepath):
        """Embed metadata into video file using FFmpeg"""
        try:
            # Read the metadata JSON file
            base_name = os.path.splitext(video_filepath)[0]
            metadata_filepath = f"{base_name}.json"

            if not os.path.exists(metadata_filepath):
                print(f"Warning: Metadata file not found: {metadata_filepath}")
                return

            with open(metadata_filepath, 'r') as f:
                metadata = json.load(f)

            # Create temporary output file
            temp_filepath = f"{base_name}_temp{os.path.splitext(video_filepath)[1]}"

            # Build FFmpeg command with metadata tags
            cmd = ['ffmpeg', '-i', video_filepath, '-codec', 'copy']

            # Add standard metadata tags
            cmd.extend(['-metadata', f'title=Raspberry Pi Target Camera Recording'])

            # Add all metadata values as individual tags
            # FFmpeg metadata keys should not have spaces or special characters
            for key, value in metadata.items():
                # Convert the key to a valid metadata tag name
                # Replace underscores and make it more readable
                if key == 'capture_timestamp':
                    cmd.extend(['-metadata', f'date={value}'])
                    cmd.extend(['-metadata', f'{key}={value}'])
                elif key.startswith('camera_'):
                    # Add both with and without 'camera_' prefix
                    clean_key = key.replace('camera_', '')
                    cmd.extend(['-metadata', f'{clean_key}={value}'])
                    cmd.extend(['-metadata', f'{key}={value}'])
                elif key == 'video_file':
                    cmd.extend(['-metadata', f'source_file={value}'])
                    cmd.extend(['-metadata', f'{key}={value}'])
                elif key == 'camera_index':
                    cmd.extend(['-metadata', f'camera={value}'])
                    cmd.extend(['-metadata', f'{key}={value}'])
                else:
                    # Add all other metadata as-is
                    cmd.extend(['-metadata', f'{key}={value}'])

            # Add complete metadata as comment field (JSON string) for programmatic access
            metadata_json = json.dumps(metadata).replace('"', '\\"')  # Escape quotes
            cmd.extend(['-metadata', f'comment={metadata_json}'])

            # Output to temp file
            cmd.extend(['-y', temp_filepath])  # -y to overwrite without asking

            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Replace original file with metadata-embedded version
                os.replace(temp_filepath, video_filepath)
                print(f"Successfully embedded metadata into {os.path.basename(video_filepath)}")

                # Delete the JSON sidecar file since metadata is now embedded
                if os.path.exists(metadata_filepath):
                    os.remove(metadata_filepath)
                    print(f"Removed JSON sidecar file: {os.path.basename(metadata_filepath)}")
            else:
                print(f"Warning: FFmpeg failed to embed metadata: {result.stderr}")
                # Clean up temp file if it exists
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)

        except Exception as e:
            print(f"Error embedding video metadata: {e}")

    def capture_image(self):
        """Capture and save current frame with camera controls metadata as EXIF tags"""
        with self.lock:
            if self.frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                filepath = os.path.join(self.captures_dir, filename)

                # Get metadata
                metadata = self._get_camera_controls_metadata()

                # First save with OpenCV
                cv2.imwrite(filepath, self.frame)

                # Load image with PIL to add EXIF data
                img = Image.open(filepath)

                # Create EXIF data
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

                # Add metadata to UserComment (can store JSON)
                metadata_json = json.dumps(metadata)
                exif_dict["Exif"][piexif.ExifIFD.UserComment] = metadata_json.encode('utf-8')

                # Add timestamp to DateTimeOriginal
                dt = datetime.now()
                datetime_str = dt.strftime("%Y:%m:%d %H:%M:%S")
                exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = datetime_str.encode('utf-8')
                exif_dict["0th"][piexif.ImageIFD.DateTime] = datetime_str.encode('utf-8')

                # Add camera make/model from metadata
                exif_dict["0th"][piexif.ImageIFD.Make] = b"Raspberry Pi"
                exif_dict["0th"][piexif.ImageIFD.Model] = f"Target Camera (Source: {self.source_type})".encode('utf-8')

                # Add image description with key settings
                description = f"Resolution: {metadata.get('resolution')}, Zoom: {metadata.get('zoom')}, Rotation: {metadata.get('rotation')}"
                exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description.encode('utf-8')

                # Encode and save
                exif_bytes = piexif.dump(exif_dict)
                img.save(filepath, "jpeg", exif=exif_bytes, quality=95)

                return filename
        return None

    def set_reference_frame(self):
        """Set current frame as reference for bullet hole detection"""
        with self.lock:
            if self.frame is not None:
                self.reference_frame = self.frame.copy()
                self.bullet_holes = []  # Clear previous detections
                return True
        return False
    
    def detect_bullet_holes(self):
        """Detect bullet holes by comparing current frame with reference"""
        if self.reference_frame is None:
            return False, "No reference frame set"
            
        with self.lock:
            if self.frame is not None:
                # Detect bullet holes
                holes = self.bullet_hole_detector.detect_bullet_holes(
                    self.reference_frame, self.frame)
                
                # Store detected holes and cache in detector for overlay
                self.bullet_holes = holes
                self.bullet_hole_detector.last_detection = holes
                return True, f"Found {len(holes)} bullet hole(s)"
        
        return False, "No current frame available"
    
    def clear_bullet_holes(self):
        """Clear all detected bullet holes"""
        self.bullet_holes = []
        self.bullet_hole_detector.last_detection = []
        return True
    
    def get_bullet_hole_debug_frame(self, frame_type='combined'):
        """Get bullet hole detection debug frame"""
        return self.bullet_hole_detector.get_debug_frame(frame_type)

    def start_recording(self):
        """Start recording video to file"""
        if self.recording:
            return False, "Already recording"

        if self.source_type == "test":
            return False, "Cannot record test pattern"

        if not self.cap or not self.cap.isOpened():
            return False, "Camera/video source not available"

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_filename = f"recording_{timestamp}.mp4"
        filepath = os.path.join(self.recordings_dir, self.recording_filename)

        # Get dimensions directly from the capture device (before any rotation/processing)
        # This ensures we record the raw frame dimensions
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            return False, f"Invalid frame dimensions: {width}x{height}"

        # Store recording dimensions for validation
        self.recording_width = width
        self.recording_height = height

        # Determine recording FPS based on source type
        if self.source_type == "video":
            # Use video file's native FPS
            fps = self.video_fps
        elif self.cap and self.cap.isOpened():
            # Try to get camera's actual FPS
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            fps = actual_fps if actual_fps > 0 else 30.0
        else:
            # Default fallback
            fps = 30.0

        # Clamp FPS to reasonable values
        fps = max(1.0, min(120.0, fps))

        # Try multiple codec options in order of preference
        # Use MKV for MJPEG (better support), MP4 for H.264
        codec_options = [
            ('MJPG', '.mkv'),   # Motion JPEG in MKV (very low compression, stable, good metadata)
            ('X264', '.mp4'),   # H.264 in MP4 (good quality, widely compatible)
            ('avc1', '.mp4'),   # H.264 variant (alternative)
            ('mp4v', '.mp4'),   # MPEG-4 (fallback)
        ]

        self.video_writer = None
        for codec, ext in codec_options:
            try:
                # Update filename extension to match codec
                test_filename = f"recording_{timestamp}{ext}"
                test_filepath = os.path.join(self.recordings_dir, test_filename)

                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(test_filepath, fourcc, fps, (width, height))

                if writer.isOpened():
                    self.video_writer = writer
                    self.recording_filename = test_filename
                    filepath = test_filepath
                    print(f"Successfully initialized VideoWriter with codec: {codec}")
                    break
                else:
                    writer.release()
                    # Remove failed file if created
                    if os.path.exists(test_filepath):
                        os.remove(test_filepath)

            except Exception as e:
                print(f"Failed to initialize codec {codec}: {e}")
                continue

        if self.video_writer is None or not self.video_writer.isOpened():
            return False, "Failed to initialize video writer with any available codec"

        self.recording = True
        self.recording_start_time = datetime.now()

        # Save metadata for recording
        metadata = self._get_camera_controls_metadata()
        metadata['recording_fps'] = fps
        metadata['recording_resolution'] = f"{width}x{height}"
        metadata['recording_codec'] = codec

        # Save metadata as JSON sidecar file
        metadata_filename = f"recording_{timestamp}.json"
        metadata_filepath = os.path.join(self.recordings_dir, metadata_filename)
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Started recording to {filepath} at {width}x{height} @ {fps} fps")
        return True, f"Recording started: {self.recording_filename} @ {fps:.1f} fps"

    def stop_recording(self):
        """Stop recording video and embed metadata"""
        if not self.recording:
            return False, "Not currently recording"

        self.recording = False

        # Release video writer
        if self.video_writer is not None:
            try:
                self.video_writer.release()
                self.video_writer = None

                # Calculate recording duration
                if self.recording_start_time:
                    duration = (datetime.now() - self.recording_start_time).total_seconds()
                    filepath = os.path.join(self.recordings_dir, self.recording_filename)
                    filesize_mb = os.path.getsize(filepath) / (1024 * 1024)
                    message = f"Recording saved: {self.recording_filename} ({duration:.1f}s, {filesize_mb:.1f}MB)"
                else:
                    message = f"Recording saved: {self.recording_filename}"

                # Embed metadata into video file using FFmpeg
                self._embed_video_metadata(filepath)

                print(message)
                return True, message

            except Exception as e:
                return False, f"Error stopping recording: {str(e)}"

        return True, "Recording stopped"

    def get_recording_status(self):
        """Get current recording status"""
        if not self.recording:
            return {
                'recording': False,
                'filename': None,
                'duration': 0
            }

        duration = 0
        if self.recording_start_time:
            duration = (datetime.now() - self.recording_start_time).total_seconds()

        return {
            'recording': True,
            'filename': self.recording_filename,
            'duration': duration
        }

    def get_camera_controls(self):
        """Get available camera controls from cached data"""
        return {
            'available': self.cached_camera_controls.get('available', False),
            'controls': self.cached_camera_controls.get('controls', {})
        }

    def set_camera_control(self, name, value):
        """Set a camera control value using the existing camera instance"""
        if not self.cached_camera_controls.get('available'):
            return False, "Camera controls not available"

        if name not in self.cached_camera_controls.get('controls', {}):
            return False, f"Control '{name}' not available"

        if not self.cap or not self.cap.isOpened():
            return False, "Camera not available for control changes"

        try:
            # Map control names to OpenCV property IDs
            control_mapping = {
                'brightness': cv2.CAP_PROP_BRIGHTNESS,
                'contrast': cv2.CAP_PROP_CONTRAST,
                'saturation': cv2.CAP_PROP_SATURATION,
                'hue': cv2.CAP_PROP_HUE,
                'gain': cv2.CAP_PROP_GAIN,
                'gamma': cv2.CAP_PROP_GAMMA,
                'exposure': cv2.CAP_PROP_EXPOSURE,
                'auto_exposure': cv2.CAP_PROP_AUTO_EXPOSURE,
                'auto_wb': cv2.CAP_PROP_AUTO_WB,
                'backlight': cv2.CAP_PROP_BACKLIGHT,
                'sharpness': cv2.CAP_PROP_SHARPNESS,
                'temperature': cv2.CAP_PROP_TEMPERATURE,
                'wb_temperature': cv2.CAP_PROP_WB_TEMPERATURE,
            }

            if name not in control_mapping:
                return False, f"Control '{name}' not supported for direct setting"

            # Validate value against cached range
            control_info = self.cached_camera_controls['controls'][name]
            min_val = control_info.get('min')
            max_val = control_info.get('max')

            if min_val is not None and max_val is not None:
                if value < min_val or value > max_val:
                    return False, f"Value {value} out of range for {name} (valid: {min_val} to {max_val})"

            # Set the control using the existing camera instance
            prop_id = control_mapping[name]
            success = self.cap.set(prop_id, float(value))

            if success:
                # Verify the value was set correctly
                actual_value = self.cap.get(prop_id)

                # Update cached value
                self.cached_camera_controls['controls'][name]['current'] = actual_value

                return True, f"Set {name} to {actual_value}"
            else:
                return False, f"Failed to set {name} (OpenCV returned false)"

        except Exception as e:
            return False, f"Error setting {name}: {str(e)}"

    def get_camera_control(self, name):
        """Get current value of a camera control from cached data"""
        if not self.cached_camera_controls.get('available'):
            return None

        if name not in self.cached_camera_controls.get('controls', {}):
            return None

        return self.cached_camera_controls['controls'][name].get('current')

    def reset_camera_controls(self):
        """Reset all camera controls to defaults using existing camera instance"""
        if not self.cached_camera_controls.get('available'):
            return False, "Camera controls not available"

        if not self.cap or not self.cap.isOpened():
            return False, "Camera not available for control reset"

        try:
            success_count = 0
            total_count = 0

            # Reset each control that has a default value
            for name, control_info in self.cached_camera_controls['controls'].items():
                default_val = control_info.get('default')
                if default_val is not None:
                    total_count += 1
                    success, message = self.set_camera_control(name, default_val)
                    if success:
                        success_count += 1

            return True, f"Reset {success_count}/{total_count} controls to defaults"

        except Exception as e:
            return False, f"Error resetting controls: {str(e)}"

    def set_camera_resolution(self, width, height, fps, format_name):
        """Set camera resolution, frame rate, and format using existing camera instance"""
        if not self.cap or not self.cap.isOpened():
            return False, "Camera not available for resolution change"

        try:
            print(f"DEBUG: Setting camera resolution to {width}x{height} @ {fps}fps ({format_name})")

            # Map format name to OpenCV fourcc code
            format_mapping = {
                'MJPG': cv2.VideoWriter_fourcc(*'MJPG'),
                'YUYV': cv2.VideoWriter_fourcc(*'YUYV'),
                'YUY2': cv2.VideoWriter_fourcc(*'YUY2'),
            }

            fourcc_code = format_mapping.get(format_name.upper())
            if fourcc_code is None:
                return False, f"Unsupported format: {format_name}"

            # Set the format first
            format_success = self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
            if not format_success:
                print(f"DEBUG: Warning - could not set format to {format_name}")

            # Set resolution
            width_success = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            height_success = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Set frame rate
            fps_success = self.cap.set(cv2.CAP_PROP_FPS, fps)

            # Verify the settings took effect
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            print(f"DEBUG: Resolution set result - Width: {width_success}, Height: {height_success}, FPS: {fps_success}")
            print(f"DEBUG: Actual values - {actual_width}x{actual_height} @ {actual_fps}fps")

            if actual_width == width and actual_height == height:
                message = f"Resolution changed to {actual_width}x{actual_height} @ {actual_fps:.1f}fps ({format_name})"
                if actual_fps != fps:
                    message += f" (requested {fps}fps)"
                return True, message
            else:
                return False, f"Resolution change failed - got {actual_width}x{actual_height}, expected {width}x{height}"

        except Exception as e:
            return False, f"Error setting resolution: {str(e)}"

    def save_camera_preset(self, preset_name):
        """Save current camera control settings as a preset"""
        if not self.cached_camera_controls.get('available'):
            return False, "Camera controls not available"

        try:
            # Create preset from cached current values
            preset = {}
            for name, control_info in self.cached_camera_controls['controls'].items():
                preset[name] = control_info.get('current')

            # Save to file for persistence
            preset_file = f"./presets/{preset_name}.json"
            os.makedirs("./presets", exist_ok=True)
            with open(preset_file, 'w') as f:
                json.dump(preset, f, indent=2)
            return True, f"Saved preset '{preset_name}' with {len(preset)} controls"
        except Exception as e:
            return False, f"Error saving preset: {str(e)}"

    def load_camera_preset(self, preset_name):
        """Load camera control settings from a preset using existing camera instance"""
        if not self.cached_camera_controls.get('available'):
            return False, "Camera controls not available"

        if not self.cap or not self.cap.isOpened():
            return False, "Camera not available for preset loading"

        try:
            preset_file = f"./presets/{preset_name}.json"
            if not os.path.exists(preset_file):
                return False, f"Preset '{preset_name}' not found"

            with open(preset_file, 'r') as f:
                preset = json.load(f)

            # Apply preset using existing camera instance
            success_count = 0
            total_count = 0

            for name, value in preset.items():
                if name in self.cached_camera_controls['controls']:
                    total_count += 1
                    success, message = self.set_camera_control(name, value)
                    if success:
                        success_count += 1

            return True, f"Loaded preset '{preset_name}': {success_count}/{total_count} controls applied"

        except Exception as e:
            return False, f"Error loading preset: {str(e)}"

    def list_camera_presets(self):
        """List available camera presets"""
        try:
            presets_dir = "./presets"
            if not os.path.exists(presets_dir):
                return []

            presets = []
            for filename in os.listdir(presets_dir):
                if filename.endswith('.json'):
                    preset_name = filename[:-5]  # Remove .json extension
                    presets.append(preset_name)
            return presets
        except Exception as e:
            print(f"Error listing presets: {e}")
            return []

    def get_camera_formats(self):
        """Get available formats and resolutions for the current camera"""
        if self.source_type != 'camera':
            return {'available': False, 'formats': []}

        # Find the current camera in available sources
        camera_source = None
        for camera in self.available_sources.get('cameras', []):
            if camera['index'] == self.camera_index:
                camera_source = camera
                break

        if not camera_source or 'controls' not in camera_source:
            return {'available': False, 'formats': []}

        formats_data = camera_source['controls'].get('formats', [])

        # Convert to UI-friendly format
        resolution_options = []

        for format_info in formats_data:
            format_name = format_info.get('name', 'Unknown')
            format_desc = format_info.get('description', '')

            for res_info in format_info.get('resolutions', []):
                size = res_info.get('size', '')
                framerates = res_info.get('framerates', [])

                if size and framerates:
                    # Create options for each framerate
                    for fps in sorted(framerates, reverse=True):  # Sort highest fps first
                        option = {
                            'value': f"{size}@{fps:.0f}fps_{format_name}",
                            'label': f"{size} @ {fps:.0f}fps ({format_name})",
                            'width': int(size.split('x')[0]) if 'x' in size else 0,
                            'height': int(size.split('x')[1]) if 'x' in size else 0,
                            'fps': fps,
                            'format': format_name,
                            'format_desc': format_desc
                        }
                        resolution_options.append(option)

        # Sort by resolution (width * height) and then by fps
        resolution_options.sort(key=lambda x: (x['width'] * x['height'], -x['fps']))

        return {
            'available': True,
            'formats': formats_data,
            'resolution_options': resolution_options
        }

    def calibrate_perspective(self):
        """Perform perspective calibration using current frame"""
        try:
            with self.lock:
                if self.frame is not None:
                    # Validate frame before calibration
                    if not self._is_valid_frame(self.frame):
                        return False, "Current frame is invalid for calibration"

                    print("DEBUG: Starting perspective calibration...")
                    print(f"DEBUG: Frame shape: {self.frame.shape}, dtype: {self.frame.dtype}")

                    # Make a copy to ensure memory safety
                    frame_copy = self.frame.copy()

                    # Ensure proper data type for calibration
                    if frame_copy.dtype != 'uint8':
                        frame_copy = frame_copy.astype('uint8')

                    # Use perspective.py directly for calibration - it handles all matrix storage internally
                    success, message = self.perspective.calibrate_perspective(frame_copy)
                    print(f"DEBUG: Calibration result: {success}, {message}")

                    # Debug frame handling can be added later if needed

                    return success, message
                else:
                    return False, "No frame available for calibration"
        except Exception as e:
            print(f"ERROR in calibrate_perspective: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Calibration failed with error: {str(e)}"


    def _detect_available_sources(self):
        """Detect available cameras and video files using V4L2"""
        print("DEBUG: Detecting available sources...")

        # Always add test pattern as the first available source
        self.available_sources['test'] = [{
            'id': 'test_pattern',
            'name': 'Test Pattern (Default)'
        }]

        # Use V4L2 to detect cameras properly
        camera_count = 0
        detected_cameras = self._detect_v4l2_cameras()

        for camera_info in detected_cameras:
            # Detect and cache camera controls during enumeration
            controls_info = self._detect_camera_controls(camera_info['index'])

            self.available_sources['cameras'].append({
                'id': f'camera_{camera_info["index"]}',
                'name': camera_info['name'],
                'index': camera_info['index'],
                'device_path': camera_info['device_path'],
                'driver': camera_info.get('driver', 'unknown'),
                'controls': controls_info
            })
            camera_count += 1

        # Check for video files in samples directory
        video_count = 0
        if os.path.exists(self.samples_dir):
            try:
                for filename in os.listdir(self.samples_dir):
                    if filename.lower().endswith(('.avi', '.mp4', '.mov', '.mkv', '.wmv')):
                        filepath = os.path.join(self.samples_dir, filename)
                        self.available_sources['videos'].append({
                            'id': f'video_{filename}',
                            'name': filename,
                            'path': filepath
                        })
                        video_count += 1
            except Exception as e:
                print(f"DEBUG: Error scanning video directory: {e}")

        print(f"DEBUG: Found test pattern, {camera_count} cameras and {video_count} video files")

    def _detect_v4l2_cameras(self):
        """Detect cameras using V4L2 enumeration"""
        cameras = []

        try:
            # Method 1: Use v4l2-ctl to list devices
            result = subprocess.run(['v4l2-ctl', '--list-devices'],
                                  capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                cameras.extend(self._parse_v4l2_devices(result.stdout))
            else:
                print(f"DEBUG: v4l2-ctl failed, using fallback detection")
                cameras.extend(self._detect_cameras_fallback())

        except Exception as e:
            print(f"DEBUG: V4L2 detection failed ({e}), using fallback")
            cameras.extend(self._detect_cameras_fallback())

        return cameras

    def _parse_v4l2_devices(self, v4l2_output):
        """Parse v4l2-ctl --list-devices output"""
        cameras = []
        current_device = None

        for line in v4l2_output.split('\n'):
            original_line = line
            line = line.strip()
            if not line:
                continue

            if original_line.startswith('\t/dev/video'):
                # This is a device path (indented with tab)
                if current_device:
                    device_path = line
                    # Extract index from /dev/videoX
                    try:
                        index = int(device_path.split('video')[1])
                        cameras.append({
                            'index': index,
                            'name': current_device['name'],
                            'device_path': device_path,
                            'driver': current_device.get('driver', 'unknown')
                        })
                    except (ValueError, IndexError):
                        continue
            elif not original_line.startswith('\t') and ':' in line:
                # This is a device description line (not indented)
                # Format: "device_name (driver_info):"
                device_line = line.rstrip(':')

                if '(' in device_line and ')' in device_line:
                    # Extract name and driver
                    name_part = device_line.split('(')[0].strip()
                    driver_part = device_line.split('(')[1].split(')')[0].strip()
                else:
                    # No driver info in parentheses
                    name_part = device_line
                    driver_part = 'unknown'

                current_device = {
                    'name': name_part,
                    'driver': driver_part
                }

        # Filter to only include actual camera devices (not platform devices)
        camera_devices = []
        for cam in cameras:
            # Skip platform devices, codec devices, etc.
            if any(skip in cam['name'].lower() for skip in ['pisp', 'hevc', 'codec', 'platform']):
                continue
            # Include USB cameras and similar, or devices with "camera" in name
            if (any(include in cam['name'].lower() for include in ['camera', 'usb', 'webcam']) or
                'usb-' in cam['driver'] or 'camera' in cam['driver'].lower()):
                camera_devices.append(cam)

        return camera_devices

    def _detect_cameras_fallback(self):
        """Fallback camera detection using direct device enumeration"""
        cameras = []

        # Check /dev/video* devices directly
        for i in range(10):  # Check first 10 video devices
            device_path = f"/dev/video{i}"
            if os.path.exists(device_path):
                try:
                    # Try to get device info without opening camera
                    result = subprocess.run(['v4l2-ctl', f'--device={device_path}', '--info'],
                                          capture_output=True, text=True, timeout=2)

                    if result.returncode == 0:
                        # Parse device info
                        device_name = f"Camera {i}"
                        driver = "unknown"

                        for line in result.stdout.split('\n'):
                            if 'Card type' in line:
                                device_name = line.split(':')[1].strip()
                            elif 'Driver name' in line:
                                driver = line.split(':')[1].strip()

                        # Skip non-camera devices
                        if any(skip in device_name.lower() for skip in ['pisp', 'hevc', 'codec']):
                            continue

                        cameras.append({
                            'index': i,
                            'name': device_name,
                            'device_path': device_path,
                            'driver': driver
                        })

                except Exception as e:
                    print(f"DEBUG: Error checking {device_path}: {e}")
                    continue

        return cameras

    def _detect_camera_controls(self, camera_index):
        """Detect camera controls and capabilities for a specific camera"""
        controls_info = {
            'available': False,
            'controls': {},
            'formats': [],
            'error': None
        }

        try:
            print(f"DEBUG: Detecting controls for camera {camera_index}...")

            # Use a temporary camera instance to detect controls
            temp_controls = CameraControlManager(camera_index, auto_detect=False)

            # Try to open camera briefly for control detection
            temp_controls.open()
            if temp_controls.cap and temp_controls.cap.isOpened():
                # Detect controls
                temp_controls.detect_controls()
                available_controls = temp_controls.list_controls()

                if len(available_controls) > 0:
                    controls_info['available'] = True

                    # Cache control information
                    for control_name in available_controls:
                        control_data = temp_controls.get_control_info(control_name)
                        current_value = temp_controls.get(control_name)

                        controls_info['controls'][control_name] = {
                            'current': current_value,
                            'min': control_data.min_value,
                            'max': control_data.max_value,
                            'default': control_data.default_value,
                            'type': control_data.control_type
                        }

                    print(f"DEBUG: Cached {len(available_controls)} controls for camera {camera_index}")
                else:
                    print(f"DEBUG: No controls found for camera {camera_index}")

            # Detect available formats using v4l2-ctl
            try:
                device_path = f"/dev/video{camera_index}"
                result = subprocess.run(['v4l2-ctl', f'--device={device_path}', '--list-formats-ext'],
                                      capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    formats = self._parse_v4l2_formats(result.stdout)
                    controls_info['formats'] = formats
                    print(f"DEBUG: Found {len(formats)} formats for camera {camera_index}")

            except Exception as e:
                print(f"DEBUG: Could not detect formats for camera {camera_index}: {e}")

            # Close temporary camera
            temp_controls.close()

        except Exception as e:
            controls_info['error'] = str(e)
            print(f"DEBUG: Error detecting controls for camera {camera_index}: {e}")

        return controls_info

    def _parse_v4l2_formats(self, format_output):
        """Parse v4l2-ctl --list-formats-ext output with frame rates"""
        formats = []
        current_format = None
        current_resolution = None

        for line in format_output.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Format line: [0]: 'MJPG' (Motion-JPEG, compressed)
            if line.startswith('[') and ']:' in line:
                # Save previous format
                if current_format and current_format.get('resolutions'):
                    formats.append(current_format)

                if 'MJPG' in line or 'YUYV' in line:
                    format_name = line.split("'")[1] if "'" in line else 'unknown'
                    format_desc = line.split('(')[1].split(')')[0] if '(' in line else 'unknown'
                    current_format = {
                        'name': format_name,
                        'description': format_desc,
                        'resolutions': []
                    }
                else:
                    current_format = None

            # Resolution line: Size: Discrete 640x480
            elif line.startswith('Size: Discrete') and current_format:
                resolution = line.split('Discrete ')[1] if 'Discrete ' in line else ''
                if 'x' in resolution:
                    current_resolution = {
                        'size': resolution,
                        'framerates': []
                    }
                    current_format['resolutions'].append(current_resolution)

            # Frame rate line: Interval: Discrete 0.033s (30.000 fps)
            elif line.startswith('Interval: Discrete') and current_resolution:
                if 'fps)' in line:
                    # Extract fps from line like "Interval: Discrete 0.033s (30.000 fps)"
                    fps_part = line.split('(')[1].split(' fps')[0] if '(' in line and ' fps' in line else ''
                    try:
                        fps = float(fps_part)
                        if fps not in current_resolution['framerates']:
                            current_resolution['framerates'].append(fps)
                    except ValueError:
                        continue

        # Add the last format if it exists
        if current_format and current_format.get('resolutions'):
            formats.append(current_format)

        return formats

    def get_available_sources(self):
        """Get list of available cameras and video files (cached)"""
        return self.available_sources

    def set_video_source(self, source_type, source_id):
        """Change video source (camera or video file) - SAFE VERSION"""
        try:
            print(f"DEBUG: Requesting source change to {source_type}: {source_id}")
            
            # For camera switching, use a safer approach that avoids OpenCV crashes
            if source_type == "camera":
                camera_index = int(source_id.split('_')[1])
                
                # Test the camera first before switching
                print(f"DEBUG: Testing camera {camera_index}...")
                test_cap = None
                try:
                    test_cap = cv2.VideoCapture(camera_index)
                    if not test_cap.isOpened():
                        if test_cap:
                            test_cap.release()
                        return False, f"Camera {camera_index} is not available"
                    
                    # Try to read a frame to ensure it works
                    ret, frame = test_cap.read()
                    if not ret or frame is None:
                        test_cap.release()
                        return False, f"Camera {camera_index} cannot capture frames"
                    
                    test_cap.release()
                    print(f"DEBUG: Camera {camera_index} test successful")
                    
                except Exception as e:
                    if test_cap:
                        try:
                            test_cap.release()
                        except:
                            pass
                    print(f"ERROR: Camera {camera_index} test failed: {e}")
                    return False, f"Camera {camera_index} test failed: {str(e)}"
            
            # Stop current capture
            print("DEBUG: Stopping current capture...")
            self.stop()
            
            # Wait longer for everything to settle
            time.sleep(1.0)
            
            # Reset all state
            self.paused = False
            self.step_frame = False
            self.current_frame_number = 0
            self.display_frame_number = 0
            self.total_frames = 0
            
            # Configure new source
            if source_type == "camera":
                camera_index = int(source_id.split('_')[1])
                self.camera_index = camera_index
                self.source_type = "camera"
                self.video_file = None
                self.native_video_resolution = None
                self.native_video_fps = None
                print(f"DEBUG: Configured for camera {camera_index}")
                
            elif source_type == "video":
                filename = source_id.replace('video_', '', 1)
                video_path = os.path.join(self.samples_dir, filename)
                if not os.path.exists(video_path):
                    return False, f"Video file not found: {filename}"

                self.source_type = "video"
                self.video_file = video_path
                print(f"DEBUG: Configured for video file {video_path}")

            elif source_type == "test":
                self.source_type = "test"
                self.video_file = None
                self.native_video_resolution = None
                self.native_video_fps = None
                print("DEBUG: Configured for test pattern")

            else:
                return False, f"Invalid source type: {source_type}"
            
            # Start new capture
            print("DEBUG: Starting new capture...")
            success = self.start_capture()
            
            if success:
                print("DEBUG: Source change successful")
                return True, f"Successfully changed to {source_type}: {source_id}"
            else:
                print("DEBUG: Source change failed - reverting to test mode")
                # Fallback to test mode
                self.source_type = "test"
                self.video_file = None
                self.start_capture()
                return False, f"Failed to start {source_type}, reverted to test pattern"
                
        except Exception as e:
            print(f"CRITICAL ERROR in set_video_source: {e}")
            import traceback
            traceback.print_exc()
            
            # Emergency fallback to test mode
            try:
                self.source_type = "test"
                self.video_file = None
                self.start_capture()
            except:
                pass
                
            return False, f"Critical error during source change: {str(e)}"

    def get_status(self):
        """Get current camera status"""
        # Get actual current FPS and resolution from the capture device
        current_fps = self.video_fps if self.source_type == "video" else 30.0  # Default camera FPS
        current_resolution = self.resolution

        # Try to get actual values from capture device if available
        if self.cap and self.cap.isOpened():
            try:
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

                if actual_width > 0 and actual_height > 0:
                    current_resolution = (actual_width, actual_height)
                if actual_fps > 0:
                    current_fps = actual_fps
            except:
                pass  # Use defaults if unable to get actual values

        # Handle current source display based on type
        if self.source_type == "camera":
            current_source = self.camera_index
        elif self.source_type == "video":
            current_source = self.video_file
        else:  # test mode
            current_source = "Test Pattern"
        
        status = {
            'resolution': self.resolution,
            'actual_resolution': current_resolution,
            'actual_fps': current_fps,
            'zoom': self.zoom,
            'pan_x': self.pan_x,
            'pan_y': self.pan_y,
            'rotation': self.rotation,
            'running': self.running,
            'captures_dir': self.captures_dir,
            'source_type': self.source_type,
            'current_source': current_source,
            'perspective_correction_enabled': self.perspective_correction_enabled,
            'perspective_correction_method': 'ellipse-to-circle' if self.perspective.saved_ellipse_data else 'matrix-based',
            'is_video_mode': self.source_type == "video",
            'native_video_resolution': self.native_video_resolution,
            'native_video_fps': self.native_video_fps
        }

        # Add target detection status
        target_status = self.target_detector.get_detection_status()
        status.update(target_status)

        # Add playback control status
        playback_status = self.get_playback_info()
        status.update(playback_status)

        # Add camera controls status
        camera_controls_status = self.get_camera_controls()
        status['camera_controls'] = camera_controls_status

        return status

    def stop(self):
        """Stop camera capture"""
        self.running = False

        # Stop any ongoing recording
        if self.recording:
            try:
                self.stop_recording()
            except Exception as e:
                print(f"WARNING: Error stopping recording: {e}")

        # Simple wait for capture loop to notice running=False
        time.sleep(0.5)

        # Release capture device
        if self.cap:
            try:
                self.cap.release()
                self.cap = None
            except Exception as e:
                print(f"WARNING: Error releasing capture device: {e}")

        # Clear frame buffer
        try:
            self.frame_buffer.clear()
        except:
            pass

        # Close camera controls
        if self.camera_controls:
            try:
                self.camera_controls.close()
                self.camera_controls = None
            except Exception as e:
                print(f"WARNING: Error closing camera controls: {e}")


class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP request handler for both streaming and web interface"""

    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path

            if path == '/stream.mjpg':
                self._serve_mjpeg_stream()
            elif path == '/debug.mjpg':
                # Parse debug type parameter
                query_params = parse_qs(parsed_path.query)
                debug_type = query_params.get('type', ['combined'])[0]  # Default to 'combined'
                self._serve_debug_stream(debug_type)
            elif path == '/' or path == '/index.html':
                self._serve_file('camera_interface.html')
            elif path == '/api/status':
                self._serve_api_status()
            elif path == '/api/sources':
                self._serve_api_sources()
            elif path == '/api/camera_controls':
                controls_info = camera_controller.get_camera_controls()
                self._send_json_response({'success': True, 'data': controls_info})
            elif path == '/api/camera_formats':
                formats_info = camera_controller.get_camera_formats()
                self._send_json_response({'success': True, 'data': formats_info})
            elif path == '/api/start_recording':
                success, message = camera_controller.start_recording()
                self._send_json_response({'success': success, 'message': message})
            elif path == '/api/stop_recording':
                success, message = camera_controller.stop_recording()
                self._send_json_response({'success': success, 'message': message})
            elif path == '/api/recording_status':
                recording_status = camera_controller.get_recording_status()
                self._send_json_response({'success': True, 'data': recording_status})
            elif path == '/api/list_camera_presets':
                presets = camera_controller.list_camera_presets()
                self._send_json_response({'success': True, 'presets': presets})
            elif path.endswith('.html') or path.endswith('.css') or path.endswith('.js'):
                self._serve_file(path[1:])  # Remove leading slash
            else:
                self.send_error(404)
        except Exception as e:
            print(f"ERROR in do_GET({self.path}): {e}")
            import traceback
            traceback.print_exc()
            try:
                self.send_error(500)
            except:
                pass  # Connection might already be closed

    def do_POST(self):
        """Handle POST requests for API calls"""
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path

            if path.startswith('/api/'):
                self._handle_api_request(path)
            else:
                self.send_error(404)
        except Exception as e:
            print(f"ERROR in do_POST({self.path}): {e}")
            import traceback
            traceback.print_exc()
            try:
                self.send_error(500)
            except:
                pass  # Connection might already be closed

    def _serve_mjpeg_stream(self):
        """Serve MJPEG video stream"""
        global camera_controller
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            frame_count = 0
            while True:
                try:
                    frame_data = camera_controller.get_frame_jpeg()
                    if frame_data:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame_data)))
                        self.end_headers()
                        self.wfile.write(frame_data)
                        self.wfile.write(b'\r\n')

                        frame_count += 1
                        if frame_count % 100 == 0:  # Log every 100 frames
                            print(f"DEBUG: Served {frame_count} stream frames")
                    time.sleep(0.03)
                except Exception as e:
                    print(f"ERROR in stream frame delivery: {e}")
                    break  # Exit the streaming loop on error
        except Exception as e:
            print(f"ERROR in _serve_mjpeg_stream: {e}")
            import traceback
            traceback.print_exc()

    def _serve_debug_stream(self, debug_type='combined'):
        """Serve MJPEG debug stream for specific debug type

        Args:
            debug_type: Type of debug stream ('combined', 'perspective', 'circles', 'corrected')
        """
        global camera_controller
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            while True:
                debug_data = camera_controller.get_debug_frame_jpeg(debug_type)
                if debug_data and len(debug_data) > 100:  # Valid JPEG should be larger
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(debug_data)))
                    self.end_headers()
                    self.wfile.write(debug_data)
                    self.wfile.write(b'\r\n')
                else:
                    # Create a simple "Debug Mode Disabled" image
                    import cv2
                    import numpy as np

                    # Create a simple status image
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    img.fill(50)  # Dark gray background

                    # Display debug type and status
                    cv2.putText(img, f"DEBUG MODE: {debug_type.upper()}", (120, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.putText(img, "Enable debug mode in the web interface", (80, 280),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img, f"Stream: /debug.mjpg?type={debug_type}", (120, 320),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 255), 2)

                    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    placeholder = buffer.tobytes()

                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(placeholder)))
                    self.end_headers()
                    self.wfile.write(placeholder)
                    self.wfile.write(b'\r\n')

                time.sleep(0.2)  # 5 FPS for debug stream
        except Exception as e:
            print(f"Debug stream client disconnected: {e}")

    def _serve_file(self, filename):
        """Serve static files"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    content = f.read()

                # Determine content type
                if filename.endswith('.html'):
                    content_type = 'text/html'
                elif filename.endswith('.css'):
                    content_type = 'text/css'
                elif filename.endswith('.js'):
                    content_type = 'application/javascript'
                else:
                    content_type = 'text/plain'

                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404)
        except Exception as e:
            print(f"Error serving file {filename}: {e}")
            self.send_error(500)

    def _serve_api_status(self):
        """Serve camera status as JSON"""
        global camera_controller
        status = camera_controller.get_status()
        self._send_json_response(status)

    def _serve_api_sources(self):
        """Serve available sources as JSON"""
        global camera_controller
        sources = camera_controller.get_available_sources()
        self._send_json_response(sources)

    def _handle_api_request(self, path):
        """Handle API requests"""
        global camera_controller
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
            else:
                data = {}

            response = {'success': False, 'message': 'Unknown API endpoint'}

            if path == '/api/resolution':
                width = data.get('width', 640)
                height = data.get('height', 480)
                if camera_controller.set_resolution(width, height):
                    response = {'success': True, 'message': f'Resolution set to {width}x{height}'}

            elif path == '/api/zoom':
                zoom = data.get('zoom', 1.0)
                if camera_controller.set_zoom(zoom):
                    response = {'success': True, 'message': f'Zoom set to {zoom}x'}

            elif path == '/api/pan':
                pan_x = data.get('pan_x', 0)
                pan_y = data.get('pan_y', 0)
                if camera_controller.set_pan(pan_x, pan_y):
                    response = {'success': True, 'message': f'Pan set to ({pan_x}, {pan_y})'}

            elif path == '/api/rotation':
                rotation = data.get('rotation', 0)
                if camera_controller.set_rotation(rotation):
                    response = {'success': True, 'message': f'Rotation set to {rotation}°'}
                else:
                    response = {'success': False, 'message': 'Invalid rotation (must be 0, 90, 180, or 270)'}

            elif path == '/api/target_detection':
                enabled = data.get('enabled', True)
                if camera_controller.set_target_detection(enabled):
                    status = "enabled" if enabled else "disabled"
                    response = {'success': True, 'message': f'Target detection {status}'}
                else:
                    response = {'success': False, 'message': 'Failed to set target detection'}

            elif path == '/api/debug_mode':
                enabled = data.get('enabled', False)
                if camera_controller.set_debug_mode(enabled):
                    status = "enabled" if enabled else "disabled"
                    response = {'success': True, 'message': f'Debug mode {status}'}
                else:
                    response = {'success': False, 'message': 'Failed to set debug mode'}

            elif path == '/api/debug_type':
                debug_type = data.get('debug_type', 'combined')
                if camera_controller.target_detector.set_debug_type(debug_type):
                    response = {'success': True, 'message': f'Debug type set to {debug_type}'}
                else:
                    response = {'success': False, 'message': 'Invalid debug type (must be combined, corners, or circles)'}

            elif path == '/api/force_detection':
                if camera_controller.target_detector.force_detection():
                    response = {'success': True, 'message': 'Target re-detection forced successfully'}
                else:
                    response = {'success': False, 'message': 'Failed to force target re-detection'}

            elif path == '/api/calibrate_perspective':
                success, message = camera_controller.calibrate_perspective()
                response = {'success': success, 'message': message}

            elif path == '/api/save_calibration':
                # Get current camera resolution and save calibration
                camera_resolution = camera_controller.resolution
                success, message = camera_controller.perspective.save_calibration(camera_resolution)
                response = {'success': success, 'message': message}

            elif path == '/api/calibration_mode':
                enabled = data.get('enabled', False)
                if camera_controller.target_detector.set_calibration_mode(enabled):
                    status = "enabled" if enabled else "disabled"
                    response = {'success': True, 'message': f'Calibration mode {status}'}
                else:
                    response = {'success': False, 'message': 'Failed to set calibration mode'}

            elif path == '/api/capture':
                filename = camera_controller.capture_image()
                if filename:
                    response = {'success': True, 'message': f'Image captured: {filename}', 'filename': filename}
                else:
                    response = {'success': False, 'message': 'Failed to capture image'}

            elif path == '/api/set_reference_frame':
                if camera_controller.set_reference_frame():
                    response = {'success': True, 'message': 'Reference frame set for bullet hole detection'}
                else:
                    response = {'success': False, 'message': 'Failed to set reference frame'}

            elif path == '/api/detect_bullet_holes':
                success, message = camera_controller.detect_bullet_holes()
                response = {'success': success, 'message': message}
                if success and camera_controller.bullet_holes:
                    # Add bullet hole data to response
                    holes_data = []
                    for x, y, radius, score, area, circularity in camera_controller.bullet_holes:
                        holes_data.append({
                            'x': int(x), 'y': int(y), 'radius': int(radius),
                            'score': float(score), 'area': float(area), 'circularity': float(circularity)
                        })
                    response['bullet_holes'] = holes_data

            elif path == '/api/clear_bullet_holes':
                if camera_controller.clear_bullet_holes():
                    response = {'success': True, 'message': 'Bullet holes cleared'}
                else:
                    response = {'success': False, 'message': 'Failed to clear bullet holes'}

            elif path == '/api/change_source':
                source_type = data.get('source_type', '')
                source_id = data.get('source_id', '')
                success, message = camera_controller.set_video_source(source_type, source_id)
                response = {'success': success, 'message': message}

            elif path == '/api/perspective_correction':
                enabled = data.get('enabled', False)
                if camera_controller.set_perspective_correction(enabled):
                    status = "enabled" if enabled else "disabled"
                    response = {'success': True, 'message': f'Perspective correction {status}'}
                else:
                    response = {'success': False, 'message': 'Failed to set perspective correction'}

            elif path == '/api/playback_pause':
                if camera_controller.pause_playback():
                    response = {'success': True, 'message': 'Video playback paused'}
                else:
                    response = {'success': False, 'message': 'Cannot pause (not playing video or already paused)'}

            elif path == '/api/playback_resume':
                if camera_controller.resume_playback():
                    response = {'success': True, 'message': 'Video playback resumed'}
                else:
                    response = {'success': False, 'message': 'Cannot resume (not playing video or not paused)'}

            elif path == '/api/playback_step_forward':
                if camera_controller.step_frame_forward():
                    response = {'success': True, 'message': 'Stepped forward one frame'}
                else:
                    response = {'success': False, 'message': 'Cannot step forward (not paused or no frames available)'}

            elif path == '/api/playback_step_backward':
                if camera_controller.step_frame_backward():
                    response = {'success': True, 'message': 'Stepped backward one frame'}
                else:
                    response = {'success': False, 'message': 'Cannot step backward (not paused or at beginning)'}

            elif path == '/api/seek_to_frame':
                frame_number = data.get('frame', 0)
                success, message = camera_controller.seek_to_frame(frame_number)
                response = {'success': success, 'message': message}

            elif path == '/api/playback_info':
                playback_info = camera_controller.get_playback_info()
                response = {'success': True, 'data': playback_info}

            # Camera controls endpoints
            elif path == '/api/camera_controls':
                controls_info = camera_controller.get_camera_controls()
                response = {'success': True, 'data': controls_info}

            elif path == '/api/set_camera_control':
                control_name = data.get('name', '')
                control_value = data.get('value', 0)
                if control_name:
                    success, message = camera_controller.set_camera_control(control_name, control_value)
                    response = {'success': success, 'message': message}
                else:
                    response = {'success': False, 'message': 'Control name is required'}

            elif path == '/api/get_camera_control':
                control_name = data.get('name', '')
                if control_name:
                    current_value = camera_controller.get_camera_control(control_name)
                    if current_value is not None:
                        response = {'success': True, 'value': current_value}
                    else:
                        response = {'success': False, 'message': f'Control {control_name} not available'}
                else:
                    response = {'success': False, 'message': 'Control name is required'}

            elif path == '/api/reset_camera_controls':
                success, message = camera_controller.reset_camera_controls()
                response = {'success': success, 'message': message}

            elif path == '/api/save_camera_preset':
                preset_name = data.get('name', '')
                if preset_name:
                    success, message = camera_controller.save_camera_preset(preset_name)
                    response = {'success': success, 'message': message}
                else:
                    response = {'success': False, 'message': 'Preset name is required'}

            elif path == '/api/load_camera_preset':
                preset_name = data.get('name', '')
                if preset_name:
                    success, message = camera_controller.load_camera_preset(preset_name)
                    response = {'success': success, 'message': message}
                else:
                    response = {'success': False, 'message': 'Preset name is required'}

            elif path == '/api/list_camera_presets':
                presets = camera_controller.list_camera_presets()
                response = {'success': True, 'presets': presets}

            elif path == '/api/set_resolution':
                width = data.get('width', 0)
                height = data.get('height', 0)
                fps = data.get('fps', 30)
                format_name = data.get('format', 'MJPG')

                if width > 0 and height > 0:
                    success, message = camera_controller.set_camera_resolution(width, height, fps, format_name)
                    response = {'success': success, 'message': message}
                else:
                    response = {'success': False, 'message': 'Valid width and height are required'}

            elif path == '/api/camera_formats':
                formats_info = camera_controller.get_camera_formats()
                response = {'success': True, 'data': formats_info}

            elif path == '/api/start_recording':
                success, message = camera_controller.start_recording()
                response = {'success': success, 'message': message}

            elif path == '/api/stop_recording':
                success, message = camera_controller.stop_recording()
                response = {'success': success, 'message': message}

            elif path == '/api/recording_status':
                recording_status = camera_controller.get_recording_status()
                response = {'success': True, 'data': recording_status}

            self._send_json_response(response)

        except Exception as e:
            print(f"API request error: {e}")
            self._send_json_response({'success': False, 'message': str(e)})

    def _send_json_response(self, data):
        """Send JSON response"""
        json_data = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json_data)

    def log_message(self, format, *args):
        """Custom logging"""
        # Check if args exist and first arg is a string before checking content
        if args and len(args) > 0 and isinstance(args[0], str):
            if '/stream.mjpg' not in args[0]:  # Don't log stream requests
                print(f"Web Server: {format % args}")
        else:
            # For other log messages (like errors), always log them
            print(f"Web Server: {format % args}")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    global camera_controller

    print("🎯 Starting Raspberry Pi Camera Streaming System...")
    print("🔍 Faulthandler will catch any segmentation faults")

    try:
        # Initialize camera controller
        print("DEBUG: Creating CameraController...")
        camera_controller = CameraController(camera_index=0)
        print("DEBUG: CameraController created successfully")

        print("DEBUG: Starting capture...")
        if not camera_controller.start_capture():
            print("❌ Failed to initialize camera")
            return
        print("✅ Camera initialized successfully")

        # Start unified HTTP server
        print("DEBUG: Creating HTTP server...")
        server = ThreadingHTTPServer(('0.0.0.0', 8088), StreamingHandler)
        print("🌐 Server starting on port 8088...")
        print("📺 Camera stream: http://localhost:8088/stream.mjpg")
        print("🖥️  Web interface: http://localhost:8088")
        print("🔧 API endpoints: http://localhost:8088/api/")

        try:
            print("DEBUG: Starting server.serve_forever()...")
            print("🔄 Monitoring for crashes - will show detailed traceback if segfault occurs")
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            camera_controller.stop()
            server.shutdown()
    except Exception as e:
        print(f"FATAL ERROR in main(): {e}")
        print("🔍 Dumping all thread stacks:")
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()