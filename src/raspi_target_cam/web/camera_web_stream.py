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
        with open("/tmp/camera_crash.log", "w") as f:
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
import os
from datetime import datetime
from http.server import HTTPServer
from socketserver import ThreadingMixIn
from functools import partial
from raspi_target_cam.core.target_detection import TargetDetector
from raspi_target_cam.core.perspective import Perspective
from raspi_target_cam.detection.bullet_hole_detection import BulletHoleDetector
from raspi_target_cam.detection.yolo_detector import YoloDetector
from raspi_target_cam.detection.bullet_hole_tracker import BulletHoleTracker
from raspi_target_cam.camera.camera_controls import CameraControlManager
from raspi_target_cam.core.streaming_handler import StreamingHandler
from raspi_target_cam.utils.metadata_handler import MetadataHandler
from raspi_target_cam.camera.threaded_capture import ThreadedCaptureSystem
from raspi_target_cam.camera.camera_settings import CameraSettings

# Keep FFmpeg/MJPEG error messages visible for debugging
# Don't suppress them - they help diagnose video issues
# We handle the errors in code to prevent crashes, but still log them


class CameraController:
    """Centralized camera control and streaming"""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.video_file = None
        self.source_type = "test"  # Start with test pattern to avoid camera hang
        self.cap = None
        self.frame = None  # Processed frame (with overlays, transformations, etc.)
        self.raw_frame = None  # Raw frame without any processing (for calibration)
        self.running = False
        self.lock = threading.Lock()
        self.switching_source = False  # Flag to indicate source switch in progress

        # Camera settings
        self.resolution = (2592, 1944)
        self.camera_fps = None  # None = use camera default, or set specific FPS
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.rotation = 270  # degrees clockwise (270 = 90 anti-clockwise)

        # Capture settings
        self.captures_dir = "./data/captures"
        os.makedirs(self.captures_dir, exist_ok=True)

        # Video recording settings
        self.recordings_dir = "./data/recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
        self.recording = False
        self.video_writer = None
        self.recording_filename = None
        self.recording_start_time = None
        self.recording_width = 0
        self.recording_height = 0

        # Video file settings
        self.samples_dir = "./data/samples"
        self.video_fps = 30  # Default FPS for video files
        self.video_frame_time = 1.0 / self.video_fps  # Time between frames
        self.native_video_resolution = None  # Native resolution of video file
        self.native_video_fps = None  # Native FPS of video file

        # Available sources (detected once at startup)
        self.available_sources = {"cameras": [], "videos": []}

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
        self.detector_type = "yolo"  # "traditional" or "yolo"
        self.bullet_hole_detector = BulletHoleDetector()
        self.yolo_detector = None  # Will be initialized below
        self.yolo_conf_threshold = 0.5  # Confidence threshold for YOLO (0.0-1.0, higher = fewer false positives)
        self.reference_frame = None  # Store reference frame for bullet hole detection
        self.bullet_holes = []  # Detected bullet holes (stable tracked holes)
        self.continuous_detection = False  # Run detection on every frame (for YOLO)
        self.detection_interval = 0  # Frames between detections (0 = every frame)
        self.frame_count = 0  # Counter for detection interval

        # Bullet hole tracker (for temporal filtering and position averaging)
        self.bullet_hole_tracker = BulletHoleTracker(
            match_distance_threshold=30.0,  # Max 30px distance to match holes
            max_frames_missing=5,  # Remove holes after 5 frames without detection
            min_detections_for_stability=3,  # Require 3 detections before showing
        )

        # Track previous target detection state to detect transitions
        self.previous_target_detected = False

        # Camera controls
        self.camera_controls = None  # Will be initialized when camera is opened
        self.cached_camera_controls = {"available": False, "controls": {}}  # Cached controls from detection

        # Threaded capture system
        self.capture_system = None

        # Initialize YOLO detector if using YOLO by default
        if self.detector_type == "yolo":
            try:
                print("Initializing YOLO detector...")
                self.yolo_detector = YoloDetector(
                    conf_threshold=self.yolo_conf_threshold,
                    iou_threshold=0.45,
                    target_class=0,  # bullet_hole class
                    target_detector=self.target_detector,  # Pass target detector for centered cropping
                )
                print("✅ YOLO detector initialized successfully")
            except Exception as e:
                print(f"⚠️  Failed to initialize YOLO detector: {e}")
                print("   Falling back to traditional detector")
                self.detector_type = "traditional"

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
                    int(50 + (x / width) * 100),  # Red gradient
                    int(30 + (y / height) * 80),  # Green gradient
                    100,  # Blue constant
                ]

        # Add title
        cv2.putText(frame, "🎯 Raspberry Pi Target Camera", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

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
        cv2.rectangle(
            frame, (clock_bg_x, clock_bg_y), (clock_bg_x + clock_bg_w, clock_bg_y + clock_bg_h), (0, 0, 0), -1
        )  # Black background
        cv2.rectangle(
            frame, (clock_bg_x, clock_bg_y), (clock_bg_x + clock_bg_w, clock_bg_y + clock_bg_h), (100, 100, 100), 2
        )  # Gray border

        # Digital time display
        cv2.putText(
            frame, time_str, (clock_bg_x + 10, clock_bg_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
        )  # Green digits
        cv2.putText(
            frame, date_str, (clock_bg_x + 10, clock_bg_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1
        )  # Gray date

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
            f"• {len(self.available_sources.get('videos', []))} video file(s) available",
        ]

        for line in info_lines:
            cv2.putText(frame, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 40

        # Add timestamp
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, f"Generated: {timestamp}", (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )

        # Add frame counter (to show it's "live")
        self.test_frame_counter += 1
        cv2.putText(
            frame,
            f"Frame: {self.test_frame_counter}",
            (width - 200, height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

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
                # Test mode - use ThreadedCaptureSystem with no capture device
                print("DEBUG: Test mode - will generate test frames via ThreadedCaptureSystem")
                self.running = True
                # No cap device needed for test mode
                self.cap = None

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
                            [
                                "v4l2-ctl",
                                "-d",
                                f"/dev/video{self.camera_index}",
                                "--set-ctrl",
                                "power_line_frequency=1",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=2,
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

                        # Set FPS if specified
                        if self.camera_fps is not None:
                            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
                            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                            print(f"DEBUG: Requested FPS: {self.camera_fps}, Actual FPS: {actual_fps}")

                            # Warn if FPS mismatch (common with MJPEG at high resolutions)
                            if abs(actual_fps - self.camera_fps) > 1:
                                print(f"WARNING: FPS mismatch - requested {self.camera_fps}, got {actual_fps}")
                                print(f"         High resolutions with MJPEG may be limited to 15 FPS")
                        else:
                            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                            print(f"DEBUG: Using camera default FPS: {actual_fps}")

                        print(f"DEBUG: Camera properties set")
                    except Exception as e:
                        print(f"WARNING: Could not set camera properties: {e}")
                        # Continue anyway, camera might still work with default settings

                    # Use cached camera controls from source detection
                    try:
                        print(f"DEBUG: Loading cached camera controls for camera {self.camera_index}...")

                        # Find the camera in available sources to get cached controls
                        camera_source = None
                        for camera in self.available_sources.get("cameras", []):
                            if camera["index"] == self.camera_index:
                                camera_source = camera
                                break

                        if camera_source and camera_source.get("controls", {}).get("available"):
                            # Use cached control information
                            self.cached_camera_controls = camera_source["controls"]
                            print(
                                f"DEBUG: Using cached controls - {len(self.cached_camera_controls['controls'])} controls available"
                            )
                        else:
                            self.cached_camera_controls = {"available": False, "controls": {}}
                            print(f"DEBUG: No cached controls available for camera {self.camera_index}")

                    except Exception as e:
                        print(f"WARNING: Could not load cached camera controls: {e}")
                        self.cached_camera_controls = {"available": False, "controls": {}}

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

            print("DEBUG: Starting threaded capture system...")

            # Create and start threaded capture system
            # Wrap test frame generator to match expected signature (takes frame_counter)
            test_gen = (lambda counter: self.generate_test_frame()) if self.source_type == "test" else None

            self.capture_system = ThreadedCaptureSystem(
                cap=self.cap,
                source_type=self.source_type,
                camera_index=self.camera_index,
                buffer_size=self.buffer_size,
                test_frame_generator=test_gen,
            )
            self.capture_system.start()

            # Start processing loop for transformations and detection
            threading.Thread(target=self._processing_loop, daemon=True).start()

            return True
        except Exception as e:
            print(f"Failed to initialize capture: {e}")
            return False

    def _processing_loop(self):
        """Processing loop for transformations, perspective correction, and detection"""
        print("DEBUG: Processing loop started")

        last_processed_change_counter = -1  # Track last processed frame using change counter

        while self.running:
            try:
                # Get latest frame with playback position from capture system
                if self.capture_system:
                    raw_frame, position = self.capture_system.get_latest_frame()
                else:
                    raw_frame, position = (
                        None,
                        {"current_frame": 0, "total_frames": 0, "source_type": "unknown", "change_counter": 0},
                    )

                if raw_frame is not None:
                    change_counter = position.get("change_counter", 0)
                    # print(f"processing frame {change_counter}")

                    # Skip processing if it's the same frame as last time
                    # Use change_counter which works for all source types (camera, video, test)
                    if change_counter == last_processed_change_counter:
                        # Frame hasn't changed, skip processing
                        time.sleep(0.10)
                        continue

                    # Update last processed change counter
                    last_processed_change_counter = change_counter

                    # Step 1: Apply pre-detection transformations (rotation + perspective correction)
                    # Note: _apply_transformations stores the rotated frame as self.raw_frame
                    corrected_frame = self._apply_transformations(raw_frame)

                    # Step 2: Run target detection (circle detection) if enabled
                    # This detects the target itself and determines target type (pistol/rifle)
                    target_detected = False
                    if self.target_detector.detection_enabled:
                        self.target_detector.detect_target(corrected_frame)
                        # Check if target is detected (even if not stable yet)
                        target_detected = self.target_detector.target_center is not None

                    # Clear bullet holes when target transitions from detected to not detected
                    if self.previous_target_detected and not target_detected:
                        print("Target lost - clearing bullet holes")
                        self.clear_bullet_holes()

                    # Update previous target state
                    self.previous_target_detected = target_detected

                    # Step 3: Run continuous bullet hole detection if enabled (on corrected frame, before zoom/pan)
                    # IMPORTANT: Only run if target is detected, and detection runs on perspective-corrected frame
                    if target_detected and self.continuous_detection and self.detector_type == "yolo":
                        self.frame_count += 1
                        # Check if we should run detection on this frame
                        if self.frame_count > self.detection_interval:
                            self.frame_count = 0
                            self._run_continuous_detection(corrected_frame)

                    # Step 4: Add bullet hole overlays (on corrected frame, before zoom/pan)
                    display_frame = corrected_frame
                    if self.bullet_holes:
                        display_frame = self.bullet_hole_detector.draw_bullet_hole_overlays(
                            corrected_frame, self.bullet_holes
                        )

                    # Step 5: Apply display transformations (zoom/pan) for UI
                    display_frame = self._apply_display_transformations(display_frame)

                    # Step 6: Add HUD overlay
                    display_frame = self._draw_hud_overlay(display_frame)

                    # Update display frame (thread-safe)
                    with self.lock:
                        self.frame = display_frame.copy()  # Store display frame for streaming

                    # Update playback position for video files
                    if self.source_type == "video":
                        self.current_frame_number = position["current_frame"]
                        self.total_frames = position["total_frames"]
                        if not self.paused:
                            self.display_frame_number = self.current_frame_number

                # Sleep briefly to avoid busy-waiting (~100 FPS max processing rate)
                time.sleep(0.01)

            except Exception as e:
                print(f"ERROR in _processing_loop: {e}")
                import traceback

                traceback.print_exc()
                time.sleep(0.1)

        print("DEBUG: Processing loop stopped")

    def _apply_transformations(self, frame):
        """Apply rotation and perspective correction (pre-detection transformations)

        Args:
            frame: Input frame

        Returns:
            Transformed frame ready for detection (rotation + perspective correction applied)
        """
        # Apply rotation first
        if self.rotation != 0:
            frame = self._rotate_frame(frame, self.rotation)

        # Store rotated frame as raw frame (for calibration - includes rotation but no overlays)
        with self.lock:
            self.raw_frame = frame.copy()

        # Apply perspective correction
        if self.perspective_correction_enabled:
            corrected_frame = self.perspective.apply_perspective_correction(frame)
            if corrected_frame is not None:
                frame = corrected_frame

        return frame

    def _apply_display_transformations(self, frame):
        """Apply zoom and pan for display (post-detection transformations)

        Args:
            frame: Input frame (should already have rotation and perspective correction applied)

        Returns:
            Frame with zoom/pan applied for display
        """
        # Skip if no zoom or pan
        if self.zoom == 1.0 and self.pan_x == 0 and self.pan_y == 0:
            return frame

        h, w = frame.shape[:2]

        # Apply zoom and pan
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

    def _draw_hud_overlay(self, frame):
        """Draw HUD overlay on the frame showing target information

        Args:
            frame: Input frame

        Returns:
            Frame with HUD overlay
        """
        if not self._is_valid_frame(frame):
            return frame

        # Create a copy to avoid modifying the original
        overlay_frame = frame.copy()
        h, w = overlay_frame.shape[:2]

        # Get target type from target detector (using the old attributes for compatibility)
        target_type = self.target_detector.target_type
        is_stable = self.target_detector.stable_detection is not None
        confidence = self.target_detector.detection_confidence

        # Prepare text and color based on detection state
        if target_type:
            text = f"Target: {target_type.upper()}"
            if is_stable:
                text_color = (0, 255, 0)  # Green for stable detection
            else:
                text_color = (0, 255, 255)  # Yellow for unstable detection
                text += " (detecting...)"
        else:
            text = "No Target"
            text_color = (100, 100, 100)  # Gray for no target

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2

        # Get text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Calculate centered position at top
        text_x = (w - text_width) // 2
        text_y = 50 + text_height  # 50px from top

        # Draw semi-transparent background rectangle for better readability
        padding = 10
        bg_x1 = text_x - padding
        bg_y1 = text_y - text_height - padding
        bg_x2 = text_x + text_width + padding
        bg_y2 = text_y + baseline + padding

        # Create semi-transparent overlay
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        alpha = 0.6  # Transparency
        cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)

        # Draw text
        cv2.putText(overlay_frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

        return overlay_frame

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
        if not hasattr(frame, "shape") or not hasattr(frame, "dtype"):
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
            # Don't try to get frames during source switching
            if self.switching_source:
                return None

            # Don't try to get frames if not running
            if not self.running:
                return None

            with self.lock:
                if self.frame is not None:
                    # Validate frame before OpenCV operations
                    if not self._is_valid_frame(self.frame):
                        print(f"WARNING: Invalid frame detected, skipping encode")
                        return None

                    # Additional safety checks for corrupted data
                    if not hasattr(self.frame, "shape") or not hasattr(self.frame, "dtype"):
                        print(f"WARNING: Frame missing attributes, skipping encode")
                        return None

                    # Check for reasonable dimensions to prevent memory issues
                    height, width = self.frame.shape[:2]
                    if width > 10000 or height > 10000 or width < 10 or height < 10:
                        print(f"WARNING: Frame dimensions unreasonable ({width}x{height}), skipping encode")
                        return None

                    # Make a copy to avoid memory issues
                    frame_copy = self.frame.copy()

                    # Ensure proper data type and layout
                    if frame_copy.dtype != "uint8":
                        frame_copy = frame_copy.astype("uint8")

                    # Ensure contiguous memory layout
                    if not frame_copy.flags["C_CONTIGUOUS"]:
                        frame_copy = frame_copy.copy()

                    # Extra validation before imencode
                    if frame_copy.size == 0 or not frame_copy.data.contiguous:
                        print(f"WARNING: Frame data invalid before encode, skipping")
                        return None

                    try:
                        # Try to encode - catch FFmpeg decoder errors
                        _, buffer = cv2.imencode(".jpg", frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        return buffer.tobytes()
                    except cv2.error as e:
                        print(f"WARNING: OpenCV error during JPEG encoding: {e}")
                        return None
                    except Exception as e:
                        print(f"WARNING: Unexpected error during JPEG encoding: {e}")
                        return None

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

    def set_fps(self, fps):
        """Change camera FPS (only works for cameras, not video files)

        Args:
            fps: Desired FPS value, or None to use camera default

        Returns:
            True if successful, False otherwise
        """
        if self.source_type == "video":
            # Cannot change FPS for video files
            return False

        self.camera_fps = fps
        if self.cap and self.cap.isOpened():
            if fps is not None:
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                print(f"FPS set: requested {fps}, actual {actual_fps}")
            else:
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                print(f"FPS using camera default: {actual_fps}")
            return True
        else:
            # Camera not active yet, just store the setting
            print(f"FPS setting stored (will apply when camera starts): {fps if fps is not None else 'default'}")
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
        """Pause video playback using threaded system"""
        if self.source_type == "video":
            self.paused = True
            if self.capture_system:
                self.capture_system.pause()
            # Record the frame number when we paused
            self.pause_frame_number = self.display_frame_number
            return True
        return False

    def resume_playback(self):
        """Resume video playback using threaded system"""
        if self.source_type == "video":
            self.paused = False
            if self.capture_system:
                self.capture_system.resume()
            return True
        return False

    def step_frame_forward(self):
        """Step one frame forward when paused using threaded system"""
        if self.source_type != "video" or not self.paused:
            return False

        if not self.capture_system:
            return False

        success = self.capture_system.step_forward()
        if success:
            # Update display frame number
            pos = self.capture_system.get_playback_position()
            self.display_frame_number = pos["current_frame"]
            print(f"DEBUG: Step forward to frame {self.display_frame_number}")
        return success

    def step_frame_backward(self):
        """Step one frame backward when paused using threaded system"""
        if self.source_type != "video" or not self.paused:
            return False

        if not self.capture_system:
            return False

        success = self.capture_system.step_backward()
        if success:
            # Update display frame number
            pos = self.capture_system.get_playback_position()
            self.display_frame_number = pos["current_frame"]
            print(f"DEBUG: Step backward to frame {self.display_frame_number}")
        return success

    def step_seconds_forward(self, seconds):
        """Step forward by specified seconds when paused using threaded system"""
        if self.source_type != "video" or not self.paused:
            return False

        if not self.capture_system:
            return False

        success = self.capture_system.step_seconds(seconds, forward=True)
        if success:
            # Update display frame number
            pos = self.capture_system.get_playback_position()
            self.display_frame_number = pos["current_frame"]
            print(f"DEBUG: Step forward {seconds}s to frame {self.display_frame_number}")
        return success

    def step_seconds_backward(self, seconds):
        """Step backward by specified seconds when paused using threaded system"""
        if self.source_type != "video" or not self.paused:
            return False

        if not self.capture_system:
            return False

        success = self.capture_system.step_seconds(seconds, forward=False)
        if success:
            # Update display frame number
            pos = self.capture_system.get_playback_position()
            self.display_frame_number = pos["current_frame"]
            print(f"DEBUG: Step backward {seconds}s to frame {self.display_frame_number}")
        return success

    def seek_to_frame(self, target_frame):
        """Seek to a specific frame number using threaded system"""
        if self.source_type != "video":
            return False, "Seek only works with video files"

        if not self.capture_system:
            return False, "Capture system not available"

        # Validate frame number
        if target_frame < 0:
            target_frame = 0
        elif self.total_frames > 0 and target_frame >= self.total_frames:
            target_frame = self.total_frames - 1

        print(f"DEBUG: Seeking to frame {target_frame}")

        # Send seek command to capture system
        success = self.capture_system.seek_to_frame(target_frame)

        if success:
            # Update frame counters
            time.sleep(0.2)  # Give time for seek to complete
            pos = self.capture_system.get_playback_position()
            self.current_frame_number = pos["current_frame"]
            self.display_frame_number = pos["current_frame"]
            print(f"DEBUG: Seeked to frame {self.current_frame_number}")
            return True, f"Seeked to frame {self.current_frame_number}"
        else:
            return False, f"Failed to seek to frame {target_frame}"

    def is_paused(self):
        """Check if playback is paused"""
        return self.paused

    def get_playback_info(self):
        """Get current playback information"""
        return {
            "paused": self.paused,
            "current_frame": self.display_frame_number,  # Use display frame number instead
            "total_frames": self.total_frames,
            "buffer_size": len(self.frame_buffer),
            "pause_buffer_index": self.pause_buffer_index if self.paused else None,
            "can_step_backward": self.paused and self.display_frame_number > 0,
            "can_step_forward": self.paused and self.display_frame_number < self.total_frames - 1,
            "supports_playback_controls": self.source_type == "video",
        }

    def get_debug_frame_jpeg(self, debug_type="combined"):
        """Get debug frame as JPEG bytes for specific debug type

        Args:
            debug_type: Type of debug frame ('combined', 'perspective', 'circles', 'corrected')
        """
        # Generate fresh debug frame from current frame if debug mode is on
        with self.lock:
            if self.frame is not None and self.target_detector.debug_mode:
                self.target_detector.generate_debug_frame(self.frame)

                # For corrected debug type, provide the perspective-corrected frame
                if debug_type == "corrected" and self.perspective_correction_enabled:
                    # Fallback to generic perspective correction
                    corrected_frame = self.perspective.apply_perspective_correction(self.frame)
                    if corrected_frame is not None:
                        # Temporarily store corrected frame for debug access
                        self.target_detector.corrected_debug_frame = corrected_frame

        return self.target_detector.get_debug_frame_jpeg(debug_type)

    def _get_camera_controls_metadata(self):
        """Get current camera controls as metadata dictionary"""
        source_info = {}
        if self.source_type == "video":
            source_info["video_file"] = self.video_file or "unknown"
        elif self.source_type == "camera":
            source_info["camera_index"] = self.camera_index

        settings = {
            "resolution": self.resolution,
            "zoom": self.zoom,
            "rotation": self.rotation,
            "perspective_correction": self.perspective_correction_enabled,
        }

        return MetadataHandler.get_camera_controls_metadata(
            self.source_type, source_info, settings, self.cached_camera_controls
        )

    def _embed_video_metadata(self, video_filepath):
        """Embed metadata into video file using FFmpeg"""
        MetadataHandler.embed_video_metadata(video_filepath)

    def capture_image(self):
        """Capture and save current frame with camera controls metadata as EXIF tags"""
        with self.lock:
            if self.frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                filepath = os.path.join(self.captures_dir, filename)

                # Get metadata
                metadata = self._get_camera_controls_metadata()

                # Save with OpenCV at 100% JPEG quality (lossless compression)
                cv2.imwrite(filepath, self.frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

                # Embed metadata as EXIF
                MetadataHandler.embed_image_metadata(filepath, metadata)

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
        # Check if target is detected first
        if self.target_detector.target_center is None:
            return False, "No target detected. Please ensure a target is visible in the frame."

        if self.reference_frame is None and self.detector_type == "traditional":
            return False, "No reference frame set"

        with self.lock:
            if self.frame is not None:
                # Select detector based on type
                if self.detector_type == "yolo":
                    # Initialize YOLO detector if not already loaded
                    if self.yolo_detector is None:
                        try:
                            print(f"Loading YOLO detector with confidence threshold {self.yolo_conf_threshold}...")
                            self.yolo_detector = YoloDetector(conf_threshold=self.yolo_conf_threshold, target_class=0)
                            print("YOLO detector loaded successfully")
                        except Exception as e:
                            return False, f"Failed to load YOLO detector: {e}"

                    # YOLO detector works on single frame (doesn't need reference)
                    holes = self.yolo_detector.detect_bullet_holes(
                        self.reference_frame if self.reference_frame is not None else self.frame, self.frame
                    )
                else:
                    # Traditional detector needs reference frame
                    holes = self.bullet_hole_detector.detect_bullet_holes(self.reference_frame, self.frame)

                # Store detected holes and cache in detector for overlay
                self.bullet_holes = holes
                # Also cache in traditional detector for compatibility with overlay system
                self.bullet_hole_detector.last_detection = holes
                return True, f"Found {len(holes)} bullet hole(s) using {self.detector_type} detector"

        return False, "No current frame available"

    def clear_bullet_holes(self):
        """Clear all detected bullet holes"""
        self.bullet_holes = []
        self.bullet_hole_detector.last_detection = []
        self.bullet_hole_tracker.reset()  # Also reset the tracker
        return True

    def set_detector_type(self, detector_type):
        """Set the bullet hole detector type"""
        if detector_type not in ["traditional", "yolo"]:
            return False, f"Invalid detector type: {detector_type}"

        self.detector_type = detector_type

        # Pre-load YOLO detector if selected
        if detector_type == "yolo" and self.yolo_detector is None:
            try:
                print(f"Pre-loading YOLO detector with confidence threshold {self.yolo_conf_threshold}...")
                self.yolo_detector = YoloDetector(
                    conf_threshold=self.yolo_conf_threshold, target_class=0, target_detector=self.target_detector
                )
                print("YOLO detector loaded successfully")
            except Exception as e:
                return False, f"Failed to load YOLO detector: {e}"

        return True, f"Detector type set to: {detector_type}"

    def get_detector_type(self):
        """Get current detector type"""
        return self.detector_type

    def list_yolo_models(self):
        """List all available YOLO models in data/models/ directory"""
        from pathlib import Path

        models_dir = Path("./data/models")
        if not models_dir.exists():
            return []

        models = []

        # Find .pt files
        for pt_file in models_dir.glob("*.pt"):
            models.append(
                {"name": pt_file.name, "path": str(pt_file), "type": "PyTorch", "size": pt_file.stat().st_size}
            )

        # Find NCNN model directories (contain .param and .bin files)
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                param_files = list(model_dir.glob("*.param"))
                bin_files = list(model_dir.glob("*.bin"))
                if param_files and bin_files:
                    # Calculate total size of all files in directory
                    total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                    models.append({"name": model_dir.name, "path": str(model_dir), "type": "NCNN", "size": total_size})

        # Sort by name
        models.sort(key=lambda x: x["name"])

        return models

    def get_current_yolo_model(self):
        """Get the currently loaded YOLO model path"""
        if self.yolo_detector:
            return str(self.yolo_detector.model_path)
        return None

    def set_yolo_model(self, model_path):
        """Set/load a specific YOLO model"""
        from pathlib import Path

        # Validate model path exists
        model_path = Path(model_path)
        if not model_path.exists():
            return False, f"Model not found: {model_path}"

        try:
            print(f"Loading YOLO model: {model_path}")
            # Create new detector with the specified model
            self.yolo_detector = YoloDetector(
                model_path=str(model_path),
                conf_threshold=self.yolo_conf_threshold,
                iou_threshold=0.45,
                target_class=0,
                target_detector=self.target_detector,
            )
            print(f"✅ Successfully loaded YOLO model: {model_path.name}")
            return True, f"Loaded model: {model_path.name}"
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            return False, f"Failed to load model: {str(e)}"

    def set_continuous_detection(self, enabled):
        """Enable or disable continuous detection (runs on every frame)"""
        self.continuous_detection = enabled
        if enabled and self.detector_type == "yolo":
            # Pre-load YOLO detector
            if self.yolo_detector is None:
                try:
                    print(
                        f"Pre-loading YOLO detector for continuous detection with confidence threshold {self.yolo_conf_threshold}..."
                    )
                    self.yolo_detector = YoloDetector(
                        conf_threshold=self.yolo_conf_threshold, target_class=0, target_detector=self.target_detector
                    )
                    print("YOLO detector loaded successfully")
                except Exception as e:
                    return False, f"Failed to load YOLO detector: {e}"
        return True, f"Continuous detection {'enabled' if enabled else 'disabled'}"

    def get_continuous_detection(self):
        """Get continuous detection status"""
        return self.continuous_detection

    def set_yolo_confidence(self, confidence):
        """Set YOLO confidence threshold and reload detector if already loaded"""
        if not (0.0 <= confidence <= 1.0):
            return False, "Confidence must be between 0.0 and 1.0"

        self.yolo_conf_threshold = confidence

        # Reload detector if it's already loaded
        if self.yolo_detector is not None:
            try:
                print(f"Reloading YOLO detector with new confidence threshold {confidence}...")
                self.yolo_detector = YoloDetector(conf_threshold=confidence, target_class=0)
                print("YOLO detector reloaded successfully")
            except Exception as e:
                return False, f"Failed to reload YOLO detector: {e}"

        return True, f"YOLO confidence threshold set to {confidence:.2f}"

    def get_yolo_confidence(self):
        """Get current YOLO confidence threshold"""
        return self.yolo_conf_threshold

    def _run_continuous_detection(self, frame):
        """Run detection on current frame (called from processing loop)"""
        try:
            if self.yolo_detector is None:
                return

            # Run YOLO detection on current frame (raw detections)
            raw_detections = self.yolo_detector.detect(frame, debug=False)

            # Update tracker with raw detections
            # Tracker will filter out noise and average positions
            stable_holes = self.bullet_hole_tracker.update(raw_detections)

            # Update bullet holes with stable tracked holes (thread-safe)
            with self.lock:
                self.bullet_holes = stable_holes
                self.bullet_hole_detector.last_detection = stable_holes

        except Exception as e:
            print(f"Error in continuous detection: {e}")

    def get_bullet_hole_debug_frame(self, frame_type="combined"):
        """Get bullet hole detection debug frame"""
        return self.bullet_hole_detector.get_debug_frame(frame_type)

    def start_recording(self):
        """Start recording video to file using threaded system"""
        if self.recording:
            return False, "Already recording"

        if self.source_type == "test":
            return False, "Cannot record test pattern"

        if not self.capture_system:
            return False, "Capture system not available"

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"recording_{timestamp}"
        filepath = os.path.join(self.recordings_dir, base_filename)

        # Determine recording FPS based on source type
        if self.source_type == "video":
            fps = self.video_fps
        else:
            fps = 30.0

        # Clamp FPS to reasonable values
        fps = max(1.0, min(120.0, fps))

        # Get camera controls metadata
        metadata = self._get_camera_controls_metadata()

        # Start recording using threaded system (handles codec fallback automatically)
        success, message, actual_filepath = self.capture_system.start_recording(filepath, fps=fps, metadata=metadata)

        if success:
            self.recording = True
            self.recording_start_time = datetime.now()
            self.recording_filename = os.path.basename(actual_filepath)
            print(f"Started recording: {actual_filepath}")
            return True, message
        else:
            return False, message

    def stop_recording(self):
        """Stop recording video using threaded system"""
        if not self.recording:
            return False, "Not currently recording"

        self.recording = False

        # Stop recording in threaded system (metadata already saved automatically)
        if self.capture_system:
            self.capture_system.stop_recording()

        # Calculate recording duration and get file info
        if self.recording_start_time and self.recording_filename:
            duration = (datetime.now() - self.recording_start_time).total_seconds()
            filepath = os.path.join(self.recordings_dir, self.recording_filename)

            try:
                if os.path.exists(filepath):
                    filesize_mb = os.path.getsize(filepath) / (1024 * 1024)
                    message = f"Recording saved: {self.recording_filename} ({duration:.1f}s, {filesize_mb:.1f}MB)"

                    # Embed metadata into video file using FFmpeg
                    self._embed_video_metadata(filepath)
                else:
                    message = f"Recording saved: {self.recording_filename}"

                print(message)
                return True, message

            except Exception as e:
                return False, f"Error processing recording: {str(e)}"

        return True, "Recording stopped"

    def get_recording_status(self):
        """Get current recording status"""
        if not self.recording:
            return {"recording": False, "filename": None, "duration": 0}

        duration = 0
        if self.recording_start_time:
            duration = (datetime.now() - self.recording_start_time).total_seconds()

        return {"recording": True, "filename": self.recording_filename, "duration": duration}

    def get_camera_controls(self):
        """Get available camera controls from cached data"""
        return {
            "available": self.cached_camera_controls.get("available", False),
            "controls": self.cached_camera_controls.get("controls", {}),
        }

    def set_camera_control(self, name, value):
        """Set a camera control value using the existing camera instance"""
        if not self.cached_camera_controls.get("available"):
            return False, "Camera controls not available"

        if name not in self.cached_camera_controls.get("controls", {}):
            return False, f"Control '{name}' not available"

        if not self.cap or not self.cap.isOpened():
            return False, "Camera not available for control changes"

        try:
            # Map control names to OpenCV property IDs
            control_mapping = {
                "brightness": cv2.CAP_PROP_BRIGHTNESS,
                "contrast": cv2.CAP_PROP_CONTRAST,
                "saturation": cv2.CAP_PROP_SATURATION,
                "hue": cv2.CAP_PROP_HUE,
                "gain": cv2.CAP_PROP_GAIN,
                "gamma": cv2.CAP_PROP_GAMMA,
                "exposure": cv2.CAP_PROP_EXPOSURE,
                "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
                "auto_wb": cv2.CAP_PROP_AUTO_WB,
                "backlight": cv2.CAP_PROP_BACKLIGHT,
                "sharpness": cv2.CAP_PROP_SHARPNESS,
                "temperature": cv2.CAP_PROP_TEMPERATURE,
                "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
            }

            if name not in control_mapping:
                return False, f"Control '{name}' not supported for direct setting"

            # Validate value against cached range
            control_info = self.cached_camera_controls["controls"][name]
            min_val = control_info.get("min")
            max_val = control_info.get("max")

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
                self.cached_camera_controls["controls"][name]["current"] = actual_value

                return True, f"Set {name} to {actual_value}"
            else:
                return False, f"Failed to set {name} (OpenCV returned false)"

        except Exception as e:
            return False, f"Error setting {name}: {str(e)}"

    def get_camera_control(self, name):
        """Get current value of a camera control from cached data"""
        if not self.cached_camera_controls.get("available"):
            return None

        if name not in self.cached_camera_controls.get("controls", {}):
            return None

        return self.cached_camera_controls["controls"][name].get("current")

    def reset_camera_controls(self):
        """Reset all camera controls to defaults using existing camera instance"""
        if not self.cached_camera_controls.get("available"):
            return False, "Camera controls not available"

        if not self.cap or not self.cap.isOpened():
            return False, "Camera not available for control reset"

        try:
            success_count = 0
            total_count = 0

            # Reset each control that has a default value
            for name, control_info in self.cached_camera_controls["controls"].items():
                default_val = control_info.get("default")
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
                "MJPG": cv2.VideoWriter_fourcc(*"MJPG"),
                "YUYV": cv2.VideoWriter_fourcc(*"YUYV"),
                "YUY2": cv2.VideoWriter_fourcc(*"YUY2"),
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

            print(
                f"DEBUG: Resolution set result - Width: {width_success}, Height: {height_success}, FPS: {fps_success}"
            )
            print(f"DEBUG: Actual values - {actual_width}x{actual_height} @ {actual_fps}fps")

            if actual_width == width and actual_height == height:
                message = f"Resolution changed to {actual_width}x{actual_height} @ {actual_fps:.1f}fps ({format_name})"
                if actual_fps != fps:
                    message += f" (requested {fps}fps)"
                return True, message
            else:
                return (
                    False,
                    f"Resolution change failed - got {actual_width}x{actual_height}, expected {width}x{height}",
                )

        except Exception as e:
            return False, f"Error setting resolution: {str(e)}"

    def save_camera_preset(self, preset_name):
        """Save current camera control settings as a preset"""
        if not self.cached_camera_controls.get("available"):
            return False, "Camera controls not available"

        try:
            # Create preset from cached current values
            preset = {}
            for name, control_info in self.cached_camera_controls["controls"].items():
                preset[name] = control_info.get("current")

            # Save to file for persistence
            preset_file = f"./data/presets/{preset_name}.json"
            os.makedirs("./data/presets", exist_ok=True)
            with open(preset_file, "w") as f:
                json.dump(preset, f, indent=2)
            return True, f"Saved preset '{preset_name}' with {len(preset)} controls"
        except Exception as e:
            return False, f"Error saving preset: {str(e)}"

    def load_camera_preset(self, preset_name):
        """Load camera control settings from a preset using existing camera instance"""
        if not self.cached_camera_controls.get("available"):
            return False, "Camera controls not available"

        if not self.cap or not self.cap.isOpened():
            return False, "Camera not available for preset loading"

        try:
            preset_file = f"./data/presets/{preset_name}.json"
            if not os.path.exists(preset_file):
                return False, f"Preset '{preset_name}' not found"

            with open(preset_file, "r") as f:
                preset = json.load(f)

            # Apply preset using existing camera instance
            success_count = 0
            total_count = 0

            for name, value in preset.items():
                if name in self.cached_camera_controls["controls"]:
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
            presets_dir = "./data/presets"
            if not os.path.exists(presets_dir):
                return []

            presets = []
            for filename in os.listdir(presets_dir):
                if filename.endswith(".json"):
                    preset_name = filename[:-5]  # Remove .json extension
                    presets.append(preset_name)
            return presets
        except Exception as e:
            print(f"Error listing presets: {e}")
            return []

    def save_camera_defaults(self):
        """Save current camera settings as defaults.json"""
        try:
            # Build defaults dictionary with all camera settings
            defaults = {
                "source_type": self.source_type,
                "camera_index": self.camera_index if self.source_type == "camera" else None,
                "video_file": self.video_file if self.source_type == "video" else None,
                "resolution": list(self.resolution),
                "camera_fps": self.camera_fps,
                "zoom": self.zoom,
                "pan_x": self.pan_x,
                "pan_y": self.pan_y,
                "rotation": self.rotation,
                "target_detection_enabled": self.target_detector.detection_enabled,
                "perspective_correction_enabled": self.perspective_correction_enabled,
                "debug_mode": self.target_detector.debug_mode,
                "detector_type": self.detector_type,
                "yolo_conf_threshold": self.yolo_conf_threshold,
                "yolo_model_path": self.get_current_yolo_model(),
                "continuous_detection": self.continuous_detection,
            }

            # Add V4L2 camera controls if available
            if self.cached_camera_controls.get("available"):
                camera_controls = {}
                for name, control_info in self.cached_camera_controls["controls"].items():
                    camera_controls[name] = control_info.get("current")
                defaults["camera_controls"] = camera_controls

            # Save to defaults.json file
            defaults_file = "./defaults.json"
            with open(defaults_file, "w") as f:
                json.dump(defaults, f, indent=2)

            print(f"Saved camera settings to {defaults_file}")
            return True, f"Saved camera settings as defaults"
        except Exception as e:
            print(f"Error saving defaults: {e}")
            return False, f"Error saving defaults: {str(e)}"

    def load_camera_defaults(self):
        """Load camera settings from defaults.json and apply them"""
        try:
            defaults_file = "./defaults.json"
            if not os.path.exists(defaults_file):
                print(f"No defaults file found at {defaults_file}")
                return False, "No defaults file found"

            with open(defaults_file, "r") as f:
                defaults = json.load(f)

            print(f"Loading defaults from {defaults_file}")

            # Apply video source first if different from current
            if "source_type" in defaults:
                source_type = defaults["source_type"]
                if source_type == "camera" and defaults.get("camera_index") is not None:
                    camera_index = defaults["camera_index"]
                    if self.source_type != "camera" or self.camera_index != camera_index:
                        print(f"  Switching to camera {camera_index}")
                        # Format: set_video_source expects string camera index
                        success, message = self.set_video_source("camera", str(camera_index))
                        if not success:
                            print(f"  WARNING: Failed to switch to camera: {message}")
                elif source_type == "video" and defaults.get("video_file"):
                    video_file = defaults["video_file"]
                    if self.source_type != "video" or self.video_file != video_file:
                        print(f"  Switching to video file: {video_file}")
                        # Extract just the filename from the path
                        filename = os.path.basename(video_file)
                        success, message = self.set_video_source("video", filename)
                        if not success:
                            print(f"  WARNING: Failed to switch to video: {message}")
                elif source_type == "test":
                    if self.source_type != "test":
                        print(f"  Switching to test pattern")
                        success, message = self.set_video_source("test", "")
                        if not success:
                            print(f"  WARNING: Failed to switch to test: {message}")

            # Apply resolution and FPS if specified
            if "resolution" in defaults and "camera_fps" in defaults:
                resolution = tuple(defaults["resolution"])
                fps = defaults["camera_fps"]
                if self.source_type == "camera":
                    self.set_camera_resolution(resolution[0], resolution[1], fps if fps else 30, "MJPG")
                    print(f"  Applied resolution: {resolution[0]}x{resolution[1]}@{fps}fps")

            # Apply basic camera settings
            if "zoom" in defaults:
                self.zoom = defaults["zoom"]
                print(f"  Applied zoom: {self.zoom}")

            if "pan_x" in defaults and "pan_y" in defaults:
                self.pan_x = defaults["pan_x"]
                self.pan_y = defaults["pan_y"]
                print(f"  Applied pan: ({self.pan_x}, {self.pan_y})")

            if "rotation" in defaults:
                self.rotation = defaults["rotation"]
                print(f"  Applied rotation: {self.rotation}")

            if "target_detection_enabled" in defaults:
                self.target_detector.set_detection_enabled(defaults["target_detection_enabled"])
                print(f"  Applied target detection: {defaults['target_detection_enabled']}")

            if "perspective_correction_enabled" in defaults:
                self.perspective_correction_enabled = defaults["perspective_correction_enabled"]
                print(f"  Applied perspective correction: {defaults['perspective_correction_enabled']}")

            if "detector_type" in defaults:
                detector_type = defaults["detector_type"]
                success, message = self.set_detector_type(detector_type)
                if success:
                    print(f"  Applied detector type: {detector_type}")
                else:
                    print(f"  WARNING: Failed to set detector type: {message}")

            if "yolo_conf_threshold" in defaults:
                self.yolo_conf_threshold = defaults["yolo_conf_threshold"]
                print(f"  Applied YOLO confidence threshold: {self.yolo_conf_threshold}")

            if "yolo_model_path" in defaults and defaults["yolo_model_path"]:
                model_path = defaults["yolo_model_path"]
                success, message = self.set_yolo_model(model_path)
                if success:
                    print(f"  Applied YOLO model: {model_path}")
                else:
                    print(f"  WARNING: Failed to load YOLO model: {message}")

            if "continuous_detection" in defaults:
                self.continuous_detection = defaults["continuous_detection"]
                print(f"  Applied continuous detection: {self.continuous_detection}")

            if "debug_mode" in defaults:
                self.target_detector.set_debug_mode(defaults["debug_mode"])
                print(f"  Applied debug mode: {defaults['debug_mode']}")

            # Apply V4L2 camera controls if available and camera is open
            if "camera_controls" in defaults and self.cached_camera_controls.get("available"):
                if self.cap and self.cap.isOpened():
                    success_count = 0
                    total_count = 0

                    for name, value in defaults["camera_controls"].items():
                        if name in self.cached_camera_controls["controls"]:
                            total_count += 1
                            success, message = self.set_camera_control(name, value)
                            if success:
                                success_count += 1
                            else:
                                print(f"  Failed to apply default for {name}: {message}")

                    print(f"  Applied {success_count}/{total_count} V4L2 camera controls")

            return True, "Applied default settings"

        except Exception as e:
            print(f"Error loading defaults: {e}")
            import traceback

            traceback.print_exc()
            return False, f"Error loading defaults: {str(e)}"

    def get_camera_formats(self):
        """Get available formats and resolutions for the current camera"""
        if self.source_type != "camera":
            return {"available": False, "formats": []}

        # Find the current camera in available sources
        camera_source = None
        for camera in self.available_sources.get("cameras", []):
            if camera["index"] == self.camera_index:
                camera_source = camera
                break

        if not camera_source or "controls" not in camera_source:
            return {"available": False, "formats": []}

        formats_data = camera_source["controls"].get("formats", [])

        # Convert to UI-friendly format
        resolution_options = []

        for format_info in formats_data:
            format_name = format_info.get("name", "Unknown")
            format_desc = format_info.get("description", "")

            for res_info in format_info.get("resolutions", []):
                size = res_info.get("size", "")
                framerates = res_info.get("framerates", [])

                if size and framerates:
                    # Create options for each framerate
                    for fps in sorted(framerates, reverse=True):  # Sort highest fps first
                        option = {
                            "value": f"{size}@{fps:.0f}fps_{format_name}",
                            "label": f"{size} @ {fps:.0f}fps ({format_name})",
                            "width": int(size.split("x")[0]) if "x" in size else 0,
                            "height": int(size.split("x")[1]) if "x" in size else 0,
                            "fps": fps,
                            "format": format_name,
                            "format_desc": format_desc,
                        }
                        resolution_options.append(option)

        # Sort by resolution (width * height) and then by fps
        resolution_options.sort(key=lambda x: (x["width"] * x["height"], -x["fps"]))

        return {"available": True, "formats": formats_data, "resolution_options": resolution_options}

    def calibrate_perspective(
        self, method="auto", pattern_size=(9, 6), iterative=True, target_circularity=0.95, max_iterations=3
    ):
        """Perform perspective calibration using current frame

        Args:
            method: Calibration method - 'auto', 'ellipse', or 'checkerboard'
            pattern_size: For checkerboard method, tuple of (columns, rows) of internal corners
            iterative: Use iterative refinement for ellipse method
            target_circularity: Target circularity for iterative refinement (0.0-1.0)
            max_iterations: Maximum number of iterations for refinement

        Returns:
            success: True if calibration succeeded
            message: Status message
        """
        try:
            with self.lock:
                # Use raw frame (without target detection overlays) for calibration
                if self.raw_frame is not None:
                    # Validate frame before calibration
                    if not self._is_valid_frame(self.raw_frame):
                        return False, "Current frame is invalid for calibration"

                    print(
                        f"DEBUG: Starting perspective calibration ({method}, iterative={iterative}, "
                        f"target_circularity={target_circularity}, max_iterations={max_iterations}) with raw frame..."
                    )
                    print(f"DEBUG: Frame shape: {self.raw_frame.shape}, dtype: {self.raw_frame.dtype}")

                    # Make a copy to ensure memory safety
                    frame_copy = self.raw_frame.copy()

                    # Ensure proper data type for calibration
                    if frame_copy.dtype != "uint8":
                        frame_copy = frame_copy.astype("uint8")

                    # Use perspective.py directly for calibration - it handles all matrix storage internally
                    success, message = self.perspective.calibrate_perspective(
                        frame_copy,
                        method=method,
                        pattern_size=pattern_size,
                        iterative=iterative,
                        target_circularity=target_circularity,
                        max_iterations=max_iterations,
                    )
                    print(f"DEBUG: Calibration result: {success}, {message}")

                    # Debug frame handling can be added later if needed

                    return success, message
                else:
                    return False, "No raw frame available for calibration"
        except Exception as e:
            print(f"ERROR in calibrate_perspective: {e}")
            import traceback

            traceback.print_exc()
            return False, f"Calibration failed with error: {str(e)}"

    def _detect_available_sources(self):
        """Detect available cameras and video files using V4L2"""
        print("DEBUG: Detecting available sources...")

        # Always add test pattern as the first available source
        self.available_sources["test"] = [{"id": "test_pattern", "name": "Test Pattern (Default)"}]

        # Use V4L2 to detect cameras properly
        camera_count = 0
        detected_cameras = self._detect_v4l2_cameras()

        for camera_info in detected_cameras:
            # Detect and cache camera controls during enumeration
            controls_info = self._detect_camera_controls(camera_info["index"])

            self.available_sources["cameras"].append(
                {
                    "id": f"camera_{camera_info['index']}",
                    "name": camera_info["name"],
                    "index": camera_info["index"],
                    "device_path": camera_info["device_path"],
                    "driver": camera_info.get("driver", "unknown"),
                    "controls": controls_info,
                }
            )
            camera_count += 1

        # Check for video files in samples directory
        video_count = 0
        if os.path.exists(self.samples_dir):
            try:
                for filename in os.listdir(self.samples_dir):
                    if filename.lower().endswith((".avi", ".mp4", ".mov", ".mkv", ".wmv")):
                        filepath = os.path.join(self.samples_dir, filename)
                        self.available_sources["videos"].append(
                            {"id": f"video_{filename}", "name": filename, "path": filepath}
                        )
                        video_count += 1
            except Exception as e:
                print(f"DEBUG: Error scanning video directory: {e}")

        print(f"DEBUG: Found test pattern, {camera_count} cameras and {video_count} video files")

    def _detect_v4l2_cameras(self):
        """Detect cameras using V4L2 enumeration"""
        cameras = []

        try:
            # Method 1: Use v4l2-ctl to list devices
            result = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, timeout=5)

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

        for line in v4l2_output.split("\n"):
            original_line = line
            line = line.strip()
            if not line:
                continue

            if original_line.startswith("\t/dev/video"):
                # This is a device path (indented with tab)
                if current_device:
                    device_path = line
                    # Extract index from /dev/videoX
                    try:
                        index = int(device_path.split("video")[1])
                        cameras.append(
                            {
                                "index": index,
                                "name": current_device["name"],
                                "device_path": device_path,
                                "driver": current_device.get("driver", "unknown"),
                            }
                        )
                    except (ValueError, IndexError):
                        continue
            elif not original_line.startswith("\t") and ":" in line:
                # This is a device description line (not indented)
                # Format: "device_name (driver_info):"
                device_line = line.rstrip(":")

                if "(" in device_line and ")" in device_line:
                    # Extract name and driver
                    name_part = device_line.split("(")[0].strip()
                    driver_part = device_line.split("(")[1].split(")")[0].strip()
                else:
                    # No driver info in parentheses
                    name_part = device_line
                    driver_part = "unknown"

                current_device = {"name": name_part, "driver": driver_part}

        # Filter to only include actual camera devices (not platform devices)
        camera_devices = []
        for cam in cameras:
            # Skip platform devices, codec devices, etc.
            if any(skip in cam["name"].lower() for skip in ["pisp", "hevc", "codec", "platform"]):
                continue
            # Include USB cameras and similar, or devices with "camera" in name
            if (
                any(include in cam["name"].lower() for include in ["camera", "usb", "webcam"])
                or "usb-" in cam["driver"]
                or "camera" in cam["driver"].lower()
            ):
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
                    result = subprocess.run(
                        ["v4l2-ctl", f"--device={device_path}", "--info"], capture_output=True, text=True, timeout=2
                    )

                    if result.returncode == 0:
                        # Parse device info
                        device_name = f"Camera {i}"
                        driver = "unknown"

                        for line in result.stdout.split("\n"):
                            if "Card type" in line:
                                device_name = line.split(":")[1].strip()
                            elif "Driver name" in line:
                                driver = line.split(":")[1].strip()

                        # Skip non-camera devices
                        if any(skip in device_name.lower() for skip in ["pisp", "hevc", "codec"]):
                            continue

                        cameras.append({"index": i, "name": device_name, "device_path": device_path, "driver": driver})

                except Exception as e:
                    print(f"DEBUG: Error checking {device_path}: {e}")
                    continue

        return cameras

    def _detect_camera_controls(self, camera_index):
        """Detect camera controls and capabilities for a specific camera"""
        controls_info = {"available": False, "controls": {}, "formats": [], "error": None}

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
                    controls_info["available"] = True

                    # Cache control information
                    for control_name in available_controls:
                        control_data = temp_controls.get_control_info(control_name)
                        current_value = temp_controls.get(control_name)

                        controls_info["controls"][control_name] = {
                            "current": current_value,
                            "min": control_data.min_value,
                            "max": control_data.max_value,
                            "default": control_data.default_value,
                            "type": control_data.control_type,
                        }

                    print(f"DEBUG: Cached {len(available_controls)} controls for camera {camera_index}")
                else:
                    print(f"DEBUG: No controls found for camera {camera_index}")

            # Detect available formats using v4l2-ctl
            try:
                device_path = f"/dev/video{camera_index}"
                result = subprocess.run(
                    ["v4l2-ctl", f"--device={device_path}", "--list-formats-ext"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    formats = self._parse_v4l2_formats(result.stdout)
                    controls_info["formats"] = formats
                    print(f"DEBUG: Found {len(formats)} formats for camera {camera_index}")

            except Exception as e:
                print(f"DEBUG: Could not detect formats for camera {camera_index}: {e}")

            # Close temporary camera
            temp_controls.close()

        except Exception as e:
            controls_info["error"] = str(e)
            print(f"DEBUG: Error detecting controls for camera {camera_index}: {e}")

        return controls_info

    def _parse_v4l2_formats(self, format_output):
        """Parse v4l2-ctl --list-formats-ext output with frame rates"""
        formats = []
        current_format = None
        current_resolution = None

        for line in format_output.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Format line: [0]: 'MJPG' (Motion-JPEG, compressed)
            if line.startswith("[") and "]:" in line:
                # Save previous format
                if current_format and current_format.get("resolutions"):
                    formats.append(current_format)

                if "MJPG" in line or "YUYV" in line:
                    format_name = line.split("'")[1] if "'" in line else "unknown"
                    format_desc = line.split("(")[1].split(")")[0] if "(" in line else "unknown"
                    current_format = {"name": format_name, "description": format_desc, "resolutions": []}
                else:
                    current_format = None

            # Resolution line: Size: Discrete 640x480
            elif line.startswith("Size: Discrete") and current_format:
                resolution = line.split("Discrete ")[1] if "Discrete " in line else ""
                if "x" in resolution:
                    current_resolution = {"size": resolution, "framerates": []}
                    current_format["resolutions"].append(current_resolution)

            # Frame rate line: Interval: Discrete 0.033s (30.000 fps)
            elif line.startswith("Interval: Discrete") and current_resolution:
                if "fps)" in line:
                    # Extract fps from line like "Interval: Discrete 0.033s (30.000 fps)"
                    fps_part = line.split("(")[1].split(" fps")[0] if "(" in line and " fps" in line else ""
                    try:
                        fps = float(fps_part)
                        if fps not in current_resolution["framerates"]:
                            current_resolution["framerates"].append(fps)
                    except ValueError:
                        continue

        # Add the last format if it exists
        if current_format and current_format.get("resolutions"):
            formats.append(current_format)

        return formats

    def get_available_sources(self):
        """Get list of available cameras and video files (cached)"""
        return self.available_sources

    def set_video_source(self, source_type, source_id):
        """Change video source (camera or video file) - SAFE VERSION"""
        try:
            print(f"DEBUG: Requesting source change to {source_type}: {source_id}")

            # Set flag to prevent frame access during switching
            self.switching_source = True

            # For camera switching, use a safer approach that avoids OpenCV crashes
            if source_type == "camera":
                camera_index = int(source_id.split("_")[1])

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

            # Stop current capture and wait for threads to finish
            print("DEBUG: Stopping current capture...")
            self.stop(timeout=3.0)

            # Clear frames to prevent serving stale data
            with self.lock:
                self.frame = None
                self.raw_frame = None

            # Additional settling time for device cleanup
            print("DEBUG: Waiting for device cleanup...")
            time.sleep(0.5)

            # Reset all state
            self.paused = False
            self.step_frame = False
            self.current_frame_number = 0
            self.display_frame_number = 0
            self.total_frames = 0
            self.frame_buffer.clear()

            # Configure new source
            if source_type == "camera":
                camera_index = int(source_id.split("_")[1])
                self.camera_index = camera_index
                self.source_type = "camera"
                self.video_file = None
                self.native_video_resolution = None
                self.native_video_fps = None
                print(f"DEBUG: Configured for camera {camera_index}")

            elif source_type == "video":
                filename = source_id.replace("video_", "", 1)
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

            # Clear switching flag once done
            self.switching_source = False

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

            # Clear switching flag on error
            self.switching_source = False

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
            "resolution": self.resolution,
            "actual_resolution": current_resolution,
            "actual_fps": current_fps,
            "camera_fps": self.camera_fps,  # Configured FPS (None = default)
            "zoom": self.zoom,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y,
            "rotation": self.rotation,
            "running": self.running,
            "captures_dir": self.captures_dir,
            "source_type": self.source_type,
            "current_source": current_source,
            "perspective_correction_enabled": self.perspective_correction_enabled,
            "perspective_correction_method": "ellipse-to-circle"
            if self.perspective.saved_ellipse_data
            else "matrix-based",
            "is_video_mode": self.source_type == "video",
            "native_video_resolution": self.native_video_resolution,
            "native_video_fps": self.native_video_fps,
        }

        # Add target detection status
        target_status = self.target_detector.get_detection_status()
        status.update(target_status)

        # Add playback control status
        playback_status = self.get_playback_info()
        status.update(playback_status)

        # Add camera controls status
        camera_controls_status = self.get_camera_controls()
        status["camera_controls"] = camera_controls_status

        # Add recording status
        recording_status = self.get_recording_status()
        status["recording"] = recording_status["recording"]
        status["recording_filename"] = recording_status["filename"]
        status["recording_duration"] = recording_status["duration"]

        return status

    def stop(self, timeout=3.0):
        """Stop camera capture and wait for all threads to finish

        Args:
            timeout: Maximum time to wait for threads to stop (seconds)
        """
        print("DEBUG: Stopping camera capture...")
        self.running = False

        # Stop any ongoing recording
        if self.recording:
            try:
                self.stop_recording()
            except Exception as e:
                print(f"WARNING: Error stopping recording: {e}")

        # Stop the threaded capture system (this waits for threads)
        if self.capture_system:
            try:
                print("DEBUG: Stopping capture system...")
                self.capture_system.stop(timeout=timeout)
                self.capture_system = None
            except Exception as e:
                print(f"WARNING: Error stopping capture system: {e}")

        # Additional safety wait for processing loop to finish
        # The processing loop checks self.running flag
        time.sleep(0.2)

        # Release capture device
        if self.cap:
            try:
                print("DEBUG: Releasing capture device...")
                self.cap.release()
                self.cap = None
                print("DEBUG: Capture device released")
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


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    print("🎯 Starting Raspberry Pi Camera Streaming System...")
    print("🔍 Faulthandler will catch any segmentation faults")

    try:
        # Initialize camera controller
        print("DEBUG: Creating CameraController...")
        camera_controller = CameraController(camera_index=0)
        print("DEBUG: CameraController created successfully")

        # Load and apply defaults
        print("DEBUG: Loading defaults...")
        success, message = camera_controller.load_camera_defaults()
        if success:
            print(f"✅ {message}")
        else:
            print(f"⚠️  {message}")

        print("DEBUG: Starting capture...")
        if not camera_controller.start_capture():
            print("❌ Failed to initialize camera")
            return
        print("✅ Camera initialized successfully")

        # Start unified HTTP server
        print("DEBUG: Creating HTTP server...")
        # Create handler factory that binds camera_controller to StreamingHandler
        handler_factory = partial(StreamingHandler, camera_controller)
        server = ThreadingHTTPServer(("0.0.0.0", 8088), handler_factory)
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


if __name__ == "__main__":
    main()
