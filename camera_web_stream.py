#!/usr/bin/env python3
"""
Unified Camera HTTP Streaming and Web Interface Server
Combines MJPEG camera stream and web interface in a single threaded application
"""

import cv2
import threading
import time
import json
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from target_detection import TargetDetector
from perspective import Perspective


class CameraController:
    """Centralized camera control and streaming"""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.video_file = None  # For video file input
        self.source_type = "camera"  # "camera" or "video"
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

        # Camera settings
        self.resolution = (2592, 1944)
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.rotation = 90  # degrees clockwise (0, 90, 180, 270)

        # Capture settings
        self.captures_dir = "./captures"
        os.makedirs(self.captures_dir, exist_ok=True)

        # Video file settings
        self.samples_dir = "./samples"
        self.video_fps = 30  # Default FPS for video files
        self.video_frame_time = 1.0 / self.video_fps  # Time between frames
        self.native_video_resolution = None  # Native resolution of video file
        self.native_video_fps = None  # Native FPS of video file

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
        self.perspective_correction_enabled = False

    def start_capture(self):
        """Initialize and start camera or video file capture"""
        try:
            if self.source_type == "camera":
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Could not open camera {self.camera_index}")

                # Set initial resolution for camera
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
            self.perspective.load_calibration()
            threading.Thread(target=self._capture_loop, daemon=True).start()
            return True
        except Exception as e:
            print(f"Failed to initialize capture: {e}")
            return False

    def _capture_loop(self):
        """Continuous frame capture and processing loop"""
        while self.running:
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
                # Update current frame number for video files
                if self.source_type == "video":
                    self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                # Apply zoom and pan transformations
                processed_frame = self._apply_transformations(frame)

                # Add to frame buffer (maintain rolling buffer)
                self.frame_buffer.append(processed_frame.copy())
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)

                # Handle stepping
                if self.step_frame:
                    self.step_frame = False
                    if self.paused and self._handle_frame_step():
                        continue

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
        if self.perspective_correction_enabled and False:
            corrected_frame = self.perspective.apply_perspective_correction(frame)
            if corrected_frame is not None:
                frame = corrected_frame
                # For perspective-corrected streams, detect targets on the corrected frame
                # This ensures target overlays are accurate for the corrected view
                frame = self.target_detector.draw_target_overlay(frame, frame_is_corrected=True)
            else:
                # Perspective correction failed, use original frame with target overlay
                frame = self.target_detector.draw_target_overlay(frame)
        else:
            # No perspective correction, use original frame with target overlay
            pass
            # frame = self.target_detector.draw_target_overlay(frame)

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
            frame = cv2.resize(cropped, (w, h))

        return frame

    def _rotate_frame(self, frame, degrees):
        """Rotate frame by specified degrees clockwise"""
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
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -degrees, 1.0)
            return cv2.warpAffine(frame, rotation_matrix, (w, h))

    def get_frame_jpeg(self):
        """Get the latest frame as JPEG bytes"""
        with self.lock:
            if self.frame is not None:
                _, buffer = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return buffer.tobytes()
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
        """Step one frame forward when paused"""
        if self.source_type == "video" and self.paused:
            self.step_direction = 1
            self.step_frame = True
            return True
        return False

    def step_frame_backward(self):
        """Step one frame backward when paused"""
        if self.source_type == "video" and self.paused:
            self.step_direction = -1
            self.step_frame = True
            return True
        return False

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
            'can_step_backward': self.paused and self.pause_buffer_index > 0,
            'can_step_forward': self.paused and self.pause_buffer_index < len(self.frame_buffer) - 1,
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

    def capture_image(self):
        """Capture and save current frame"""
        with self.lock:
            if self.frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                filepath = os.path.join(self.captures_dir, filename)

                cv2.imwrite(filepath, self.frame)
                return filename
        return None

    def calibrate_perspective(self):
        """Perform perspective calibration using current frame"""
        with self.lock:
            if self.frame is not None:
                # Use perspective.py directly for calibration - it handles all matrix storage internally
                success, message = self.perspective.calibrate_perspective(self.frame)

                # Debug frame handling can be added later if needed

                return success, message
            else:
                return False, "No frame available for calibration"


    def get_available_sources(self):
        """Get list of available cameras and video files"""
        sources = {
            'cameras': [],
            'videos': []
        }

        # Check for available cameras (0-5)
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                sources['cameras'].append({
                    'id': f'camera_{i}',
                    'name': f'Camera {i}',
                    'index': i
                })
                cap.release()

        # Check for video files in samples directory
        if os.path.exists(self.samples_dir):
            for filename in os.listdir(self.samples_dir):
                if filename.lower().endswith(('.avi', '.mp4', '.mov', '.mkv', '.wmv')):
                    filepath = os.path.join(self.samples_dir, filename)
                    sources['videos'].append({
                        'id': f'video_{filename}',
                        'name': filename,
                        'path': filepath
                    })

        return sources

    def set_video_source(self, source_type, source_id):
        """Change video source (camera or video file)"""
        try:
            # Stop current capture
            self.stop()
            time.sleep(0.5)  # Wait for capture loop to stop

            if source_type == "camera":
                # Extract camera index from source_id like "camera_0"
                camera_index = int(source_id.split('_')[1])
                self.camera_index = camera_index
                self.source_type = "camera"
                self.video_file = None
                # Clear video-specific properties
                self.native_video_resolution = None
                self.native_video_fps = None
            elif source_type == "video":
                # Extract filename from source_id like "video_filename.avi"
                filename = source_id.replace('video_', '', 1)
                video_path = os.path.join(self.samples_dir, filename)
                if not os.path.exists(video_path):
                    return False, f"Video file not found: {filename}"

                self.source_type = "video"
                self.video_file = video_path
            else:
                return False, f"Invalid source type: {source_type}"

            # Start capture with new source
            if self.start_capture():
                return True, f"Source changed to {source_type}: {source_id}"
            else:
                return False, f"Failed to start capture with new source"

        except Exception as e:
            return False, f"Error changing source: {str(e)}"

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
            'current_source': self.camera_index if self.source_type == "camera" else self.video_file,
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

        return status

    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()


class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP request handler for both streaming and web interface"""

    def do_GET(self):
        """Handle GET requests"""
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
        elif path.endswith('.html') or path.endswith('.css') or path.endswith('.js'):
            self._serve_file(path[1:])  # Remove leading slash
        else:
            self.send_error(404)

    def do_POST(self):
        """Handle POST requests for API calls"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path.startswith('/api/'):
            self._handle_api_request(path)
        else:
            self.send_error(404)

    def _serve_mjpeg_stream(self):
        """Serve MJPEG video stream"""
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            while True:
                frame_data = camera_controller.get_frame_jpeg()
                if frame_data:
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_data)))
                    self.end_headers()
                    self.wfile.write(frame_data)
                    self.wfile.write(b'\r\n')
                time.sleep(0.03)
        except Exception as e:
            print(f"Stream client disconnected: {e}")

    def _serve_debug_stream(self, debug_type='combined'):
        """Serve MJPEG debug stream for specific debug type

        Args:
            debug_type: Type of debug stream ('combined', 'perspective', 'circles', 'corrected')
        """
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
        status = camera_controller.get_status()
        self._send_json_response(status)

    def _serve_api_sources(self):
        """Serve available sources as JSON"""
        sources = camera_controller.get_available_sources()
        self._send_json_response(sources)

    def _handle_api_request(self, path):
        """Handle API requests"""
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

            elif path == '/api/playback_info':
                playback_info = camera_controller.get_playback_info()
                response = {'success': True, 'data': playback_info}

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

    # Initialize camera controller
    camera_controller = CameraController(camera_index=0)
    if not camera_controller.start_capture():
        print("❌ Failed to initialize camera")
        return

    print("✅ Camera initialized successfully")

    # Start unified HTTP server
    server = ThreadingHTTPServer(('0.0.0.0', 8088), StreamingHandler)
    print("🌐 Server starting on port 8088...")
    print("📺 Camera stream: http://localhost:8088/stream.mjpg")
    print("🖥️  Web interface: http://localhost:8088")
    print("🔧 API endpoints: http://localhost:8088/api/")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        camera_controller.stop()
        server.shutdown()


if __name__ == '__main__':
    main()