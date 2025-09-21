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


class CameraController:
    """Centralized camera control and streaming"""

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
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

        # Target detection
        self.target_detector = TargetDetector()

    def start_capture(self):
        """Initialize and start camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera {self.camera_index}")

            # Set initial resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.running = True
            threading.Thread(target=self._capture_loop, daemon=True).start()
            return True
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False

    def _capture_loop(self):
        """Continuous frame capture and processing loop"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Apply zoom and pan transformations
                processed_frame = self._apply_transformations(frame)

                with self.lock:
                    self.frame = processed_frame.copy()
            time.sleep(0.03)  # ~30 FPS

    def _apply_transformations(self, frame):
        """Apply rotation, zoom and pan transformations to frame"""
        # Apply rotation first
        if self.rotation != 0:
            frame = self._rotate_frame(frame, self.rotation)

        # Apply target detection overlay
        frame = self.target_detector.draw_target_overlay(frame)

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
        """Change camera resolution"""
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

    def get_status(self):
        """Get current camera status"""
        status = {
            'resolution': self.resolution,
            'zoom': self.zoom,
            'pan_x': self.pan_x,
            'pan_y': self.pan_y,
            'rotation': self.rotation,
            'running': self.running,
            'captures_dir': self.captures_dir
        }

        # Add target detection status
        target_status = self.target_detector.get_detection_status()
        status.update(target_status)

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
        elif path == '/' or path == '/index.html':
            self._serve_file('camera_interface.html')
        elif path == '/api/status':
            self._serve_api_status()
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

            elif path == '/api/capture':
                filename = camera_controller.capture_image()
                if filename:
                    response = {'success': True, 'message': f'Image captured: {filename}', 'filename': filename}
                else:
                    response = {'success': False, 'message': 'Failed to capture image'}

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
    server = ThreadingHTTPServer(('0.0.0.0', 8080), StreamingHandler)
    print("🌐 Server starting on port 8080...")
    print("📺 Camera stream: http://localhost:8080/stream.mjpg")
    print("🖥️  Web interface: http://localhost:8080")
    print("🔧 API endpoints: http://localhost:8080/api/")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        camera_controller.stop()
        server.shutdown()


if __name__ == '__main__':
    main()