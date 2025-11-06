#!/usr/bin/env python3
"""
Camera HTTP Streaming Server
Provides MJPEG stream from camera on HTTP port 8088
"""

import cv2
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import io


class CameraStream:
    def __init__(self, camera_index=0, resolution=(640, 480)):
        self.camera_index = camera_index
        self.resolution = resolution
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start_capture(self):
        """Initialize and start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Set buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        """Continuous frame capture loop"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()
            time.sleep(0.03)  # ~30 FPS

    def get_frame(self):
        """Get the latest frame as JPEG bytes"""
        with self.lock:
            if self.frame is not None:
                _, buffer = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return buffer.tobytes()
        return None

    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                while True:
                    frame_data = camera_stream.get_frame()
                    if frame_data:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame_data)))
                        self.end_headers()
                        self.wfile.write(frame_data)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.03)
            except Exception as e:
                print(f"Client disconnected: {e}")
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress default logging
        pass


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    global camera_stream

    # Initialize camera stream
    try:
        camera_stream = CameraStream(camera_index=0, resolution=(640, 480))
        camera_stream.start_capture()
        print("Camera initialized successfully")
    except Exception as e:
        print(f"Failed to initialize camera: {e}")
        return

    # Start streaming server
    server = ThreadingHTTPServer(('0.0.0.0', 8088), StreamingHandler)
    print("Camera stream server starting on port 8088...")
    print("Stream URL: http://localhost:8088/stream.mjpg")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down camera stream server...")
        camera_stream.stop()
        server.shutdown()


if __name__ == '__main__':
    main()