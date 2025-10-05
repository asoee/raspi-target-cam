#!/usr/bin/env python3
"""
Example of integrating ThreadedCaptureSystem into CameraController.

This shows how to modify the existing CameraController to use threaded capture.
"""

import cv2
import threading
import time
from threaded_capture import ThreadedCaptureSystem


class CameraControllerThreaded:
    """
    Modified CameraController using threaded capture system.

    Key changes from original:
    1. Frame reading happens in separate thread (FrameReader)
    2. Video recording happens in separate thread (VideoWriter)
    3. Main thread only handles transformations and serving frames
    """

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.video_file = None
        self.source_type = "camera"
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

        # Threaded capture system
        self.capture_system = None

        # Camera settings
        self.resolution = (2592, 1944)
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.rotation = 270

    def start_capture(self):
        """Initialize and start threaded camera capture"""
        try:
            print(f"Initializing {self.source_type} capture...")

            if self.source_type == "camera":
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap or not self.cap.isOpened():
                    raise RuntimeError(f"Could not open camera {self.camera_index}")

                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            elif self.source_type == "video":
                if not self.video_file:
                    raise RuntimeError("No video file specified")
                self.cap = cv2.VideoCapture(self.video_file)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Could not open video file: {self.video_file}")

            # Create and start threaded capture system
            self.capture_system = ThreadedCaptureSystem(
                self.cap,
                source_type=self.source_type,
                buffer_size=100
            )
            self.capture_system.start()

            self.running = True

            # Start processing thread (for transformations, detection, etc.)
            threading.Thread(target=self._processing_loop, daemon=True).start()

            print("Threaded capture system started")
            return True

        except Exception as e:
            print(f"Failed to initialize capture: {e}")
            return False

    def _processing_loop(self):
        """
        Processing loop - runs in separate thread.

        This thread:
        1. Gets latest frame from capture system
        2. Applies transformations (rotation, zoom, pan, perspective)
        3. Runs detection algorithms
        4. Updates the display frame
        """
        while self.running:
            try:
                # Get latest frame from capture system
                raw_frame = self.capture_system.get_latest_frame()

                if raw_frame is not None:
                    # Apply transformations
                    processed_frame = self._apply_transformations(raw_frame)

                    # Update display frame (thread-safe)
                    with self.lock:
                        self.frame = processed_frame.copy()

                # Sleep to avoid busy-waiting
                time.sleep(0.01)  # 100 FPS max processing rate

            except Exception as e:
                print(f"ERROR in _processing_loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _apply_transformations(self, frame):
        """Apply rotation, zoom, pan transformations to frame"""
        # Apply rotation
        if self.rotation != 0:
            frame = self._rotate_frame(frame, self.rotation)

        # Apply zoom and pan
        if self.zoom != 1.0 or self.pan_x != 0 or self.pan_y != 0:
            frame = self._zoom_and_pan(frame)

        return frame

    def _rotate_frame(self, frame, angle):
        """Rotate frame by specified angle"""
        if angle == 0:
            return frame
        elif angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # For arbitrary angles, use getRotationMatrix2D
            height, width = frame.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(frame, matrix, (width, height))

    def _zoom_and_pan(self, frame):
        """Apply zoom and pan to frame"""
        height, width = frame.shape[:2]

        # Calculate crop region based on zoom
        crop_width = int(width / self.zoom)
        crop_height = int(height / self.zoom)

        # Apply pan offset
        center_x = width // 2 + self.pan_x
        center_y = height // 2 + self.pan_y

        # Calculate crop coordinates
        x1 = max(0, center_x - crop_width // 2)
        y1 = max(0, center_y - crop_height // 2)
        x2 = min(width, x1 + crop_width)
        y2 = min(height, y1 + crop_height)

        # Crop and resize
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (width, height))

    def start_recording(self, filename):
        """Start video recording in separate thread"""
        if self.capture_system:
            # Recording happens automatically from the frame buffer
            return self.capture_system.start_recording(
                filename,
                fps=30,
                frame_size=None  # Auto-detect from frame
            )
        return False

    def stop_recording(self):
        """Stop video recording"""
        if self.capture_system:
            self.capture_system.stop_recording()

    def get_frame(self):
        """Get current processed frame (thread-safe)"""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

    def stop(self):
        """Stop all capture and processing"""
        self.running = False
        if self.capture_system:
            self.capture_system.stop()
        if self.cap:
            self.cap.release()


# Example usage
if __name__ == "__main__":
    print("=== Threaded Camera Capture Example ===\n")

    # Create controller
    controller = CameraControllerThreaded(camera_index=0)

    # Start capture
    if controller.start_capture():
        print("Capture started successfully")

        # Let it run for 5 seconds
        print("Capturing frames for 5 seconds...")
        time.sleep(5)

        # Start recording
        print("\nStarting recording...")
        controller.start_recording("test_recording.avi")

        # Record for 10 seconds
        print("Recording for 10 seconds...")
        time.sleep(10)

        # Stop recording
        print("\nStopping recording...")
        controller.stop_recording()

        # Continue capturing for 3 more seconds
        print("Capturing for 3 more seconds...")
        time.sleep(3)

        # Stop everything
        print("\nStopping capture...")
        controller.stop()
        print("Done!")
    else:
        print("Failed to start capture")
