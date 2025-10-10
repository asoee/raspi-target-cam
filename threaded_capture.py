#!/usr/bin/env python3
"""
Threaded frame capture and video recording system.

This module provides a thread-safe architecture for:
1. Frame Reader Thread - Continuously reads frames from camera/video
2. Video Writer Thread - Records frames from buffer to video file
3. Shared frame buffer - Thread-safe queue for frame distribution
"""

import cv2
import threading
import queue
import time
from datetime import datetime
from collections import deque
from camera_settings import (
    CameraSettings,
    CameraCommand,
    ThreadSafeCameraSettings,
    ApplySettingsCommand,
    SeekCommand
)


class FrameBuffer:
    """Thread-safe frame buffer with multiple consumer support"""

    def __init__(self, maxsize=100):
        self.maxsize = maxsize
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_count = 0
        self.frame_change_counter = 0  # Increments each time a new frame is added

    def put(self, frame):
        """Add frame to buffer (thread-safe)"""
        with self.lock:
            self.buffer.append(frame.copy())
            self.latest_frame = frame.copy()
            self.frame_count += 1
            self.frame_change_counter += 1  # Increment change counter

    def get_latest(self):
        """
        Get the most recent frame with change counter (non-blocking).

        Returns:
            Tuple of (frame, change_counter) where change_counter increments
            each time a new frame is added to the buffer
        """
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.frame_change_counter
            return None, 0

    def get_buffer_copy(self):
        """Get a copy of the entire buffer"""
        with self.lock:
            return list(self.buffer)

    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
            self.latest_frame = None


class FrameReader(threading.Thread):
    """Thread that continuously reads frames from camera/video source"""

    def __init__(self, cap, frame_buffer, source_type="camera", camera_index=0,
                 initial_settings=None, test_frame_generator=None):
        super().__init__(daemon=True)
        self.cap = cap
        self.frame_buffer = frame_buffer
        self.source_type = source_type
        self.camera_index = camera_index
        self.running = False
        self.paused = False
        self.fps = 30
        self.frame_time = 1.0 / self.fps

        # Command queue for camera control (only this thread accesses capture device)
        self.command_queue = queue.Queue()

        # Camera settings (thread-safe)
        self.settings = ThreadSafeCameraSettings(initial_settings or CameraSettings())

        # Video playback state (for seek support)
        self.current_frame_number = 0
        self.total_frames = 0
        self.loop_video = True  # Loop video when it reaches end

        # Test pattern support
        self.test_frame_generator = test_frame_generator
        self.test_frame_counter = 0

        # Get video properties if available
        if self.cap is not None and self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.fps = fps
                self.frame_time = 1.0 / fps

            # Get video file properties
            if self.source_type == "video":
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"FrameReader: Video has {self.total_frames} frames")

    def run(self):
        """Main frame reading loop"""
        self.running = True
        print(f"FrameReader: Starting capture loop ({self.source_type} @ {self.fps:.1f} FPS)")

        frame_count = 0
        start_time = time.time()

        while self.running:
            try:
                # Process commands from queue (non-blocking)
                self._process_commands()

                # Skip reading if paused (but keep thread alive)
                if self.paused:
                    time.sleep(0.1)
                    continue

                # Handle test pattern mode
                if self.source_type == "test":
                    frame = self._generate_test_frame()
                    if frame is not None:
                        self.frame_buffer.put(frame)
                    time.sleep(0.2)  # 5 FPS for test pattern
                    continue

                # Check if capture is available
                if not self.cap or not self.cap.isOpened():
                    print("FrameReader: Capture device not available")
                    time.sleep(0.1)
                    continue

                # Read frame
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    # Validate frame
                    if len(frame.shape) >= 2 and frame.shape[0] > 0 and frame.shape[1] > 0:
                        # Update frame number for video files
                        if self.source_type == "video":
                            self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                        # Add to buffer
                        self.frame_buffer.put(frame)
                        frame_count += 1

                        # Debug output every 100 frames
                        if frame_count % 100 == 0:
                            elapsed = time.time() - start_time
                            actual_fps = frame_count / elapsed if elapsed > 0 else 0
                            print(f"FrameReader: {frame_count} frames captured ({actual_fps:.1f} FPS)")
                else:
                    # Handle end of video file
                    if self.source_type == "video":
                        if self.loop_video:
                            print("FrameReader: Video ended, looping...")
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.current_frame_number = 0
                        else:
                            print("FrameReader: Video ended, pausing...")
                            self.paused = True
                        continue
                    else:
                        print("FrameReader: Failed to read frame")

                # Sleep to maintain target FPS
                time.sleep(self.frame_time)

            except Exception as e:
                print(f"FrameReader ERROR: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

        print("FrameReader: Capture loop stopped")

    def stop(self):
        """Stop the frame reader thread"""
        self.running = False

    def pause(self):
        """Pause frame reading"""
        self.paused = True

    def resume(self):
        """Resume frame reading"""
        self.paused = False

    def send_command(self, command: CameraCommand) -> bool:
        """
        Send a command to be executed by the frame reader thread.

        Args:
            command: CameraCommand instance to execute

        Returns:
            True if command was queued successfully
        """
        try:
            self.command_queue.put(command, block=False)
            return True
        except queue.Full:
            print("FrameReader: Command queue full, dropping command")
            return False

    def _process_commands(self):
        """Process all pending commands from the queue (called by frame reader thread)"""
        processed = 0
        max_commands_per_cycle = 10  # Prevent command processing from blocking frame capture

        while processed < max_commands_per_cycle:
            try:
                # Non-blocking get
                command = self.command_queue.get_nowait()

                # Execute command on capture device (only this thread accesses cap)
                if self.cap and self.cap.isOpened():
                    command.execute(self.cap)

                processed += 1

            except queue.Empty:
                break
            except Exception as e:
                print(f"FrameReader: Error executing command: {e}")
                import traceback
                traceback.print_exc()

    def get_settings(self) -> CameraSettings:
        """Get current camera settings (thread-safe)"""
        return self.settings.get()

    def apply_settings(self, new_settings: CameraSettings):
        """
        Apply new camera settings via command queue.

        Args:
            new_settings: CameraSettings object with desired settings
        """
        # Update stored settings
        self.settings.update(new_settings)

        # Create device path for v4l2 commands
        device_path = f"/dev/video{self.camera_index}" if self.source_type == "camera" else None

        # Send command to apply settings
        command = ApplySettingsCommand(new_settings, device_path)
        self.send_command(command)

    def seek_to_frame(self, frame_number: int) -> bool:
        """
        Seek to specific frame in video file.

        Args:
            frame_number: Target frame number

        Returns:
            True if command was queued successfully
        """
        from camera_settings import SeekCommand
        return self.send_command(SeekCommand(frame_number))

    def step_forward(self) -> bool:
        """Step one frame forward (for video files)"""
        if self.source_type != "video":
            return False
        target_frame = self.current_frame_number + 1
        if target_frame >= self.total_frames:
            return False
        return self.seek_to_frame(target_frame)

    def step_backward(self) -> bool:
        """Step one frame backward (for video files)"""
        if self.source_type != "video":
            return False
        target_frame = self.current_frame_number - 1
        if target_frame < 0:
            return False
        return self.seek_to_frame(target_frame)

    def get_playback_position(self) -> dict:
        """
        Get current playback position.

        Returns:
            Dictionary with current_frame, total_frames, progress
        """
        if self.source_type == "video":
            progress = (self.current_frame_number / self.total_frames * 100) if self.total_frames > 0 else 0
            return {
                'current_frame': self.current_frame_number,
                'total_frames': self.total_frames,
                'progress': progress
            }
        return {'current_frame': 0, 'total_frames': 0, 'progress': 0}

    def _generate_test_frame(self):
        """Generate test pattern frame"""
        if self.test_frame_generator:
            # Use provided generator function
            return self.test_frame_generator(self.test_frame_counter)
        else:
            # Simple default test pattern
            import numpy as np
            width, height = 1920, 1080
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Draw grid
            for i in range(0, width, 100):
                cv2.line(frame, (i, 0), (i, height), (50, 50, 50), 1)
            for i in range(0, height, 100):
                cv2.line(frame, (0, i), (width, i), (50, 50, 50), 1)

            # Draw center crosshair
            cv2.line(frame, (width//2, 0), (width//2, height), (0, 255, 0), 2)
            cv2.line(frame, (0, height//2), (width, height//2), (0, 255, 0), 2)

            # Add text
            cv2.putText(frame, "TEST PATTERN", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(frame, f"Frame: {self.test_frame_counter}", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            self.test_frame_counter += 1
            return frame


class VideoWriter(threading.Thread):
    """Thread that writes frames from buffer to video file"""

    def __init__(self, frame_buffer, output_file, fourcc, fps, frame_size, metadata=None):
        super().__init__(daemon=True)
        self.frame_buffer = frame_buffer
        self.output_file = output_file
        self.fourcc = fourcc
        self.fps = fps
        self.frame_size = frame_size  # (width, height)
        self.metadata = metadata or {}  # Recording metadata
        self.running = False
        self.writer = None
        self.frames_written = 0
        self.start_time = None
        self.codec_name = None

    def run(self):
        """Main video writing loop"""
        try:
            # Create video writer
            self.writer = cv2.VideoWriter(
                self.output_file,
                self.fourcc,
                self.fps,
                self.frame_size
            )

            if not self.writer.isOpened():
                print(f"VideoWriter ERROR: Could not open video writer for {self.output_file}")
                return

            self.running = True
            self.start_time = time.time()
            print(f"VideoWriter: Started recording to {self.output_file} ({self.frame_size[0]}x{self.frame_size[1]} @ {self.fps} FPS)")

            last_frame_time = time.time()
            frame_interval = 1.0 / self.fps

            while self.running:
                try:
                    # Get latest frame from buffer
                    frame = self.frame_buffer.get_latest()

                    if frame is not None:
                        # Ensure frame is correct size
                        frame_height, frame_width = frame.shape[:2]
                        if frame_width != self.frame_size[0] or frame_height != self.frame_size[1]:
                            frame = cv2.resize(frame, self.frame_size)

                        # Ensure frame is in correct format (BGR, 3 channels)
                        if len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        elif frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                        # Ensure contiguous memory and uint8
                        if not frame.flags['C_CONTIGUOUS']:
                            frame = frame.copy()
                        if frame.dtype != 'uint8':
                            frame = frame.astype('uint8')

                        # Write frame
                        self.writer.write(frame)
                        self.frames_written += 1

                        # Debug output every 100 frames
                        if self.frames_written % 100 == 0:
                            elapsed = time.time() - self.start_time
                            actual_fps = self.frames_written / elapsed if elapsed > 0 else 0
                            print(f"VideoWriter: {self.frames_written} frames written ({actual_fps:.1f} FPS)")

                    # Maintain target FPS for writing
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)
                    last_frame_time = time.time()

                except Exception as e:
                    print(f"VideoWriter ERROR (frame write): {e}")
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            print(f"VideoWriter ERROR (init): {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    def stop(self):
        """Stop recording and cleanup"""
        self.running = False

    def _cleanup(self):
        """Cleanup video writer resources and save metadata"""
        if self.writer is not None:
            try:
                self.writer.release()
                elapsed = time.time() - self.start_time if self.start_time else 0
                print(f"VideoWriter: Recording stopped. {self.frames_written} frames written in {elapsed:.1f}s")

                # Save metadata as JSON sidecar file
                if self.metadata:
                    self._save_metadata(elapsed)

            except Exception as e:
                print(f"VideoWriter ERROR (cleanup): {e}")
            finally:
                self.writer = None

    def _save_metadata(self, duration):
        """Save recording metadata to JSON sidecar file"""
        import json
        import os

        try:
            # Add recording statistics to metadata
            self.metadata['recording_duration_seconds'] = duration
            self.metadata['frames_written'] = self.frames_written
            self.metadata['actual_fps'] = self.frames_written / duration if duration > 0 else 0

            # Create metadata filename (same as video but .json)
            base_name = os.path.splitext(self.output_file)[0]
            metadata_file = f"{base_name}.json"

            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)

            print(f"VideoWriter: Metadata saved to {metadata_file}")

        except Exception as e:
            print(f"VideoWriter: Failed to save metadata: {e}")


class ThreadedCaptureSystem:
    """
    Orchestrates threaded frame capture and video recording.

    Usage:
        system = ThreadedCaptureSystem(cap)
        system.start()

        # Start recording
        system.start_recording("output.avi")

        # Get frames for display
        frame = system.get_latest_frame()

        # Stop recording
        system.stop_recording()

        # Cleanup
        system.stop()
    """

    def __init__(self, cap, source_type="camera", camera_index=0, buffer_size=100,
                 initial_settings=None, test_frame_generator=None):
        self.cap = cap
        self.source_type = source_type
        self.camera_index = camera_index
        self.frame_buffer = FrameBuffer(maxsize=buffer_size)
        self.frame_reader = None
        self.video_writer = None
        self.recording = False
        self.initial_settings = initial_settings
        self.test_frame_generator = test_frame_generator

    def start(self):
        """Start the frame reading thread"""
        if self.frame_reader is None or not self.frame_reader.is_alive():
            self.frame_reader = FrameReader(
                self.cap,
                self.frame_buffer,
                self.source_type,
                self.camera_index,
                self.initial_settings,
                self.test_frame_generator
            )
            self.frame_reader.start()

    def stop(self):
        """Stop all threads"""
        if self.frame_reader:
            self.frame_reader.stop()
            self.frame_reader = None
        self.stop_recording()

    def pause(self):
        """Pause frame reading"""
        if self.frame_reader:
            self.frame_reader.pause()

    def resume(self):
        """Resume frame reading"""
        if self.frame_reader:
            self.frame_reader.resume()

    def get_latest_frame(self):
        """
        Get the most recent frame with playback position and change counter.

        Returns:
            Tuple of (frame, position_dict) where position_dict contains:
            - current_frame: Frame number (0 for camera, actual frame for video)
            - total_frames: Total frames (0 for camera)
            - source_type: Type of source ('camera', 'video', 'test')
            - change_counter: Increments each time a new frame is captured
        """
        frame, change_counter = self.frame_buffer.get_latest()
        position = self.get_playback_position()
        position['source_type'] = self.source_type
        position['change_counter'] = change_counter
        return frame, position

    def start_recording(self, output_file, fps=30, frame_size=None, metadata=None, codec_priority=None):
        """
        Start recording frames to video file with codec fallback.

        Args:
            output_file: Path to output video file
            fps: Frames per second for video
            frame_size: (width, height) tuple, or None to use latest frame size
            metadata: Dictionary of metadata to save with recording
            codec_priority: List of (codec, extension) tuples to try in order

        Returns:
            Tuple of (success, message, actual_filename)
        """
        if self.recording:
            return False, "Already recording", None

        # Get frame size from latest frame if not specified
        if frame_size is None:
            latest_frame = self.frame_buffer.get_latest()
            if latest_frame is None:
                return False, "No frames available to determine size", None
            frame_size = (latest_frame.shape[1], latest_frame.shape[0])

        # Default codec priority if not specified
        if codec_priority is None:
            codec_priority = [
                ('MJPG', '.mkv'),   # Motion JPEG (stable, low compression)
                ('X264', '.mp4'),   # H.264 (good quality, widely compatible)
                ('avc1', '.mp4'),   # H.264 variant
                ('mp4v', '.mp4'),   # MPEG-4 (fallback)
                ('XVID', '.avi'),   # Xvid (fallback)
            ]

        # Try each codec until one works
        import os
        base_name = os.path.splitext(output_file)[0]
        actual_output_file = None
        actual_codec = None

        for codec, ext in codec_priority:
            try:
                test_output = f"{base_name}{ext}"
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_writer = cv2.VideoWriter(test_output, fourcc, fps, frame_size)

                if test_writer.isOpened():
                    # Success! Use this codec
                    test_writer.release()
                    actual_output_file = test_output
                    actual_codec = codec
                    print(f"VideoWriter: Using codec {codec} -> {test_output}")
                    break
                else:
                    # Failed, clean up and try next
                    test_writer.release()
                    if os.path.exists(test_output):
                        os.remove(test_output)

            except Exception as e:
                print(f"VideoWriter: Codec {codec} failed: {e}")
                continue

        if actual_output_file is None:
            return False, "Failed to initialize video writer with any available codec", None

        # Add codec info to metadata
        if metadata is None:
            metadata = {}
        metadata['recording_fps'] = fps
        metadata['recording_resolution'] = f"{frame_size[0]}x{frame_size[1]}"
        metadata['recording_codec'] = actual_codec

        # Create video writer thread with selected codec
        fourcc = cv2.VideoWriter_fourcc(*actual_codec)
        self.video_writer = VideoWriter(
            self.frame_buffer,
            actual_output_file,
            fourcc,
            fps,
            frame_size,
            metadata
        )
        self.video_writer.codec_name = actual_codec
        self.video_writer.start()
        self.recording = True

        return True, f"Recording started: {os.path.basename(actual_output_file)} @ {fps:.1f} fps", actual_output_file

    def stop_recording(self):
        """Stop video recording"""
        if self.video_writer:
            self.video_writer.stop()
            self.video_writer.join(timeout=2.0)  # Wait for thread to finish
            self.video_writer = None
        self.recording = False

    def is_recording(self):
        """Check if currently recording"""
        return self.recording and self.video_writer is not None and self.video_writer.running

    # ========== Camera Settings API ==========

    def get_settings(self) -> CameraSettings:
        """
        Get current camera settings (thread-safe).

        Returns:
            CameraSettings object with current settings
        """
        if self.frame_reader:
            return self.frame_reader.get_settings()
        return self.initial_settings or CameraSettings()

    def apply_settings(self, new_settings: CameraSettings):
        """
        Apply new camera settings.

        This sends a command to the frame reader thread to update settings.
        The frame reader thread is the only thread that accesses the capture device.

        Args:
            new_settings: CameraSettings object with desired settings
        """
        if self.frame_reader:
            self.frame_reader.apply_settings(new_settings)
        else:
            print("ThreadedCaptureSystem: Cannot apply settings - frame reader not started")

    def update_settings(self, **changes) -> CameraSettings:
        """
        Update specific camera settings.

        Args:
            **changes: Keyword arguments for settings to change
                      (e.g., width=1920, height=1080, fps=60)

        Returns:
            New CameraSettings object with changes applied

        Example:
            system.update_settings(width=1920, height=1080, fps=60)
        """
        current = self.get_settings()
        new_settings = current.copy(**changes)
        self.apply_settings(new_settings)
        return new_settings

    def send_command(self, command: CameraCommand) -> bool:
        """
        Send a custom command to the frame reader thread.

        Args:
            command: CameraCommand instance to execute

        Returns:
            True if command was queued successfully
        """
        if self.frame_reader:
            return self.frame_reader.send_command(command)
        return False

    # ========== Video Playback Controls ==========

    def seek_to_frame(self, frame_number: int) -> bool:
        """
        Seek to specific frame (video files only).

        Args:
            frame_number: Target frame number

        Returns:
            True if seek command was queued
        """
        if self.frame_reader:
            return self.frame_reader.seek_to_frame(frame_number)
        return False

    def step_forward(self) -> bool:
        """Step one frame forward (video files only)"""
        if self.frame_reader:
            return self.frame_reader.step_forward()
        return False

    def step_backward(self) -> bool:
        """Step one frame backward (video files only)"""
        if self.frame_reader:
            return self.frame_reader.step_backward()
        return False

    def get_playback_position(self) -> dict:
        """
        Get current playback position for video files.

        Returns:
            Dictionary with current_frame, total_frames, progress
        """
        if self.frame_reader:
            return self.frame_reader.get_playback_position()
        return {'current_frame': 0, 'total_frames': 0, 'progress': 0}

    def set_loop_video(self, loop: bool):
        """Enable or disable video looping"""
        if self.frame_reader:
            self.frame_reader.loop_video = loop
