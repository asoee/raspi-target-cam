"""
HTTP request handler for streaming and web interface
"""

import os
import time
import json
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP request handler for both streaming and web interface"""

    def __init__(self, camera_controller, *args, **kwargs):
        """Initialize handler with camera controller reference

        Args:
            camera_controller: CameraController instance to use for all operations
        """
        self.camera_controller = camera_controller
        super().__init__(*args, **kwargs)

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
                controls_info = self.camera_controller.get_camera_controls()
                self._send_json_response({'success': True, 'data': controls_info})
            elif path == '/api/camera_formats':
                formats_info = self.camera_controller.get_camera_formats()
                self._send_json_response({'success': True, 'data': formats_info})
            elif path == '/api/start_recording':
                success, message = self.camera_controller.start_recording()
                self._send_json_response({'success': success, 'message': message})
            elif path == '/api/stop_recording':
                success, message = self.camera_controller.stop_recording()
                self._send_json_response({'success': success, 'message': message})
            elif path == '/api/recording_status':
                recording_status = self.camera_controller.get_recording_status()
                self._send_json_response({'success': True, 'data': recording_status})
            elif path == '/api/list_camera_presets':
                presets = self.camera_controller.list_camera_presets()
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
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            frame_count = 0
            while True:
                try:
                    frame_data = self.camera_controller.get_frame_jpeg()
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
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            while True:
                debug_data = self.camera_controller.get_debug_frame_jpeg(debug_type)
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
        status = self.camera_controller.get_status()
        self._send_json_response(status)

    def _serve_api_sources(self):
        """Serve available sources as JSON"""
        sources = self.camera_controller.get_available_sources()
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
                if self.camera_controller.set_resolution(width, height):
                    response = {'success': True, 'message': f'Resolution set to {width}x{height}'}

            elif path == '/api/zoom':
                zoom = data.get('zoom', 1.0)
                if self.camera_controller.set_zoom(zoom):
                    response = {'success': True, 'message': f'Zoom set to {zoom}x'}

            elif path == '/api/pan':
                pan_x = data.get('pan_x', 0)
                pan_y = data.get('pan_y', 0)
                if self.camera_controller.set_pan(pan_x, pan_y):
                    response = {'success': True, 'message': f'Pan set to ({pan_x}, {pan_y})'}

            elif path == '/api/rotation':
                rotation = data.get('rotation', 0)
                if self.camera_controller.set_rotation(rotation):
                    response = {'success': True, 'message': f'Rotation set to {rotation}°'}
                else:
                    response = {'success': False, 'message': 'Invalid rotation (must be 0, 90, 180, or 270)'}

            elif path == '/api/fps':
                fps = data.get('fps')
                if self.camera_controller.set_fps(fps):
                    if fps is None:
                        response = {'success': True, 'message': 'FPS set to camera default'}
                    else:
                        response = {'success': True, 'message': f'FPS set to {fps}'}
                else:
                    response = {'success': False, 'message': 'Failed to set FPS (camera must be active)'}

            elif path == '/api/target_detection':
                enabled = data.get('enabled', True)
                if self.camera_controller.set_target_detection(enabled):
                    status = "enabled" if enabled else "disabled"
                    response = {'success': True, 'message': f'Target detection {status}'}
                else:
                    response = {'success': False, 'message': 'Failed to set target detection'}

            elif path == '/api/debug_mode':
                enabled = data.get('enabled', False)
                if self.camera_controller.set_debug_mode(enabled):
                    status = "enabled" if enabled else "disabled"
                    response = {'success': True, 'message': f'Debug mode {status}'}
                else:
                    response = {'success': False, 'message': 'Failed to set debug mode'}

            elif path == '/api/debug_type':
                debug_type = data.get('debug_type', 'combined')
                if self.camera_controller.target_detector.set_debug_type(debug_type):
                    response = {'success': True, 'message': f'Debug type set to {debug_type}'}
                else:
                    response = {'success': False, 'message': 'Invalid debug type (must be combined, corners, or circles)'}

            elif path == '/api/force_detection':
                if self.camera_controller.target_detector.force_detection():
                    response = {'success': True, 'message': 'Target re-detection forced successfully'}
                else:
                    response = {'success': False, 'message': 'Failed to force target re-detection'}

            elif path == '/api/calibrate_perspective':
                success, message = self.camera_controller.calibrate_perspective()
                response = {'success': success, 'message': message}

            elif path == '/api/save_calibration':
                # Get current camera resolution and save calibration
                camera_resolution = self.camera_controller.resolution
                success, message = self.camera_controller.perspective.save_calibration(camera_resolution)
                response = {'success': success, 'message': message}

            elif path == '/api/calibration_mode':
                enabled = data.get('enabled', False)
                if self.camera_controller.target_detector.set_calibration_mode(enabled):
                    status = "enabled" if enabled else "disabled"
                    response = {'success': True, 'message': f'Calibration mode {status}'}
                else:
                    response = {'success': False, 'message': 'Failed to set calibration mode'}

            elif path == '/api/capture':
                filename = self.camera_controller.capture_image()
                if filename:
                    response = {'success': True, 'message': f'Image captured: {filename}', 'filename': filename}
                else:
                    response = {'success': False, 'message': 'Failed to capture image'}

            elif path == '/api/set_reference_frame':
                if self.camera_controller.set_reference_frame():
                    response = {'success': True, 'message': 'Reference frame set for bullet hole detection'}
                else:
                    response = {'success': False, 'message': 'Failed to set reference frame'}

            elif path == '/api/detect_bullet_holes':
                success, message = self.camera_controller.detect_bullet_holes()
                response = {'success': success, 'message': message}
                if success and self.camera_controller.bullet_holes:
                    # Add bullet hole data to response
                    holes_data = []
                    for x, y, radius, score, area, circularity in self.camera_controller.bullet_holes:
                        holes_data.append({
                            'x': int(x), 'y': int(y), 'radius': int(radius),
                            'score': float(score), 'area': float(area), 'circularity': float(circularity)
                        })
                    response['bullet_holes'] = holes_data

            elif path == '/api/clear_bullet_holes':
                if self.camera_controller.clear_bullet_holes():
                    response = {'success': True, 'message': 'Bullet holes cleared'}
                else:
                    response = {'success': False, 'message': 'Failed to clear bullet holes'}

            elif path == '/api/change_source':
                source_type = data.get('source_type', '')
                source_id = data.get('source_id', '')
                success, message = self.camera_controller.set_video_source(source_type, source_id)
                response = {'success': success, 'message': message}

                # Include camera formats if switching to camera
                if success and source_type == 'camera':
                    try:
                        formats = self.camera_controller.get_camera_formats()
                        if formats and formats.get('available'):
                            response['camera_formats'] = formats
                            # Include current camera settings - read ACTUAL values from camera
                            if self.camera_controller.cap and self.camera_controller.cap.isOpened():
                                actual_width = int(self.camera_controller.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                actual_height = int(self.camera_controller.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                actual_fps = self.camera_controller.cap.get(cv2.CAP_PROP_FPS)
                                print(f"DEBUG: Actual camera settings - {actual_width}x{actual_height}@{actual_fps}fps")

                                # Find matching option by width, height, and fps
                                current_resolution = None
                                for option in formats['resolution_options']:
                                    if (option['width'] == actual_width and
                                        option['height'] == actual_height and
                                        abs(option['fps'] - actual_fps) < 0.1):  # Allow small fps difference
                                        current_resolution = option['value']
                                        print(f"DEBUG: Matched resolution option: {current_resolution}")
                                        break

                                if current_resolution:
                                    response['current_resolution'] = current_resolution
                                else:
                                    print(f"WARNING: Could not find matching resolution for {actual_width}x{actual_height}@{actual_fps}fps")
                                    print(f"DEBUG: Available options:")
                                    for option in formats['resolution_options'][:5]:  # Print first 5
                                        print(f"  - {option['width']}x{option['height']}@{option['fps']}fps = {option['value']}")

                        # Include camera controls
                        controls = self.camera_controller.get_camera_controls()
                        if controls and controls.get('available'):
                            response['camera_controls'] = controls
                    except Exception as e:
                        print(f"WARNING: Could not get camera formats: {e}")

            elif path == '/api/perspective_correction':
                enabled = data.get('enabled', False)
                if self.camera_controller.set_perspective_correction(enabled):
                    status = "enabled" if enabled else "disabled"
                    response = {'success': True, 'message': f'Perspective correction {status}'}
                else:
                    response = {'success': False, 'message': 'Failed to set perspective correction'}

            elif path == '/api/playback_pause':
                if self.camera_controller.pause_playback():
                    response = {'success': True, 'message': 'Video playback paused'}
                else:
                    response = {'success': False, 'message': 'Cannot pause (not playing video or already paused)'}

            elif path == '/api/playback_resume':
                if self.camera_controller.resume_playback():
                    response = {'success': True, 'message': 'Video playback resumed'}
                else:
                    response = {'success': False, 'message': 'Cannot resume (not playing video or not paused)'}

            elif path == '/api/playback_step_forward':
                if self.camera_controller.step_frame_forward():
                    response = {'success': True, 'message': 'Stepped forward one frame'}
                else:
                    response = {'success': False, 'message': 'Cannot step forward (not paused or no frames available)'}

            elif path == '/api/playback_step_backward':
                if self.camera_controller.step_frame_backward():
                    response = {'success': True, 'message': 'Stepped backward one frame'}
                else:
                    response = {'success': False, 'message': 'Cannot step backward (not paused or at beginning)'}

            elif path == '/api/seek_to_frame':
                frame_number = data.get('frame', 0)
                success, message = self.camera_controller.seek_to_frame(frame_number)
                response = {'success': success, 'message': message}

            elif path == '/api/playback_info':
                playback_info = self.camera_controller.get_playback_info()
                response = {'success': True, 'data': playback_info}

            # Camera controls endpoints
            elif path == '/api/camera_controls':
                controls_info = self.camera_controller.get_camera_controls()
                response = {'success': True, 'data': controls_info}

            elif path == '/api/set_camera_control':
                control_name = data.get('name', '')
                control_value = data.get('value', 0)
                if control_name:
                    success, message = self.camera_controller.set_camera_control(control_name, control_value)
                    response = {'success': success, 'message': message}
                else:
                    response = {'success': False, 'message': 'Control name is required'}

            elif path == '/api/get_camera_control':
                control_name = data.get('name', '')
                if control_name:
                    current_value = self.camera_controller.get_camera_control(control_name)
                    if current_value is not None:
                        response = {'success': True, 'value': current_value}
                    else:
                        response = {'success': False, 'message': f'Control {control_name} not available'}
                else:
                    response = {'success': False, 'message': 'Control name is required'}

            elif path == '/api/reset_camera_controls':
                success, message = self.camera_controller.reset_camera_controls()
                response = {'success': success, 'message': message}

            elif path == '/api/save_camera_preset':
                preset_name = data.get('name', '')
                if preset_name:
                    success, message = self.camera_controller.save_camera_preset(preset_name)
                    response = {'success': success, 'message': message}
                else:
                    response = {'success': False, 'message': 'Preset name is required'}

            elif path == '/api/load_camera_preset':
                preset_name = data.get('name', '')
                if preset_name:
                    success, message = self.camera_controller.load_camera_preset(preset_name)
                    response = {'success': success, 'message': message}
                else:
                    response = {'success': False, 'message': 'Preset name is required'}

            elif path == '/api/list_camera_presets':
                presets = self.camera_controller.list_camera_presets()
                response = {'success': True, 'presets': presets}

            elif path == '/api/set_resolution':
                width = data.get('width', 0)
                height = data.get('height', 0)
                fps = data.get('fps', 30)
                format_name = data.get('format', 'MJPG')

                if width > 0 and height > 0:
                    success, message = self.camera_controller.set_camera_resolution(width, height, fps, format_name)
                    response = {'success': success, 'message': message}
                else:
                    response = {'success': False, 'message': 'Valid width and height are required'}

            elif path == '/api/camera_formats':
                formats_info = self.camera_controller.get_camera_formats()
                response = {'success': True, 'data': formats_info}

            elif path == '/api/start_recording':
                success, message = self.camera_controller.start_recording()
                response = {'success': success, 'message': message}

            elif path == '/api/stop_recording':
                success, message = self.camera_controller.stop_recording()
                response = {'success': success, 'message': message}

            elif path == '/api/recording_status':
                recording_status = self.camera_controller.get_recording_status()
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
