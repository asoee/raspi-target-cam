import cv2
import numpy as np
from datetime import datetime
import os
from typing import List, Tuple
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time


class WebcamProcessor:
    def __init__(self):
        # Initialize variables
        self.video_capture = None
        self.all_possible_resolutions_found = False
        self.number_frames_after_connect = 0
        self.is_started = False
        self.can_save_image_now = False
        self.last_directory_path_save_image_video = "./captures"  # Default path

        # Create captures directory if it doesn't exist
        os.makedirs(self.last_directory_path_save_image_video, exist_ok=True)

        # Standard resolutions list
        self.standard_resolutions = [
            [160, 120],
            [192, 144], [256, 144],
            [240, 160],
            [320, 240], [360, 240], [384, 240], [400, 240], [432, 240],
            [480, 320],
            [480, 360], [640, 360],
            [600, 480], [640, 480], [720, 480], [768, 480], [800, 480], [854, 480], [960, 480],
            [675, 540], [960, 540],
            [720, 576], [768, 576], [1024, 576],
            [750, 600], [800, 600], [1024, 600],
            [960, 640], [1024, 640], [1136, 640],
            [960, 720], [1152, 720], [1280, 720], [1440, 720],
            [960, 768], [1024, 768], [1152, 768], [1280, 768], [1366, 768],
            [1280, 800],
            [1152, 864], [1280, 864], [1536, 864],
            [1200, 900], [1440, 900], [1600, 900],
            [1280, 960], [1440, 960], [1536, 960],
            [1280, 1024], [1600, 1024],
            [1400, 1050], [1680, 1050],
            [1440, 1080], [1920, 1080], [2160, 1080], [2280, 1080], [2560, 1080],
            [2048, 1152],
            [1500, 1200], [1600, 1200], [1920, 1200],
            [1920, 1280], [2048, 1280],
            [1920, 1440], [2160, 1440], [2304, 1440], [2560, 1440], [2880, 1440],
            [2960, 1440], [3040, 1440], [3120, 1440], [3200, 1440], [3440, 1440], [5120, 1440],
            [2048, 1536],
            [2400, 1600], [2560, 1600], [3840, 1600],
            [2880, 1620],
            [2880, 1800], [3200, 1800],
            [2560, 1920], [2880, 1920], [3072, 1920],
            [2560, 2048], [2732, 2048], [3200, 2048],
            [2880, 2160], [3240, 2160], [3840, 2160], [4320, 2160], [5120, 2160],
            [3200, 2400], [3840, 2400],
            [3840, 2560], [4096, 2560],
            [5120, 2880], [5760, 2880],
            [4096, 3072],
            [7680, 4320], [10240, 4320]
        ]

        self.supported_resolutions = []
        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI using tkinter (Python's built-in GUI library)"""
        self.root = tk.Tk()
        self.root.title("Webcam Processor")
        self.root.geometry("800x600")

        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Resolution combobox
        ttk.Label(main_frame, text="Possible Resolutions:").grid(row=0, column=0, sticky=tk.W)
        self.resolution_combobox = ttk.Combobox(main_frame, state="readonly")
        self.resolution_combobox.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))

        # Contrast slider
        ttk.Label(main_frame, text="Contrast (1-5):").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.contrast_slider = tk.Scale(main_frame, from_=100, to=500, orient=tk.HORIZONTAL)
        self.contrast_slider.set(100)  # Default contrast = 1.0
        self.contrast_slider.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=(10, 0))

        # Brightness slider
        ttk.Label(main_frame, text="Brightness (-255 to 255):").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.brightness_slider = tk.Scale(main_frame, from_=-255, to=255, orient=tk.HORIZONTAL)
        self.brightness_slider.set(0)  # Default brightness = 0
        self.brightness_slider.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=(10, 0))

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(20, 0))

        self.start_button = ttk.Button(button_frame, text="Start Capture", command=self.toggle_capture)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        self.save_button = ttk.Button(button_frame, text="Save Frame", command=self.save_frame)
        self.save_button.pack(side=tk.LEFT)

        # Image display label
        self.image_label = ttk.Label(main_frame, text="Camera feed will appear here")
        self.image_label.grid(row=4, column=0, columnspan=2, pady=(20, 0))

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def show_message_on_status_bar(self, message="Processing..."):
        """Update status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def find_supported_resolutions(self):
        """Find all supported webcam resolutions"""
        if not self.video_capture or not self.video_capture.isOpened():
            return

        print("Finding supported resolutions...")
        self.show_message_on_status_bar("Finding supported resolutions...")

        for i, (w, h) in enumerate(self.standard_resolutions):
            print("Trying resolution: ",w,h)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

            actual_w = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if w == actual_w and h == actual_h:
                resolution_str = f"{w}x{h}"
                self.supported_resolutions.append(resolution_str)

        # Update combobox with supported resolutions
        self.resolution_combobox['values'] = self.supported_resolutions
        if self.supported_resolutions:
            self.resolution_combobox.set(self.supported_resolutions[0])

        print("Supported resolutions found!")
        self.show_message_on_status_bar("Ready")
        self.all_possible_resolutions_found = True

    def apply_effects(self, frame):
        """Apply contrast and brightness effects to frame"""
        contrast = self.contrast_slider.get() / 100.0  # Convert to 1.0-5.0 range
        brightness = self.brightness_slider.get()  # -255 to 255 range

        # Apply contrast and brightness: new_image = contrast * image + brightness
        adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        return adjusted

    def save_frame(self):
        """Enable frame saving"""
        self.can_save_image_now = True

    def toggle_capture(self):
        """Toggle frame capture/saving"""
        self.is_started = not self.is_started
        if self.is_started:
            self.start_button.config(text="Stop Capture")
            self.show_message_on_status_bar("Capturing frames...")
        else:
            self.start_button.config(text="Start Capture")
            self.show_message_on_status_bar("Capture stopped")

    def process_webcam_frames(self, camera_index=1):
        """Main processing loop - equivalent to the original C++ function"""
        # Initialize video capture
        self.video_capture = cv2.VideoCapture(camera_index)

        if not self.video_capture.isOpened():
            print("Error: Could not open webcam")
            self.show_message_on_status_bar("Error: Could not open webcam")
            return

        print("Webcam opened successfully")

        try:
            while True:
                self.show_message_on_status_bar("Processing...")

                if self.video_capture.isOpened():
                    # Find all possible webcam resolutions (only once)
                    if not self.all_possible_resolutions_found:
                        self.find_supported_resolutions()

                    # Read next frame
                    ret, frame = self.video_capture.read()

                    if ret and frame is not None:
                        # Increase frame counter
                        self.number_frames_after_connect += 1

                        # Apply effects on frame
                        frame = self.apply_effects(frame)

                        # Convert BGR to RGB for display (OpenCV uses BGR, GUI expects RGB)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Convert to PIL Image for tkinter display
                        pil_image = Image.fromarray(frame_rgb)
                        # Resize for display if too large
                        display_size = (640, 480)
                        pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(pil_image)

                        # Update GUI in main thread
                        self.image_label.config(image=photo)
                        self.image_label.image = photo  # Keep a reference

                        # Save frame as image if requested
                        if self.is_started and self.can_save_image_now:
                            current_time = datetime.now().strftime("%H-%M-%S.%f")[:-3]  # Format: hh-mm-ss.zzz
                            filename = os.path.join(
                                self.last_directory_path_save_image_video,
                                f"{self.number_frames_after_connect}) {current_time}.jpg"
                            )

                            # Save the original frame (before RGB conversion) as JPEG
                            cv2.imwrite(filename, frame)
                            print(f"Frame saved: {filename}")

                            self.can_save_image_now = False

                # Update GUI
                self.root.update_idletasks()

                # Small delay to prevent excessive CPU usage
                time.sleep(0.033)  # ~30 FPS

        except KeyboardInterrupt:
            print("Processing stopped by user")
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            # Cleanup
            if self.video_capture:
                self.video_capture.release()
            cv2.destroyAllWindows()

    def run(self):
        """Start the application"""
        # Start webcam processing in a separate thread
        webcam_thread = threading.Thread(target=self.process_webcam_frames, daemon=True)
        webcam_thread.start()

        # Start GUI main loop
        self.root.mainloop()


# Usage example
if __name__ == "__main__":
    app = WebcamProcessor()
    app.run()