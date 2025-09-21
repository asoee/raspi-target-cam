import cv2

def demonstrate_properties(cap):
    """
    Test and display OpenCV properties of the video capture device.
    """
    # List of properties to demonstrate
    properties = {
        "CAP_PROP_POS_MSEC": cv2.CAP_PROP_POS_MSEC,
        "CAP_PROP_POS_FRAMES": cv2.CAP_PROP_POS_FRAMES,
        "CAP_PROP_POS_AVI_RATIO": cv2.CAP_PROP_POS_AVI_RATIO,
        "CAP_PROP_FRAME_WIDTH": cv2.CAP_PROP_FRAME_WIDTH,
        "CAP_PROP_FRAME_HEIGHT": cv2.CAP_PROP_FRAME_HEIGHT,
        "CAP_PROP_FPS": cv2.CAP_PROP_FPS,
        "CAP_PROP_FOURCC": cv2.CAP_PROP_FOURCC,
        "CAP_PROP_FRAME_COUNT": cv2.CAP_PROP_FRAME_COUNT,
        "CAP_PROP_BRIGHTNESS": cv2.CAP_PROP_BRIGHTNESS,
        "CAP_PROP_CONTRAST": cv2.CAP_PROP_CONTRAST,
        "CAP_PROP_SATURATION": cv2.CAP_PROP_SATURATION,
        "CAP_PROP_HUE": cv2.CAP_PROP_HUE,
        "CAP_PROP_GAIN": cv2.CAP_PROP_GAIN,
        "CAP_PROP_EXPOSURE": cv2.CAP_PROP_EXPOSURE,
        "CAP_PROP_CONVERT_RGB": cv2.CAP_PROP_CONVERT_RGB,
        "CAP_PROP_ZOOM": cv2.CAP_PROP_ZOOM,
        "CAP_PROP_FOCUS": cv2.CAP_PROP_FOCUS,
    }

    print("\nTesting Video Capture Properties:")
    for prop_name, prop_id in properties.items():
        value = cap.get(prop_id)
        if value == -1:
            print(f"{prop_name}: Not supported by this device.")
        else:
            print(f"{prop_name}: {value}")


def real_time_demo(cap):
    """
    Demonstrates real-time properties while playing the video/camera feed.
    """
    w,h = 3264, 2448
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    print("\nReal-Time Video Properties Demonstration (Press 'q' to exit):")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or no frames to capture.")
            break

        # Display the current frame
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Show the video frame
        cv2.imshow("Video", frame)
        print(f"Time: {current_time:.2f} ms | Frame: {current_frame}/{total_frames:.0f}", end="\r")

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def unsupported_properties_section():
    """
    Highlights properties that are hard to demonstrate in a standard environment.
    """
    print("\nProperties Not Demonstrated Directly:")
    unsupported_properties = {
        "CAP_PROP_WHITE_BALANCE_BLUE_U": "Not widely supported.",
        "CAP_PROP_RECTIFICATION": "Specific to stereo cameras.",
        "CAP_PROP_MONOCHROME": "Device-dependent; not commonly supported.",
        "CAP_PROP_SHARPNESS": "Requires specific hardware.",
        "CAP_PROP_TRIGGER": "Relevant for hardware triggers (e.g., industrial cameras).",
        "CAP_PROP_TRIGGER_DELAY": "Hardware-specific.",
        "CAP_PROP_TEMPERATURE": "Depends on thermal cameras.",
        "CAP_PROP_GUID": "Used for device identification.",
        "CAP_PROP_IRIS": "Applicable to advanced cameras with iris control.",
        "CAP_PROP_SETTINGS": "Opens camera settings dialog (Windows-only).",
    }

    for prop, description in unsupported_properties.items():
        print(f"{prop}: {description}")


# Main script
if __name__ == "__main__":
    # Video source (0 for webcam, or provide a video file path)
    video_source = 0  # Replace with 'sample_video.mp4' for video file
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    print(f"Using backend: {cap.getBackendName()}")

    # Demonstrate properties
    demonstrate_properties(cap)

    # Real-time demonstration
    real_time_demo(cap)

    # Close video capture and windows
    cap.release()
    cv2.destroyAllWindows()

    # Unsupported properties section
    unsupported_properties_section()
