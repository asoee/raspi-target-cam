#!/usr/bin/env python3
"""
Start both camera stream and web interface servers
"""

import subprocess
import sys
import time
import signal
import os


def start_servers():
    """Start both camera stream and web servers"""
    processes = []

    try:
        print("Starting camera stream server on port 8080...")
        camera_process = subprocess.Popen([sys.executable, 'camera_stream.py'])
        processes.append(camera_process)
        time.sleep(2)  # Give camera time to initialize

        print("Starting web interface server on port 8081...")
        web_process = subprocess.Popen([sys.executable, 'web_server.py'])
        processes.append(web_process)

        print("\n" + "="*50)
        print("🎯 Raspberry Pi Camera Streaming System")
        print("="*50)
        print("Camera Stream: http://localhost:8080/stream.mjpg")
        print("Web Interface: http://localhost:8081")
        print("="*50)
        print("\nPress Ctrl+C to stop both servers...")

        # Wait for processes
        for process in processes:
            process.wait()

    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
        for process in processes:
            process.terminate()

        # Give processes time to cleanup
        time.sleep(2)

        # Force kill if still running
        for process in processes:
            if process.poll() is None:
                process.kill()

        print("Servers stopped.")


if __name__ == '__main__':
    start_servers()