#!/usr/bin/env python3
"""
Stress test to try to trigger segfaults under various conditions
"""

import sys
import time
import threading
from camera_web_stream import CameraController
import concurrent.futures

# Try to import requests, but don't fail if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

def stress_test_frame_operations():
    """Stress test frame operations that might cause segfaults"""
    print("🔥 Starting frame operations stress test...")

    controller = CameraController()
    controller.video_file = 'samples/video_20250918_182102.avi'
    controller.source_type = 'video'

    if not controller.start_capture():
        print("❌ Failed to start capture")
        return False

    success = True
    try:
        # Rapid frame requests from multiple threads
        def worker():
            for i in range(20):
                frame = controller.get_frame_jpeg()
                if frame and len(frame) > 1000:
                    # Rapid successive calls
                    for j in range(3):
                        controller.get_frame_jpeg()
                time.sleep(0.01)  # Very short sleep

        # Start multiple threads doing frame operations
        threads = []
        for i in range(3):  # 3 concurrent threads
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        # Main thread also does calibration operations
        for i in range(10):
            try:
                success_cal, msg = controller.calibrate_perspective()
                print(f"Calibration {i+1}: {success_cal}")
            except Exception as e:
                print(f"Calibration error: {e}")

            time.sleep(0.2)

        # Wait for all threads
        for t in threads:
            t.join()

    except Exception as e:
        print(f"💥 Exception in stress test: {e}")
        success = False

    finally:
        controller.stop()

    return success

def stress_test_http_requests():
    """Stress test HTTP requests that might trigger segfaults"""
    print("🌐 Starting HTTP stress test...")

    # Start server in background
    from camera_web_stream import ThreadingHTTPServer, StreamingHandler

    global camera_controller
    camera_controller = CameraController()
    camera_controller.video_file = 'samples/video_20250918_182102.avi'
    camera_controller.source_type = 'video'

    if not camera_controller.start_capture():
        return False

    server = ThreadingHTTPServer(('localhost', 8090), StreamingHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    time.sleep(2)  # Let server start

    success = True
    try:
        # Rapid API calls
        urls = [
            'http://localhost:8090/api/status',
            'http://localhost:8090/api/sources',
        ]

        def make_requests():
            for i in range(10):
                for url in urls:
                    try:
                        response = requests.get(url, timeout=1)
                        print(f"Request {i+1}: {response.status_code}")
                    except Exception as e:
                        print(f"Request error: {e}")
                time.sleep(0.1)

        # Multiple threads making requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_requests) for _ in range(2)]
            concurrent.futures.wait(futures)

    except Exception as e:
        print(f"💥 Exception in HTTP stress test: {e}")
        success = False

    finally:
        server.shutdown()
        camera_controller.stop()

    return success

def main():
    """Run various stress tests"""
    print("=== SEGFAULT STRESS TEST ===")
    print("🎯 This will try to trigger segfaults through various operations")

    tests = [
        ("Frame Operations Stress", stress_test_frame_operations),
    ]

    # Skip HTTP test if requests not available
    if HAS_REQUESTS:
        tests.append(("HTTP Requests Stress", stress_test_http_requests))
    else:
        print("⚠️  Skipping HTTP stress test (requests not available)")

    all_passed = True
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            result = test_func()
            print(f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
            if not result:
                all_passed = False
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            all_passed = False

    print(f"\n🏁 All tests: {'✅ PASSED' if all_passed else '❌ SOME FAILED'}")

    # Check for crash log
    import os
    crash_log = "/tmp/camera_crash.log"
    if os.path.exists(crash_log):
        print("\n💥 CRASH LOG DETECTED!")
        with open(crash_log, 'r') as f:
            print(f.read())
    else:
        print("\n✅ No segfault crash log found")

if __name__ == "__main__":
    main()