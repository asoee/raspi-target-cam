#!/usr/bin/env python3
"""
Test script to trigger segfaults and examine crash logs
"""

import subprocess
import time
import os
import sys

def run_application():
    """Run the application and monitor for crashes"""
    print("🔍 Starting application with segfault detection...")

    # Clean up any previous crash logs
    crash_log = "/tmp/camera_crash.log"
    if os.path.exists(crash_log):
        os.remove(crash_log)

    try:
        # Start the application
        process = subprocess.Popen([
            sys.executable, "camera_web_stream.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        print(f"🆔 Process started with PID: {process.pid}")

        # Wait up to 30 seconds for either completion or crash
        try:
            stdout, stderr = process.communicate(timeout=30)
            print("🏁 Process completed normally")
            if stdout:
                print("STDOUT:", stdout[-500:])  # Last 500 chars
            if stderr:
                print("STDERR:", stderr[-500:])

        except subprocess.TimeoutExpired:
            print("⏰ Process still running after 30 seconds")
            process.kill()
            stdout, stderr = process.communicate()

    except Exception as e:
        print(f"❌ Error running process: {e}")

    # Check for crash log
    if os.path.exists(crash_log):
        print("\n💥 CRASH LOG FOUND!")
        print("📋 Crash details:")
        print("-" * 60)
        with open(crash_log, 'r') as f:
            print(f.read())
        print("-" * 60)
        return True
    else:
        print("✅ No crash detected")
        return False

def main():
    """Main test function"""
    print("=== SEGFAULT DETECTION TEST ===")

    crash_detected = run_application()

    if crash_detected:
        print("\n🎯 ANALYSIS RECOMMENDATIONS:")
        print("1. Look at the stack trace in the crash log above")
        print("2. Identify the exact function and line where segfault occurred")
        print("3. Check if it's in OpenCV operations or memory management")
        print("4. Add additional validation around that specific operation")
    else:
        print("\n✅ Application ran without segfaults")
        print("💡 If segfaults occur intermittently, run this test multiple times")

if __name__ == "__main__":
    main()