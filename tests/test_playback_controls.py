#!/usr/bin/env python3
"""
Test script for playback controls functionality
"""

import time
import requests
import json

def test_playback_controls():
    """Test the playback control endpoints"""
    base_url = "http://localhost:8088"

    print("🎯 Testing Playback Controls")
    print("=" * 50)

    # First, get status
    print("1. Getting initial status...")
    try:
        response = requests.get(f"{base_url}/api/status")
        status = response.json()
        print(f"   Source type: {status['source_type']}")
        print(f"   Supports playback controls: {status['supports_playback_controls']}")

        if not status['supports_playback_controls']:
            print("   ❌ Playback controls not supported (likely using camera, not video)")
            return

        print(f"   Current frame: {status['current_frame']} / {status['total_frames']}")
        print(f"   Paused: {status['paused']}")

    except Exception as e:
        print(f"   ❌ Failed to get status: {e}")
        return

    # Test pause
    print("\n2. Testing pause...")
    try:
        response = requests.post(f"{base_url}/api/playback_pause",
                               headers={'Content-Type': 'application/json'},
                               data='{}')
        result = response.json()
        print(f"   Response: {result['message']}")
        print(f"   Success: {result['success']}")
    except Exception as e:
        print(f"   ❌ Failed to pause: {e}")

    time.sleep(1)

    # Get frame position after pause
    print("\n   Getting frame position after pause...")
    try:
        response = requests.get(f"{base_url}/api/status")
        status = response.json()
        initial_frame = status['current_frame']
        print(f"   Frame after pause: {initial_frame}")
    except Exception as e:
        print(f"   ❌ Failed to get frame position: {e}")

    # Test step forward
    print("\n3. Testing step forward...")
    try:
        response = requests.post(f"{base_url}/api/playback_step_forward",
                               headers={'Content-Type': 'application/json'},
                               data='{}')
        result = response.json()
        print(f"   Response: {result['message']}")
        print(f"   Success: {result['success']}")

        # Check frame position
        response = requests.get(f"{base_url}/api/status")
        status = response.json()
        print(f"   Frame after step forward: {status['current_frame']} (should be {initial_frame + 1})")
    except Exception as e:
        print(f"   ❌ Failed to step forward: {e}")

    time.sleep(1)

    # Test step backward
    print("\n4. Testing step backward...")
    try:
        response = requests.post(f"{base_url}/api/playback_step_backward",
                               headers={'Content-Type': 'application/json'},
                               data='{}')
        result = response.json()
        print(f"   Response: {result['message']}")
        print(f"   Success: {result['success']}")

        # Check frame position
        response = requests.get(f"{base_url}/api/status")
        status = response.json()
        expected_frame = initial_frame  # Should be back to where we started
        print(f"   Frame after step backward: {status['current_frame']} (should be {expected_frame})")
    except Exception as e:
        print(f"   ❌ Failed to step backward: {e}")

    time.sleep(1)

    # Test resume
    print("\n5. Testing resume...")
    try:
        response = requests.post(f"{base_url}/api/playback_resume",
                               headers={'Content-Type': 'application/json'},
                               data='{}')
        result = response.json()
        print(f"   Response: {result['message']}")
        print(f"   Success: {result['success']}")
    except Exception as e:
        print(f"   ❌ Failed to resume: {e}")

    # Final status check
    print("\n6. Final status check...")
    try:
        response = requests.get(f"{base_url}/api/status")
        status = response.json()
        print(f"   Current frame: {status['current_frame']} / {status['total_frames']}")
        print(f"   Paused: {status['paused']}")
        print(f"   Buffer size: {status['buffer_size']}")
        if status['pause_buffer_index'] is not None:
            print(f"   Pause buffer index: {status['pause_buffer_index']}")
        print(f"   Can step backward: {status['can_step_backward']}")
        print(f"   Can step forward: {status['can_step_forward']}")
    except Exception as e:
        print(f"   ❌ Failed to get final status: {e}")

    print("\n✅ Playback controls test completed!")

if __name__ == "__main__":
    print("Make sure camera_web_stream.py is running on localhost:8088")
    print("And that you're using a video file (not camera) for testing")
    print()

    try:
        test_playback_controls()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")