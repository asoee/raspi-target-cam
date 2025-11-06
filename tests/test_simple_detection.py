#!/usr/bin/env python3
"""
Simple Bullet Hole Detection Test
Tests detection on extracted frame pairs with known differences
"""

import cv2
import numpy as np
from raspi_target_cam.detection.bullet_hole_detection import BulletHoleDetector
from raspi_target_cam.core.target_detection import TargetDetector
from raspi_target_cam.utils.target_scoring import TargetScoringSystem

def test_frame_pair(before_path, after_path, test_name):
    """Test bullet hole detection on a pair of frames"""
    print(f"\n{'='*60}")
    print(f"🎯 Test: {test_name}")
    print(f"{'='*60}")

    # Load frames
    before = cv2.imread(before_path)
    after = cv2.imread(after_path)

    if before is None or after is None:
        print(f"❌ Could not load frames:")
        print(f"   Before: {before_path} (exists: {before is not None})")
        print(f"   After:  {after_path} (exists: {after is not None})")
        return

    print(f"✅ Loaded frames:")
    print(f"   Before: {before_path}")
    print(f"   After:  {after_path}")
    print(f"   Resolution: {before.shape[1]}x{before.shape[0]}")

    # Detect target in before frame
    print(f"\n📍 Detecting target...")
    target_detector = TargetDetector()
    inner_circle = target_detector.detect_black_circle_improved(before)
    outer_circle = target_detector.detect_outer_circle(before)

    if inner_circle:
        target_center = (int(inner_circle[0]), int(inner_circle[1]))
        inner_radius = int(inner_circle[2])
        print(f"   ✅ Inner circle: center={target_center}, radius={inner_radius}px")
    else:
        print(f"   ⚠️  No inner circle detected")
        target_center = (before.shape[1] // 2, before.shape[0] // 2)
        inner_radius = 300

    if outer_circle:
        outer_radius = int(outer_circle[2])
        print(f"   ✅ Outer circle: radius={outer_radius}px")
    else:
        print(f"   ⚠️  No outer circle detected")
        outer_radius = inner_radius * 2

    # Create scoring system
    print(f"\n🎯 Setting up scoring system...")
    scoring = TargetScoringSystem()
    scoring.calibrate_profile('test', outer_radius, inner_radius)
    session = scoring.start_session('test', target_center, before)

    # Detect bullet holes
    print(f"\n🔍 Detecting bullet holes...")
    detector = BulletHoleDetector()
    detector.debug_mode = True

    holes = detector.detect_bullet_holes(before, after)

    print(f"   Found {len(holes)} potential bullet holes")

    # Score holes
    if holes:
        print(f"\n📊 Scoring detected holes:")
        for i, hole_data in enumerate(holes):
            x, y, radius, confidence = hole_data[:4]

            # Calculate distance from center
            dx = x - target_center[0]
            dy = y - target_center[1]
            distance = np.sqrt(dx**2 + dy**2)

            # Get score
            score = scoring.profiles['test'].get_score_for_distance(distance)

            # Add to session
            shot_data = scoring.add_shot_to_current_session(x, y, radius, confidence)

            if shot_data:
                print(f"      Hole #{i+1}:")
                print(f"         Position: ({x}, {y})")
                print(f"         Distance from center: {distance:.1f}px")
                print(f"         Score: {score}")
                print(f"         Confidence: {confidence:.3f}")

                # Additional details
                if len(hole_data) >= 6:
                    area = hole_data[4]
                    circularity = hole_data[5]
                    print(f"         Area: {area}px, Circularity: {circularity:.3f}")

        # Session summary
        print(f"\n📈 Session Summary:")
        status = scoring.get_session_status()
        print(f"   Total shots scored: {status['shot_count']}")
        print(f"   Total score: {status['total_score']}")

    # Save debug frames
    print(f"\n💾 Saving debug visualization...")
    debug_output = f"test_outputs/test_{test_name.replace(' ', '_')}"
    detector.save_debug_frames(debug_output)

    scoring.end_session()

    return holes

def main():
    """Run all frame pair tests"""
    print("🎯 Simple Bullet Hole Detection Tests")
    print("=" * 60)
    print()

    # Test pairs (using corrected frames)
    tests = [
        {
            'name': 'Clean to Shot 1',
            'before': 'test_frames/frame_0000_clean_target_corrected.jpg',
            'after': 'test_frames/frame_0100_after_shot_1_approx_corrected.jpg'
        },
        {
            'name': 'Clean to Mid Session',
            'before': 'test_frames/frame_0000_clean_target_corrected.jpg',
            'after': 'test_frames/frame_0500_mid_session_corrected.jpg'
        },
        {
            'name': 'Mid Session to Near End',
            'before': 'test_frames/frame_0500_mid_session_corrected.jpg',
            'after': 'test_frames/frame_0900_near_end_corrected.jpg'
        },
        {
            'name': 'Clean to Near End',
            'before': 'test_frames/frame_0000_clean_target_corrected.jpg',
            'after': 'test_frames/frame_0900_near_end_corrected.jpg'
        }
    ]

    results = {}
    for test in tests:
        holes = test_frame_pair(test['before'], test['after'], test['name'])
        results[test['name']] = holes

    # Final summary
    print(f"\n\n{'='*60}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*60}")

    for test_name, holes in results.items():
        if holes:
            print(f"   {test_name:30s}: {len(holes)} holes detected")
        else:
            print(f"   {test_name:30s}: 0 holes detected")

    print(f"\n✅ All tests complete!")
    print(f"   Check test_outputs/ for debug visualizations")

if __name__ == "__main__":
    main()
