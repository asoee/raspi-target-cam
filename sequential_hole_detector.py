#!/usr/bin/env python3
"""
Sequential Hole Detector
Detects bullet holes one at a time by comparing each frame to the previous state
This approach should be more precise than trying to find all holes at once
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from target_detection import TargetDetector
from improved_dark_detector import ImprovedDarkDetector
import json


class SequentialHoleDetector:
    """
    Detects holes sequentially by comparing each frame against the previous frame
    """

    def __init__(self):
        self.dark_detector = ImprovedDarkDetector()
        self.target_detector = TargetDetector()

        # Detection parameters
        self.min_frames_between_shots = 5  # Minimum frames between detecting new shots
        self.similarity_threshold = 0.95   # Frames must be >95% similar to be considered "no change"

        # State tracking
        self.detected_holes = []
        self.last_detection_frame = -999
        self.reference_frame = None
        self.target_center = None
        self.inner_radius = None

    def initialize(self, first_frame):
        """
        Initialize detector with the clean target frame
        """
        if len(first_frame.shape) == 3:
            self.reference_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        else:
            self.reference_frame = first_frame.copy()

        # Detect target
        inner_circle = self.target_detector.detect_black_circle_improved(first_frame)
        if inner_circle:
            self.target_center = (int(inner_circle[0]), int(inner_circle[1]))
            self.inner_radius = int(inner_circle[2])
            print(f"✅ Initialized: target center={self.target_center}, radius={self.inner_radius}px")
        else:
            print("⚠️  Warning: Could not detect target circle")

        # Create dark mask
        _, dark_mask = cv2.threshold(self.reference_frame, 60, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        # Detect text zones
        self.dark_detector.detect_text_zones(self.reference_frame, self.dark_mask)

    def process_frame(self, frame_num: int, current_frame, debug=False) -> List[Tuple]:
        """
        Process a single frame and detect if a new hole appeared

        Returns list of newly detected holes in this frame
        """
        if len(current_frame.shape) == 3:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = current_frame.copy()

        # Check if enough frames have passed since last detection
        frames_since_last = frame_num - self.last_detection_frame
        if frames_since_last < self.min_frames_between_shots:
            return []

        # Calculate difference from reference
        diff = cv2.absdiff(self.reference_frame, current_gray)
        diff_masked = cv2.bitwise_and(diff, diff, mask=self.dark_mask)

        # Check if there's significant change
        mean_diff = np.mean(diff_masked[self.dark_mask > 0])

        if debug:
            print(f"   Frame {frame_num}: mean_diff={mean_diff:.2f}")

        # If very little change, no new hole
        if mean_diff < 0.5:  # Very low threshold - almost no change
            return []

        # Detect holes in current frame compared to reference
        holes, _ = self.dark_detector.detect_darker_holes(
            self.reference_frame,
            current_gray,
            self.dark_mask,
            self.target_center,
            self.inner_radius
        )

        # Find NEW holes (not already detected)
        new_holes = []
        for hole in holes:
            x, y = hole[0], hole[1]

            # Check if this hole is already in our detected list
            is_duplicate = False
            for existing_hole in self.detected_holes:
                ex, ey = existing_hole['x'], existing_hole['y']
                distance = np.sqrt((x - ex)**2 + (y - ey)**2)

                if distance < 40:  # Same hole
                    is_duplicate = True
                    break

            if not is_duplicate:
                new_hole = {
                    'hole_number': len(self.detected_holes) + 1,
                    'x': x,
                    'y': y,
                    'radius': hole[2],
                    'confidence': hole[3],
                    'frame_detected': frame_num
                }
                self.detected_holes.append(new_hole)
                new_holes.append(new_hole)

                if debug:
                    print(f"      ✓ NEW HOLE #{new_hole['hole_number']} at ({x},{y}) in frame {frame_num}")

        # If we detected new holes, update reference and last detection frame
        if new_holes:
            self.last_detection_frame = frame_num
            # Update reference to current frame (now includes the new hole)
            self.reference_frame = current_gray.copy()

        return new_holes

    def get_all_holes(self):
        """Return all detected holes"""
        return self.detected_holes


def test_sequential_detection():
    """
    Test sequential detection on the 10-shot video
    """
    print("🎯 Testing Sequential Hole Detection")
    print("=" * 60)

    video_path = "samples/10-shot-1.mkv"

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Video: {total_frames} frames @ {fps:.1f} fps")

    # Read first frame (clean target)
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read first frame")
        return

    # Apply rotation and perspective correction
    from perspective import detect_ellipse_and_correct_perspective

    # Rotate 90° counter-clockwise
    first_frame = cv2.rotate(first_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Correct perspective
    result = detect_ellipse_and_correct_perspective(first_frame)
    if result and result['corrected_frame'] is not None:
        first_frame_corrected = result['corrected_frame']
        transform_matrix = result['transform_matrix']
        print("✅ Perspective correction applied")
    else:
        print("⚠️  Using uncorrected frame")
        first_frame_corrected = first_frame
        transform_matrix = None

    # Initialize detector
    detector = SequentialHoleDetector()
    detector.initialize(first_frame_corrected)

    print(f"\n🔍 Processing frames...")

    frame_num = 0

    # Process every frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Apply same transformations
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if transform_matrix is not None:
            h, w = first_frame_corrected.shape[:2]
            frame = cv2.warpPerspective(frame, transform_matrix, (w, h))

        # Process frame
        new_holes = detector.process_frame(frame_num, frame, debug=False)

        # Print progress every 30 frames
        if frame_num % 30 == 0:
            print(f"   Frame {frame_num}/{total_frames} - Total holes: {len(detector.detected_holes)}")

        # If we found new holes, print immediately
        if new_holes:
            for hole in new_holes:
                print(f"   🎯 Shot #{hole['hole_number']} detected at frame {frame_num} ({frame_num/fps:.1f}s) - pos=({hole['x']},{hole['y']})")

    cap.release()

    # Final results
    all_holes = detector.get_all_holes()

    print(f"\n📊 Detection Complete")
    print(f"   Total holes detected: {len(all_holes)}")
    print(f"   Expected: 10 holes")

    # Compare with ground truth
    print(f"\n🔍 Comparing with ground truth...")

    try:
        with open('ground_truth_holes.json', 'r') as f:
            ground_truth = json.load(f)

        gt_holes = ground_truth['holes']

        # Match detections to ground truth
        matched = 0
        for gt_hole in gt_holes:
            gt_x, gt_y = gt_hole['x'], gt_hole['y']

            # Find closest detection
            best_distance = float('inf')
            best_match = None

            for detected_hole in all_holes:
                det_x, det_y = detected_hole['x'], detected_hole['y']
                distance = np.sqrt((gt_x - det_x)**2 + (gt_y - det_y)**2)

                if distance < best_distance:
                    best_distance = distance
                    best_match = detected_hole

            if best_distance < 50:  # Within 50px threshold
                matched += 1
                print(f"   ✓ GT #{gt_hole['hole_number']}: matched to Shot #{best_match['hole_number']} (distance={best_distance:.1f}px)")
            else:
                print(f"   ✗ GT #{gt_hole['hole_number']}: no match (closest={best_distance:.1f}px)")

        print(f"\n📈 Performance:")
        print(f"   Matched: {matched}/{len(gt_holes)} ({100*matched/len(gt_holes):.1f}%)")
        print(f"   False positives: {len(all_holes) - matched}")

    except FileNotFoundError:
        print("   (No ground truth file found)")

    # Save results
    output = {
        'total_holes': len(all_holes),
        'holes': all_holes
    }

    with open('sequential_detection_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Saved results to: sequential_detection_results.json")
    print(f"✅ Test complete!")


if __name__ == "__main__":
    test_sequential_detection()
