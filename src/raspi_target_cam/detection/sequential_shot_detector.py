#!/usr/bin/env python3
"""
Sequential Shot Detector - detects bullet holes one at a time
Uses concentrated darkness changes to find individual shots
"""

import cv2
import numpy as np
from raspi_target_cam.core.perspective import Perspective
from raspi_target_cam.core.target_detection import TargetDetector
import json


class SequentialShotDetector:
    """
    Detects shots sequentially by finding concentrated dark changes
    """

    def __init__(self):
        self.perspective = Perspective()
        self.target_detector = TargetDetector()

        # Detection thresholds
        self.min_darkness_threshold = 10  # Minimum intensity decrease
        self.min_mean_darkness = 8        # Minimum average darkness in region
        self.min_max_darkness = 25        # Minimum peak darkness
        self.min_area = 50
        self.max_area = 5000

        # State
        self.reference_frame = None
        self.reference_gray = None
        self.target_center = None
        self.inner_radius = None
        self.detected_shots = []

    def set_reference(self, frame, frame_num=0):
        """Set the reference (clean target) frame"""
        # Apply transformations
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.reference_frame = self.perspective.apply_perspective_correction(frame)
        self.reference_gray = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)

        # Detect target
        inner_circle = self.target_detector.detect_black_circle_improved(self.reference_frame)
        if inner_circle:
            self.target_center = (int(inner_circle[0]), int(inner_circle[1]))
            self.inner_radius = int(inner_circle[2])

        print(f"✅ Reference set (frame {frame_num}): target center={self.target_center}, radius={self.inner_radius}px")

    def detect_new_shots(self, frame, frame_num):
        """
        Detect new shots in the current frame compared to reference
        Returns list of newly detected shots
        """
        # Apply transformations
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        current_frame = self.perspective.apply_perspective_correction(frame)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Find concentrated dark changes
        candidates = self._find_concentrated_changes(self.reference_gray, current_gray)

        # Filter out already detected shots
        new_shots = []
        for cand in candidates:
            cx, cy = cand['center']

            # Check if too close to existing shot
            is_duplicate = False
            for existing in self.detected_shots:
                ex, ey = existing['x'], existing['y']
                dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                if dist < 50:  # Within 50px = same hole
                    is_duplicate = True
                    break

            if not is_duplicate:
                shot = {
                    'shot_number': len(self.detected_shots) + 1,
                    'x': cx,
                    'y': cy,
                    'frame_detected': frame_num,
                    'score': cand['score'],
                    'darkness': cand['mean_darkness']
                }
                self.detected_shots.append(shot)
                new_shots.append(shot)

        return new_shots

    def _find_concentrated_changes(self, before_gray, after_gray):
        """Find localized concentrated dark changes"""
        # Find regions that became DARKER
        darker = cv2.subtract(before_gray, after_gray)

        # Threshold for significant darkness
        _, darker_thresh = cv2.threshold(darker, self.min_darkness_threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        darker_closed = cv2.morphologyEx(darker_thresh, cv2.MORPH_CLOSE, kernel)
        darker_closed = cv2.morphologyEx(darker_closed, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(darker_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area or area > self.max_area:
                continue

            # Get center
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w//2, y + h//2

            # Location filter
            if self.target_center:
                dist_from_center = np.sqrt((cx - self.target_center[0])**2 + (cy - self.target_center[1])**2)
                if dist_from_center > self.inner_radius * 0.9:
                    continue

            # Measure darkness
            mask = np.zeros(before_gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            darker_values = darker[mask > 0]
            if len(darker_values) == 0:
                continue

            mean_darkness = np.mean(darker_values)
            max_darkness = np.max(darker_values)

            # Concentration
            very_dark_pixels = np.sum(darker_values > 15)
            concentration = very_dark_pixels / area if area > 0 else 0

            # Filter by minimum darkness
            if mean_darkness < self.min_mean_darkness or max_darkness < self.min_max_darkness:
                continue

            # Calculate score
            score = mean_darkness * 0.4 + max_darkness * 0.3 + concentration * 100 * 0.3

            candidates.append({
                'center': (cx, cy),
                'area': area,
                'mean_darkness': mean_darkness,
                'max_darkness': max_darkness,
                'concentration': concentration,
                'score': score
            })

        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)

        return candidates


def test_sequential_detection(video_path, reference_frame=50, scan_frames=None):
    """
    Test sequential detection on video
    """
    print("🎯 Sequential Shot Detection")
    print("=" * 70)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if scan_frames is None:
        scan_frames = range(reference_frame + 1, min(total_frames, 1000))  # Scan up to frame 1000

    detector = SequentialShotDetector()

    # Set reference
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame)
    ret, ref = cap.read()
    detector.set_reference(ref, frame_num=reference_frame)

    print(f"\n🔍 Scanning frames {scan_frames.start} to {scan_frames.stop}...")

    # Scan frames
    for frame_num in scan_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        new_shots = detector.detect_new_shots(frame, frame_num)

        if new_shots:
            for shot in new_shots:
                time_sec = frame_num / fps
                print(f"   🎯 Shot #{shot['shot_number']} detected at frame {frame_num} ({time_sec:.2f}s)")
                print(f"       Position: ({shot['x']}, {shot['y']}), score={shot['score']:.1f}")

        # Print progress every 100 frames
        if frame_num % 100 == 0:
            print(f"   ... frame {frame_num} ({frame_num/fps:.1f}s) - {len(detector.detected_shots)} shots so far")

    cap.release()

    # Final results
    print(f"\n📊 Detection Complete")
    print(f"   Total shots detected: {len(detector.detected_shots)}")
    print(f"   Expected: 10 shots")

    # Compare with ground truth
    try:
        with open('ground_truth_holes.json', 'r') as f:
            gt = json.load(f)

        print(f"\n🔍 Comparing with ground truth...")

        matched = 0
        for gt_hole in gt['holes']:
            gt_x, gt_y = gt_hole['x'], gt_hole['y']

            # Find closest detection
            best_dist = float('inf')
            best_match = None

            for shot in detector.detected_shots:
                dist = np.sqrt((gt_x - shot['x'])**2 + (gt_y - shot['y'])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_match = shot

            if best_dist < 50:
                matched += 1
                print(f"   ✓ GT #{gt_hole['hole_number']}: matched to Shot #{best_match['shot_number']} (dist={best_dist:.0f}px)")
            else:
                print(f"   ✗ GT #{gt_hole['hole_number']}: no match (closest={best_dist:.0f}px)")

        precision = matched / len(detector.detected_shots) if detector.detected_shots else 0
        recall = matched / len(gt['holes'])
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n📈 Performance:")
        print(f"   Matched: {matched}/{len(gt['holes'])} ({100*recall:.1f}%)")
        print(f"   Precision: {100*precision:.1f}%")
        print(f"   Recall: {100*recall:.1f}%")
        print(f"   F1 Score: {100*f1:.1f}%")

    except FileNotFoundError:
        print("   (No ground truth file)")

    # Save results
    output = {
        'reference_frame': reference_frame,
        'total_shots': len(detector.detected_shots),
        'shots': detector.detected_shots
    }

    with open('sequential_detection_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to: sequential_detection_results.json")

    return detector.detected_shots


if __name__ == "__main__":
    video_path = "samples/10-shot-1.mkv"

    # Test with frame 50 as reference, scan frames 51-1000
    shots = test_sequential_detection(video_path, reference_frame=50, scan_frames=range(51, 1000))

    print(f"\n✅ Sequential detection complete!")
