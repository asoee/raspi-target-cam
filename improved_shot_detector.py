#!/usr/bin/env python3
"""
Improved Shot Detector
- Handles both dark-to-darker changes (black center) and light-to-dark changes (white area)
- Detects when target is removed and stops scanning
- Filters out printed numbers and false positives
"""

import cv2
import numpy as np
from perspective import Perspective
from target_detection import TargetDetector
import json


class ImprovedShotDetector:
    """
    Detects shots in both black and white areas of the target
    """

    def __init__(self):
        self.perspective = Perspective()
        self.target_detector = TargetDetector()

        # Dark area detection (black center)
        self.dark_threshold = 80
        self.dark_min_darkness_increase = 8  # Lowered from 15 to catch GT #2
        self.dark_min_relative_change = 0.20
        self.dark_min_concentration = 0.30

        # Light area detection (white area)
        self.light_threshold = 120  # Pixels brighter than this are "light areas"
        self.light_min_darkness_increase = 20  # Need more absolute change in white area
        self.light_min_relative_change = 0.15  # 15% darker in white area
        self.light_min_concentration = 0.40  # Higher concentration needed

        # Common parameters
        self.min_area = 50
        self.max_area = 5000

        # Target removal detection
        self.target_removed_threshold = 0.15  # If >15% of target area changes significantly

        # State
        self.reference_frame = None
        self.reference_gray = None
        self.dark_mask = None
        self.light_mask = None
        self.target_center = None
        self.inner_radius = None
        self.outer_radius = None
        self.detected_shots = []
        self.target_removed_frame = None

    def set_reference(self, frame, frame_num=0):
        """Set the reference (clean target) frame"""
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.reference_frame = self.perspective.apply_perspective_correction(frame)
        self.reference_gray = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)

        # Detect target
        inner_circle = self.target_detector.detect_black_circle_improved(self.reference_frame)
        if inner_circle:
            self.target_center = (int(inner_circle[0]), int(inner_circle[1]))
            self.inner_radius = int(inner_circle[2])
            # Estimate outer radius (white area extends beyond black center)
            self.outer_radius = int(self.inner_radius * 2.5)

        # Create mask of dark areas (black center)
        _, self.dark_mask = cv2.threshold(self.reference_gray, self.dark_threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.dark_mask = cv2.morphologyEx(self.dark_mask, cv2.MORPH_CLOSE, kernel)

        # Create mask of light areas (white area)
        _, self.light_mask = cv2.threshold(self.reference_gray, self.light_threshold, 255, cv2.THRESH_BINARY)
        self.light_mask = cv2.morphologyEx(self.light_mask, cv2.MORPH_CLOSE, kernel)

        # Only consider light areas within the outer target circle
        if self.target_center and self.outer_radius:
            target_mask = np.zeros_like(self.light_mask)
            cv2.circle(target_mask, self.target_center, self.outer_radius, 255, -1)
            self.light_mask = cv2.bitwise_and(self.light_mask, target_mask)

        print(f"✅ Reference set (frame {frame_num}): target center={self.target_center}, "
              f"inner_radius={self.inner_radius}px, outer_radius={self.outer_radius}px")
        print(f"   Dark area threshold: {self.dark_threshold}, Light area threshold: {self.light_threshold}")

    def check_target_removed(self, frame, frame_num):
        """
        Check if target has been removed (massive change across entire target)
        """
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        current_frame = self.perspective.apply_perspective_correction(frame)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Create mask for entire target area
        target_mask = np.zeros_like(current_gray)
        if self.target_center and self.outer_radius:
            cv2.circle(target_mask, self.target_center, self.outer_radius, 255, -1)
        else:
            return False

        # Calculate difference in target area
        diff = cv2.absdiff(self.reference_gray, current_gray)
        diff_in_target = cv2.bitwise_and(diff, diff, mask=target_mask)

        # Check what percentage of target area has significant change
        significant_change = np.sum(diff_in_target > 30)
        target_area = np.sum(target_mask > 0)

        change_ratio = significant_change / target_area if target_area > 0 else 0

        if change_ratio > self.target_removed_threshold:
            self.target_removed_frame = frame_num
            print(f"\n🚨 TARGET REMOVED at frame {frame_num} ({frame_num/30:.2f}s)")
            print(f"   {change_ratio*100:.1f}% of target area changed significantly")
            return True

        return False

    def detect_new_shots(self, frame, frame_num):
        """
        Detect new shots in the current frame
        Returns list of newly detected shots
        """
        # Check if target was removed
        if self.target_removed_frame is not None:
            return []

        if self.check_target_removed(frame, frame_num):
            return []

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        current_frame = self.perspective.apply_perspective_correction(frame)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Find candidates in both dark and light areas
        dark_candidates = self._find_dark_to_darker_changes(self.reference_gray, current_gray)
        light_candidates = self._find_light_to_dark_changes(self.reference_gray, current_gray)

        # Combine candidates
        all_candidates = dark_candidates + light_candidates

        # Filter out already detected shots and printed numbers
        new_shots = []
        for cand in all_candidates:
            cx, cy = cand['center']

            # Check if too close to existing shot
            is_duplicate = False
            for existing in self.detected_shots:
                ex, ey = existing['x'], existing['y']
                dist = np.sqrt((cx - ex)**2 + (cy - ey)**2)
                if dist < 50:  # Within 50px = same hole
                    is_duplicate = True
                    break

            # Check if in printed number area (near edge of black center, lower score)
            if self._is_likely_printed_number(cx, cy, cand):
                continue

            if not is_duplicate:
                shot = {
                    'shot_number': len(self.detected_shots) + 1,
                    'x': cx,
                    'y': cy,
                    'frame_detected': frame_num,
                    'score': cand['score'],
                    'type': cand['type'],  # 'dark' or 'light'
                    'mean_darkness_increase': cand['mean_darkness_increase'],
                    'relative_change': cand['relative_change'],
                    'concentration': cand['concentration']
                }
                self.detected_shots.append(shot)
                new_shots.append(shot)

        return new_shots

    def _is_likely_printed_number(self, cx, cy, cand):
        """
        Check if detection is likely a printed number rather than bullet hole
        Printed numbers are near the edge of the black circle and have lower scores
        """
        if not self.target_center or not self.inner_radius:
            return False

        # Distance from center
        dist = np.sqrt((cx - self.target_center[0])**2 + (cy - self.target_center[1])**2)

        # Numbers are typically at 0.7-0.95 of radius, with lower concentration
        radius_ratio = dist / self.inner_radius

        if 0.7 < radius_ratio < 0.95 and cand['concentration'] < 0.65:
            return True

        return False

    def _find_dark_to_darker_changes(self, before_gray, after_gray):
        """Find dark-to-darker changes in black center"""
        darker = cv2.subtract(before_gray, after_gray)
        darker_in_dark_areas = cv2.bitwise_and(darker, darker, mask=self.dark_mask)

        _, darker_thresh = cv2.threshold(darker_in_dark_areas, self.dark_min_darkness_increase,
                                        255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        darker_closed = cv2.morphologyEx(darker_thresh, cv2.MORPH_CLOSE, kernel)
        darker_closed = cv2.morphologyEx(darker_closed, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(darker_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area or area > self.max_area:
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Must be within inner circle (black area)
            if self.target_center:
                dist = np.sqrt((cx - self.target_center[0])**2 + (cy - self.target_center[1])**2)
                if dist > self.inner_radius * 0.95:
                    continue

            mask = np.zeros(before_gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            ref_values = before_gray[mask > 0]
            curr_values = after_gray[mask > 0]

            if len(ref_values) == 0:
                continue

            mean_ref = np.mean(ref_values)
            mean_curr = np.mean(curr_values)
            mean_darkness_increase = mean_ref - mean_curr

            relative_change = mean_darkness_increase / mean_ref if mean_ref > 0 else 0

            darker_values = darker_in_dark_areas[mask > 0]
            significantly_darker = np.sum(darker_values >= self.dark_min_darkness_increase)
            concentration = significantly_darker / len(darker_values) if len(darker_values) > 0 else 0

            if mean_darkness_increase < self.dark_min_darkness_increase:
                continue
            if relative_change < self.dark_min_relative_change:
                continue
            if concentration < self.dark_min_concentration:
                continue

            score = (mean_darkness_increase * 0.4 +
                    relative_change * 100 * 0.3 +
                    concentration * 100 * 0.3)

            candidates.append({
                'center': (cx, cy),
                'area': area,
                'type': 'dark',
                'mean_darkness_increase': mean_darkness_increase,
                'relative_change': relative_change,
                'concentration': concentration,
                'score': score
            })

        return sorted(candidates, key=lambda x: x['score'], reverse=True)

    def _find_light_to_dark_changes(self, before_gray, after_gray):
        """Find light-to-dark changes in white area"""
        darker = cv2.subtract(before_gray, after_gray)
        darker_in_light_areas = cv2.bitwise_and(darker, darker, mask=self.light_mask)

        _, darker_thresh = cv2.threshold(darker_in_light_areas, self.light_min_darkness_increase,
                                        255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        darker_closed = cv2.morphologyEx(darker_thresh, cv2.MORPH_CLOSE, kernel)
        darker_closed = cv2.morphologyEx(darker_closed, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(darker_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area or area > self.max_area:
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Must be within outer circle but outside inner circle (white area)
            if self.target_center:
                dist = np.sqrt((cx - self.target_center[0])**2 + (cy - self.target_center[1])**2)
                if dist > self.outer_radius * 0.95:
                    continue
                if dist < self.inner_radius * 1.05:  # Outside black center
                    continue

            mask = np.zeros(before_gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            ref_values = before_gray[mask > 0]
            curr_values = after_gray[mask > 0]

            if len(ref_values) == 0:
                continue

            mean_ref = np.mean(ref_values)
            mean_curr = np.mean(curr_values)
            mean_darkness_increase = mean_ref - mean_curr

            relative_change = mean_darkness_increase / mean_ref if mean_ref > 0 else 0

            darker_values = darker_in_light_areas[mask > 0]
            significantly_darker = np.sum(darker_values >= self.light_min_darkness_increase)
            concentration = significantly_darker / len(darker_values) if len(darker_values) > 0 else 0

            if mean_darkness_increase < self.light_min_darkness_increase:
                continue
            if relative_change < self.light_min_relative_change:
                continue
            if concentration < self.light_min_concentration:
                continue

            score = (mean_darkness_increase * 0.4 +
                    relative_change * 100 * 0.3 +
                    concentration * 100 * 0.3)

            candidates.append({
                'center': (cx, cy),
                'area': area,
                'type': 'light',
                'mean_darkness_increase': mean_darkness_increase,
                'relative_change': relative_change,
                'concentration': concentration,
                'score': score
            })

        return sorted(candidates, key=lambda x: x['score'], reverse=True)


def test_improved_detector(video_path, reference_frame=50, scan_frames=None):
    """
    Test improved detector on video
    """
    print("🎯 Improved Sequential Shot Detection")
    print("=" * 70)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if scan_frames is None:
        scan_frames = range(reference_frame + 1, min(total_frames, 1000))

    detector = ImprovedShotDetector()

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
                print(f"       Position: ({shot['x']}, {shot['y']}), type={shot['type']}")
                print(f"       Darkness: {shot['mean_darkness_increase']:.1f} ({shot['relative_change']*100:.1f}% relative)")
                print(f"       Concentration: {shot['concentration']*100:.1f}%, score={shot['score']:.1f}")

        # Stop if target removed
        if detector.target_removed_frame is not None:
            print(f"\n🛑 Stopping scan - target removed at frame {detector.target_removed_frame}")
            break

        # Print progress every 100 frames
        if frame_num % 100 == 0:
            print(f"   ... frame {frame_num} ({frame_num/fps:.1f}s) - {len(detector.detected_shots)} shots so far")

    cap.release()

    # Final results
    print(f"\n📊 Detection Complete")
    print(f"   Total shots detected: {len(detector.detected_shots)}")
    print(f"   Expected: 10 shots")
    if detector.target_removed_frame:
        print(f"   Target removed at frame: {detector.target_removed_frame}")

    # Compare with ground truth
    try:
        with open('ground_truth_holes.json', 'r') as f:
            gt = json.load(f)

        print(f"\n🔍 Comparing with ground truth...")

        matched = 0
        for gt_hole in gt['holes']:
            gt_x, gt_y = gt_hole['x'], gt_hole['y']

            best_dist = float('inf')
            best_match = None

            for shot in detector.detected_shots:
                dist = np.sqrt((gt_x - shot['x'])**2 + (gt_y - shot['y'])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_match = shot

            if best_dist < 50:
                matched += 1
                print(f"   ✓ GT #{gt_hole['hole_number']}: matched to Shot #{best_match['shot_number']} "
                      f"(dist={best_dist:.0f}px, type={best_match['type']})")
            else:
                print(f"   ✗ GT #{gt_hole['hole_number']}: no match (closest={best_dist:.0f}px)")

        precision = matched / len(detector.detected_shots) if detector.detected_shots else 0
        recall = matched / len(gt['holes'])
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n📈 Performance:")
        print(f"   Matched: {matched}/{len(gt['holes'])} ground truth holes")
        print(f"   Precision: {100*precision:.1f}% ({matched}/{len(detector.detected_shots)} detections correct)")
        print(f"   Recall: {100*recall:.1f}%")
        print(f"   F1 Score: {100*f1:.1f}%")

    except FileNotFoundError:
        print("   (No ground truth file)")

    # Save results
    output = {
        'reference_frame': reference_frame,
        'total_shots': len(detector.detected_shots),
        'target_removed_frame': detector.target_removed_frame,
        'shots': detector.detected_shots
    }

    with open('improved_detection_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to: improved_detection_results.json")

    return detector.detected_shots


if __name__ == "__main__":
    video_path = "samples/10-shot-1.mkv"

    shots = test_improved_detector(video_path, reference_frame=50, scan_frames=range(51, 1000))

    print(f"\n✅ Improved detection complete!")
