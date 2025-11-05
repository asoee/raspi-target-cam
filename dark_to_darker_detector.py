#!/usr/bin/env python3
"""
Dark-to-Darker Detector
Focuses on finding areas that became DARKER in already-dark regions
This is more sensitive to bullet holes in the black target center
"""

import cv2
import numpy as np
from perspective import Perspective
from target_detection import TargetDetector
import json
import os


class DarkToDarkerDetector:
    """
    Specialized detector for finding darkness increases in already-dark areas
    """

    def __init__(self):
        self.perspective = Perspective()
        self.target_detector = TargetDetector()

        # Focus on dark regions
        self.dark_threshold = 80  # Pixels darker than this are "dark areas"

        # Detection thresholds for dark regions
        self.min_darkness_increase = 15  # Must get at least 15 units darker
        self.min_relative_change = 0.20   # Must get at least 20% darker (relative)
        self.min_area = 50
        self.max_area = 5000

        # Concentration - how much of the region is actually darker
        self.min_concentration = 0.30  # At least 30% of pixels must be significantly darker

    def set_reference(self, frame, frame_num=0):
        """Set reference frame"""
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.reference_frame = self.perspective.apply_perspective_correction(frame)
        self.reference_gray = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)

        # Detect target
        inner_circle = self.target_detector.detect_black_circle_improved(self.reference_frame)
        if inner_circle:
            self.target_center = (int(inner_circle[0]), int(inner_circle[1]))
            self.inner_radius = int(inner_circle[2])

        # Create mask of dark areas in reference
        _, self.dark_mask = cv2.threshold(self.reference_gray, self.dark_threshold, 255, cv2.THRESH_BINARY_INV)

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.dark_mask = cv2.morphologyEx(self.dark_mask, cv2.MORPH_CLOSE, kernel)

        print(f"✅ Reference set: target center={self.target_center}, radius={self.inner_radius}px")
        print(f"   Dark area threshold: {self.dark_threshold} (darker pixels are in 'dark area')")

    def detect_in_frame(self, frame, frame_num):
        """
        Detect new dark-to-darker changes in current frame
        """
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        current_frame = self.perspective.apply_perspective_correction(frame)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Focus on dark areas only
        # Calculate difference: reference - current (positive = got darker)
        darker = cv2.subtract(self.reference_gray, current_gray)

        # Only look in areas that were already dark in reference
        darker_in_dark_areas = cv2.bitwise_and(darker, darker, mask=self.dark_mask)

        # Threshold for significant darkness increase
        _, darker_thresh = cv2.threshold(darker_in_dark_areas, self.min_darkness_increase,
                                        255, cv2.THRESH_BINARY)

        # Morphological operations to connect nearby darker pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        darker_closed = cv2.morphologyEx(darker_thresh, cv2.MORPH_CLOSE, kernel)
        darker_closed = cv2.morphologyEx(darker_closed, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(darker_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Size filter
            if area < self.min_area or area > self.max_area:
                continue

            # Get center
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Spatial filter - must be within target
            if self.target_center:
                dist = np.sqrt((cx - self.target_center[0])**2 + (cy - self.target_center[1])**2)
                if dist > self.inner_radius * 0.9:
                    continue

            # Create mask for this contour
            mask = np.zeros(self.reference_gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # Measure darkness change in this region
            ref_values = self.reference_gray[mask > 0]
            curr_values = current_gray[mask > 0]

            if len(ref_values) == 0:
                continue

            # Calculate various darkness metrics
            mean_ref = np.mean(ref_values)
            mean_curr = np.mean(curr_values)
            mean_darkness_increase = mean_ref - mean_curr

            # Relative change (percentage)
            relative_change = mean_darkness_increase / mean_ref if mean_ref > 0 else 0

            # Concentration: how many pixels got significantly darker?
            darker_values = darker_in_dark_areas[mask > 0]
            significantly_darker = np.sum(darker_values >= self.min_darkness_increase)
            concentration = significantly_darker / len(darker_values) if len(darker_values) > 0 else 0

            # Maximum darkness increase in this region
            max_darkness_increase = np.max(darker_values)

            # Filters based on dark-to-darker criteria
            if mean_darkness_increase < self.min_darkness_increase:
                continue

            if relative_change < self.min_relative_change:
                continue

            if concentration < self.min_concentration:
                continue

            # Calculate score
            # Emphasize: absolute darkness increase, relative change, and concentration
            score = (mean_darkness_increase * 0.4 +
                    relative_change * 100 * 0.3 +
                    concentration * 100 * 0.3)

            candidates.append({
                'center': (cx, cy),
                'area': area,
                'mean_darkness_increase': mean_darkness_increase,
                'relative_change': relative_change,
                'concentration': concentration,
                'max_darkness_increase': max_darkness_increase,
                'mean_ref_intensity': mean_ref,
                'mean_curr_intensity': mean_curr,
                'score': score
            })

        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)

        return candidates


def test_dark_to_darker(video_path, ref_frame_num=50, test_frame_num=68):
    """
    Test the dark-to-darker detector
    """
    print("🎯 Dark-to-Darker Detection Test")
    print("=" * 70)

    cap = cv2.VideoCapture(video_path)

    detector = DarkToDarkerDetector()

    # Set reference
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_num)
    ret, ref = cap.read()
    detector.set_reference(ref, ref_frame_num)

    # Test on specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_num)
    ret, test = cap.read()

    cap.release()

    print(f"\n🔍 Testing on frame {test_frame_num}...")
    candidates = detector.detect_in_frame(test, test_frame_num)

    print(f"\n📊 Results: {len(candidates)} candidates found")

    if candidates:
        print(f"\nTop 10 candidates:")
        for i, cand in enumerate(candidates[:10]):
            cx, cy = cand['center']
            print(f"\n  #{i+1}: pos=({cx},{cy}), score={cand['score']:.1f}")
            print(f"      Ref intensity: {cand['mean_ref_intensity']:.1f} -> Curr: {cand['mean_curr_intensity']:.1f}")
            print(f"      Darkness increase: {cand['mean_darkness_increase']:.1f} ({cand['relative_change']*100:.1f}% relative)")
            print(f"      Concentration: {cand['concentration']*100:.1f}% of pixels significantly darker")
            print(f"      Max darkness increase: {cand['max_darkness_increase']:.0f}")

    # Compare with ground truth
    try:
        with open('ground_truth_holes.json', 'r') as f:
            gt = json.load(f)

        print(f"\n🔍 Comparing with ground truth...")

        for cand in candidates[:5]:  # Check top 5
            cx, cy = cand['center']

            # Find closest GT hole
            min_dist = float('inf')
            closest_gt = None

            for hole in gt['holes']:
                dist = np.sqrt((cx - hole['x'])**2 + (cy - hole['y'])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_gt = hole['hole_number']

            if min_dist < 50:
                print(f"   ✓ Candidate at ({cx},{cy}) matches GT #{closest_gt} (dist={min_dist:.0f}px)")
            else:
                print(f"   ✗ Candidate at ({cx},{cy}) is FALSE POSITIVE (closest GT={min_dist:.0f}px)")

    except FileNotFoundError:
        pass

    # Create visualization
    test_corrected = cv2.rotate(test, cv2.ROTATE_90_COUNTERCLOCKWISE)
    test_corrected = detector.perspective.apply_perspective_correction(test_corrected)
    result = test_corrected.copy()

    # Draw target
    cv2.circle(result, detector.target_center, detector.inner_radius, (100, 100, 100), 2)

    # Draw candidates
    for i, cand in enumerate(candidates[:5]):
        cx, cy = cand['center']
        color = (0, 255, 0) if i == 0 else (0, 255, 255)

        cv2.circle(result, (cx, cy), 25, color, 3)
        cv2.drawMarker(result, (cx, cy), color, cv2.MARKER_CROSS, 25, 2)

        label = f"#{i+1}"
        cv2.putText(result, label, (cx + 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Add info
    cv2.putText(result, f"Frame {ref_frame_num} -> {test_frame_num}: {len(candidates)} dark-to-darker detections",
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Save
    output_dir = "test_outputs/dark_to_darker"
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(f"{output_dir}/test_frame_{test_frame_num}.jpg", result)

    print(f"\n💾 Saved visualization to: {output_dir}/test_frame_{test_frame_num}.jpg")

    return candidates


if __name__ == "__main__":
    video_path = "samples/10-shot-1.mkv"

    # Test on frame 68 (where we detected shots 1 and 2)
    print("Testing frame 68 (where shot #1 and false positive #2 were detected):")
    print("=" * 70)
    candidates = test_dark_to_darker(video_path, ref_frame_num=50, test_frame_num=68)

    print(f"\n" + "=" * 70)
    print("\nTesting frame 93 (where multiple false positives were detected):")
    print("=" * 70)
    candidates = test_dark_to_darker(video_path, ref_frame_num=50, test_frame_num=93)

    print(f"\n✅ Test complete!")
