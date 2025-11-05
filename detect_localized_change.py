#!/usr/bin/env python3
"""
Detect localized concentrated changes (bullet holes) rather than diffuse changes
"""

import cv2
import numpy as np
from perspective import Perspective
from target_detection import TargetDetector


def detect_concentrated_changes(before_gray, after_gray, target_center, inner_radius):
    """
    Find localized areas with concentrated dark changes (bullet holes)
    Ignore diffuse changes (lighting/noise)
    """
    # Calculate difference
    diff = cv2.absdiff(before_gray, after_gray)

    # Find regions that became DARKER (bullet holes)
    darker = cv2.subtract(before_gray, after_gray)

    # Only look at significant darkness (> 10 intensity units darker)
    # Bullet holes create substantial darkness, not just minor variations
    _, darker_thresh = cv2.threshold(darker, 10, 255, cv2.THRESH_BINARY)

    # Use morphological closing to connect nearby dark pixels (bullet hole region)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    darker_closed = cv2.morphologyEx(darker_thresh, cv2.MORPH_CLOSE, kernel)

    # Remove small noise
    darker_closed = cv2.morphologyEx(darker_closed, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(darker_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} darker regions")

    # Filter contours by:
    # 1. Size (reasonable bullet hole size)
    # 2. Concentration (high darkness in small area)
    # 3. Location (within target area)

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Size filter
        if area < 50 or area > 5000:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w//2, y + h//2

        # Location filter - must be within inner black circle
        dist_from_center = np.sqrt((cx - target_center[0])**2 + (cy - target_center[1])**2)
        if dist_from_center > inner_radius * 0.9:
            continue

        # Create mask for this contour
        mask = np.zeros(before_gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Measure concentration: how dark is this region?
        darker_values = darker[mask > 0]
        if len(darker_values) == 0:
            continue

        mean_darkness = np.mean(darker_values)
        max_darkness = np.max(darker_values)

        # Also measure "compactness" - is the darkness concentrated or spread out?
        # Use the ratio of dark pixels (> 15 units) to total area
        very_dark_pixels = np.sum(darker_values > 15)
        concentration = very_dark_pixels / area if area > 0 else 0

        # Require minimum darkness to be considered a bullet hole
        if mean_darkness < 8 or max_darkness < 25:
            continue

        # Score based on darkness and concentration
        score = mean_darkness * 0.4 + max_darkness * 0.3 + concentration * 100 * 0.3

        candidates.append({
            'contour': contour,
            'center': (cx, cy),
            'area': area,
            'mean_darkness': mean_darkness,
            'max_darkness': max_darkness,
            'concentration': concentration,
            'score': score,
            'distance_from_center': dist_from_center
        })

    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)

    return candidates, darker_closed


def test_frames(video_path, ref_frame, test_frame):
    """
    Test detection on specific frames
    """
    cap = cv2.VideoCapture(video_path)
    perspective = Perspective()

    # Get reference
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)
    ret, before = cap.read()
    before = cv2.rotate(before, cv2.ROTATE_90_COUNTERCLOCKWISE)
    before = perspective.apply_perspective_correction(before)
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)

    # Get test frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
    ret, after = cap.read()
    after = cv2.rotate(after, cv2.ROTATE_90_COUNTERCLOCKWISE)
    after = perspective.apply_perspective_correction(after)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    cap.release()

    # Detect target
    target_detector = TargetDetector()
    inner_circle = target_detector.detect_black_circle_improved(before)
    target_center = (int(inner_circle[0]), int(inner_circle[1]))
    inner_radius = int(inner_circle[2])

    print(f"🎯 Detecting concentrated changes")
    print(f"   Reference: Frame {ref_frame}")
    print(f"   Test: Frame {test_frame}")
    print(f"   Target: center={target_center}, radius={inner_radius}px")
    print("=" * 70)

    candidates, darker_map = detect_concentrated_changes(before_gray, after_gray, target_center, inner_radius)

    print(f"\n📊 Results: {len(candidates)} concentrated dark regions found")

    if candidates:
        print(f"\nTop 10 candidates:")
        for i, cand in enumerate(candidates[:10]):
            print(f"  #{i+1}: pos={cand['center']}, area={cand['area']:.0f}px, "
                  f"score={cand['score']:.1f}")
            print(f"        darkness: mean={cand['mean_darkness']:.1f}, max={cand['max_darkness']:.0f}, "
                  f"concentration={cand['concentration']:.2f}")

    # Create visualization
    result = after.copy()

    # Draw target
    cv2.circle(result, target_center, inner_radius, (100, 100, 100), 2)

    # Draw candidates
    for i, cand in enumerate(candidates[:5]):  # Top 5
        cx, cy = cand['center']
        color = (0, 255, 0) if i == 0 else (0, 255, 255)

        cv2.circle(result, (cx, cy), 20, color, 3)
        cv2.drawMarker(result, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)

        label = f"#{i+1} (score={cand['score']:.0f})"
        cv2.putText(result, label, (cx + 25, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add info
    cv2.putText(result, f"Frame {ref_frame} -> {test_frame}: {len(candidates)} candidates",
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Save
    import os
    output_dir = "test_outputs/concentrated_change"
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(f"{output_dir}/detection_result.jpg", result)
    cv2.imwrite(f"{output_dir}/darker_map.jpg", darker_map)

    print(f"\n💾 Saved to {output_dir}/")

    return candidates


if __name__ == "__main__":
    video_path = "samples/10-shot-1.mkv"

    # Test frame 50 -> 93 (suspected first shot)
    candidates = test_frames(video_path, ref_frame=50, test_frame=93)

    print(f"\n✅ Analysis complete!")
