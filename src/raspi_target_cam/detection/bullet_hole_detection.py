#!/usr/bin/env python3
"""
Bullet Hole Detection System
Compares before/after frames to detect new bullet holes in targets
"""

import cv2
import numpy as np
import os
from datetime import datetime


class BulletHoleDetector:
    """Detects bullet holes by comparing before/after images"""

    def __init__(self):
        self.debug_mode = True
        self.debug_frames = {}

        # Detection parameters
        self.min_hole_area = 50  # Minimum pixels for a bullet hole (lowered for 10-ring detection)
        self.max_hole_area = 2000  # Maximum pixels for a bullet hole
        self.circularity_threshold = 0.3  # How circular the hole should be (lowered for 10-ring)
        self.difference_threshold = 20  # Minimum brightness difference (lowered for dark areas)

    def detect_bullet_holes(self, before_image, after_image):
        """
        Detect bullet holes by comparing before and after images

        Args:
            before_image: Image before shooting (numpy array or file path)
            after_image: Image after shooting (numpy array or file path)

        Returns:
            List of detected holes: [(x, y, radius, score), ...]
        """
        # Load images if paths provided
        if isinstance(before_image, str):
            before_image = cv2.imread(before_image)
        if isinstance(after_image, str):
            after_image = cv2.imread(after_image)

        if before_image is None or after_image is None:
            return []

        # Ensure images are same size
        if before_image.shape != after_image.shape:
            print(f"WARNING: Image sizes don't match: {before_image.shape} vs {after_image.shape}")
            # Resize after image to match before image
            after_image = cv2.resize(after_image, (before_image.shape[1], before_image.shape[0]))

        # Convert to grayscale
        before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        before_blur = cv2.GaussianBlur(before_gray, (5, 5), 0)
        after_blur = cv2.GaussianBlur(after_gray, (5, 5), 0)

        # Calculate difference
        diff = cv2.absdiff(before_blur, after_blur)

        # Apply threshold to get binary difference
        _, thresh = cv2.threshold(diff, self.difference_threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours of differences
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to find bullet holes
        bullet_holes = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by size
            if area < self.min_hole_area or area > self.max_hole_area:
                continue

            # Calculate circularity (4π*area/perimeter²)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Filter by circularity (bullet holes should be roughly circular)
            if circularity < self.circularity_threshold:
                continue

            # Get center and radius
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y = int(x), int(y)
            radius = int(radius)

            # Score based on area, circularity, and darkness
            # Check if the hole is darker in the after image (bullet holes are dark)
            mask = np.zeros(after_gray.shape, np.uint8)
            cv2.circle(mask, (x, y), radius, 255, -1)

            before_mean = cv2.mean(before_gray, mask=mask)[0]
            after_mean = cv2.mean(after_gray, mask=mask)[0]

            # For dark areas (like 10-ring), bullet holes might not get much darker
            # Instead, look for texture changes and slight darkness differences
            darkness_diff = before_mean - after_mean

            # Special handling for dark areas (mean < 50)
            if before_mean < 50 and after_mean < 50:
                # In dark areas, look for any detectable change and texture disruption
                if abs(darkness_diff) < 1:  # Very sensitive change threshold for dark areas
                    continue
                # For dark areas, texture change is more important than darkness
                darkness_score = min(abs(darkness_diff) / 30.0, 1.0)  # More sensitive normalization
                score = circularity * 0.8 + darkness_score * 0.2  # Weight shape heavily in dark areas
            else:
                # Normal processing for light areas - require significant darkening
                if darkness_diff < 15:  # Require at least 15 units darker
                    continue
                darkness_score = darkness_diff / 255.0
                score = circularity * 0.4 + darkness_score * 0.6  # Weight darkness more heavily

            bullet_holes.append((x, y, radius, score, area, circularity))

        # Sort by score (best first)
        bullet_holes.sort(key=lambda h: h[3], reverse=True)

        # Store debug frames
        if self.debug_mode:
            self._create_debug_frames(before_image, after_image, diff, thresh, bullet_holes)

        return bullet_holes

    def draw_bullet_hole_overlays(self, frame, bullet_holes=None):
        """
        Draw bullet hole overlays on the provided frame

        Args:
            frame: Frame to draw on
            bullet_holes: List of detected holes, or None to use last detection

        Returns:
            Frame with bullet hole overlays drawn
        """
        if bullet_holes is None:
            bullet_holes = getattr(self, "last_detection", [])

        if not bullet_holes:
            return frame

        # Make a copy to avoid modifying the original
        overlay_frame = frame.copy()

        for i, hole_data in enumerate(bullet_holes):
            if len(hole_data) >= 6:
                x, y, radius, score, area, circularity = hole_data
            else:
                # Handle shorter tuple format
                x, y, radius, score = hole_data[:4]

            # Color coding based on confidence
            if score > 0.5:
                color = (0, 255, 0)  # Green for high confidence
            elif score > 0.3:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 165, 255)  # Orange for low confidence

            # Draw main circle around bullet hole
            cv2.circle(overlay_frame, (int(x), int(y)), int(radius + 5), color, 3)

            # Draw inner circle for precise location
            # cv2.circle(overlay_frame, (int(x), int(y)), max(3, int(radius // 3)), color, 2)

            # Draw center dot
            cv2.circle(overlay_frame, (int(x), int(y)), 2, color, -1)

            # Add bullet hole number label
            label = f"#{i + 1}"
            label_pos = (int(x - 15), int(y - radius - 15))

            # Add background for text
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(
                overlay_frame,
                (label_pos[0] - 5, label_pos[1] - text_height - 5),
                (label_pos[0] + text_width + 5, label_pos[1] + 5),
                (0, 0, 0),
                -1,
            )  # Black background

            # Add text
            cv2.putText(overlay_frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Add confidence score
            # score_text = f"{score:.2f}"
            # score_pos = (int(x - 20), int(y + radius + 25))
            # cv2.putText(overlay_frame, score_text, score_pos,
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return overlay_frame

    def _create_debug_frames(self, before_img, after_img, diff, thresh, holes):
        """Create debug visualization frames"""

        # Difference visualization
        diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)

        # Binary threshold visualization
        thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Detection results on after image
        result_img = after_img.copy()

        for i, (x, y, radius, score, area, circularity) in enumerate(holes):
            # Draw detected hole
            color = (0, 255, 0) if i == 0 else (0, 255, 255)  # Green for best, yellow for others
            cv2.circle(result_img, (x, y), radius, color, 2)
            cv2.circle(result_img, (x, y), 3, color, -1)  # Center dot

            # Add text with detection info
            text = f"#{i + 1} S:{score:.2f} A:{area} C:{circularity:.2f}"
            cv2.putText(result_img, text, (x - 50, y - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Create combined debug frame
        h, w = before_img.shape[:2]
        combined = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        # Top row: before and after
        combined[0:h, 0:w] = before_img
        combined[0:h, w : w * 2] = after_img

        # Bottom row: difference and result
        combined[h : h * 2, 0:w] = diff_colored
        combined[h : h * 2, w : w * 2] = result_img

        # Add labels
        cv2.putText(combined, "BEFORE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "AFTER", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "DIFFERENCE", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "DETECTED HOLES", (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        self.debug_frames = {
            "combined": combined,
            "difference": diff_colored,
            "threshold": thresh_colored,
            "result": result_img,
        }

    def get_debug_frame(self, frame_type="combined"):
        """Get debug visualization frame"""
        return self.debug_frames.get(frame_type)

    def save_debug_frames(self, output_dir="test_outputs"):
        """Save all debug frames to disk"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for frame_type, frame in self.debug_frames.items():
            filename = f"bullet_detection_{frame_type}_{timestamp}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved debug frame: {filepath}")


def test_bullet_hole_detection():
    """Test bullet hole detection with the captured images"""

    print("🎯 Testing Bullet Hole Detection")
    print("=" * 50)

    detector = BulletHoleDetector()

    # Test with multiple scenarios
    scenarios = [
        {
            "name": "Single Hole Detection",
            "before": "captures/capture_20250925_154125.jpg",
            "after": "captures/capture_20250925_154133.jpg",
        },
        {
            "name": "Double Hole Detection",
            "before": "captures/capture_20250925_154125.jpg",
            "after": "captures/capture_20250925_160527.jpg",
        },
        {
            "name": "Center Hole Detection (10-ring)",
            "before": "captures/capture_20250925_160527.jpg",
            "after": "captures/capture_20250925_163839.jpg",
        },
    ]

    for scenario in scenarios:
        print(f"\n📊 {scenario['name']}")
        print("-" * 40)

        before_path = scenario["before"]
        after_path = scenario["after"]

        if not os.path.exists(before_path) or not os.path.exists(after_path):
            print(f"ERROR: Could not find captured images")
            print(f"Before: {before_path} (exists: {os.path.exists(before_path)})")
            print(f"After: {after_path} (exists: {os.path.exists(after_path)})")
            continue

        print(f"Comparing images:")
        print(f"Before: {before_path}")
        print(f"After:  {after_path}")
        print()

        # Detect bullet holes
        holes = detector.detect_bullet_holes(before_path, after_path)

        print(f"Detection Results:")
        print(f"Found {len(holes)} potential bullet hole(s)")
        print()

        for i, (x, y, radius, score, area, circularity) in enumerate(holes):
            print(f"Hole #{i + 1}:")
            print(f"  Position: ({x}, {y})")
            print(f"  Radius: {radius} pixels")
            print(f"  Score: {score:.3f}")
            print(f"  Area: {area} pixels")
            print(f"  Circularity: {circularity:.3f}")
            print()

        # Save debug frames with scenario name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "test_outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for frame_type, frame in detector.debug_frames.items():
            filename = f"bullet_detection_{scenario['name'].replace(' ', '_')}_{frame_type}_{timestamp}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved debug frame: {filepath}")

    return holes


if __name__ == "__main__":
    test_bullet_hole_detection()
