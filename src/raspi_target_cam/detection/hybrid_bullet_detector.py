#!/usr/bin/env python3
"""
Hybrid Bullet Hole Detector
Combines standard detection with dark-area specialized detection
"""

import cv2
import numpy as np
from typing import List, Tuple
from raspi_target_cam.detection.bullet_hole_detection import BulletHoleDetector
from raspi_target_cam.detection.dark_area_detector import DarkAreaBulletDetector


class HybridBulletDetector:
    """
    Hybrid bullet hole detector that uses:
    1. Standard detection for light areas (outer rings)
    2. Dark-area specialized detection for black center (10-ring)
    3. Intelligent filtering to remove false positives
    """

    def __init__(self):
        # Standard detector
        self.standard_detector = BulletHoleDetector()

        # Dark area detector
        self.dark_detector = DarkAreaBulletDetector()

        # Hybrid settings
        self.dark_threshold = 60  # Pixels darker than this use dark detector
        self.max_total_holes = 20  # Maximum reasonable number of holes
        self.min_confidence = 0.25  # Minimum confidence to keep

    def detect_bullet_holes(self, before_image, after_image,
                           target_center=None) -> List[Tuple]:
        """
        Detect bullet holes using hybrid approach

        Args:
            before_image: Reference image
            after_image: Current image
            target_center: Optional (x, y) center of target for spatial filtering

        Returns:
            List of holes: [(x, y, radius, score, area, circularity), ...]
        """
        # Convert to grayscale if needed
        if len(before_image.shape) == 3:
            before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
        else:
            before_gray = before_image

        if len(after_image.shape) == 3:
            after_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)
        else:
            after_gray = after_image

        # Create dark area mask
        dark_mask = self._create_dark_mask(before_gray)

        # Create light area mask (inverse of dark)
        light_mask = cv2.bitwise_not(dark_mask)

        # Detect in light areas using standard detector
        print("   🔍 Detecting in light areas (standard method)...")
        light_holes = self._detect_in_light_areas(
            before_image, after_image, light_mask
        )
        print(f"      Found {len(light_holes)} holes in light areas")

        # Detect in dark areas using specialized detector
        print("   🔍 Detecting in dark areas (specialized method)...")
        dark_holes_raw = self.dark_detector.detect_in_dark_area(
            before_gray, after_gray, dark_mask
        )
        print(f"      Found {len(dark_holes_raw)} raw candidates in dark areas")

        # Filter dark area detections more aggressively
        dark_holes_filtered = self._filter_dark_area_holes(
            dark_holes_raw, target_center, dark_mask
        )
        print(f"      Filtered to {len(dark_holes_filtered)} high-confidence dark holes")

        # Combine results
        all_holes = light_holes + dark_holes_filtered

        # Final filtering and ranking
        final_holes = self._final_filter_and_rank(all_holes, target_center)

        print(f"   ✅ Total holes after filtering: {len(final_holes)}")

        return final_holes

    def _create_dark_mask(self, gray_image):
        """Create mask of dark areas"""
        _, dark_mask = cv2.threshold(
            gray_image, self.dark_threshold, 255, cv2.THRESH_BINARY_INV
        )

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        return dark_mask

    def _detect_in_light_areas(self, before_image, after_image, light_mask):
        """Detect holes in light areas using standard detector"""
        # Use standard detector
        holes = self.standard_detector.detect_bullet_holes(before_image, after_image)

        # Filter to only keep holes in light areas
        filtered_holes = []
        for hole in holes:
            x, y = int(hole[0]), int(hole[1])

            # Check if this hole is in light area
            if y < light_mask.shape[0] and x < light_mask.shape[1]:
                if light_mask[y, x] > 0:  # In light area
                    filtered_holes.append(hole)

        return filtered_holes

    def _filter_dark_area_holes(self, dark_holes, target_center, dark_mask):
        """
        Aggressively filter dark area holes to remove false positives

        Filtering criteria:
        1. Minimum confidence threshold
        2. Maximum distance from target center
        3. Size constraints
        4. Spatial clustering (remove isolated outliers)
        """
        if not dark_holes:
            return []

        filtered = []

        # Get dark mask dimensions to find center of dark area
        if target_center is None:
            # Find center of dark mask
            M = cv2.moments(dark_mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                target_center = (cx, cy)
            else:
                # Fallback
                h, w = dark_mask.shape
                target_center = (w // 2, h // 2)

        # Calculate max reasonable distance (radius of dark circle)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (_, _), radius = cv2.minEnclosingCircle(largest_contour)
            max_distance = radius * 1.2  # Allow slightly outside dark circle
        else:
            max_distance = 400  # Default fallback

        for hole in dark_holes:
            x, y, radius, confidence, area, circularity = hole

            # Filter 1: Minimum confidence
            if confidence < 0.35:  # Raised threshold for dark areas
                continue

            # Filter 2: Distance from target center
            dx = x - target_center[0]
            dy = y - target_center[1]
            distance = np.sqrt(dx**2 + dy**2)

            if distance > max_distance:
                continue

            # Filter 3: Size constraints (bullet holes should be reasonably sized)
            if area < 50 or area > 1500:  # Tighter bounds
                continue

            # Filter 4: Radius constraints
            if radius < 5 or radius > 35:
                continue

            filtered.append(hole)

        # Filter 5: Keep only top N by confidence
        filtered.sort(key=lambda h: h[3], reverse=True)  # Sort by confidence
        filtered = filtered[:15]  # Keep max 15 from dark area

        return filtered

    def _final_filter_and_rank(self, all_holes, target_center):
        """
        Final filtering and ranking of all detected holes

        Args:
            all_holes: Combined list from light and dark detection
            target_center: Center of target for spatial validation

        Returns:
            Filtered and ranked list of holes
        """
        if not all_holes:
            return []

        # Remove duplicates (holes detected by multiple methods)
        unique_holes = self._remove_duplicates(all_holes, distance_threshold=25)

        # Filter by confidence
        high_confidence = [h for h in unique_holes if h[3] >= self.min_confidence]

        # Sort by confidence
        high_confidence.sort(key=lambda h: h[3], reverse=True)

        # Limit total number
        final = high_confidence[:self.max_total_holes]

        return final

    def _remove_duplicates(self, holes, distance_threshold=25):
        """Remove duplicate detections"""
        if not holes:
            return []

        # Sort by confidence (keep best)
        sorted_holes = sorted(holes, key=lambda h: h[3], reverse=True)

        unique = []

        for hole in sorted_holes:
            x, y = hole[0], hole[1]

            # Check if too close to existing hole
            is_duplicate = False
            for existing in unique:
                ex, ey = existing[0], existing[1]
                distance = np.sqrt((x - ex)**2 + (y - ey)**2)

                if distance < distance_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(hole)

        return unique


def test_hybrid_detector():
    """Test hybrid detector on sample frames"""
    print("🎯 Testing Hybrid Bullet Hole Detector")
    print("=" * 60)

    # Load test frames
    before_path = "test_frames/frame_0000_clean_target_corrected.jpg"
    after_path = "test_frames/frame_0900_near_end_corrected.jpg"

    before = cv2.imread(before_path)
    after = cv2.imread(after_path)

    if before is None or after is None:
        print("❌ Could not load test frames")
        return

    print(f"✅ Loaded test frames:")
    print(f"   Before: {before_path}")
    print(f"   After:  {after_path}")
    print(f"   Resolution: {before.shape[1]}x{before.shape[0]}")

    # Detect target center (for spatial filtering)
    from target_detection import TargetDetector
    target_detector = TargetDetector()

    inner_circle = target_detector.detect_black_circle_improved(before)
    if inner_circle:
        target_center = (int(inner_circle[0]), int(inner_circle[1]))
        print(f"\n📍 Target center: {target_center}")
    else:
        target_center = None
        print(f"\n⚠️  Could not detect target center")

    # Create hybrid detector
    detector = HybridBulletDetector()

    # Detect holes
    print(f"\n🔍 Detecting bullet holes (hybrid method)...")
    holes = detector.detect_bullet_holes(before, after, target_center)

    # Display results
    print(f"\n📊 Detection Results:")
    print(f"   Total holes detected: {len(holes)}")

    if holes:
        print(f"\n   Hole details:")
        for i, (x, y, radius, confidence, area, circularity) in enumerate(holes):
            # Calculate distance from center if known
            if target_center:
                dx = x - target_center[0]
                dy = y - target_center[1]
                distance = np.sqrt(dx**2 + dy**2)
                dist_str = f"{distance:.1f}px"
            else:
                dist_str = "unknown"

            print(f"      Hole #{i+1}:")
            print(f"         Position: ({x}, {y})")
            print(f"         Distance from center: {dist_str}")
            print(f"         Radius: {radius}px")
            print(f"         Area: {area}px")
            print(f"         Circularity: {circularity:.3f}")
            print(f"         Confidence: {confidence:.3f}")

    # Create visualization
    result = after.copy()

    for i, (x, y, radius, confidence, area, circularity) in enumerate(holes):
        # Color based on confidence
        if confidence > 0.6:
            color = (0, 255, 0)  # Green - high confidence
        elif confidence > 0.4:
            color = (0, 255, 255)  # Yellow - medium
        else:
            color = (0, 165, 255)  # Orange - lower confidence

        # Draw circle
        cv2.circle(result, (x, y), radius + 5, color, 3)
        cv2.circle(result, (x, y), 3, color, -1)

        # Add label
        cv2.putText(result, f"#{i+1}", (x - 15, y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw target center
    if target_center:
        cv2.circle(result, target_center, 10, (255, 0, 0), 2)
        cv2.line(result, (target_center[0] - 20, target_center[1]),
                (target_center[0] + 20, target_center[1]), (255, 0, 0), 2)
        cv2.line(result, (target_center[0], target_center[1] - 20),
                (target_center[0], target_center[1] + 20), (255, 0, 0), 2)

    # Save result
    import os
    output_dir = "test_outputs/hybrid_detection"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "hybrid_detection_result.jpg")
    cv2.imwrite(output_path, result)

    print(f"\n💾 Saved result to: {output_path}")
    print(f"✅ Test complete!")

    return holes


if __name__ == "__main__":
    test_hybrid_detector()
