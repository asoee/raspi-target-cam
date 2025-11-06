#!/usr/bin/env python3
"""
Dark Area Bullet Hole Detection
Specialized algorithms for detecting bullet holes in dark target areas (10-ring black center)
"""

import cv2
import numpy as np
from typing import List, Tuple


class DarkAreaBulletDetector:
    """
    Specialized detector for bullet holes in dark areas (black 10-ring)

    Uses multiple techniques:
    1. Texture analysis (local binary patterns)
    2. Edge enhancement
    3. Local adaptive thresholding
    4. Morphological analysis
    """

    def __init__(self):
        self.min_hole_area = 30       # Smaller minimum for subtle holes
        self.max_hole_area = 2000
        self.dark_threshold = 60       # Pixels darker than this are "dark area"
        self.texture_sensitivity = 5   # Sensitivity for texture change detection

    def detect_in_dark_area(self, before_image, after_image,
                           dark_mask=None) -> List[Tuple]:
        """
        Detect bullet holes specifically in dark areas

        Args:
            before_image: Reference image (grayscale or BGR)
            after_image: Current image (grayscale or BGR)
            dark_mask: Optional mask of dark regions (if None, auto-detect)

        Returns:
            List of detected holes: [(x, y, radius, score, area, circularity), ...]
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

        # Create dark area mask if not provided
        if dark_mask is None:
            dark_mask = self._create_dark_mask(before_gray)

        # Apply multiple detection methods
        holes_texture = self._detect_by_texture_change(before_gray, after_gray, dark_mask)
        holes_edge = self._detect_by_edge_enhancement(before_gray, after_gray, dark_mask)
        holes_local = self._detect_by_local_threshold(before_gray, after_gray, dark_mask)

        # Merge results from all methods
        all_holes = holes_texture + holes_edge + holes_local

        # Remove duplicates (holes detected by multiple methods)
        merged_holes = self._merge_duplicate_detections(all_holes)

        return merged_holes

    def _create_dark_mask(self, gray_image):
        """Create binary mask of dark areas"""
        _, dark_mask = cv2.threshold(gray_image, self.dark_threshold, 255, cv2.THRESH_BINARY_INV)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        return dark_mask

    def _detect_by_texture_change(self, before_gray, after_gray, dark_mask):
        """
        Detect holes by texture disruption in dark areas
        Uses local standard deviation to measure texture
        """
        # Calculate local texture (standard deviation in local windows)
        def local_texture(img, ksize=7):
            """Calculate local standard deviation"""
            mean = cv2.blur(img.astype(np.float32), (ksize, ksize))
            mean_sq = cv2.blur((img.astype(np.float32) ** 2), (ksize, ksize))
            variance = mean_sq - (mean ** 2)
            variance = np.maximum(variance, 0)  # Handle numerical errors
            std_dev = np.sqrt(variance)
            return std_dev.astype(np.uint8)

        texture_before = local_texture(before_gray)
        texture_after = local_texture(after_gray)

        # Calculate texture difference
        texture_diff = cv2.absdiff(texture_before, texture_after)

        # Apply dark mask
        texture_diff = cv2.bitwise_and(texture_diff, texture_diff, mask=dark_mask)

        # Threshold texture changes
        _, texture_thresh = cv2.threshold(texture_diff, self.texture_sensitivity, 255, cv2.THRESH_BINARY)

        # Find contours
        holes = self._extract_hole_contours(texture_thresh, after_gray)

        return holes

    def _detect_by_edge_enhancement(self, before_gray, after_gray, dark_mask):
        """
        Detect holes by edge enhancement in dark areas
        Enhances subtle edges that appear from bullet holes
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to dark regions
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        before_enhanced = before_gray.copy()
        after_enhanced = after_gray.copy()

        # Only enhance dark areas
        before_enhanced = cv2.bitwise_and(before_enhanced, before_enhanced, mask=dark_mask)
        after_enhanced = cv2.bitwise_and(after_enhanced, after_enhanced, mask=dark_mask)

        before_enhanced = clahe.apply(before_enhanced)
        after_enhanced = clahe.apply(after_enhanced)

        # Detect edges using Canny
        edges_before = cv2.Canny(before_enhanced, 30, 100)
        edges_after = cv2.Canny(after_enhanced, 30, 100)

        # Find new edges (appeared in after image)
        new_edges = cv2.subtract(edges_after, edges_before)

        # Apply dark mask
        new_edges = cv2.bitwise_and(new_edges, new_edges, mask=dark_mask)

        # Dilate edges slightly to form regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        new_edges = cv2.dilate(new_edges, kernel, iterations=2)

        # Find contours
        holes = self._extract_hole_contours(new_edges, after_gray)

        return holes

    def _detect_by_local_threshold(self, before_gray, after_gray, dark_mask):
        """
        Detect holes using local adaptive thresholding
        Sensitive to subtle brightness changes in dark regions
        """
        # Calculate absolute difference
        diff = cv2.absdiff(before_gray, after_gray)

        # Apply dark mask
        diff = cv2.bitwise_and(diff, diff, mask=dark_mask)

        # Use very sensitive threshold for dark areas
        _, diff_thresh = cv2.threshold(diff, 3, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
        diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        holes = self._extract_hole_contours(diff_thresh, after_gray)

        return holes

    def _extract_hole_contours(self, binary_mask, reference_gray):
        """
        Extract hole candidates from binary mask

        Args:
            binary_mask: Binary image with potential holes
            reference_gray: Grayscale image for additional validation

        Returns:
            List of holes: [(x, y, radius, score, area, circularity), ...]
        """
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        holes = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by size
            if area < self.min_hole_area or area > self.max_hole_area:
                continue

            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # More lenient circularity for dark areas (holes may appear irregular)
            if circularity < 0.15:  # Very lenient
                continue

            # Get center and radius
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y = int(x), int(y)
            radius = int(radius)

            # Calculate confidence score
            # For dark areas, prioritize area and shape over darkness
            area_score = min(area / 500.0, 1.0)  # Normalize to 0-1
            shape_score = circularity

            confidence = area_score * 0.5 + shape_score * 0.5

            holes.append((x, y, radius, confidence, area, circularity))

        # Sort by confidence
        holes.sort(key=lambda h: h[3], reverse=True)

        return holes

    def _merge_duplicate_detections(self, all_holes, distance_threshold=20):
        """
        Merge holes detected by multiple methods (remove duplicates)

        Args:
            all_holes: List of all detected holes from multiple methods
            distance_threshold: Max distance in pixels to consider holes as duplicates

        Returns:
            Merged list of unique holes
        """
        if not all_holes:
            return []

        # Sort by confidence
        sorted_holes = sorted(all_holes, key=lambda h: h[3], reverse=True)

        merged = []

        for hole in sorted_holes:
            x, y = hole[0], hole[1]

            # Check if this hole is a duplicate of an already added hole
            is_duplicate = False
            for merged_hole in merged:
                mx, my = merged_hole[0], merged_hole[1]
                distance = np.sqrt((x - mx)**2 + (y - my)**2)

                if distance < distance_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                merged.append(hole)

        return merged


def test_dark_area_detection():
    """Test dark area detection on sample frames"""
    print("🎯 Testing Dark Area Bullet Hole Detection")
    print("=" * 60)

    # Load test frames (use corrected frames)
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

    # Convert to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Create detector
    detector = DarkAreaBulletDetector()

    # Detect holes
    print(f"\n🔍 Detecting bullet holes in dark areas...")
    holes = detector.detect_in_dark_area(before_gray, after_gray)

    print(f"   Found {len(holes)} holes in dark areas")

    # Display results
    if holes:
        print(f"\n📊 Detected holes:")
        for i, (x, y, radius, confidence, area, circularity) in enumerate(holes):
            print(f"      Hole #{i+1}:")
            print(f"         Position: ({x}, {y})")
            print(f"         Radius: {radius}px")
            print(f"         Area: {area}px")
            print(f"         Circularity: {circularity:.3f}")
            print(f"         Confidence: {confidence:.3f}")

    # Create visualization
    result = after.copy()

    for i, (x, y, radius, confidence, area, circularity) in enumerate(holes):
        # Color based on confidence
        if confidence > 0.5:
            color = (0, 255, 0)  # Green
        elif confidence > 0.3:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange

        # Draw circle
        cv2.circle(result, (x, y), radius + 5, color, 2)
        cv2.circle(result, (x, y), 3, color, -1)

        # Add label
        cv2.putText(result, f"#{i+1}", (x - 15, y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save result
    import os
    output_dir = "test_outputs/dark_area_detection"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "dark_area_detection_result.jpg")
    cv2.imwrite(output_path, result)

    print(f"\n💾 Saved result to: {output_path}")
    print(f"✅ Test complete!")

    return holes


if __name__ == "__main__":
    test_dark_area_detection()
