#!/usr/bin/env python3
"""
Improved Dark Area Detector
Uses darkness-based filtering to detect bullet holes in black target areas

Key insight: Bullet holes are DARKER than the surrounding black area
"""

import cv2
import numpy as np
from typing import List, Tuple


class ImprovedDarkDetector:
    """
    Improved detector for dark areas using darkness as primary criterion

    Filters:
    1. Hole must be DARKER than surrounding area
    2. Exclude ring number labels (text regions)
    3. Size constraints (not too small/large)
    4. Reasonable circularity
    """

    def __init__(self):
        self.min_hole_area = 50         # Lower threshold to detect smaller holes
        self.max_hole_area = 2000
        self.min_darkness_diff = 1.5    # Hole must be at least 1.5 units darker
        self.min_circularity = 0.10     # Very lenient for irregular/torn holes
        self.min_confidence = 0.10      # Very low to catch edge cases like merged holes
        self.darkness_check_margin = 5  # Pixels around hole to check darkness
        self.duplicate_distance = 40    # Holes closer than this are duplicates
        self.text_exclusion_zones = []  # Areas with ring numbers

        # Merged hole detection
        self.merged_hole_min_area = 800  # Large holes might be multiple merged
        self.merged_hole_max_circularity = 0.35  # Irregular shape suggests merging

    def detect_darker_holes(self, before_gray, after_gray, dark_mask,
                           target_center=None, inner_radius=None) -> List[Tuple]:
        """
        Detect holes that are darker than surrounding area

        Args:
            before_gray: Reference grayscale image
            after_gray: Current grayscale image
            dark_mask: Mask of dark target area
            target_center: Optional target center for spatial filtering

        Returns:
            List of holes: [(x, y, radius, confidence, area, circularity), ...]
        """
        # Calculate absolute difference
        diff = cv2.absdiff(before_gray, after_gray)

        # Apply dark mask - only look in dark areas
        diff_masked = cv2.bitwise_and(diff, diff, mask=dark_mask)

        # Find regions where after image is DARKER (bullet holes)
        darker_regions = cv2.subtract(before_gray, after_gray)
        darker_regions = cv2.bitwise_and(darker_regions, darker_regions, mask=dark_mask)

        # Threshold to find significantly darker regions
        _, darker_thresh = cv2.threshold(darker_regions, self.min_darkness_diff,
                                        255, cv2.THRESH_BINARY)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        darker_thresh = cv2.morphologyEx(darker_thresh, cv2.MORPH_OPEN, kernel)
        darker_thresh = cv2.morphologyEx(darker_thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(darker_thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        print(f"      🔍 DEBUG: Found {len(contours)} contours in darker_thresh")

        holes = []
        filtered_out = {
            'too_small': 0,
            'too_large': 0,
            'low_circularity': 0,
            'text_zone': 0,
            'spatial': 0,
            'not_dark_enough': 0
        }

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Get center and radius for tracking
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y = int(x), int(y)
            radius = int(radius)

            # Log large contours for debugging
            if area > 800:
                print(f"         Large contour #{idx}: pos=({x},{y}), area={area:.0f}px, checking filters...")

            # Filter by size
            if area < self.min_hole_area:
                filtered_out['too_small'] += 1
                continue
            if area > self.max_hole_area:
                if area > 800:
                    print(f"           FILTERED: too large (>{self.max_hole_area})")
                filtered_out['too_large'] += 1
                continue

            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Circularity check - now with lowered threshold
            if circularity < self.min_circularity:
                if area > 800:
                    print(f"           FILTERED: low circularity={circularity:.2f} (<{self.min_circularity})")
                filtered_out['low_circularity'] += 1
                continue

            # Log if large irregular contour passed
            if area >= self.merged_hole_min_area and circularity < 0.20:
                if area > 800:
                    print(f"           Large irregular contour PASSED (circ={circularity:.2f})")

            # Check if this is in a text exclusion zone (ring number)
            if self._is_in_text_zone(x, y):
                if area > 800:
                    print(f"           FILTERED: in text zone")
                filtered_out['text_zone'] += 1
                continue

            # Spatial filtering: must be within reasonable distance from target center
            if target_center:
                dx_spatial = x - target_center[0]
                dy_spatial = y - target_center[1]
                distance_from_center = np.sqrt(dx_spatial**2 + dy_spatial**2)

                # Assume dark mask radius is approximately the inner black circle
                # Holes should be well within the black circle
                # Use 0.9x to exclude edge detections (likely ring numbers)
                max_distance = inner_radius * 0.9 if inner_radius else 600

                if distance_from_center > max_distance:
                    if area > 800:
                        print(f"           FILTERED: spatial (dist={distance_from_center:.0f}px > max={max_distance:.0f}px)")
                    filtered_out['spatial'] += 1
                    continue

            # Verify darkness in after image
            # Compare hole region to surrounding ring
            inner_mask = np.zeros(after_gray.shape, np.uint8)
            cv2.circle(inner_mask, (x, y), radius, 255, -1)

            outer_mask = np.zeros(after_gray.shape, np.uint8)
            cv2.circle(outer_mask, (x, y), radius + self.darkness_check_margin, 255, -1)

            # Ring = outer circle minus inner circle
            ring_mask = cv2.subtract(outer_mask, inner_mask)

            # Darkness in before/after for hole area
            before_hole_mean = cv2.mean(before_gray, mask=inner_mask)[0]
            after_hole_mean = cv2.mean(after_gray, mask=inner_mask)[0]

            # Darkness in surrounding ring (after image)
            after_ring_mean = cv2.mean(after_gray, mask=ring_mask)[0]

            # Two checks:
            # 1. Hole became darker (before -> after)
            # 2. Hole is darker than surrounding area (in after image)
            darkness_diff_temporal = before_hole_mean - after_hole_mean
            darkness_diff_spatial = after_ring_mean - after_hole_mean

            # Use the maximum of both (either temporal or spatial darkness)
            darkness_diff = max(darkness_diff_temporal, darkness_diff_spatial)

            # Must show some darkness
            # BUT: for large irregular contours (possible merged holes), skip darkness check
            is_large_irregular = (area >= self.merged_hole_min_area and circularity < 0.20)

            if is_large_irregular:
                # Skip darkness check for large irregular contours - they're likely merged holes
                # with complex lighting
                if area > 800:
                    print(f"           Skipping darkness check for large irregular contour (darkness={darkness_diff:.1f})")
            elif darkness_diff < self.min_darkness_diff:
                if area > 800:
                    print(f"           FILTERED: not dark enough (darkness={darkness_diff:.1f} < {min_darkness_required})")
                filtered_out['not_dark_enough'] += 1
                continue

            # Passed all filters!
            if area > 800:
                print(f"           PASSED all filters! conf={confidence:.2f}, darkness={darkness_diff:.1f}")

            # Calculate confidence score based on:
            # - How much darker (primary)
            # - Circularity (secondary)
            # - Size (tertiary)
            darkness_score = min(darkness_diff / 15.0, 1.0)  # Normalize
            shape_score = circularity
            size_score = min(area / 500.0, 1.0)

            confidence = (darkness_score * 0.6 +  # Darkness is most important
                         shape_score * 0.3 +
                         size_score * 0.1)

            holes.append((x, y, radius, confidence, area, circularity, darkness_diff))

        # Print filtering statistics
        print(f"      🔍 DEBUG: Filtering statistics:")
        print(f"         Passed: {len(holes)}")
        print(f"         Filtered out: too_small={filtered_out['too_small']}, too_large={filtered_out['too_large']}")
        print(f"                       low_circ={filtered_out['low_circularity']}, text_zone={filtered_out['text_zone']}")
        print(f"                       spatial={filtered_out['spatial']}, not_dark={filtered_out['not_dark_enough']}")

        # Sort by confidence
        holes.sort(key=lambda h: h[3], reverse=True)

        print(f"      🔍 DEBUG: Before duplicate removal: {len(holes)} holes")

        # Remove duplicates (keep highest confidence)
        holes = self._remove_duplicates(holes)

        print(f"      🔍 DEBUG: After duplicate removal: {len(holes)} holes")
        print(f"      🔍 DEBUG: Before confidence filter: showing all holes:")
        for i, h in enumerate(holes):
            print(f"         #{i+1}: pos=({h[0]},{h[1]}), conf={h[3]:.2f}, area={h[4]:.0f}, circ={h[5]:.2f}")

        # Filter by minimum confidence
        holes = [h for h in holes if h[3] >= self.min_confidence]

        print(f"      🔍 DEBUG: After confidence filter (>={self.min_confidence}): {len(holes)} holes")

        # TEMPORARILY disable merged hole splitting
        #merged_candidates, split_holes = self._detect_merged_holes(holes, after_gray)
        merged_candidates = []
        split_holes = []

        # If we successfully split merged holes, remove the merged one and add splits
        if False and split_holes:
            # Remove merged holes that were split
            merged_hole_positions = [(m['hole'][0], m['hole'][1]) for m in merged_candidates]

            holes_filtered = []
            for hole in holes:
                is_merged = False
                for mx, my in merged_hole_positions:
                    if abs(hole[0] - mx) < 10 and abs(hole[1] - my) < 10:
                        is_merged = True
                        break

                if not is_merged:
                    holes_filtered.append(hole)

            # Add split holes
            holes = holes_filtered + split_holes

            # Remove duplicates again after adding splits
            holes = self._remove_duplicates(holes)

        return holes, merged_candidates

    def detect_text_zones(self, image, dark_mask):
        """
        Detect areas that contain ring number labels
        These should be excluded from bullet hole detection

        Uses edge detection to find text-like patterns
        """
        # Apply CLAHE to enhance contrast in dark areas
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Only process dark areas
        dark_region = cv2.bitwise_and(image, image, mask=dark_mask)
        enhanced = clahe.apply(dark_region)

        # Detect edges (text has strong edges)
        edges = cv2.Canny(enhanced, 50, 150)

        # Dilate to connect text edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours of potential text
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        text_zones = []

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h

            # Text typically has specific aspect ratio and size
            aspect_ratio = w / float(h) if h > 0 else 0

            # Ring numbers are typically small, roughly square
            if (20 < w < 100 and 20 < h < 100 and
                0.3 < aspect_ratio < 3.0 and
                area > 400):

                # Expand zone slightly to ensure exclusion
                margin = 10
                text_zones.append({
                    'x1': max(0, x - margin),
                    'y1': max(0, y - margin),
                    'x2': x + w + margin,
                    'y2': y + h + margin
                })

        self.text_exclusion_zones = text_zones

        return text_zones

    def _is_in_text_zone(self, x, y):
        """Check if point (x,y) is inside any text exclusion zone"""
        for zone in self.text_exclusion_zones:
            if (zone['x1'] <= x <= zone['x2'] and
                zone['y1'] <= y <= zone['y2']):
                return True
        return False

    def _remove_duplicates(self, holes):
        """
        Remove duplicate hole detections
        Holes closer than duplicate_distance are considered duplicates
        Keep the one with highest confidence
        """
        if not holes:
            return []

        unique_holes = []

        for hole in holes:
            x, y = hole[0], hole[1]

            # Check if too close to existing hole
            is_duplicate = False
            for existing in unique_holes:
                ex, ey = existing[0], existing[1]
                distance = np.sqrt((x - ex)**2 + (y - ey)**2)

                if distance < self.duplicate_distance:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_holes.append(hole)

        return unique_holes

    def _detect_merged_holes(self, holes, gray_image):
        """
        Detect holes that might actually be 2+ close holes merged/torn together
        Attempts to split them into individual holes

        Characteristics of merged holes:
        - Large area (> merged_hole_min_area)
        - Low circularity (irregular shape from tearing)
        - High aspect ratio (elongated)

        Returns list of split holes
        """
        merged_candidates = []
        split_holes = []

        for hole in holes:
            x, y, radius, confidence, area, circularity = hole[:6]

            # Check if this might be a merged hole
            if (area >= self.merged_hole_min_area and
                circularity <= self.merged_hole_max_circularity):

                merged_candidates.append({
                    'hole': hole,
                    'reason': f'Large irregular (area={area}, circ={circularity:.2f})'
                })

                # Try to split this hole into multiple individual holes
                split_result = self._split_merged_hole(hole, gray_image)
                if split_result:
                    split_holes.extend(split_result)
                    print(f"      🔍 Split merged hole at ({x},{y}) into {len(split_result)} holes")
                else:
                    print(f"      ⚠️  Could not split merged hole at ({x},{y})")

        return merged_candidates, split_holes

    def _split_large_contour_early(self, contour, gray_image, x, y, radius):
        """
        Split a large irregular contour into individual holes during initial filtering
        This is called BEFORE circularity filtering for large contours

        Returns list of split holes, or None if splitting fails
        """
        # Create ROI around the contour
        margin = int(radius * 2.5)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(gray_image.shape[1], x + margin)
        y2 = min(gray_image.shape[0], y + margin)

        roi = gray_image[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        # Create mask from contour
        contour_mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)

        # Translate contour to ROI coordinates
        contour_translated = contour.copy()
        contour_translated[:, 0, 0] -= x1
        contour_translated[:, 0, 1] -= y1

        cv2.drawContours(contour_mask, [contour_translated], -1, 255, -1)

        # Apply mask to ROI
        masked_roi = cv2.bitwise_and(roi, roi, mask=contour_mask)

        # Use distance transform to separate touching holes
        # This is better than simple thresholding for merged objects

        # First, apply moderate threshold to get the general dark region
        mean_intensity = np.mean(masked_roi[contour_mask > 0])
        _, binary = cv2.threshold(masked_roi, mean_intensity * 0.7, 255, cv2.THRESH_BINARY_INV)

        # Apply distance transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Threshold the distance transform to find peaks (hole centers)
        # Peaks are the points furthest from edges
        _, peaks = cv2.threshold(dist_transform, dist_transform.max() * 0.5, 255, cv2.THRESH_BINARY)
        peaks = np.uint8(peaks)

        print(f"             Distance transform: max_dist={dist_transform.max():.1f}, peak_pixels={np.count_nonzero(peaks)}")

        # Find connected components in peaks
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks, connectivity=8)
        print(f"             Found {num_labels-1} peak components")

        # Filter components
        min_component_area = 50
        valid_components = []

        for i in range(1, num_labels):
            component_area = stats[i, cv2.CC_STAT_AREA]

            if component_area >= min_component_area:
                cx_roi = int(centroids[i][0])
                cy_roi = int(centroids[i][1])

                cx_full = x1 + cx_roi
                cy_full = y1 + cy_roi

                valid_components.append({
                    'center': (cx_full, cy_full),
                    'area': component_area
                })

        print(f"             Valid components (>={min_component_area}px): {len(valid_components)}")

        # Need at least 2 components to be considered a split
        if len(valid_components) >= 2:
            split_holes = []

            for comp in valid_components:
                cx, cy = comp['center']
                comp_radius = int(np.sqrt(comp['area'] / np.pi))
                comp_radius = max(10, min(comp_radius, 30))

                # Estimate darkness at this position
                # Sample in small region around center
                sample_size = 5
                sy1 = max(0, cy - sample_size)
                sy2 = min(gray_image.shape[0], cy + sample_size)
                sx1 = max(0, cx - sample_size)
                sx2 = min(gray_image.shape[1], cx + sample_size)

                darkness = np.mean(gray_image[sy1:sy2, sx1:sx2])

                # Create hole tuple with estimated values
                split_hole = (
                    cx, cy,              # x, y
                    comp_radius,         # radius
                    0.5,                 # moderate confidence (will be recalculated if needed)
                    float(comp['area']), # area
                    0.7,                 # assume reasonable circularity for splits
                    10.0                 # placeholder darkness diff
                )

                split_holes.append(split_hole)

            return split_holes

        return None

    def _split_merged_hole(self, merged_hole, gray_image):
        """
        Attempt to split a merged hole into individual holes
        Uses distance transform to find individual hole centers

        Returns list of split holes, or None if splitting fails
        """
        x, y, radius, confidence, area, circularity, darkness = merged_hole

        print(f"         🔬 DEBUG: Attempting to split hole at ({x},{y}), radius={radius}")

        # Create ROI around the merged hole - use larger margin
        margin = int(radius * 2.5)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(gray_image.shape[1], x + margin)
        y2 = min(gray_image.shape[0], y + margin)

        roi = gray_image[y1:y2, x1:x2]
        print(f"         🔬 DEBUG: ROI size={roi.shape}, coords=({x1},{y1}) to ({x2},{y2})")

        if roi.size == 0:
            print(f"         🔬 DEBUG: ROI is empty, returning None")
            return None

        # Create binary mask of dark regions in ROI
        # Use adaptive threshold to find dark areas
        mean_intensity = np.mean(roi)
        threshold_val = max(mean_intensity * 0.8, 20)  # Dark holes should be < 80% of mean
        _, dark_mask = cv2.threshold(roi, threshold_val, 255, cv2.THRESH_BINARY_INV)

        print(f"         🔬 DEBUG: mean_intensity={mean_intensity:.1f}, threshold={threshold_val:.1f}")
        print(f"         🔬 DEBUG: dark_mask has {np.count_nonzero(dark_mask)} dark pixels")

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        # Find separate connected components (individual holes)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)

        # Filter components by size
        min_component_area = 100
        valid_components = []

        for i in range(1, num_labels):  # Skip background (label 0)
            component_area = stats[i, cv2.CC_STAT_AREA]

            if component_area >= min_component_area:
                # Get centroid in ROI coordinates
                cx_roi = int(centroids[i][0])
                cy_roi = int(centroids[i][1])

                # Convert to full image coordinates
                cx_full = x1 + cx_roi
                cy_full = y1 + cy_roi

                valid_components.append({
                    'center': (cx_full, cy_full),
                    'area': component_area
                })

        print(f"         🔬 DEBUG: Found {len(valid_components)} valid components")

        if len(valid_components) > 0:
            for i, comp in enumerate(valid_components):
                cx, cy = comp['center']
                print(f"         🔬 DEBUG:   Component {i+1}: center=({cx}, {cy}), area={comp['area']}")

        # If we found 2+ components, create split holes
        if len(valid_components) >= 2:
            split_holes = []

            for comp in valid_components:
                cx, cy = comp['center']

                # Estimate radius based on component area
                comp_radius = int(np.sqrt(comp['area'] / np.pi))
                comp_radius = max(10, min(comp_radius, 30))

                # Create hole tuple
                split_hole = (
                    cx, cy,                      # x, y
                    comp_radius,                 # radius
                    confidence * 0.8,            # slightly lower confidence for split holes
                    comp['area'],                # area
                    0.7,                         # assume reasonable circularity
                    darkness / len(valid_components)  # darkness divided among splits
                )

                split_holes.append(split_hole)

            print(f"         🔬 DEBUG: Created {len(split_holes)} split holes")
            return split_holes
        else:
            print(f"         🔬 DEBUG: Only {len(valid_components)} component(s) found, need 2+, returning None")

        return None


def test_improved_detector():
    """Test improved detector on frame with all 10 holes"""
    print("🎯 Testing Improved Dark Area Detector")
    print("=" * 60)

    before_path = "test_frames/frame_0000_clean_target_corrected.jpg"
    after_path = "test_frames/frame_0930_all_10_shots_corrected.jpg"

    before = cv2.imread(before_path)
    after = cv2.imread(after_path)

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Detect target
    from target_detection import TargetDetector
    target_detector = TargetDetector()

    inner_circle = target_detector.detect_black_circle_improved(before)
    if inner_circle:
        target_center = (int(inner_circle[0]), int(inner_circle[1]))
        inner_radius = int(inner_circle[2])
    else:
        target_center = (before.shape[1] // 2, before.shape[0] // 2)
        inner_radius = 300

    print(f"✅ Target center: {target_center}, radius: {inner_radius}px")

    # Create dark mask
    _, dark_mask = cv2.threshold(before_gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    # Create improved detector
    detector = ImprovedDarkDetector()

    # Detect text zones (ring numbers)
    print(f"\n🔍 Detecting ring number text zones...")
    text_zones = detector.detect_text_zones(before_gray, dark_mask)
    print(f"   Found {len(text_zones)} text zones to exclude")

    # Detect darker holes
    print(f"\n🔍 Detecting darker holes in black area...")
    holes, merged_candidates = detector.detect_darker_holes(before_gray, after_gray, dark_mask,
                                                            target_center, inner_radius)

    print(f"   Found {len(holes)} darker holes")
    if merged_candidates:
        print(f"   ⚠️  Found {len(merged_candidates)} possible merged/torn holes:")
        for mc in merged_candidates:
            hole = mc['hole']
            print(f"      - pos=({hole[0]},{hole[1]}), {mc['reason']}")

    # Also detect in light areas using standard method
    print(f"\n🔍 Detecting in light areas (standard method)...")
    from bullet_hole_detection import BulletHoleDetector
    standard_detector = BulletHoleDetector()

    # Create light mask
    light_mask = cv2.bitwise_not(dark_mask)

    # Detect in light areas
    all_holes_standard = standard_detector.detect_bullet_holes(before, after)

    # Filter to light areas only
    light_holes = []
    for hole in all_holes_standard:
        x, y = int(hole[0]), int(hole[1])
        if y < light_mask.shape[0] and x < light_mask.shape[1]:
            if light_mask[y, x] > 0:
                light_holes.append(hole)

    print(f"   Found {len(light_holes)} holes in light areas")

    # Combine results
    all_holes = holes + light_holes

    print(f"\n📊 Total Detection Results:")
    print(f"   Dark area (improved): {len(holes)} holes")
    print(f"   Light area (standard): {len(light_holes)} holes")
    print(f"   TOTAL: {len(all_holes)} holes")
    print(f"   Expected: ~10 holes")

    # Show details
    if holes:
        print(f"\n🎯 Dark Area Holes (with darkness values):")
        for i, hole in enumerate(holes):
            x, y, radius, conf, area, circ, darkness = hole
            dx = x - target_center[0]
            dy = y - target_center[1]
            dist = np.sqrt(dx**2 + dy**2)
            print(f"      #{i+1}: pos=({x},{y}), dist={dist:.0f}px, "
                  f"conf={conf:.2f}, darkness={darkness:.1f}, area={area:.0f}px")

    # Create visualization
    result = after.copy()

    # Draw text exclusion zones
    for zone in text_zones:
        cv2.rectangle(result,
                     (zone['x1'], zone['y1']),
                     (zone['x2'], zone['y2']),
                     (0, 0, 255), 2)  # Red rectangles for text zones

    # Draw target
    cv2.circle(result, target_center, inner_radius, (100, 100, 100), 2)
    cv2.drawMarker(result, target_center, (0, 255, 0), cv2.MARKER_CROSS, 30, 3)

    # Draw dark area holes
    for i, hole in enumerate(holes):
        x, y, radius, conf = int(hole[0]), int(hole[1]), int(hole[2]), hole[3]

        color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)

        cv2.circle(result, (x, y), radius + 5, color, 3)
        cv2.circle(result, (x, y), 3, color, -1)

        label = f"D{i+1}"
        cv2.putText(result, label, (x - 15, y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw light area holes
    for i, hole in enumerate(light_holes):
        x, y, radius, conf = int(hole[0]), int(hole[1]), int(hole[2]), hole[3]

        color = (0, 165, 255)  # Orange for light areas

        cv2.circle(result, (x, y), radius + 5, color, 3)
        cv2.circle(result, (x, y), 3, color, -1)

        label = f"L{i+1}"
        cv2.putText(result, label, (x - 15, y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add legend
    legend_height = 150
    legend = np.zeros((legend_height, result.shape[1], 3), dtype=np.uint8)

    y_off = 30
    cv2.putText(legend, f"Improved Detection Results - {len(all_holes)} Total Holes",
               (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    y_off += 40
    cv2.putText(legend, f"Green (D#): Dark area holes ({len(holes)}) - using darkness filtering",
               (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    y_off += 35
    cv2.putText(legend, f"Orange (L#): Light area holes ({len(light_holes)}) - standard detection",
               (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    y_off += 35
    cv2.putText(legend, f"Red boxes: Excluded text zones (ring numbers) - {len(text_zones)} zones",
               (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    final = np.vstack([legend, result])

    # Save
    import os
    output_dir = "test_outputs/improved_detection"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/improved_darkness_based.jpg"
    cv2.imwrite(output_path, final)

    print(f"\n💾 Saved to: {output_path}")
    print(f"✅ Test complete!")

    return all_holes


if __name__ == "__main__":
    test_improved_detector()
