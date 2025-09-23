#!/usr/bin/env python3
"""
Target Detection Module
Detects shooting target circles and bullet holes using OpenCV
"""

import cv2
import numpy as np
import time
import yaml
import os


class TargetDetector:
    """Detects shooting targets and bullet holes"""

    def __init__(self):
        self.target_center = None
        self.target_radius = None
        self.detection_enabled = True

        # Detection stability
        self.detection_history = []
        self.max_history = 5
        self.stable_detection = None
        self.detection_confidence = 0

        # Outer circle detection
        self.outer_circle = None
        self.outer_circle_history = []
        self.stable_outer_circle = None
        self.outer_confidence = 0

        # Performance optimization - periodic detection
        self.last_detection_time = 0
        self.detection_interval = 1.0  # Run detection every 1 second
        self.cached_inner_result = None
        self.cached_outer_result = None

        # Frame change detection
        self.last_frame_gray = None
        self.frame_change_threshold = 0.05  # 5% change triggers re-detection

        # Perspective correction
        self.target_corners = None
        self.perspective_matrix = None
        self.corrected_target_size = (800, 800)  # Standard square output
        self.min_target_area_ratio = 0.01  # Minimum 1% of frame area

        # Debug visualization
        self.debug_mode = False
        self.debug_type = "combined"  # "combined", "perspective", "circles"
        self.debug_frame = None
        self.debug_contours = None
        self.corner_debug_frame = None
        self.circle_debug_frame = None

        # Perspective calibration
        self.calibration_file = "perspective_calibration.yaml"
        self.calibration_mode = False  # When true, performs detection on demand
        self.saved_perspective_matrix = None
        self.saved_ellipse_data = None
        self.calibration_resolution = None

        # Debug frames
        self.corrected_debug_frame = None  # For perspective-corrected debug view

        # Load saved calibration on startup
        self.load_perspective_calibration()

    def detect_black_circle_improved(self, frame):
        """
        Improved detection specifically for the large black circle target
        Automatically scales parameters based on frame resolution
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get frame dimensions for scaling
        h, w = frame.shape[:2]

        # Scale parameters based on resolution
        # Base resolution: 640x480, scaling factor for larger resolutions
        scale_factor = max(w / 640, h / 480)

        # Apply median blur to reduce noise while preserving edges
        blur_size = max(5, int(5 * scale_factor) | 1)  # Ensure odd number
        blurred = cv2.medianBlur(gray, blur_size)

        # Create binary mask for very dark regions (black target area)
        # This threshold isolates the black circle from lighter areas
        _, binary = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

        # Remove small noise with morphological operations
        kernel_size = max(5, int(5 * scale_factor))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Create circle detection debug visualization
        if self.debug_mode:
            self.circle_debug_frame = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            cv2.putText(self.circle_debug_frame, "CIRCLE DETECTION DEBUG",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(self.circle_debug_frame, f"Threshold: 80, Scale: {scale_factor:.1f}",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours for the main black circle
        best_circle = None
        best_score = 0

        # Scale size thresholds based on resolution
        min_area = int(5000 * scale_factor * scale_factor)
        max_area = int(100000 * scale_factor * scale_factor)
        min_radius = int(40 * scale_factor)
        max_radius = int(400 * scale_factor)

        for contour in contours:
            area = cv2.contourArea(contour)

            # Size filtering - target should be reasonably large
            if area < min_area or area > max_area:
                continue

            # Fit circle to contour
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Check if circle is reasonably sized and positioned
            if (radius < min_radius or radius > max_radius or
                x < radius or y < radius or
                x + radius > w or y + radius > h):
                continue

            # Calculate circularity (how round the contour is)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Check how well the contour fits a circle
            circle_area = np.pi * radius * radius
            area_ratio = area / circle_area if circle_area > 0 else 0

            # Position preference - prefer circles closer to center of frame
            center_x, center_y = w // 2, h // 2
            distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            center_score = 1.0 - (distance_from_center / max_distance)

            # Calculate darkness score for this region
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            mean_intensity = cv2.mean(gray, mask=mask)[0]
            darkness_score = (255 - mean_intensity) / 255  # Higher for darker regions

            # Combined score: circularity + area fit + darkness + center position
            score = (circularity * 0.3 +
                    area_ratio * 0.3 +
                    darkness_score * 0.3 +
                    center_score * 0.1)

            # Must be reasonably circular and dark
            if circularity > 0.6 and darkness_score > 0.3 and score > best_score:
                best_score = score
                best_circle = (center[0], center[1], radius)

        # Add debug visualization for circle detection
        if self.debug_mode and self.circle_debug_frame is not None:
            # Draw all contours in blue
            cv2.drawContours(self.circle_debug_frame, contours, -1, (255, 0, 0), 2)

            # Draw the best circle in green if found
            if best_circle is not None:
                center = (int(best_circle[0]), int(best_circle[1]))
                radius = int(best_circle[2])
                cv2.circle(self.circle_debug_frame, center, radius, (0, 255, 0), 3)
                cv2.circle(self.circle_debug_frame, center, 5, (0, 0, 255), -1)
                cv2.putText(self.circle_debug_frame, f"Circle: r={radius}",
                           (center[0] - 80, center[1] - radius - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            cv2.putText(self.circle_debug_frame, f"Contours: {len(contours)}, Best Score: {best_score:.2f}",
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        return best_circle

    def detect_outer_circle(self, frame, inner_circle=None):
        """
        Detect the outermost thin black circle using edge detection
        Automatically scales parameters based on frame resolution
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get frame dimensions for scaling
        h, w = frame.shape[:2]
        scale_factor = max(w / 640, h / 480)

        # Apply Gaussian blur to reduce noise - scale blur kernel
        blur_size = max(5, int(5 * scale_factor) | 1)  # Ensure odd number
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 1)

        # Use Canny edge detection to find edges
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Scale parameters for HoughCircles
        min_dist = int(50 * scale_factor)
        min_radius = int(100 * scale_factor)
        max_radius = int(500 * scale_factor)
        param2 = max(15, int(25 * scale_factor / 2))  # Lower threshold for larger images

        # Use HoughCircles specifically for edge-based circle detection
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,                    # Inverse ratio of accumulator resolution
            minDist=min_dist,        # Minimum distance between circle centers
            param1=50,               # Upper threshold for edge detection (already applied)
            param2=param2,           # Accumulator threshold for center detection
            minRadius=min_radius,    # Minimum radius for outer circle
            maxRadius=max_radius     # Maximum radius for outer circle
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            best_outer_circle = None
            best_score = 0

            for (x, y, r) in circles:
                # Check if circle is within frame bounds
                h, w = frame.shape[:2]
                if (x - r < 0 or y - r < 0 or
                    x + r >= w or y + r >= h):
                    continue

                # If we have an inner circle, outer should be concentric and larger
                if inner_circle is not None:
                    inner_x, inner_y, inner_r = inner_circle
                    distance_between_centers = np.sqrt((x - inner_x)**2 + (y - inner_y)**2)

                    # Scale distance thresholds
                    max_center_distance = int(30 * scale_factor)
                    min_radius_diff = int(20 * scale_factor)

                    # Centers should be close (concentric) and outer should be larger
                    if distance_between_centers > max_center_distance or r <= inner_r + min_radius_diff:
                        continue

                # Prefer circles closer to center of frame
                center_x, center_y = w // 2, h // 2
                distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                center_score = 1.0 - (distance_from_center / max_distance)

                # Check edge density around the circle perimeter
                edge_score = self._calculate_edge_score(edges, (x, y), r)

                # Combined score
                score = center_score * 0.4 + edge_score * 0.6

                if score > best_score:
                    best_score = score
                    best_outer_circle = (x, y, r)

            return best_outer_circle

        return None

    def _calculate_edge_score(self, edges, center, radius):
        """
        Calculate how many edge pixels are on the circle perimeter
        """
        x, y = center
        edge_count = 0
        total_points = 0

        # Sample points around the circle
        for angle in range(0, 360, 5):  # Sample every 5 degrees
            angle_rad = np.radians(angle)
            px = int(x + radius * np.cos(angle_rad))
            py = int(y + radius * np.sin(angle_rad))

            if 0 <= px < edges.shape[1] and 0 <= py < edges.shape[0]:
                total_points += 1
                if edges[py, px] > 0:  # Edge pixel found
                    edge_count += 1

        return edge_count / total_points if total_points > 0 else 0

    def _has_significant_frame_change(self, frame):
        """
        Check if the current frame has changed significantly from the last one
        """
        # Convert to grayscale for comparison
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.last_frame_gray is None:
            self.last_frame_gray = current_gray.copy()
            return True  # First frame, always detect

        # Resize for faster comparison (quarter resolution)
        h, w = current_gray.shape
        small_current = cv2.resize(current_gray, (w//4, h//4))
        small_last = cv2.resize(self.last_frame_gray, (w//4, h//4))

        # Calculate absolute difference
        diff = cv2.absdiff(small_current, small_last)

        # Calculate percentage of changed pixels
        total_pixels = diff.shape[0] * diff.shape[1]
        changed_pixels = np.count_nonzero(diff > 30)  # Threshold for "significant" change
        change_percentage = changed_pixels / total_pixels

        # Update last frame
        self.last_frame_gray = current_gray.copy()

        return change_percentage > self.frame_change_threshold

    def _should_run_detection(self, frame):
        """
        Determine if we should run detection based on time and frame changes
        """
        current_time = time.time()

        # Always run detection if interval has passed
        if current_time - self.last_detection_time >= self.detection_interval:
            return True

        # Run detection if significant frame change detected
        if self._has_significant_frame_change(frame):
            return True

        return False

    def detect_ellipse_perspective(self, frame):
        """
        Detect perspective transformation based on outer circle appearing as ellipse
        Returns transformation matrix to correct perspective distortion
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # Use edge detection to find the outer ring
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Debug tracking for rejection reasons
        rejection_reasons = []
        analyzed_contours = 0

        best_ellipse = None
        best_score = 0

        # Look for elliptical contours that could be the outer circle
        for contour in contours:
            if len(contour) < 5:  # Need at least 5 points to fit ellipse
                rejection_reasons.append(f"Contour {len(rejection_reasons)+1}: <5 points ({len(contour)})")
                continue

            area = cv2.contourArea(contour)
            if area < 1000:  # Skip small contours
                # Skip logging "too small" contours to reduce clutter
                continue

            analyzed_contours += 1

            try:
                # Fit ellipse to contour
                ellipse = cv2.fitEllipse(contour)
                (center_x, center_y), (minor_axis, major_axis), angle = ellipse

                # Calculate aspect ratio (how "stretched" the ellipse is)
                aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)

                # Skip if too circular (no perspective) or too elongated (likely noise)
                if aspect_ratio < 1.02:
                    rejection_reasons.append(f"Contour {analyzed_contours}: too circular (ratio={aspect_ratio:.2f})")
                    continue
                elif aspect_ratio > 4.0:
                    rejection_reasons.append(f"Contour {analyzed_contours}: too elongated (ratio={aspect_ratio:.2f})")
                    continue

                # Score based on size and reasonable ellipse properties
                size_score = area / (w * h)  # Prefer larger ellipses
                aspect_score = 1.0 / aspect_ratio  # Prefer moderate distortion

                # Check if ellipse is reasonably centered
                center_score = 1.0 - (abs(center_x - w/2) + abs(center_y - h/2)) / (w + h)

                score = size_score * 0.5 + aspect_score * 0.3 + center_score * 0.2

                if score > best_score:
                    best_score = score
                    best_ellipse = ellipse
                    rejection_reasons.append(f"Contour {analyzed_contours}: ACCEPTED (score={score:.3f}, ratio={aspect_ratio:.2f})")
                else:
                    rejection_reasons.append(f"Contour {analyzed_contours}: low score ({score:.3f} vs {best_score:.3f})")

            except cv2.error as e:
                rejection_reasons.append(f"Contour {analyzed_contours}: ellipse fit failed ({str(e)[:20]})")
                continue

        # Create debug visualization for ellipse detection
        if self.debug_mode:
            self.corner_debug_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.putText(self.corner_debug_frame, "ELLIPSE PERSPECTIVE DETECTION",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(self.corner_debug_frame, f"Edge contours: {len(contours)} (analyzed: {analyzed_contours})",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # Add debug visualization for ellipse detection
        if self.debug_mode and self.corner_debug_frame is not None:
            # Draw all significant contours in blue
            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    cv2.drawContours(self.corner_debug_frame, [contour], -1, (255, 0, 0), 1)

            # Draw the best ellipse in green if found
            if best_ellipse is not None:
                cv2.ellipse(self.corner_debug_frame, best_ellipse, (0, 255, 0), 4)
                (center_x, center_y), (minor_axis, major_axis), angle = best_ellipse
                aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)
                cv2.putText(self.corner_debug_frame, f"Ellipse: ratio={aspect_ratio:.2f}",
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.putText(self.corner_debug_frame, f"Score: {best_score:.3f}",
                           (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

                # Add visual debugging for transformation points
                # Calculate the same points that will be used for transformation
                semi_major = major_axis / 2
                semi_minor = minor_axis / 2
                angle_rad = np.radians(angle)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

                # Calculate extreme points
                a, b = semi_major, semi_minor
                x_extent = np.sqrt((a * cos_a)**2 + (b * sin_a)**2)
                y_extent = np.sqrt((a * sin_a)**2 + (b * cos_a)**2)

                # Mark the transformation points on the debug frame
                debug_points = [
                    (int(center_x + x_extent), int(center_y)),      # Right
                    (int(center_x), int(center_y - y_extent)),      # Top
                    (int(center_x - x_extent), int(center_y)),      # Left
                    (int(center_x), int(center_y + y_extent))       # Bottom
                ]

                colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 128, 0)]  # Different colors for each point
                labels = ["R", "T", "L", "B"]

                for i, (point, color, label) in enumerate(zip(debug_points, colors, labels)):
                    cv2.circle(self.corner_debug_frame, point, 8, color, -1)
                    cv2.putText(self.corner_debug_frame, label, (point[0] + 10, point[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            else:
                cv2.putText(self.corner_debug_frame, "No suitable ellipse found",
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Display rejection reasons (show first few)
            y_offset = 250
            max_reasons = min(6, len(rejection_reasons))  # Show max 6 reasons (fewer due to larger font)
            for i in range(max_reasons):
                reason = rejection_reasons[i]
                if len(reason) > 45:  # Truncate shorter due to larger font
                    reason = reason[:42] + "..."
                color = (0, 255, 0) if "ACCEPTED" in reason else (255, 255, 255)
                cv2.putText(self.corner_debug_frame, reason,
                           (10, y_offset + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if len(rejection_reasons) > max_reasons:
                cv2.putText(self.corner_debug_frame, f"... and {len(rejection_reasons) - max_reasons} more",
                           (10, y_offset + max_reasons * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

        if best_ellipse is not None:
            # Calculate perspective transformation from ellipse
            perspective_matrix = self._ellipse_to_perspective_transform(best_ellipse, (w, h))

            # Create side-by-side debug frame with correction preview
            # Use the newly calculated matrix, or fall back to saved matrix if available
            preview_matrix = perspective_matrix if perspective_matrix is not None else self.perspective_matrix
            if self.debug_mode and self.corner_debug_frame is not None and preview_matrix is not None:
                self._create_perspective_debug_with_preview(frame, preview_matrix)

            return perspective_matrix

        # If no ellipse found but we have a saved matrix, still show preview for reference
        elif self.debug_mode and self.corner_debug_frame is not None and self.perspective_matrix is not None:
            self._create_perspective_debug_with_preview(frame, self.perspective_matrix)

        return None

    def _create_perspective_debug_with_preview(self, original_frame, perspective_matrix):
        """Create side-by-side perspective debug frame with correction preview"""
        h, w = original_frame.shape[:2]

        # Create side-by-side canvas (double width)
        combined_width = w * 2
        combined_height = h
        side_by_side_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        # Left side: Current corner debug frame (detection visualization)
        if self.corner_debug_frame is not None:
            resized_debug = cv2.resize(self.corner_debug_frame, (w, h))
            side_by_side_frame[:h, :w] = resized_debug

        # Right side: Perspective-corrected preview
        try:
            # Check if this is an affine or perspective matrix and apply accordingly
            if perspective_matrix.shape == (2, 3):
                # Affine transformation
                corrected_preview = cv2.warpAffine(original_frame, perspective_matrix, (w, h))
            elif perspective_matrix.shape == (3, 3):
                # Perspective transformation
                corrected_preview = cv2.warpPerspective(original_frame, perspective_matrix, (w, h))
            else:
                raise ValueError(f"Unsupported matrix shape: {perspective_matrix.shape}")

            side_by_side_frame[:h, w:] = corrected_preview

            # Determine if this is using saved or new calibration
            is_saved_matrix = np.array_equal(perspective_matrix, self.perspective_matrix) if self.perspective_matrix is not None else False
            matrix_source = "SAVED CALIBRATION" if is_saved_matrix else "LIVE DETECTION"

            # Add labels
            cv2.putText(side_by_side_frame, "DETECTION", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(side_by_side_frame, "CORRECTED PREVIEW", (w + 10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(side_by_side_frame, f"({matrix_source})", (w + 10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Show where the transformation points map to in the corrected image
            margin = 50
            square_size = min(w, h) - 2 * margin
            square_half = square_size / 2
            center_x_dst = w / 2
            center_y_dst = h / 2

            corrected_points = [
                (int(w + center_x_dst + square_half), int(center_y_dst)),      # Right
                (int(w + center_x_dst), int(center_y_dst - square_half)),      # Top
                (int(w + center_x_dst - square_half), int(center_y_dst)),      # Left
                (int(w + center_x_dst), int(center_y_dst + square_half))       # Bottom
            ]

            colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 128, 0)]
            labels = ["R", "T", "L", "B"]

            for point, color, label in zip(corrected_points, colors, labels):
                cv2.circle(side_by_side_frame, point, 8, color, -1)
                cv2.putText(side_by_side_frame, label, (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw expected outer ring circle in the corrected preview
            # This shows where we expect the outer ring to be after transformation
            expected_ring_center = (int(w + center_x_dst), int(center_y_dst))
            expected_ring_radius = int(square_half * 0.85)  # Slightly smaller than the square to account for margin
            cv2.circle(side_by_side_frame, expected_ring_center, expected_ring_radius, (0, 255, 0), 3)
            cv2.putText(side_by_side_frame, "Expected outer ring",
                       (w + 10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add a dividing line
            cv2.line(side_by_side_frame, (w, 0), (w, h), (255, 255, 255), 2)

        except cv2.error as e:
            # If perspective correction fails, show error message on right side
            error_frame = np.zeros((h, w, 3), dtype=np.uint8)
            error_frame.fill(50)  # Dark gray background
            cv2.putText(error_frame, "CORRECTION FAILED", (w//4, h//2 - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(error_frame, str(e)[:40], (10, h//2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            side_by_side_frame[:h, w:] = error_frame

            # Add labels
            cv2.putText(side_by_side_frame, "DETECTION", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(side_by_side_frame, "ERROR", (w + 10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Add a dividing line
            cv2.line(side_by_side_frame, (w, 0), (w, h), (255, 255, 255), 2)

        # Replace the corner debug frame with the side-by-side version
        self.corner_debug_frame = side_by_side_frame

    def _ellipse_to_perspective_transform(self, ellipse, frame_size, output_size=None):
        """
        Calculate perspective transformation matrix from ellipse parameters
        to correct the perspective so the ellipse becomes a centered circle
        """
        (center_x, center_y), (minor_axis, major_axis), angle = ellipse
        w, h = frame_size

        # Get ellipse parameters
        semi_major = major_axis / 2
        semi_minor = minor_axis / 2
        angle_rad = np.radians(angle)

        # Calculate the major axis direction and find the ellipse extremes
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Use a simpler affine approach: scale directly along the minor axis direction
        # This preserves image orientation while making the ellipse circular

        # Calculate the scaling factor needed to make ellipse circular
        scale_factor = major_axis / minor_axis

        # Calculate the minor axis direction vector
        # OpenCV ellipse angle is the angle of the major axis from horizontal
        minor_angle_rad = np.radians(angle + 90)  # Minor axis is perpendicular to major axis
        minor_dx = np.cos(minor_angle_rad)
        minor_dy = np.sin(minor_angle_rad)

        # Create transformation matrix that scales along the minor axis direction
        # We'll use a more direct approach: create a matrix that stretches in the minor axis direction

        # Translation matrices
        # Translate ellipse center to origin
        T_to_origin = np.array([[1, 0, -center_x],
                               [0, 1, -center_y],
                               [0, 0, 1]], dtype=np.float32)

        # Translate to center of corrected target output (800x800)
        # This ensures compatibility with the overall system architecture
        output_center_x = self.corrected_target_size[0] / 2  # 400 for 800x800 output
        output_center_y = self.corrected_target_size[1] / 2  # 400 for 800x800 output
        T_to_output_center = np.array([[1, 0, output_center_x],
                                      [0, 1, output_center_y],
                                      [0, 0, 1]], dtype=np.float32)

        # Rotation to align minor axis with Y-axis
        rotation_angle = np.radians(angle + 90)  # Rotate so minor axis aligns with Y
        cos_r = np.cos(-rotation_angle)  # Negative to rotate minor axis to Y
        sin_r = np.sin(-rotation_angle)

        R_align = np.array([[cos_r, -sin_r, 0],
                           [sin_r, cos_r, 0],
                           [0, 0, 1]], dtype=np.float32)

        # Scale matrix - stretch along Y (which is now the minor axis direction)
        S = np.array([[1, 0, 0],
                      [0, scale_factor, 0],
                      [0, 0, 1]], dtype=np.float32)

        # Rotation back to original orientation
        R_back = np.array([[cos_r, sin_r, 0],
                          [-sin_r, cos_r, 0],
                          [0, 0, 1]], dtype=np.float32)

        # Combine all transformations: T_to_output_center * R_back * S * R_align * T_to_origin
        combined_matrix = T_to_output_center @ R_back @ S @ R_align @ T_to_origin

        # Extract the 2x3 affine matrix (remove the last row)
        affine_matrix = combined_matrix[:2, :].astype(np.float32)

        # Add debug information
        if self.debug_mode:
            print(f"Ellipse: center=({center_x:.1f}, {center_y:.1f}), axes=({major_axis:.1f}, {minor_axis:.1f}), angle={angle:.1f}°")
            print(f"Scale factor: {scale_factor:.3f}")
            print(f"Minor axis direction: ({minor_dx:.3f}, {minor_dy:.3f})")
            print(f"Affine matrix:\n{affine_matrix}")
            print(f"T_to_origin:\n{T_to_origin}")
            print(f"R_align:\n{R_align}")
            print(f"S (scale):\n{S}")
            print(f"R_back:\n{R_back}")
            print(f"T_to_output_center:\n{T_to_output_center}")

        return affine_matrix

    def detect_rectangular_target(self, frame):
        """
        Detect the rectangular target paper and find its corners
        Returns the four corners of the target in order: [top-left, top-right, bottom-right, bottom-left]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Multiple approaches to detect paper edges against similar background

        # 1. Canny edge detection - better for subtle edges
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

        # 2. Adaptive threshold with different parameters
        thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # 3. More sensitive adaptive threshold
        thresh2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 21, 5)

        # 4. Otsu's threshold for global differences
        _, thresh3 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Combine edge detection with thresholding
        # Dilate edges to make them more prominent
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        # Combine multiple approaches
        combined = cv2.bitwise_or(thresh1, thresh2)
        combined = cv2.bitwise_or(combined, edges_dilated)

        # Use the combined result as our threshold
        thresh = combined

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Store debug information
        if self.debug_mode:
            # Only create corner debug frame if ellipse detection hasn't already created one
            if self.corner_debug_frame is None:
                # Create enhanced corner detection debug visualization
                self.corner_debug_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

                # Always draw basic debug info even if no target found
                cv2.putText(self.corner_debug_frame, f"Contours found: {len(contours)}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(self.corner_debug_frame, "CORNER DETECTION DEBUG (FALLBACK)",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(self.corner_debug_frame, "Combined: Adaptive + Canny edges",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # Ellipse detection already created debug frame, just add fallback info
                cv2.putText(self.corner_debug_frame, "FALLBACK: Corner detection active",
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw all contours in blue
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Only show significant contours
                    cv2.drawContours(self.corner_debug_frame, [contour], -1, (255, 0, 0), 2)
                else:
                    cv2.drawContours(self.corner_debug_frame, [contour], -1, (100, 100, 100), 1)

        # Filter contours for rectangular target
        best_contour = None
        best_score = 0
        min_area = w * h * self.min_target_area_ratio
        max_area = w * h * 0.8  # Don't take up more than 80% of frame

        # Also filter by perimeter - paper should have significant perimeter
        min_perimeter = 2 * (w + h) * 0.1  # At least 10% of frame perimeter

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Skip if too small or too large
            if area < min_area or area > max_area:
                continue

            # Skip if perimeter is too small (likely noise)
            if perimeter < min_perimeter:
                continue

            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Look for quadrilaterals (4 corners)
            if len(approx) == 4:
                # Calculate aspect ratio and area ratio
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box)

                # Check if it's reasonably rectangular
                area_ratio = area / box_area if box_area > 0 else 0

                # Get aspect ratio
                width, height = rect[1]
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0

                # Enhanced scoring for paper-like shapes
                rectangularity_score = area_ratio
                size_score = area / (w * h)  # Normalized by frame size

                # Prefer common paper aspect ratios (A4 ≈ 1.4, Letter ≈ 1.3, Square ≈ 1.0)
                aspect_score = 0
                if 0.9 <= aspect_ratio <= 1.1:  # Square-ish
                    aspect_score = 1.0
                elif 1.2 <= aspect_ratio <= 1.6:  # Rectangular paper
                    aspect_score = 0.9
                else:
                    aspect_score = max(0, 1.0 - abs(aspect_ratio - 1.4) * 0.5)

                # Convexity test - paper should be roughly convex
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                convexity_score = area / hull_area if hull_area > 0 else 0

                # Perimeter to area ratio - paper should be efficient
                efficiency = (4 * np.pi * area) / (perimeter * perimeter)

                # Combined score with better weighting
                score = (rectangularity_score * 0.3 +
                        size_score * 0.2 +
                        aspect_score * 0.2 +
                        convexity_score * 0.2 +
                        efficiency * 0.1)

                # Must meet minimum quality thresholds
                if (score > best_score and
                    area_ratio > 0.6 and  # Must be reasonably rectangular
                    convexity_score > 0.8 and  # Must be mostly convex
                    rectangularity_score > 0.5):  # Must approximate rectangle well
                    best_score = score
                    best_contour = approx

        if best_contour is not None:
            # Order corners: top-left, top-right, bottom-right, bottom-left
            corners = best_contour.reshape(4, 2).astype(np.float32)
            corners = self._order_corners(corners)

            # Add debug visualization
            if self.debug_mode and self.debug_frame is not None:
                self._draw_debug_info(best_contour, corners)

            return corners

        return None

    def _draw_debug_info(self, best_contour, corners):
        """Draw debug information on the corner debug frame"""
        if self.corner_debug_frame is None:
            return

        # Draw the selected best contour in green (over the blue contours)
        cv2.drawContours(self.corner_debug_frame, [best_contour], -1, (0, 255, 0), 3)

        # Draw corners as red circles
        for i, corner in enumerate(corners):
            cv2.circle(self.corner_debug_frame, tuple(corner.astype(int)), 8, (0, 0, 255), -1)
            # Label corners
            label = ['TL', 'TR', 'BR', 'BL'][i]
            cv2.putText(self.corner_debug_frame, label,
                       (int(corner[0]) + 12, int(corner[1]) - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add target found text
        cv2.putText(self.debug_frame, "TARGET FOUND!",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def _order_corners(self, corners):
        """
        Order corners in consistent order: top-left, top-right, bottom-right, bottom-left
        """
        # Calculate center point
        center = np.mean(corners, axis=0)

        # Calculate angles from center to each corner
        angles = []
        for corner in corners:
            angle = np.arctan2(corner[1] - center[1], corner[0] - center[0])
            angles.append(angle)

        # Sort corners by angle (starting from top-left, going clockwise)
        sorted_indices = np.argsort(angles)

        # Find the top-left corner (smallest x + y sum)
        sums = [corners[i][0] + corners[i][1] for i in sorted_indices]
        tl_idx = sorted_indices[np.argmin(sums)]

        # Reorder starting from top-left, going clockwise
        ordered_corners = []
        start_idx = list(sorted_indices).index(tl_idx)

        for i in range(4):
            idx = sorted_indices[(start_idx + i) % 4]
            ordered_corners.append(corners[idx])

        return np.array(ordered_corners, dtype=np.float32)

    def get_perspective_transform(self, corners, output_size=(800, 800)):
        """
        Calculate perspective transformation matrix from target corners
        """
        if corners is None or len(corners) != 4:
            return None

        # Define destination points (perfect rectangle)
        width, height = output_size
        dst_corners = np.array([
            [0, 0],                    # top-left
            [width - 1, 0],           # top-right
            [width - 1, height - 1],  # bottom-right
            [0, height - 1]           # bottom-left
        ], dtype=np.float32)

        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(corners, dst_corners)
        return matrix

    def apply_perspective_correction(self, frame, matrix, output_size=(800, 800)):
        """
        Apply perspective correction to frame
        Handles both affine (2x3) and perspective (3x3) transformation matrices
        """
        if matrix is None:
            return None

        # Check matrix dimensions to determine transformation type
        if matrix.shape == (2, 3):
            # Affine transformation (from ellipse method)
            corrected = cv2.warpAffine(frame, matrix, output_size)
        elif matrix.shape == (3, 3):
            # Perspective transformation (from corner method)
            corrected = cv2.warpPerspective(frame, matrix, output_size)
        else:
            print(f"Unsupported matrix shape: {matrix.shape}")
            return None

        return corrected

    def _transform_coordinates_back(self, circle_data, matrix, original_shape):
        """
        Transform coordinates from perspective-corrected frame back to original frame
        Handles both affine (2x3) and perspective (3x3) transformation matrices
        """
        if circle_data is None or matrix is None:
            return circle_data

        x, y, radius = circle_data

        # Handle different matrix types
        if matrix.shape == (2, 3):
            # Affine transformation
            try:
                # For 2x3 affine matrix, we need to compute the inverse
                # Add homogeneous row to make it 3x3 for inversion
                full_matrix = np.vstack([matrix, [0, 0, 1]])
                inv_matrix = np.linalg.inv(full_matrix)

                # Transform center point
                point = np.array([x, y, 1])
                transformed_point = inv_matrix @ point
                new_x, new_y = transformed_point[0], transformed_point[1]

                # For affine transformation, use simpler radius scaling
                scale_factor = np.sqrt(abs(np.linalg.det(matrix[:2, :2])))
                new_radius = radius / scale_factor

            except np.linalg.LinAlgError:
                # If inverse fails, return original coordinates
                return circle_data

        elif matrix.shape == (3, 3):
            # Perspective transformation
            try:
                inv_matrix = np.linalg.inv(matrix)

                # Transform center point back to original coordinates
                point = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
                transformed_point = cv2.perspectiveTransform(point, inv_matrix)
                new_x, new_y = transformed_point[0][0]

                # Scale radius approximately (this is an approximation since perspective changes sizes)
                scale_factor = np.sqrt(abs(np.linalg.det(matrix)))
                original_area = original_shape[0] * original_shape[1]
                corrected_area = self.corrected_target_size[0] * self.corrected_target_size[1]
                area_ratio = np.sqrt(original_area / corrected_area)

                new_radius = radius * area_ratio / scale_factor

            except (np.linalg.LinAlgError, cv2.error):
                # If transformation fails, return original coordinates
                return circle_data
        else:
            # Unsupported matrix type
            return circle_data

        return (int(new_x), int(new_y), int(new_radius))

    def _transform_coordinates_to_corrected(self, circle_data):
        """
        Transform coordinates from original frame to perspective-corrected frame
        This is the inverse of _transform_coordinates_back
        """
        if circle_data is None or self.perspective_matrix is None:
            return circle_data

        x, y, radius = circle_data

        try:
            if self.perspective_matrix.shape == (2, 3):
                # Affine transformation
                # Add homogeneous coordinate
                point = np.array([x, y, 1])
                transformed_point = self.perspective_matrix @ point
                new_x, new_y = transformed_point[0], transformed_point[1]

                # For affine transformation, use simpler radius scaling
                scale_factor = np.sqrt(abs(np.linalg.det(self.perspective_matrix[:2, :2])))
                new_radius = radius * scale_factor

            elif self.perspective_matrix.shape == (3, 3):
                # Perspective transformation
                # Transform center point to corrected coordinates
                point = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
                transformed_point = cv2.perspectiveTransform(point, self.perspective_matrix)
                new_x, new_y = transformed_point[0][0]

                # Scale radius approximately
                scale_factor = np.sqrt(abs(np.linalg.det(self.perspective_matrix)))
                new_radius = radius * scale_factor

            else:
                # Unsupported matrix type
                return circle_data

            return (int(new_x), int(new_y), int(new_radius))

        except (np.linalg.LinAlgError, cv2.error):
            # If transformation fails, return original coordinates
            return circle_data

    def detect_enhanced_target(self, frame):
        """
        Enhanced target detection using edge detection and contours
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Apply threshold to isolate dark regions (black target area)
        _, thresh = cv2.threshold(filtered, 100, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest circular contour
        best_circle = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                # Fit a circle to the contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                # Check circularity (how round the contour is)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                    # Accept if reasonably circular and largest so far
                    if circularity > 0.3 and area > max_area:
                        max_area = area
                        best_circle = (center[0], center[1], radius)

        return best_circle

    def _add_to_history(self, detection):
        """Add detection to history for stability filtering"""
        if detection is not None:
            self.detection_history.append(detection)
        else:
            self.detection_history.append(None)

        # Keep only recent history
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)

    def _add_outer_to_history(self, detection):
        """Add outer circle detection to history for stability filtering"""
        if detection is not None:
            self.outer_circle_history.append(detection)
        else:
            self.outer_circle_history.append(None)

        # Keep only recent history
        if len(self.outer_circle_history) > self.max_history:
            self.outer_circle_history.pop(0)

    def _get_stable_detection(self):
        """Get stable detection from history using averaging"""
        valid_detections = [d for d in self.detection_history if d is not None]

        if len(valid_detections) < 2:  # Need at least 2 valid detections
            return None

        # Calculate average position and radius
        x_coords = [d[0] for d in valid_detections]
        y_coords = [d[1] for d in valid_detections]
        radii = [d[2] for d in valid_detections]

        avg_x = int(np.mean(x_coords))
        avg_y = int(np.mean(y_coords))
        avg_radius = int(np.mean(radii))

        # Check consistency - if detections are too spread out, not stable
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        r_std = np.std(radii)

        # Scale stability thresholds based on current resolution
        # Use the last valid detection to estimate scale
        if valid_detections:
            sample_detection = valid_detections[0]
            estimated_radius = sample_detection[2]
            # Estimate scale from radius (assuming base radius ~80 at 640x480)
            scale_estimate = max(1.0, estimated_radius / 80.0)

            max_x_std = 20 * scale_estimate
            max_y_std = 20 * scale_estimate
            max_r_std = 15 * scale_estimate
        else:
            max_x_std = 20
            max_y_std = 20
            max_r_std = 15

        # If variations are too large, detection is not stable
        if x_std > max_x_std or y_std > max_y_std or r_std > max_r_std:
            return None

        # Calculate confidence based on consistency and number of valid detections
        consistency_score = 1.0 / (1.0 + x_std + y_std + r_std/10)
        detection_ratio = len(valid_detections) / len(self.detection_history)
        self.detection_confidence = consistency_score * detection_ratio

        return (avg_x, avg_y, avg_radius)

    def _get_stable_outer_detection(self):
        """Get stable outer circle detection from history using averaging"""
        valid_detections = [d for d in self.outer_circle_history if d is not None]

        if len(valid_detections) < 2:  # Need at least 2 valid detections
            return None

        # Calculate average position and radius
        x_coords = [d[0] for d in valid_detections]
        y_coords = [d[1] for d in valid_detections]
        radii = [d[2] for d in valid_detections]

        avg_x = int(np.mean(x_coords))
        avg_y = int(np.mean(y_coords))
        avg_radius = int(np.mean(radii))

        # Check consistency - if detections are too spread out, not stable
        x_std = np.std(x_coords)
        y_std = np.std(y_coords)
        r_std = np.std(radii)

        # Scale stability thresholds for outer circle
        if valid_detections:
            sample_detection = valid_detections[0]
            estimated_radius = sample_detection[2]
            # Estimate scale from radius (assuming base outer radius ~200 at 640x480)
            scale_estimate = max(1.0, estimated_radius / 200.0)

            max_x_std = 25 * scale_estimate
            max_y_std = 25 * scale_estimate
            max_r_std = 20 * scale_estimate
        else:
            max_x_std = 25
            max_y_std = 25
            max_r_std = 20

        # If variations are too large, detection is not stable
        if x_std > max_x_std or y_std > max_y_std or r_std > max_r_std:
            return None

        # Calculate confidence
        consistency_score = 1.0 / (1.0 + x_std + y_std + r_std/10)
        detection_ratio = len(valid_detections) / len(self.outer_circle_history)
        self.outer_confidence = consistency_score * detection_ratio

        return (avg_x, avg_y, avg_radius)

    def detect_target(self, frame):
        """
        Main target detection method with caching and periodic detection
        """
        if not self.detection_enabled:
            return self.cached_inner_result

        # Check if we should run detection or use cached results
        should_detect = self._should_run_detection(frame)

        if should_detect:
            # Update detection timestamp
            self.last_detection_time = time.time()

            # Clear debug frames for fresh detection (but not during calibration mode)
            if self.debug_mode and not self.calibration_mode:
                self.corner_debug_frame = None
                self.circle_debug_frame = None

            # Use saved perspective calibration or detect on demand
            current_resolution = (frame.shape[1], frame.shape[0])  # (width, height)

            if self.calibration_mode:
                # In calibration mode - only detect when explicitly requested
                if self.saved_perspective_matrix is not None:
                    self.perspective_matrix = self.get_scaled_perspective_matrix(current_resolution)
                    self.target_corners = None
                else:
                    self.perspective_matrix = None
                    self.target_corners = None
            else:
                # Normal mode - use saved calibration if available, otherwise detect
                if self.saved_perspective_matrix is not None:
                    self.perspective_matrix = self.get_scaled_perspective_matrix(current_resolution)
                    self.target_corners = None
                else:
                    # Fallback to detection if no saved calibration
                    ellipse_matrix = self.detect_ellipse_perspective(frame)
                    if ellipse_matrix is not None:
                        self.perspective_matrix = ellipse_matrix
                        self.target_corners = None
                    else:
                        self.perspective_matrix = None
                        self.target_corners = None

            # Use perspective-corrected frame for circle detection if available
            detection_frame = frame
            if self.perspective_matrix is not None:
                corrected_frame = self.apply_perspective_correction(frame, self.perspective_matrix, self.corrected_target_size)
                if corrected_frame is not None:
                    detection_frame = corrected_frame

            # Detect inner black circle
            current_inner = self.detect_black_circle_improved(detection_frame)

            # If we used perspective correction, transform coordinates back to original frame
            if self.perspective_matrix is not None and current_inner is not None:
                current_inner = self._transform_coordinates_back(current_inner, self.perspective_matrix, frame.shape[:2])

            self._add_to_history(current_inner)
            stable_inner = self._get_stable_detection()

            if stable_inner is not None:
                self.stable_detection = stable_inner
                self.target_center = (stable_inner[0], stable_inner[1])
                self.target_radius = stable_inner[2]
                self.cached_inner_result = stable_inner
            elif self.stable_detection is not None and self.detection_confidence > 0.3:
                stable_inner = self.stable_detection
                self.cached_inner_result = stable_inner

            # Detect outer circle using the inner circle as reference
            current_outer = self.detect_outer_circle(detection_frame, stable_inner)

            # Transform outer circle coordinates back if needed
            if self.perspective_matrix is not None and current_outer is not None:
                current_outer = self._transform_coordinates_back(current_outer, self.perspective_matrix, frame.shape[:2])

            self._add_outer_to_history(current_outer)
            stable_outer = self._get_stable_outer_detection()

            if stable_outer is not None:
                self.stable_outer_circle = stable_outer
                self.outer_circle = stable_outer
                self.cached_outer_result = stable_outer
            elif self.stable_outer_circle is not None and self.outer_confidence > 0.3:
                stable_outer = self.stable_outer_circle
                self.cached_outer_result = stable_outer

        else:
            # Use cached results - no detection needed
            stable_inner = self.cached_inner_result
            if self.cached_outer_result:
                self.outer_circle = self.cached_outer_result

        return stable_inner  # Return inner circle as main target

    def draw_target_overlay(self, frame, target_info=None, frame_is_corrected=False):
        """
        Draw target detection overlay on frame - both inner and outer circles
        Uses cached results for performance

        Args:
            frame: The frame to draw on
            target_info: Pre-computed target info (optional)
            frame_is_corrected: True if frame is already perspective-corrected
        """
        if not self.detection_enabled:
            return frame

        # Use provided target info, cached result, or detect new
        if target_info is None:
            if frame_is_corrected:
                # For corrected frames, use cached detection results but transform coordinates
                # This maintains the caching system while working on corrected frames
                if self.cached_inner_result is not None:
                    # Transform cached coordinates to corrected frame space
                    target_info = self._transform_coordinates_to_corrected(self.cached_inner_result)
                else:
                    # No cached result available, run detection on original frame
                    # This will populate the cache for subsequent frames
                    target_info = None  # Will trigger overlay without detection
            else:
                # For original frames, use normal detection with perspective correction
                # Try using cached result first to avoid unnecessary detection
                if self.cached_inner_result is not None:
                    target_info = self.cached_inner_result
                else:
                    target_info = self.detect_target(frame)

        # Draw inner circle (main target)
        if target_info is not None:
            x, y, radius = target_info

            # Color based on confidence
            if self.detection_confidence > 0.7:
                inner_color = (0, 255, 0)  # Green - high confidence
                thickness = 3
            elif self.detection_confidence > 0.4:
                inner_color = (0, 255, 255)  # Yellow - medium confidence
                thickness = 2
            else:
                inner_color = (0, 165, 255)  # Orange - low confidence
                thickness = 2

            # Draw circle around detected inner target
            cv2.circle(frame, (x, y), radius, inner_color, thickness)

            # Draw center crosshair
            cv2.line(frame, (x-15, y), (x+15, y), inner_color, 2)
            cv2.line(frame, (x, y-15), (x, y+15), inner_color, 2)

            # Add center dot
            cv2.circle(frame, (x, y), 3, inner_color, -1)

            # Add text label with confidence
            confidence_pct = int(self.detection_confidence * 100)
            label = f"INNER R:{radius} ({confidence_pct}%)"

            # Background rectangle for text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame,
                         (x - label_size[0]//2 - 5, y - radius - 25),
                         (x + label_size[0]//2 + 5, y - radius - 5),
                         (0, 0, 0), -1)

            cv2.putText(frame, label,
                       (x - label_size[0]//2, y - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, inner_color, 1)

        # Draw outer circle
        outer_circle_info = self.outer_circle
        if frame_is_corrected and outer_circle_info is not None:
            # Transform outer circle coordinates to corrected frame space
            outer_circle_info = self._transform_coordinates_to_corrected(outer_circle_info)

        if outer_circle_info is not None:
            ox, oy, oradius = outer_circle_info

            # Color based on outer confidence
            if self.outer_confidence > 0.5:
                outer_color = (255, 0, 255)  # Magenta - good detection
                outer_thickness = 2
            elif self.outer_confidence > 0.3:
                outer_color = (255, 255, 0)  # Cyan - medium detection
                outer_thickness = 2
            else:
                outer_color = (128, 128, 255)  # Light purple - low confidence
                outer_thickness = 1

            # Draw outer circle
            cv2.circle(frame, (ox, oy), oradius, outer_color, outer_thickness)

            # Add outer circle label
            outer_confidence_pct = int(self.outer_confidence * 100)
            outer_label = f"OUTER R:{oradius} ({outer_confidence_pct}%)"

            # Position outer label below the outer circle
            outer_label_size = cv2.getTextSize(outer_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(frame,
                         (ox - outer_label_size[0]//2 - 3, oy + oradius + 5),
                         (ox + outer_label_size[0]//2 + 3, oy + oradius + 20),
                         (0, 0, 0), -1)

            cv2.putText(frame, outer_label,
                       (ox - outer_label_size[0]//2, oy + oradius + 17),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, outer_color, 1)

        # Draw rectangular target outline if detected
        if self.target_corners is not None:
            # Draw target outline
            target_color = (0, 255, 255)  # Yellow for target outline
            corners_int = self.target_corners.astype(np.int32)

            # Draw the rectangle outline
            cv2.polylines(frame, [corners_int], True, target_color, 2)

            # Draw corner markers
            for i, corner in enumerate(corners_int):
                cv2.circle(frame, tuple(corner), 5, target_color, -1)
                # Label corners
                label = ['TL', 'TR', 'BR', 'BL'][i]
                cv2.putText(frame, label, (corner[0] + 8, corner[1] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, target_color, 1)

            # Add perspective correction status
            if self.perspective_matrix is not None:
                cv2.putText(frame, "PERSPECTIVE CORRECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2)

        return frame

    def detect_bullet_holes(self, frame, target_center=None, target_radius=None):
        """
        Detect bullet holes within the target area
        Returns list of (x, y, radius) tuples for detected holes
        """
        if target_center is None or target_radius is None:
            return []

        # Create region of interest around target
        x_center, y_center = target_center
        roi_size = int(target_radius * 1.2)

        x1 = max(0, x_center - roi_size)
        y1 = max(0, y_center - roi_size)
        x2 = min(frame.shape[1], x_center + roi_size)
        y2 = min(frame.shape[0], y_center + roi_size)

        roi = frame[y1:y2, x1:x2]

        # Convert ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Detect small dark circles (bullet holes)
        holes = cv2.HoughCircles(
            gray_roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,              # Small distance between holes
            param1=50,
            param2=15,               # Lower threshold for small holes
            minRadius=2,             # Very small holes
            maxRadius=15             # Maximum hole size
        )

        bullet_holes = []
        if holes is not None:
            holes = np.round(holes[0, :]).astype("int")
            for (x, y, r) in holes:
                # Convert back to full frame coordinates
                full_x = x + x1
                full_y = y + y1

                # Check if hole is within target circle
                distance = np.sqrt((full_x - x_center)**2 + (full_y - y_center)**2)
                if distance <= target_radius:
                    bullet_holes.append((full_x, full_y, r))

        return bullet_holes

    def set_detection_enabled(self, enabled):
        """Enable or disable target detection"""
        self.detection_enabled = enabled

    def force_detection(self):
        """Force immediate detection on next frame (bypasses timing)"""
        self.last_detection_time = 0
        # Clear cached results to force fresh detection
        self.cached_inner_result = None
        self.cached_outer_result = None
        self.last_frame_gray = None
        return True

    def set_detection_interval(self, interval):
        """Set detection interval in seconds"""
        self.detection_interval = max(0.1, interval)  # Minimum 0.1 seconds

    def get_corrected_target_image(self, frame):
        """
        Get perspective-corrected target image for detailed analysis
        """
        if self.perspective_matrix is not None:
            return self.apply_perspective_correction(frame, self.perspective_matrix, self.corrected_target_size)
        return None

    def set_debug_mode(self, enabled):
        """Enable or disable debug visualization mode"""
        self.debug_mode = enabled

    def set_debug_type(self, debug_type):
        """Set debug visualization type: 'combined', 'perspective', 'circles'"""
        if debug_type in ["combined", "perspective", "circles"]:
            # Clear existing debug frames when changing type to ensure fresh view
            self.corner_debug_frame = None
            self.circle_debug_frame = None
            self.debug_frame = None
            self.debug_type = debug_type
            return True
        return False

    def get_debug_frame(self, debug_type=None):
        """Get the current debug frame based on debug type

        Args:
            debug_type: Override debug type ('combined', 'perspective', 'circles', 'corrected')
                       If None, uses self.debug_type
        """
        if debug_type is None:
            debug_type = self.debug_type

        if debug_type == "perspective":
            return self.corner_debug_frame
        elif debug_type == "circles":
            return self.circle_debug_frame
        elif debug_type == "corrected":
            return self.get_corrected_debug_frame()
        else:  # combined or fallback
            return self.debug_frame

    def get_debug_frame_jpeg(self, debug_type='combined'):
        """Get debug frame as JPEG bytes for streaming

        Args:
            debug_type: Type of debug frame ('combined', 'perspective', 'circles', 'corrected')
        """
        debug_frame = self.get_debug_frame(debug_type)
        if debug_frame is not None:
            _, buffer = cv2.imencode('.jpg', debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes()
        return None

    def generate_debug_frame(self, frame):
        """Force generation of debug frame for current frame"""
        if not self.debug_mode:
            return None

        # During calibration mode, preserve existing debug frames to prevent shuffling
        if self.calibration_mode:
            # Only create debug frames if they don't exist
            if self.debug_type == "perspective" and self.corner_debug_frame is None:
                self.detect_ellipse_perspective(frame)
            elif self.debug_type == "circles" and self.circle_debug_frame is None:
                self.detect_black_circle_improved(frame)
            elif self.debug_type == "combined":
                # For combined mode, ensure both frames exist
                if self.corner_debug_frame is None:
                    self.detect_ellipse_perspective(frame)
                if self.circle_debug_frame is None:
                    self.detect_black_circle_improved(frame)
                self._create_combined_debug_frame(frame)
        else:
            # Normal mode - regenerate debug frames
            if self.debug_type in ["combined", "perspective"]:
                self.detect_rectangular_target(frame)
                self.detect_ellipse_perspective(frame)

            if self.debug_type in ["combined", "circles"]:
                self.detect_black_circle_improved(frame)

            # Create combined debug frame if needed
            if self.debug_type == "combined":
                self._create_combined_debug_frame(frame)

        return self.get_debug_frame()

    def get_corrected_debug_frame(self):
        """Get perspective-corrected frame for debugging"""
        return self.corrected_debug_frame

    def _create_combined_debug_frame(self, frame):
        """Create a combined debug frame showing both corner and circle detection"""
        h, w = frame.shape[:2]

        # Create a combined frame with side-by-side views
        combined_width = w * 2
        combined_height = h
        self.debug_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        # Left side: ellipse detection
        if self.corner_debug_frame is not None:
            resized_corner = cv2.resize(self.corner_debug_frame, (w, h))
            self.debug_frame[:h, :w] = resized_corner
        else:
            # Fill with gray and add text
            cv2.putText(self.debug_frame, "Ellipse Detection Not Available",
                       (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 128, 128), 3)

        # Right side: circle detection
        if self.circle_debug_frame is not None:
            resized_circle = cv2.resize(self.circle_debug_frame, (w, h))
            self.debug_frame[:h, w:] = resized_circle
        else:
            # Fill with gray and add text
            cv2.putText(self.debug_frame, "Circle Detection Not Available",
                       (w + 10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 128, 128), 3)

        # Add labels with larger fonts
        cv2.putText(self.debug_frame, "ELLIPSE DETECTION", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        cv2.putText(self.debug_frame, "CIRCLE DETECTION", (w + 10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # Add separator line
        cv2.line(self.debug_frame, (w, 0), (w, h), (255, 255, 255), 2)

    def get_detection_status(self):
        """Get current detection status"""
        current_time = time.time()
        time_since_detection = current_time - self.last_detection_time

        return {
            'detection_enabled': self.detection_enabled,
            'target_center': self.target_center,
            'target_radius': self.target_radius,
            'detection_confidence': self.detection_confidence,
            'stable_detection': self.stable_detection is not None,
            'outer_circle': self.outer_circle,
            'outer_confidence': self.outer_confidence,
            'stable_outer_detection': self.stable_outer_circle is not None,
            'detection_interval': self.detection_interval,
            'time_since_detection': time_since_detection,
            'using_cached_result': time_since_detection < self.detection_interval,
            'target_corners': self.target_corners.tolist() if self.target_corners is not None else None,
            'perspective_correction_enabled': self.perspective_matrix is not None,
            'debug_mode': self.debug_mode,
            'debug_type': self.debug_type,
            'calibration_mode': self.calibration_mode,
            'has_saved_calibration': self.saved_perspective_matrix is not None,
            'calibration_resolution': self.calibration_resolution
        }

    def save_perspective_calibration(self, camera_resolution=None):
        """Save current perspective calibration to YAML file"""
        if self.perspective_matrix is None:
            return False, "No perspective calibration to save"

        try:
            # Prepare calibration data
            calibration_data = {
                'timestamp': time.time(),
                'perspective_matrix': self.perspective_matrix.tolist(),
                'ellipse_data': self.saved_ellipse_data,
                'corrected_target_size': list(self.corrected_target_size),
                'camera_resolution': list(camera_resolution) if camera_resolution else None,
                'notes': 'Perspective calibration for fixed camera installation'
            }

            # Save to YAML file
            with open(self.calibration_file, 'w') as f:
                yaml.dump(calibration_data, f, default_flow_style=False)

            print(f"Perspective calibration saved to {self.calibration_file}")
            return True, f"Calibration saved successfully"

        except Exception as e:
            error_msg = f"Failed to save calibration: {str(e)}"
            print(error_msg)
            return False, error_msg

    def load_perspective_calibration(self):
        """Load perspective calibration from YAML file"""
        if not os.path.exists(self.calibration_file):
            print(f"No calibration file found at {self.calibration_file}")
            return False, "No calibration file found"

        try:
            with open(self.calibration_file, 'r') as f:
                calibration_data = yaml.safe_load(f)

            # Load perspective matrix and resolution
            if 'perspective_matrix' in calibration_data:
                self.saved_perspective_matrix = np.array(calibration_data['perspective_matrix'], dtype=np.float32)
                self.calibration_resolution = calibration_data.get('camera_resolution', None)
                self.perspective_matrix = self.saved_perspective_matrix.copy()

            # Load ellipse data
            if 'ellipse_data' in calibration_data:
                self.saved_ellipse_data = calibration_data['ellipse_data']

            # Load target size
            if 'corrected_target_size' in calibration_data:
                self.corrected_target_size = tuple(calibration_data['corrected_target_size'])

            timestamp = calibration_data.get('timestamp', 0)
            age_hours = (time.time() - timestamp) / 3600

            print(f"Perspective calibration loaded from {self.calibration_file} (age: {age_hours:.1f} hours)")
            return True, f"Calibration loaded successfully (age: {age_hours:.1f}h)"

        except Exception as e:
            error_msg = f"Failed to load calibration: {str(e)}"
            print(error_msg)
            return False, error_msg

    def calibrate_perspective(self, frame):
        """Perform on-demand perspective calibration"""
        # Store the ellipse data when calibration succeeds
        ellipse_matrix = self.detect_ellipse_perspective(frame)
        if ellipse_matrix is not None:
            self.perspective_matrix = ellipse_matrix
            # Store ellipse data from the debug info if available
            self.saved_ellipse_data = {
                'detection_method': 'ellipse',
                'calibration_timestamp': time.time()
            }
            return True, "Perspective calibration successful"
        else:
            return False, "No suitable ellipse found for calibration"

    def set_calibration_mode(self, enabled):
        """Enable or disable calibration mode"""
        self.calibration_mode = enabled
        return True

    def get_scaled_perspective_matrix(self, current_resolution):
        """Get perspective matrix scaled for current resolution"""
        if (self.saved_perspective_matrix is None or
            self.calibration_resolution is None or
            current_resolution is None):
            return self.saved_perspective_matrix

        # Calculate scaling factors
        cal_width, cal_height = self.calibration_resolution
        cur_width, cur_height = current_resolution

        scale_x = cur_width / cal_width
        scale_y = cur_height / cal_height

        # Scale the perspective matrix
        scaled_matrix = self.saved_perspective_matrix.copy()

        if scaled_matrix.shape == (2, 3):
            # Affine transformation matrix
            scaled_matrix[0, 0] *= scale_x  # Scale x-direction transform
            scaled_matrix[0, 2] *= scale_x  # Scale x translation
            scaled_matrix[1, 1] *= scale_y  # Scale y-direction transform
            scaled_matrix[1, 2] *= scale_y  # Scale y translation
        elif scaled_matrix.shape == (3, 3):
            # Perspective transformation matrix
            scaled_matrix[0, 0] *= scale_x
            scaled_matrix[0, 2] *= scale_x
            scaled_matrix[1, 1] *= scale_y
            scaled_matrix[1, 2] *= scale_y

        return scaled_matrix

    def get_perspective_matrix(self):
        """Get the current perspective matrix for main stream correction"""
        return self.perspective_matrix