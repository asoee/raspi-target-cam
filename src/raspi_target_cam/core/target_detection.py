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
        self.min_target_area_ratio = 0.01  # Minimum 1% of frame area

        # Debug visualization
        self.debug_mode = False
        self.debug_type = "combined"  # "combined", "perspective", "circles"
        self.debug_frame = None
        self.debug_contours = None
        self.corner_debug_frame = None
        self.circle_debug_frame = None

        # Calibration mode for perspective detection
        self.calibration_mode = False  # When true, performs detection on demand

        # Debug frames
        self.corrected_debug_frame = None  # For perspective-corrected debug view

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

    def detect_outer_circle_ellipse_based(self, frame, inner_circle=None):
        """
        Detect outer ring using ellipse fitting on contours (inspired by perspective detection)
        This method handles partial occlusion better than HoughCircles
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        scale_factor = max(w / 640, h / 480)
        
        # Edge detection with parameters similar to perspective detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for ring-like contours that could be the outer ring
        min_area = int(5000 * scale_factor * scale_factor)  # Larger minimum area for outer ring
        min_circularity = 0.2  # More lenient for partially occluded rings
        
        ring_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > min_circularity:
                        ring_contours.append(contour)
        
        if not ring_contours:
            return None
            
        best_circle = None
        best_score = 0
        
        for contour in ring_contours:
            if len(contour) < 5:  # Need at least 5 points to fit ellipse
                continue
                
            try:
                # Fit ellipse to contour
                ellipse = cv2.fitEllipse(contour)
                (center_x, center_y), (minor_axis, major_axis), angle = ellipse
                
                # Convert ellipse to approximate circle (use average of axes)
                avg_radius = (minor_axis + major_axis) / 4.0  # Divide by 4 because axes are diameters
                
                # Check if this could be an outer ring
                aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)
                
                # More lenient aspect ratio for partially occluded outer ring
                if aspect_ratio > 3.0:  # Too stretched, probably not a ring
                    continue
                
                # If we have inner circle, check relationship
                if inner_circle is not None:
                    inner_x, inner_y, inner_r = inner_circle
                    distance_between_centers = np.sqrt((center_x - inner_x)**2 + (center_y - inner_y)**2)
                    
                    max_center_distance = int(50 * scale_factor)  # More lenient for ellipse fitting
                    min_radius_diff = int(15 * scale_factor)
                    
                    # Check if outer and larger
                    if distance_between_centers > max_center_distance or avg_radius <= inner_r + min_radius_diff:
                        continue
                
                # Score based on size, position, and shape quality
                area = cv2.contourArea(contour)
                size_score = area / (w * h)  # Prefer larger rings
                
                # Prefer rings closer to center
                center_score = 1.0 - (abs(center_x - w/2) + abs(center_y - h/2)) / (w + h)
                
                # Prefer more circular shapes (penalize high aspect ratios)
                shape_score = 1.0 / aspect_ratio
                
                score = size_score * 0.4 + center_score * 0.3 + shape_score * 0.3
                
                if score > best_score:
                    best_score = score
                    best_circle = (int(center_x), int(center_y), int(avg_radius))
                    
            except cv2.error:
                continue
                
        return best_circle

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

    def get_perspective_transform(self, corners, output_size):
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
                # Since we now work with original frame dimensions, no area scaling needed
                new_radius = radius / scale_factor

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

            # Detect inner black circle directly on provided frame
            # Note: Perspective correction is now handled by CameraController before this method is called
            current_inner = self.detect_black_circle_improved(frame)

            # No coordinate transformation needed since perspective correction is handled upstream

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

            # Detect outer circle using ellipse-based method (better for partial occlusion)
            current_outer = self.detect_outer_circle_ellipse_based(frame, stable_inner)
            
            # Fallback to original HoughCircles method if ellipse method fails
            if current_outer is None:
                current_outer = self.detect_outer_circle(frame, stable_inner)

            # No coordinate transformation needed since perspective correction is handled upstream

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
        Note: This method is deprecated - perspective correction is now handled by CameraController
        """
        # Perspective correction is now handled by CameraController before frames reach TargetDetector
        return frame

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
                # Perspective detection is now handled by CameraController
                self.corner_debug_frame = frame.copy()  # Placeholder debug frame
            elif self.debug_type == "circles" and self.circle_debug_frame is None:
                self.detect_black_circle_improved(frame)
            elif self.debug_type == "combined":
                # For combined mode, ensure both frames exist
                if self.corner_debug_frame is None:
                    # Perspective detection is now handled by CameraController
                    self.corner_debug_frame = frame.copy()  # Placeholder debug frame
                if self.circle_debug_frame is None:
                    self.detect_black_circle_improved(frame)
                self._create_combined_debug_frame(frame)
        else:
            # Normal mode - regenerate debug frames
            if self.debug_type in ["combined", "perspective"]:
                self.detect_rectangular_target(frame)
                # Perspective detection is now handled by CameraController
                self.corner_debug_frame = frame.copy()  # Placeholder debug frame

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
            # Note: perspective_correction_enabled removed - now handled by CameraController
            'debug_mode': self.debug_mode,
            'debug_type': self.debug_type,
            'calibration_mode': self.calibration_mode,
            'has_saved_calibration': False,  # Perspective now handled by CameraController
            'calibration_resolution': None  # Perspective now handled by CameraController
        }

    def save_perspective_calibration(self, camera_resolution=None):
        """Deprecated: Perspective calibration is now handled by CameraController"""
        return False, "Perspective calibration is now handled by CameraController"

    def load_perspective_calibration(self):
        """Deprecated: Perspective calibration is now handled by CameraController"""
        return False, "Perspective calibration is now handled by CameraController"

    def calibrate_perspective(self, frame):
        """Deprecated: Perspective calibration is now handled by CameraController"""
        return False, "Perspective calibration is now handled by CameraController"

    def set_calibration_mode(self, enabled):
        """Enable or disable calibration mode"""
        self.calibration_mode = enabled
        return True


    def get_perspective_matrix(self):
        """Get the current perspective matrix for main stream correction"""
        return self.perspective_matrix