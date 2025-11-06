#!/usr/bin/env python3
"""
Perspective transformation and ellipse detection utilities
Handles ellipse detection, ellipse-to-circle transformation, debug visualization, and related calculations
"""
import cv2
import numpy as np
import os
import time
import yaml


class Perspective:
    """Handles perspective correction and ellipse detection for camera targeting systems"""

    def __init__(self, calibration_file="perspective_calibration.yaml"):
        self.calibration_file = calibration_file
        self.saved_perspective_matrix = None
        self.saved_ellipse_data = None
        self.saved_checkerboard_data = None
        self.calibration_method = None  # 'ellipse' or 'checkerboard'
        self.calibration_resolution = None
        self.debug_mode = False
        self.debug_frame = None

        # Load saved calibration on startup
        self.load_calibration()

    def set_debug_mode(self, enabled):
        """Enable or disable debug visualization"""
        self.debug_mode = enabled

    def get_debug_frame(self):
        """Get the current debug frame"""
        return self.debug_frame

    def analyze_ring_detection(self, frame, save_debug=True, output_dir="test_outputs"):
        """
        Analyze why multiple rings might not be detected as separate contours
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # Try different edge detection approaches
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Multiple Canny thresholds
        edges_conservative = cv2.Canny(blurred, 100, 200, apertureSize=3)
        edges_moderate = cv2.Canny(blurred, 50, 150, apertureSize=3)  # Original
        edges_aggressive = cv2.Canny(blurred, 30, 100, apertureSize=3)

        # Try different contour retrieval modes
        contours_external, _ = cv2.findContours(edges_moderate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list, _ = cv2.findContours(edges_moderate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_tree, hierarchy = cv2.findContours(edges_moderate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print(f"\n=== RING DETECTION ANALYSIS ===")
        print(f"Edge detection results:")
        print(f"- Conservative (100-200): {cv2.countNonZero(edges_conservative)} edge pixels")
        print(f"- Moderate (50-150): {cv2.countNonZero(edges_moderate)} edge pixels")
        print(f"- Aggressive (30-100): {cv2.countNonZero(edges_aggressive)} edge pixels")

        print(f"\nContour detection results:")
        print(f"- RETR_EXTERNAL: {len(contours_external)} contours")
        print(f"- RETR_LIST: {len(contours_list)} contours")
        print(f"- RETR_TREE: {len(contours_tree)} contours")

        # Analyze contour sizes
        def analyze_contours(contours, name):
            areas = [cv2.contourArea(c) for c in contours]
            large_contours = [a for a in areas if a > 1000]
            print(f"\n{name} contour analysis:")
            print(f"- Total contours: {len(contours)}")
            print(f"- Large contours (>1000px): {len(large_contours)}")
            if large_contours:
                print(f"- Area range: {min(large_contours):.0f} to {max(large_contours):.0f}")

        analyze_contours(contours_external, "EXTERNAL")
        analyze_contours(contours_list, "LIST")
        analyze_contours(contours_tree, "TREE")

        if save_debug:
            # Save edge detection comparisons
            edge_comparison = np.hstack([edges_conservative, edges_moderate, edges_aggressive])
            cv2.putText(edge_comparison, "Conservative", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(edge_comparison, "Moderate", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.putText(edge_comparison, "Aggressive", (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

            edge_path = os.path.join(output_dir, "04b_edge_comparison.jpg")
            cv2.imwrite(edge_path, edge_comparison)
            print(f"Saved edge comparison: {edge_path}")

            # Save contour comparisons
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            contour_comparison_frame = frame.copy()

            # Draw all contours from different methods
            cv2.drawContours(contour_comparison_frame, contours_external, -1, colors[0], 2)
            cv2.drawContours(contour_comparison_frame, [c for c in contours_list if cv2.contourArea(c) > 500], -1, colors[1], 1)

            # Add legend
            cv2.putText(contour_comparison_frame, "Red: EXTERNAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[0], 2)
            cv2.putText(contour_comparison_frame, "Green: LIST (>500px)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[1], 2)

            contour_path = os.path.join(output_dir, "05b_contour_comparison.jpg")
            cv2.imwrite(contour_path, contour_comparison_frame)
            print(f"Saved contour comparison: {contour_path}")

        return contours_tree, hierarchy, edges_moderate

    def experimental_ellipse_detection(self, frame,
                                     contour_mode=cv2.RETR_LIST,
                                     canny_low=50, canny_high=150,
                                     min_area=1000,
                                     min_circularity=0.3):
        """
        Experimental ellipse detection with tunable parameters

        Args:
            contour_mode: cv2.RETR_EXTERNAL, cv2.RETR_LIST, or cv2.RETR_TREE
            canny_low, canny_high: Canny edge detection thresholds
            min_area: Minimum contour area to consider
            min_circularity: Minimum circularity (0-1) for ring candidates
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # Configurable edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)

        # Configurable contour retrieval
        if contour_mode == cv2.RETR_TREE:
            contours, hierarchy = cv2.findContours(edges, contour_mode, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv2.findContours(edges, contour_mode, cv2.CHAIN_APPROX_SIMPLE)
            hierarchy = None

        mode_name = {cv2.RETR_EXTERNAL: "EXTERNAL", cv2.RETR_LIST: "LIST", cv2.RETR_TREE: "TREE"}.get(contour_mode, "UNKNOWN")
        print(f"Original contours found: {len(contours)} (using RETR_{mode_name})")

        # Filter for ring-like contours (circular and reasonable size)
        ring_contours = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > min_area:  # Configurable minimum area
                # Check if it could be a ring by testing circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > min_circularity:  # Configurable circularity threshold
                        ring_contours.append(contour)
                        print(f"  Ring candidate {len(ring_contours)}: area={area:.0f}, circularity={circularity:.2f}")

        print(f"Ring candidates found: {len(ring_contours)}")

        # Apply same filtering logic as original detector
        rejection_reasons = []
        analyzed_contours = 0
        best_ellipse = None
        best_score = 0
        best_contour = None
        all_detected_ellipses = []  # Store all valid ellipses with their scores

        for contour in ring_contours:
            if len(contour) < 5:  # Need at least 5 points to fit ellipse
                print("skip contour: too few points")
                rejection_reasons.append(f"Contour {analyzed_contours+1}: <5 points ({len(contour)})")
                continue

            area = cv2.contourArea(contour)
            if area < 1000:  # Skip small contours
                print("skip contour: too small")
                continue

            analyzed_contours += 1

            try:
                # Fit ellipse to contour
                ellipse = cv2.fitEllipse(contour)
                (center_x, center_y), (minor_axis, major_axis), angle = ellipse

                # Calculate aspect ratio (how "stretched" the ellipse is)
                aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)

                # Use same constraints as original detector
                if aspect_ratio < -0.8:
                    rejection_reasons.append(f"Contour {analyzed_contours}: too circular (ratio={aspect_ratio:.2f})")
                    continue
                elif aspect_ratio > 1.2:
                    rejection_reasons.append(f"Contour {analyzed_contours}: too elongated (ratio={aspect_ratio:.2f})")
                    continue

                # Score based on size and reasonable ellipse properties
                size_score = area / (w * h)  # Prefer larger ellipses
                aspect_score = 1.0 / aspect_ratio  # Prefer moderate distortion

                # Check if ellipse is reasonably centered
                center_score = 1.0 - (abs(center_x - w/2) + abs(center_y - h/2)) / (w + h)

                score = size_score * 0.5 + aspect_score * 0.3 + center_score * 0.2
                
                # Add this ellipse to our collection
                all_detected_ellipses.append((ellipse, score, contour))

                if score > best_score:
                    best_score = score
                    best_ellipse = ellipse
                    best_contour = contour
                    rejection_reasons.append(f"Contour {analyzed_contours}: ACCEPTED (score={score:.3f}, ratio={aspect_ratio:.2f})")
                else:
                    rejection_reasons.append(f"Contour {analyzed_contours}: valid ellipse (score={score:.3f} vs {best_score:.3f})")

            except cv2.error as e:
                rejection_reasons.append(f"Contour {analyzed_contours}: ellipse fit failed ({str(e)[:20]})")
                continue

        return best_ellipse, best_contour, rejection_reasons, edges, all_detected_ellipses

    def draw_all_ellipses(self, frame, all_ellipses, best_ellipse=None):
        """
        Draw all detected ellipses on the frame with different colors
        """
        result_frame = frame.copy()

        # Colors for different ellipses (excluding green which we'll use for best)
        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]

        # Draw all detected ellipses
        for i, (ellipse, score, contour) in enumerate(all_ellipses):
            color = colors[i % len(colors)]

            # Draw ellipse outline
            cv2.ellipse(result_frame, ellipse, color, 2)

            # Draw center point
            center = tuple(map(int, ellipse[0]))
            cv2.circle(result_frame, center, 3, color, -1)

            # Add score label
            cv2.putText(result_frame, f"E{i+1}: {score:.3f}",
                       (center[0] + 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw best ellipse in bright green on top
        if best_ellipse is not None:
            cv2.ellipse(result_frame, best_ellipse, (0, 255, 0), 4)
            best_center = tuple(map(int, best_ellipse[0]))
            cv2.circle(result_frame, best_center, 5, (0, 255, 0), -1)
            cv2.putText(result_frame, "BEST",
                       (best_center[0] + 15, best_center[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add legend
        y_offset = 30
        cv2.putText(result_frame, f"All detected ellipses: {len(all_ellipses)}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return result_frame

    def create_ellipse_to_circle_transform(self, ellipse, output_size):
        """
        Create ellipse-to-circle transformation using 4-point perspective mapping
        Maps ellipse-aligned bounding box to circle-aligned bounding box
        """
        (center_x, center_y), (axis1, axis2), angle = ellipse
        output_width, output_height = output_size

        # Determine which is major/minor
        minor_axis = min(axis1, axis2)
        major_axis = max(axis1, axis2)

        print(f"Creating ellipse-to-circle transform:")
        print(f"  Ellipse center: ({center_x:.1f}, {center_y:.1f})")
        print(f"  Ellipse axes: axis1={axis1:.1f}, axis2={axis2:.1f}")
        print(f"  Determined: minor={minor_axis:.1f}, major={major_axis:.1f}")
        print(f"  Ellipse angle: {angle:.1f}°")

        # Extend axes to get full ellipse dimensions (diameter * sqrt(2) for bounding box)
        a1_extended = axis1 * np.sqrt(2)
        a2_extended = axis2 * np.sqrt(2)

        # Calculate major axis endpoints (perpendicular to angle)
        # axis2 is along the perpendicular direction (90° - angle)
        angle_rad = np.radians(angle)
        maj_dx = (a2_extended / 2) * np.cos(np.radians(90 - angle))
        maj_dy = (a2_extended / 2) * np.sin(np.radians(90 - angle))

        # Calculate minor axis endpoints (along the angle)
        # axis1 is along the angle direction
        min_dx = (a1_extended / 2) * np.cos(angle_rad)
        min_dy = (a1_extended / 2) * np.sin(angle_rad)

        # Four corner points of ellipse-aligned bounding box
        p1 = np.array([center_x + maj_dx, center_y - maj_dy])
        p2 = np.array([center_x - maj_dx, center_y + maj_dy])
        p3 = np.array([center_x - min_dx, center_y - min_dy])
        p4 = np.array([center_x + min_dx, center_y + min_dy])

        # For a circle, minor axis points should be rotated 90° from their current position
        # These are the target positions for p3 and p4
        p3_circle = np.array([center_x - maj_dy, center_y - maj_dx])
        p4_circle = np.array([center_x + maj_dy, center_y + maj_dx])

        # Source points (ellipse)
        pts_src = np.float32([p1, p2, p3, p4])

        # Target points (circle) - major axis stays same, minor axis becomes perpendicular
        pts_dst = np.float32([p1, p2, p3_circle, p4_circle])

        # Calculate perspective transform from these 4 point pairs
        transform = cv2.getPerspectiveTransform(pts_src, pts_dst)

        compression_factor = minor_axis / major_axis
        print(f"  Compression factor: {compression_factor:.3f}")
        print(f"  Using 4-point perspective transform")

        return transform

    def measure_circle_circularity(self, frame, center, radius):
        """
        Measure how circular a detected shape is after transformation

        Args:
            frame: Transformed frame to analyze
            center: (x, y) center point
            radius: Expected radius

        Returns:
            circularity_score: 0.0 to 1.0, where 1.0 is perfectly circular
        """
        h, w = frame.shape[:2]
        cx, cy = int(center[0]), int(center[1])
        r = int(radius)

        # Ensure circle is within bounds
        if cx - r < 0 or cy - r < 0 or cx + r >= w or cy + r >= h:
            return 0.0

        # Extract region around the circle
        roi_size = r * 2 + 20  # Add margin
        x1 = max(0, cx - roi_size // 2)
        y1 = max(0, cy - roi_size // 2)
        x2 = min(w, cx + roi_size // 2)
        y2 = min(h, cy + roi_size // 2)

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Detect edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Find the largest contour (likely the outer ring)
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) < 5:
            return 0.0

        # Fit ellipse to the detected contour
        try:
            fitted_ellipse = cv2.fitEllipse(largest_contour)
            (_, _), (minor_axis, major_axis), _ = fitted_ellipse

            # Calculate how circular it is
            aspect_ratio = max(major_axis, minor_axis) / max(min(major_axis, minor_axis), 1.0)

            # Circularity: 1.0 = perfect circle, decreases as it becomes more elliptical
            circularity = 1.0 / aspect_ratio

            return circularity
        except:
            return 0.0

    def transform_ellipse_center(self, ellipse, transform_matrix):
        """
        Calculate where the ellipse center will be after applying the transformation matrix

        Args:
            ellipse: OpenCV ellipse tuple ((center_x, center_y), (minor_axis, major_axis), angle)
            transform_matrix: 3x3 transformation matrix

        Returns:
            new_center: (x, y) tuple of the transformed center coordinates
        """
        (center_x, center_y), _, _ = ellipse

        # Create homogeneous coordinates for the center point
        center_homogeneous = np.array([center_x, center_y, 1.0], dtype=np.float32)

        # Apply transformation matrix
        transformed_homogeneous = transform_matrix @ center_homogeneous

        # Convert back to cartesian coordinates
        new_center_x = transformed_homogeneous[0] / transformed_homogeneous[2]
        new_center_y = transformed_homogeneous[1] / transformed_homogeneous[2]

        return (new_center_x, new_center_y)

    def calculate_transformed_circle_size(self, ellipse, transform_matrix):
        """
        Calculate the radius of the circle that the ellipse becomes after transformation

        Args:
            ellipse: OpenCV ellipse tuple ((center_x, center_y), (axis1, axis2), angle)
            transform_matrix: 3x3 transformation matrix

        Returns:
            circle_radius: radius of the resulting circle in pixels
        """
        (center_x, center_y), (axis1, axis2), angle = ellipse

        # Convert angle to radians
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Calculate points on the ellipse axes from the center
        # In OpenCV fitEllipse, the angle tells us the orientation of axis1
        # axis1 endpoints (along the angle direction)
        axis1_half = axis1 / 2
        axis1_end1 = np.array([center_x + axis1_half * cos_a, center_y + axis1_half * sin_a, 1.0])
        axis1_end2 = np.array([center_x - axis1_half * cos_a, center_y - axis1_half * sin_a, 1.0])

        # axis2 endpoints (perpendicular to axis1)
        axis2_half = axis2 / 2
        axis2_end1 = np.array([center_x - axis2_half * sin_a, center_y + axis2_half * cos_a, 1.0])
        axis2_end2 = np.array([center_x + axis2_half * sin_a, center_y - axis2_half * cos_a, 1.0])

        # Transform the center and axis endpoints
        center_transformed = transform_matrix @ np.array([center_x, center_y, 1.0])
        axis1_end1_transformed = transform_matrix @ axis1_end1
        axis1_end2_transformed = transform_matrix @ axis1_end2
        axis2_end1_transformed = transform_matrix @ axis2_end1
        axis2_end2_transformed = transform_matrix @ axis2_end2

        # Convert from homogeneous coordinates
        def to_cartesian(point):
            return np.array([point[0] / point[2], point[1] / point[2]])

        center_cart = to_cartesian(center_transformed)
        axis1_end1_cart = to_cartesian(axis1_end1_transformed)
        axis1_end2_cart = to_cartesian(axis1_end2_transformed)
        axis2_end1_cart = to_cartesian(axis2_end1_transformed)
        axis2_end2_cart = to_cartesian(axis2_end2_transformed)

        # Calculate distances from center to each axis endpoint
        axis1_radius1 = np.linalg.norm(axis1_end1_cart - center_cart)
        axis1_radius2 = np.linalg.norm(axis1_end2_cart - center_cart)
        axis2_radius1 = np.linalg.norm(axis2_end1_cart - center_cart)
        axis2_radius2 = np.linalg.norm(axis2_end2_cart - center_cart)

        # Average the radii (should be very similar if transformation worked correctly)
        avg_axis1_radius = (axis1_radius1 + axis1_radius2) / 2
        avg_axis2_radius = (axis2_radius1 + axis2_radius2) / 2
        overall_avg_radius = (avg_axis1_radius + avg_axis2_radius) / 2

        print(f"  Transformed axis radii:")
        print(f"    Axis 1: {avg_axis1_radius:.1f} pixels")
        print(f"    Axis 2: {avg_axis2_radius:.1f} pixels")
        print(f"    Difference: {abs(avg_axis1_radius - avg_axis2_radius):.1f} pixels")

        return overall_avg_radius

    def calculate_ellipse_to_circle_matrix(self, ellipse, output_size):
        """
        Calculate the ellipse-to-circle transformation matrix and metadata

        Args:
            ellipse: Detected ellipse
            output_size: Output image dimensions (width, height)

        Returns:
            transform_matrix: The transformation matrix to convert ellipse to circle
            circle_radius: Radius of the resulting circle
            transformed_center: Center of the transformed ellipse
        """
        if ellipse is None:
            return None, None, None

        # Get transformation matrix
        transform_matrix = self.create_ellipse_to_circle_transform(ellipse, output_size)

        # Calculate transformation results
        transformed_center = self.transform_ellipse_center(ellipse, transform_matrix)
        circle_radius = self.calculate_transformed_circle_size(ellipse, transform_matrix)

        return transform_matrix, circle_radius, transformed_center

    def apply_ellipse_to_circle_transform(self, frame, ellipse, output_size=None):
        """
        Apply the ellipse-to-circle transformation to an image

        Args:
            frame: Input image
            ellipse: Detected ellipse
            output_size: Output image dimensions (defaults to input size)

        Returns:
            transformed_image: Warped image with ellipse corrected to circle
            transform_matrix: The transformation matrix used
            circle_radius: Radius of the resulting circle
        """
        if ellipse is None:
            return None, None, None

        if output_size is None:
            output_size = (frame.shape[1], frame.shape[0])

        # Calculate transformation matrix and metadata
        transform_matrix, circle_radius, transformed_center = self.calculate_ellipse_to_circle_matrix(ellipse, output_size)

        if transform_matrix is None:
            return None, None, None

        # Apply perspective transformation
        transformed = cv2.warpPerspective(frame, transform_matrix, output_size,
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))

        return transformed, transform_matrix, circle_radius

    def apply_ellipse_matrix_transform(self, frame, transform_matrix, output_size=None):
        """
        Apply a pre-calculated ellipse-to-circle transformation matrix to a frame

        Args:
            frame: Input image
            transform_matrix: Pre-calculated 3x3 transformation matrix
            output_size: Output image dimensions (defaults to input size)

        Returns:
            transformed_image: Warped image with transformation applied
        """
        if transform_matrix is None:
            return None

        if output_size is None:
            output_size = (frame.shape[1], frame.shape[0])

        # Apply perspective transformation
        transformed = cv2.warpPerspective(frame, transform_matrix, output_size,
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))

        return transformed

    def add_transformation_visualization(self, image, ellipse, transform_matrix, circle_radius, output_size):
        """
        Add visual markers to show transformation results

        Args:
            image: Transformed image to annotate
            ellipse: Original ellipse
            transform_matrix: Transformation matrix used
            circle_radius: Radius of resulting circle
            output_size: Image dimensions

        Returns:
            annotated_image: Image with visual markers added
        """
        annotated = image.copy()
        transformed_center = self.transform_ellipse_center(ellipse, transform_matrix)
        output_width, output_height = output_size

        # Draw the calculated circle outline
        transformed_center_int = (int(transformed_center[0]), int(transformed_center[1]))
        cv2.circle(annotated, transformed_center_int, int(circle_radius), (255, 255, 0), 3)  # Yellow circle outline

        # Draw transformed center (where the ellipse center ended up)
        cv2.circle(annotated, transformed_center_int, 8, (0, 255, 0), -1)  # Green circle
        cv2.putText(annotated, "Transformed Center",
                   (transformed_center_int[0] + 15, transformed_center_int[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Also show where the image center is (should be close to transformed center)
        image_center = (output_width // 2, output_height // 2)
        cv2.circle(annotated, image_center, 5, (255, 0, 0), -1)  # Blue circle
        cv2.putText(annotated, "Image Center",
                   (image_center[0] + 15, image_center[1] + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Add circle size annotation
        cv2.putText(annotated, f"Circle: R={circle_radius:.0f}px",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return annotated

    def create_improved_perspective_debug_with_preview(self, original_frame, transform_matrix, detected_ellipse, corner_debug_frame=None):
        """Create side-by-side debug frame using improved transformation and visualization"""
        h, w = original_frame.shape[:2]

        # Create side-by-side canvas (double width)
        combined_width = w * 2
        combined_height = h
        side_by_side_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        # Left side: Current detection visualization
        if corner_debug_frame is not None:
            resized_debug = cv2.resize(corner_debug_frame, (w, h))
            side_by_side_frame[:h, :w] = resized_debug

        # Right side: Improved perspective correction preview
        try:
            if detected_ellipse is not None:
                # Apply the provided transformation matrix (don't recalculate!)
                transformed_image = cv2.warpPerspective(original_frame, transform_matrix, (w, h))

                if transformed_image is not None:
                    # Calculate circle radius for visualization
                    circle_radius = self.calculate_transformed_circle_size(detected_ellipse, transform_matrix)

                    # Add transformation visualization markers
                    annotated_image = self.add_transformation_visualization(
                        transformed_image, detected_ellipse, transform_matrix, circle_radius, (w, h))
                    side_by_side_frame[:h, w:] = annotated_image

                    # Calculate transformation details for display
                    transformed_center = self.transform_ellipse_center(detected_ellipse, transform_matrix)
                    center_distance = np.sqrt((transformed_center[0] - w/2)**2 + (transformed_center[1] - h/2)**2)

                    # Add transformation info
                    cv2.putText(side_by_side_frame, "IMPROVED CORRECTION", (w + 10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    cv2.putText(side_by_side_frame, f"Radius: {circle_radius:.0f}px", (w + 10, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(side_by_side_frame, f"Center error: {center_distance:.0f}px", (w + 10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    # Transformation failed
                    self.add_error_preview(side_by_side_frame, w, h, "Transformation failed")
            else:
                # Using saved matrix without current detection
                corrected_preview = cv2.warpPerspective(original_frame, transform_matrix, (w, h))
                side_by_side_frame[:h, w:] = corrected_preview
                cv2.putText(side_by_side_frame, "SAVED CALIBRATION", (w + 10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            # Add labels
            cv2.putText(side_by_side_frame, "DETECTION", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            # Add dividing line
            cv2.line(side_by_side_frame, (w, 0), (w, h), (255, 255, 255), 2)

        except Exception as e:
            # Error occurred during transformation
            self.add_error_preview(side_by_side_frame, w, h, f"Error: {str(e)[:30]}")

        return side_by_side_frame

    def add_error_preview(self, side_by_side_frame, w, h, error_message):
        """Add error message to the right side of debug frame"""
        error_frame = np.zeros((h, w, 3), dtype=np.uint8)
        error_frame.fill(50)  # Dark gray background
        cv2.putText(error_frame, "CORRECTION FAILED", (w//4, h//2 - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(error_frame, error_message, (10, h//2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        side_by_side_frame[:h, w:] = error_frame

        # Add labels
        cv2.putText(side_by_side_frame, "DETECTION", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(side_by_side_frame, "ERROR", (w + 10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Add dividing line
        cv2.line(side_by_side_frame, (w, 0), (w, h), (255, 255, 255), 2)

    def create_perspective_debug_with_preview(self, original_frame, perspective_matrix, corner_debug_frame=None):
        """Create side-by-side perspective debug frame with correction preview"""
        h, w = original_frame.shape[:2]

        # Create side-by-side canvas (double width)
        combined_width = w * 2
        combined_height = h
        side_by_side_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        # Left side: Current corner debug frame (detection visualization)
        if corner_debug_frame is not None:
            resized_debug = cv2.resize(corner_debug_frame, (w, h))
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
            matrix_source = "LIVE DETECTION"

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
            expected_ring_center = (int(w + center_x_dst), int(center_y_dst))
            expected_ring_radius = int(square_half * 0.85)
            cv2.circle(side_by_side_frame, expected_ring_center, expected_ring_radius, (0, 255, 0), 3)
            cv2.putText(side_by_side_frame, "Expected outer ring",
                       (w + 10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add a dividing line
            cv2.line(side_by_side_frame, (w, 0), (w, h), (255, 255, 255), 2)

        except cv2.error as e:
            # If perspective correction fails, show error message on right side
            self.add_error_preview(side_by_side_frame, w, h, str(e))

            # Add a dividing line
            cv2.line(side_by_side_frame, (w, 0), (w, h), (255, 255, 255), 2)

        return side_by_side_frame

    def create_ellipse_detection_debug_frame(self, edges, all_ellipses, best_ellipse, rejection_reasons, frame_size):
        """Create debug visualization for ellipse detection results"""
        w, h = frame_size
        debug_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        cv2.putText(debug_frame, "IMPROVED ELLIPSE DETECTION",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(debug_frame, f"Using RETR_LIST mode",
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(debug_frame, f"Ellipses found: {len(all_ellipses)}",
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # Draw all detected ellipses
        for i, (ellipse, score, contour) in enumerate(all_ellipses):
            color = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][i % 4]
            cv2.ellipse(debug_frame, ellipse, color, 2)
            center = tuple(map(int, ellipse[0]))
            cv2.putText(debug_frame, f"E{i+1}: {score:.2f}",
                       (center[0] + 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw the best ellipse in bright green
        if best_ellipse is not None:
            cv2.ellipse(debug_frame, best_ellipse, (0, 255, 0), 4)
            (center_x, center_y), (minor_axis, major_axis), angle = best_ellipse
            aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)
            cv2.putText(debug_frame, f"BEST: ratio={aspect_ratio:.3f}",
                       (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # Draw center and axes for the best ellipse
            center_int = (int(center_x), int(center_y))
            cv2.circle(debug_frame, center_int, 5, (0, 255, 0), -1)

            # Draw major and minor axes
            angle_rad = np.radians(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

            # Major axis
            major_half = major_axis / 2
            major_end1 = (int(center_x + major_half * cos_a), int(center_y + major_half * sin_a))
            major_end2 = (int(center_x - major_half * cos_a), int(center_y - major_half * sin_a))
            cv2.line(debug_frame, major_end1, major_end2, (255, 0, 0), 3)

            # Minor axis
            minor_half = minor_axis / 2
            minor_end1 = (int(center_x - minor_half * sin_a), int(center_y + minor_half * cos_a))
            minor_end2 = (int(center_x + minor_half * sin_a), int(center_y - minor_half * cos_a))
            cv2.line(debug_frame, minor_end1, minor_end2, (0, 255, 255), 3)
        else:
            cv2.putText(debug_frame, "No suitable ellipse found",
                       (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Display some rejection reasons
        y_offset = 190
        max_reasons = min(4, len(rejection_reasons))
        for i in range(max_reasons):
            reason = rejection_reasons[i]
            if len(reason) > 50:
                reason = reason[:47] + "..."
            color = (0, 255, 0) if "ACCEPTED" in reason else (255, 255, 255)
            cv2.putText(debug_frame, reason,
                       (10, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        return debug_frame

    def detect_ellipse_perspective_transform(self, frame, debug_mode=None, min_aspect_ratio=None):
        """
        Detect perspective transformation based on outer circle appearing as ellipse
        Uses the improved detection from perspective.py module

        Args:
            frame: Input frame to analyze
            debug_mode: Whether to create debug visualizations
            min_aspect_ratio: Minimum aspect ratio threshold - only accept ellipses with
                            aspect ratio better (closer to 1.0) than this value

        Returns:
            tuple: (perspective_matrix, detected_ellipse, debug_frame) where:
                - perspective_matrix: 3x3 perspective transformation matrix, or None if no ellipse found
                - detected_ellipse: OpenCV ellipse tuple ((center_x, center_y), (minor_axis, major_axis), angle), or None if no ellipse found
                - debug_frame: Debug visualization frame or None if debug_mode is False
        """
        h, w = frame.shape[:2]
        debug_frame = None

        # Use class debug mode if not explicitly overridden
        if debug_mode is None:
            debug_mode = self.debug_mode

        # Use the improved ellipse detection
        # This uses RETR_LIST and better ellipse detection logic
        best_ellipse, best_contour, rejection_reasons, edges, all_ellipses = self.experimental_ellipse_detection(
            frame,
            contour_mode=cv2.RETR_LIST,  # Use RETR_LIST for better occlusion handling
            canny_low=50, canny_high=150,
            min_area=1000,
            min_circularity=0.3
        )

        # Filter ellipses by aspect ratio if threshold provided
        if min_aspect_ratio is not None and len(all_ellipses) > 0:
            filtered_ellipses = []
            for ellipse, score, contour in all_ellipses:
                (center_x, center_y), (minor_axis, major_axis), angle = ellipse
                aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)

                # Only accept ellipses with better (lower) aspect ratio than threshold
                if aspect_ratio < min_aspect_ratio:
                    filtered_ellipses.append((ellipse, score, contour))
                    print(f"  ✓ Accepted ellipse: aspect_ratio={aspect_ratio:.3f} < {min_aspect_ratio:.3f}")
                else:
                    print(f"  ✗ Filtered ellipse: aspect_ratio={aspect_ratio:.3f} >= {min_aspect_ratio:.3f}")

            # If we filtered out all ellipses, use None
            if len(filtered_ellipses) == 0:
                print(f"  ⚠ All {len(all_ellipses)} ellipses filtered out (worse than threshold {min_aspect_ratio:.3f})")
                best_ellipse = None
                best_contour = None
            else:
                # Re-select best from filtered ellipses
                all_ellipses = filtered_ellipses
                best_ellipse, best_score, best_contour = max(filtered_ellipses, key=lambda x: x[1])
                print(f"  → Selected best from {len(filtered_ellipses)} filtered ellipses (score={best_score:.3f})")

        # Create debug visualization if requested
        if debug_mode:
            debug_frame = self.create_ellipse_detection_debug_frame(
                edges, all_ellipses, best_ellipse, rejection_reasons, (w, h)
            )

        if best_ellipse is not None:
            # Use the improved ellipse-to-circle transformation
            # This properly handles compression and scaling to eliminate black borders
            transformed_image, transform_matrix, circle_radius = self.apply_ellipse_to_circle_transform(
                frame, best_ellipse, (w, h))

            if transform_matrix is not None and debug_mode and debug_frame is not None:

                self.saved_perspective_matrix = transform_matrix

                # Create preview showing the transformation result
                debug_frame = self.create_improved_perspective_debug_with_preview(
                    frame, transform_matrix, best_ellipse, debug_frame
                )

            # Store debug frame in instance
            if debug_mode:
                self.debug_frame = debug_frame

            return transform_matrix, best_ellipse, debug_frame

        # If no ellipse found but we have a saved matrix, still show preview
        else:
            debug_frame = frame

        # Store debug frame in instance
        if debug_mode:
            self.debug_frame = debug_frame

        return None, None, debug_frame

    def apply_perspective_correction(self, frame, matrix=None, output_size=None):
        """
        Apply perspective correction to frame using the original frame dimensions
        Handles both affine (2x3) and perspective (3x3) transformation matrices
        """
        # Use saved matrix if none provided
        if matrix is None:
            matrix = self.saved_perspective_matrix

        if matrix is None:
            return None

        # Use original frame dimensions if output_size not specified
        if output_size is None:
            input_height, input_width = frame.shape[:2]
            output_size = (input_width, input_height)

        # Apply the transformation
        if matrix.shape == (2, 3):
            return cv2.warpAffine(frame, matrix, output_size)
        elif matrix.shape == (3, 3):
            return cv2.warpPerspective(frame, matrix, output_size)
        else:
            print(f"Unsupported matrix shape: {matrix.shape}")
            return None

    def save_calibration(self, camera_resolution=None):
        """Save current perspective calibration to YAML file"""
        if self.saved_perspective_matrix is None:
            return False, "No perspective calibration to save"

        try:
            # Prepare calibration data
            calibration_data = {
                'timestamp': time.time(),
                'perspective_matrix': self.saved_perspective_matrix.tolist(),
                'camera_resolution': list(camera_resolution) if camera_resolution else None,
                'calibration_method': self.calibration_method,
                'notes': 'Perspective calibration for fixed camera installation'
            }

            # Add method-specific data
            if self.calibration_method == 'ellipse' and self.saved_ellipse_data:
                calibration_data['ellipse_data'] = self.saved_ellipse_data
            elif self.calibration_method == 'checkerboard' and self.saved_checkerboard_data:
                calibration_data['checkerboard_data'] = self.saved_checkerboard_data

            # Save to YAML file
            with open(self.calibration_file, 'w') as f:
                yaml.dump(calibration_data, f, default_flow_style=False)

            method_name = self.calibration_method or 'unknown'
            print(f"Perspective calibration ({method_name}) saved to {self.calibration_file}")
            return True, f"Calibration saved successfully ({method_name})"

        except Exception as e:
            error_msg = f"Failed to save calibration: {str(e)}"
            print(error_msg)
            return False, error_msg

    def load_calibration(self):
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

            # Load calibration method
            self.calibration_method = calibration_data.get('calibration_method', 'unknown')

            # Load method-specific data
            if 'ellipse_data' in calibration_data:
                self.saved_ellipse_data = calibration_data['ellipse_data']
            if 'checkerboard_data' in calibration_data:
                self.saved_checkerboard_data = calibration_data['checkerboard_data']

            timestamp = calibration_data.get('timestamp', 0)
            age_hours = (time.time() - timestamp) / 3600

            method_name = self.calibration_method or 'unknown'
            print(f"Perspective calibration ({method_name}) loaded from {self.calibration_file} (age: {age_hours:.1f} hours)")
            return True, f"Calibration loaded successfully ({method_name}, age: {age_hours:.1f}h)"

        except Exception as e:
            error_msg = f"Failed to load calibration: {str(e)}"
            print(error_msg)
            return False, error_msg

    def detect_checkerboard_corners(self, frame, pattern_size=(9, 6)):
        """
        Detect checkerboard pattern corners for calibration

        Args:
            frame: Input image containing checkerboard pattern
            pattern_size: Tuple of (columns, rows) of internal corners (e.g., 9x6 for 10x7 squares)

        Returns:
            corners: Detected corner points, or None if detection failed
            debug_frame: Debug visualization frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # Find checkerboard corners
        # cv2.CALIB_CB_ADAPTIVE_THRESH - Use adaptive thresholding
        # cv2.CALIB_CB_NORMALIZE_IMAGE - Normalize image gamma
        # cv2.CALIB_CB_FAST_CHECK - Fast check to reject non-checkerboard patterns
        ret, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        )

        debug_frame = frame.copy()

        if ret:
            # Refine corner locations to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw detected corners for visualization
            cv2.drawChessboardCorners(debug_frame, pattern_size, corners, ret)

            # Add success message
            cv2.putText(debug_frame, f"CHECKERBOARD DETECTED ({pattern_size[0]}x{pattern_size[1]})",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(debug_frame, f"Corners found: {len(corners)}",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            print(f"Checkerboard detected: {pattern_size[0]}x{pattern_size[1]} with {len(corners)} corners")
        else:
            # Add failure message
            cv2.putText(debug_frame, f"CHECKERBOARD NOT FOUND ({pattern_size[0]}x{pattern_size[1]})",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(debug_frame, "Ensure pattern is visible and well-lit",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            print(f"Checkerboard not detected with pattern size {pattern_size}")

        return corners if ret else None, debug_frame

    def create_checkerboard_perspective_transform(self, frame, corners, pattern_size=(9, 6)):
        """
        Create perspective transformation matrix from checkerboard corners

        Args:
            frame: Input image
            corners: Detected checkerboard corners
            pattern_size: Tuple of (columns, rows) of internal corners

        Returns:
            transform_matrix: 3x3 perspective transformation matrix
            corrected_frame: Perspective-corrected image
        """
        h, w = frame.shape[:2]

        # Extract the four outermost corners of the checkerboard
        # Corners are returned in row-major order (left to right, top to bottom)
        top_left = corners[0][0]
        top_right = corners[pattern_size[0] - 1][0]
        bottom_right = corners[-1][0]
        bottom_left = corners[-pattern_size[0]][0]

        # Source points (detected corners)
        src_points = np.float32([top_left, top_right, bottom_right, bottom_left])

        # Calculate the size of the checkerboard in the output image
        # Use the average of horizontal and vertical dimensions to maintain aspect ratio
        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        max_width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(bottom_left - top_left)
        height_right = np.linalg.norm(bottom_right - top_right)
        max_height = int(max(height_left, height_right))

        # Center the corrected pattern in the frame
        # Calculate margins to center the pattern
        margin_x = (w - max_width) // 2
        margin_y = (h - max_height) // 2

        # Destination points (centered rectangle in output frame)
        dst_points = np.float32([
            [margin_x, margin_y],                           # top-left
            [margin_x + max_width, margin_y],              # top-right
            [margin_x + max_width, margin_y + max_height], # bottom-right
            [margin_x, margin_y + max_height]              # bottom-left
        ])

        # Calculate perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Apply transformation
        corrected_frame = cv2.warpPerspective(frame, transform_matrix, (w, h))

        print(f"Checkerboard transform created:")
        print(f"  Pattern dimensions: {max_width}x{max_height}")
        print(f"  Centered at: ({margin_x}, {margin_y})")

        return transform_matrix, corrected_frame

    def calibrate_perspective_checkerboard(self, frame, pattern_size=(9, 6)):
        """
        Calibrate perspective using checkerboard pattern

        Args:
            frame: Input frame containing checkerboard pattern
            pattern_size: Tuple of (columns, rows) of internal corners

        Returns:
            success: True if calibration succeeded
            message: Status message
        """
        # Detect checkerboard corners
        corners, debug_frame = self.detect_checkerboard_corners(frame, pattern_size)

        # Store debug frame
        if self.debug_mode:
            self.debug_frame = debug_frame

        if corners is None:
            return False, f"Checkerboard pattern {pattern_size[0]}x{pattern_size[1]} not detected"

        # Create perspective transformation
        try:
            transform_matrix, corrected_frame = self.create_checkerboard_perspective_transform(
                frame, corners, pattern_size
            )

            # Store the calibration
            self.saved_perspective_matrix = transform_matrix
            self.calibration_method = 'checkerboard'
            self.saved_checkerboard_data = {
                'pattern_size': list(pattern_size),  # Convert tuple to list for YAML compatibility
                'calibration_timestamp': time.time(),
            }

            # Update debug frame with corrected view
            if self.debug_mode:
                # Create side-by-side visualization
                h, w = frame.shape[:2]
                combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
                combined[:, :w] = debug_frame
                combined[:, w:] = corrected_frame

                # Add labels
                cv2.putText(combined, "DETECTION", (10, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(combined, "CORRECTED", (w + 10, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)

                self.debug_frame = combined

            return True, f"Checkerboard calibration successful ({pattern_size[0]}x{pattern_size[1]})"

        except Exception as e:
            print(f"Error creating checkerboard transform: {e}")
            return False, f"Failed to create perspective transform: {str(e)}"

    def calibrate_perspective_ellipse_iterative(self, frame, max_iterations=3, min_circularity=0.95):
        """
        Iteratively refine ellipse-based perspective calibration

        Args:
            frame: Input frame for calibration
            max_iterations: Maximum number of refinement iterations
            min_circularity: Target circularity (1.0 = perfect circle)

        Returns:
            success: True if calibration succeeded
            message: Status message with iteration details
        """
        print(f"\n=== ITERATIVE ELLIPSE CALIBRATION ===")
        print(f"Target circularity: {min_circularity:.3f}, Max iterations: {max_iterations}")

        h, w = frame.shape[:2]
        current_frame = frame.copy()
        cumulative_matrix = None
        best_matrix = None
        best_circularity = 0.0
        best_aspect_ratio = None  # Track best aspect ratio for filtering
        iteration_results = []
        iteration_debug_frames = []  # Store debug frames from all iterations

        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

            # Detect ellipse in current frame - always enable debug mode
            # For iterations after the first, only accept ellipses better than previous best
            min_aspect_threshold = best_aspect_ratio if iteration > 0 else None
            ellipse_matrix, detected_ellipse, debug_frame = self.detect_ellipse_perspective_transform(
                current_frame, debug_mode=True, min_aspect_ratio=min_aspect_threshold
            )

            # Save debug frame from each iteration
            if debug_frame is not None:
                iteration_debug_frames.append({
                    'iteration': iteration + 1,
                    'debug_frame': debug_frame,
                    'input_frame': current_frame.copy()
                })

            if ellipse_matrix is None:
                print(f"  No ellipse detected")
                if iteration == 0:
                    # First iteration failed
                    return False, "No ellipse detected in initial frame"
                else:
                    # Use best result from previous iterations
                    print(f"  Using best result from iteration {best_iteration + 1}")
                    break

            # Combine with cumulative transformation
            if cumulative_matrix is None:
                cumulative_matrix = ellipse_matrix
            else:
                cumulative_matrix = ellipse_matrix @ cumulative_matrix

            # Apply cumulative transformation to ORIGINAL frame (not current_frame)
            # This ensures we're always working from the original, not compounding transformations
            transformed = cv2.warpPerspective(frame, cumulative_matrix, (w, h))

            # Measure circularity by detecting ellipse in the TRANSFORMED result
            # This shows how circular the result actually is
            if detected_ellipse is not None:
                # The aspect ratio of the detected ellipse directly indicates circularity
                # If ellipse was detected in current_frame, it represents the remaining distortion
                minor_axis = min(detected_ellipse[1])
                major_axis = max(detected_ellipse[1])
                aspect_ratio = major_axis / max(minor_axis, 1.0)

                # Circularity based on detected ellipse aspect ratio
                # 1.0 = perfect circle, lower = more elliptical
                ellipse_circularity = 1.0 / aspect_ratio

                # Also measure circularity in the final transformed result
                # This validates that the transformation actually made it circular
                result_circularity = 0.0
                try:
                    # Detect ellipse in transformed frame to measure actual result
                    test_ellipse, _, _, _, _ = self.experimental_ellipse_detection(
                        transformed,
                        contour_mode=cv2.RETR_LIST,
                        canny_low=50, canny_high=150,
                        min_area=1000,
                        min_circularity=0.3
                    )
                    if test_ellipse is not None:
                        test_minor = min(test_ellipse[1])
                        test_major = max(test_ellipse[1])
                        test_aspect = test_major / max(test_minor, 1.0)
                        result_circularity = 1.0 / test_aspect
                        print(f"  Result ellipse aspect ratio: {test_aspect:.3f}")
                except:
                    result_circularity = ellipse_circularity

                # Use the result circularity as the true measure
                # This is what we actually achieved in the transformed frame
                circularity = result_circularity if result_circularity > 0 else ellipse_circularity

                iteration_results.append({
                    'iteration': iteration + 1,
                    'circularity': circularity,
                    'ellipse_circularity': ellipse_circularity,
                    'result_circularity': result_circularity,
                    'matrix': cumulative_matrix.copy()
                })

                print(f"  Input ellipse aspect ratio: {aspect_ratio:.3f} (circularity: {ellipse_circularity:.4f})")
                print(f"  Result circularity: {circularity:.4f}")

                # Track best result and best aspect ratio
                improved = False
                if circularity > best_circularity:
                    best_circularity = circularity
                    best_matrix = cumulative_matrix.copy()
                    best_iteration = iteration
                    best_aspect_ratio = aspect_ratio  # Update threshold for next iteration
                    improved = True
                    print(f"  ✓ Improvement: {circularity:.4f} > {best_circularity if iteration == 0 else iteration_results[-2]['circularity']:.4f}")
                    print(f"  → Next iteration will only accept ellipses with aspect_ratio < {best_aspect_ratio:.3f}")

                # Check if we've reached target circularity
                if circularity >= min_circularity:
                    print(f"  ✓ Target circularity reached!")
                    break

                # Check if we're making progress
                # If circularity isn't improving or getting worse, stop before next iteration
                if iteration > 0:
                    prev_circularity = iteration_results[-2]['circularity']
                    if circularity <= prev_circularity + 0.001:
                        # Not improving or getting worse
                        if circularity < prev_circularity:
                            print(f"  ⚠ Circularity got worse: {circularity:.4f} < {prev_circularity:.4f}")
                            print(f"  → Using best result from iteration {best_iteration + 1}")
                        else:
                            print(f"  ⚠ Circularity not improving enough, stopping iterations")
                        break

                # Prepare for next iteration - use transformed frame as new input
                current_frame = transformed
            else:
                print(f"  Warning: Could not measure circularity")
                break

        # Use best result
        if best_matrix is not None:
            self.saved_perspective_matrix = best_matrix
            self.calibration_method = 'ellipse_iterative'
            self.saved_ellipse_data = {
                'calibration_timestamp': time.time(),
                'iterations': len(iteration_results),
                'final_circularity': best_circularity,
                'iteration_results': [
                    {'iteration': r['iteration'], 'circularity': r['circularity']}
                    for r in iteration_results
                ]
            }

            # Create enhanced debug frame showing all iterations
            if len(iteration_results) > 0:
                self._create_iterative_debug_frame(frame, iteration_results, best_matrix, iteration_debug_frames)

            # Build message showing which iteration was best
            if best_iteration < len(iteration_results) - 1:
                # Best result was NOT the last iteration
                message = (f"Iterative ellipse calibration successful: "
                          f"{len(iteration_results)} iterations, "
                          f"best from iteration {best_iteration + 1}, "
                          f"circularity {best_circularity:.4f}")
            else:
                # Best result was the last iteration
                message = (f"Iterative ellipse calibration successful: "
                          f"{len(iteration_results)} iterations, "
                          f"circularity {best_circularity:.4f}")

            print(f"\n✓ {message}")
            print(f"✓ Using transformation from iteration {best_iteration + 1}")
            return True, message
        else:
            return False, "Iterative calibration failed to improve circularity"

    def _create_iterative_debug_frame(self, original_frame, iteration_results, final_matrix, iteration_debug_frames=None):
        """Create debug visualization showing all iteration progress

        Args:
            original_frame: Original input frame
            iteration_results: List of iteration results with circularity scores
            final_matrix: Final transformation matrix
            iteration_debug_frames: List of debug frames from each iteration
        """
        h, w = original_frame.shape[:2]

        if iteration_debug_frames and len(iteration_debug_frames) > 0:
            num_iterations = len(iteration_debug_frames)

            # Vertical stacking - one iteration per row
            # Debug frames are typically 2x width (side-by-side: detection + corrected)
            total_rows = num_iterations + 1  # +1 for final result

            # Get dimensions from first debug frame to determine aspect ratio
            first_debug = iteration_debug_frames[0]['debug_frame']
            debug_h, debug_w = first_debug.shape[:2]

            # Use the actual debug frame dimensions
            row_height = debug_h
            row_width = debug_w

            combined_width = row_width
            combined_height = row_height * total_rows
            combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

            # Add each iteration's debug frame as a separate row
            for idx, iter_data in enumerate(iteration_debug_frames):
                y_offset = row_height * idx
                debug_frame = iter_data['debug_frame']

                # Resize if needed to match the row dimensions (keeping aspect ratio)
                if debug_frame.shape[:2] != (row_height, row_width):
                    debug_frame = cv2.resize(debug_frame, (row_width, row_height))

                combined[y_offset:y_offset + row_height, :] = debug_frame

                # Add iteration label
                iter_num = iter_data['iteration']
                result = iteration_results[idx] if idx < len(iteration_results) else None
                if result:
                    circ = result.get('circularity', 0)
                    color = (0, 255, 0) if circ >= 0.95 else (0, 255, 255)
                    label = f"ITERATION {iter_num}: {circ:.4f}"
                else:
                    color = (255, 255, 255)
                    label = f"ITERATION {iter_num}"

                # Scale font based on image size
                font_scale = max(0.7, min(1.5, row_width / 2000))
                cv2.putText(combined, label, (10, y_offset + int(40 * font_scale)),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            # Add final result as last row
            # Create a side-by-side view for consistency (original | final)
            y_offset = row_height * num_iterations

            # Create final row with original on left, final result on right
            final_row = np.zeros((row_height, row_width, 3), dtype=np.uint8)

            # Calculate dimensions for side-by-side layout
            half_width = row_width // 2

            # Resize original and final to fit side-by-side
            original_resized = cv2.resize(original_frame, (half_width, row_height))
            final_result = cv2.warpPerspective(original_frame, final_matrix, (w, h))
            final_resized = cv2.resize(final_result, (half_width, row_height))

            final_row[:, :half_width] = original_resized
            final_row[:, half_width:] = final_resized

            combined[y_offset:y_offset + row_height, :] = final_row

            # Label final result
            final_circ = iteration_results[-1].get('circularity', 0) if iteration_results else 0
            color = (0, 255, 0) if final_circ >= 0.95 else (0, 255, 255)
            font_scale = max(0.7, min(1.5, row_width / 2000))

            # Add labels on both sides
            cv2.putText(combined, "ORIGINAL", (10, y_offset + int(40 * font_scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
            cv2.putText(combined, f"FINAL: {final_circ:.4f}", (half_width + 10, y_offset + int(40 * font_scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            # Add center dividing line for final row
            cv2.line(combined, (half_width, y_offset), (half_width, y_offset + row_height), (255, 255, 255), 2)

            # Add dividing lines between rows
            for idx in range(1, total_rows):
                y_pos = row_height * idx
                cv2.line(combined, (0, y_pos), (combined_width, y_pos), (255, 255, 255), 2)

            self.debug_frame = combined
        else:
            # Fallback: simple side-by-side comparison
            combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
            combined[:, :w] = original_frame
            combined[:, w:] = cv2.warpPerspective(original_frame, final_matrix, (w, h))

            # Add labels and iteration info
            cv2.putText(combined, "ORIGINAL", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(combined, "CORRECTED (ITERATIVE)", (w + 10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            # Show iteration progress
            y_offset = 80
            for result in iteration_results:
                iteration_num = result['iteration']
                circularity = result['circularity']
                color = (0, 255, 0) if circularity >= 0.95 else (0, 255, 255)

                cv2.putText(combined, f"Iter {iteration_num}: {circularity:.4f}",
                           (w + 10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30

            # Add dividing line
            cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)

            self.debug_frame = combined

    def calibrate_perspective(self, frame, method='auto', pattern_size=(9, 6), iterative=True, target_circularity=0.95, max_iterations=3):
        """Perform on-demand perspective calibration

        Args:
            frame: Input frame for calibration
            method: Calibration method - 'auto', 'ellipse', or 'checkerboard'
            pattern_size: For checkerboard method, tuple of (columns, rows) of internal corners
            iterative: Use iterative refinement for ellipse method
            target_circularity: Target circularity for iterative refinement (0.0-1.0)
            max_iterations: Maximum number of iterations for refinement

        Returns:
            success: True if calibration succeeded
            message: Status message
        """
        # Store the ellipse/checkerboard data when calibration succeeds
        # Always enable debug mode during calibration to generate visualization

        if method == 'checkerboard':
            return self.calibrate_perspective_checkerboard(frame, pattern_size)
        elif method == 'ellipse':
            if iterative:
                return self.calibrate_perspective_ellipse_iterative(frame, max_iterations, target_circularity)
            else:
                # Original single-pass ellipse calibration
                ellipse_matrix, detected_ellipse, debug_frame = self.detect_ellipse_perspective_transform(frame, debug_mode=True)
                if ellipse_matrix is not None:
                    self.saved_perspective_matrix = ellipse_matrix
                    self.calibration_method = 'ellipse'
                    self.saved_ellipse_data = {
                        'calibration_timestamp': time.time(),
                    }
                    return True, "Ellipse calibration successful"
                else:
                    return False, "No suitable ellipse found for calibration"
        else:  # method == 'auto'
            # Try iterative ellipse first
            if iterative:
                success, message = self.calibrate_perspective_ellipse_iterative(frame, max_iterations, target_circularity)
                if success:
                    return True, f"{message} (auto)"

            # Fallback to single-pass ellipse
            ellipse_matrix, detected_ellipse, debug_frame = self.detect_ellipse_perspective_transform(frame, debug_mode=True)
            if ellipse_matrix is not None:
                self.saved_perspective_matrix = ellipse_matrix
                self.calibration_method = 'ellipse'
                self.saved_ellipse_data = {
                    'calibration_timestamp': time.time(),
                }
                return True, "Ellipse calibration successful (auto, single-pass)"

            # Final fallback to checkerboard
            return self.calibrate_perspective_checkerboard(frame, pattern_size)

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
        """Get the current perspective matrix"""
        return self.saved_perspective_matrix