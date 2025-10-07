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
                rejection_reasons.append(f"Contour {analyzed_contours+1}: <5 points ({len(contour)})")
                continue

            area = cv2.contourArea(contour)
            if area < 1000:  # Skip small contours
                continue

            analyzed_contours += 1

            try:
                # Fit ellipse to contour
                ellipse = cv2.fitEllipse(contour)
                (center_x, center_y), (minor_axis, major_axis), angle = ellipse

                # Calculate aspect ratio (how "stretched" the ellipse is)
                aspect_ratio = max(major_axis, minor_axis) / min(major_axis, minor_axis)

                # Use same constraints as original detector
                if aspect_ratio < 1.02:
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
        Create the correct ellipse-to-circle transformation (compress major axis with inverse rotation)
        """
        (center_x, center_y), (minor_axis, major_axis), angle = ellipse
        output_width, output_height = output_size

        print(f"Creating ellipse-to-circle transform:")
        print(f"  Ellipse center: ({center_x:.1f}, {center_y:.1f})")
        print(f"  Ellipse axes: minor={minor_axis:.1f}, major={major_axis:.1f}")
        print(f"  Ellipse angle: {angle:.1f}°")

        # Convert angle to radians
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        image_center_x = output_width / 2
        image_center_y = output_height / 2

        # Translation matrices
        T_to_origin = np.array([
            [1, 0, -image_center_x],
            [0, 1, -image_center_y],
            [0, 0, 1]
        ], dtype=np.float32)

        T_from_origin = np.array([
            [1, 0, image_center_x],
            [0, 1, image_center_y],
            [0, 0, 1]
        ], dtype=np.float32)

        # Rotation matrices (inverse rotation)
        R_align_inv = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        R_back_inv = np.array([
            [cos_a, sin_a, 0],
            [-sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Compression and scaling
        compression_factor = minor_axis / major_axis
        scale_to_fill = 1.0 / compression_factor  # Scale up to fill frame

        # Combined scaling matrix
        S_combined = np.array([
            [scale_to_fill * compression_factor, 0, 0],  # This equals 1.0
            [0, scale_to_fill, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # Final transformation
        transform = T_from_origin @ R_back_inv @ S_combined @ R_align_inv @ T_to_origin

        print(f"  Compression factor: {compression_factor:.3f}")
        print(f"  Scale to fill factor: {scale_to_fill:.3f}")

        return transform

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
            ellipse: OpenCV ellipse tuple ((center_x, center_y), (minor_axis, major_axis), angle)
            transform_matrix: 3x3 transformation matrix

        Returns:
            circle_radius: radius of the resulting circle in pixels
        """
        (center_x, center_y), (minor_axis, major_axis), angle = ellipse

        # Convert angle to radians
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Calculate points on the ellipse axes from the center
        # Major axis endpoints
        major_half = major_axis / 2
        major_end1 = np.array([center_x + major_half * cos_a, center_y + major_half * sin_a, 1.0])
        major_end2 = np.array([center_x - major_half * cos_a, center_y - major_half * sin_a, 1.0])

        # Minor axis endpoints
        minor_half = minor_axis / 2
        minor_end1 = np.array([center_x - minor_half * sin_a, center_y + minor_half * cos_a, 1.0])
        minor_end2 = np.array([center_x + minor_half * sin_a, center_y - minor_half * cos_a, 1.0])

        # Transform the center and axis endpoints
        center_transformed = transform_matrix @ np.array([center_x, center_y, 1.0])
        major_end1_transformed = transform_matrix @ major_end1
        major_end2_transformed = transform_matrix @ major_end2
        minor_end1_transformed = transform_matrix @ minor_end1
        minor_end2_transformed = transform_matrix @ minor_end2

        # Convert from homogeneous coordinates
        def to_cartesian(point):
            return np.array([point[0] / point[2], point[1] / point[2]])

        center_cart = to_cartesian(center_transformed)
        major_end1_cart = to_cartesian(major_end1_transformed)
        major_end2_cart = to_cartesian(major_end2_transformed)
        minor_end1_cart = to_cartesian(minor_end1_transformed)
        minor_end2_cart = to_cartesian(minor_end2_transformed)

        # Calculate distances from center to each axis endpoint
        major_radius1 = np.linalg.norm(major_end1_cart - center_cart)
        major_radius2 = np.linalg.norm(major_end2_cart - center_cart)
        minor_radius1 = np.linalg.norm(minor_end1_cart - center_cart)
        minor_radius2 = np.linalg.norm(minor_end2_cart - center_cart)

        # Average the radii (should be very similar if transformation worked correctly)
        avg_major_radius = (major_radius1 + major_radius2) / 2
        avg_minor_radius = (minor_radius1 + minor_radius2) / 2
        overall_avg_radius = (avg_major_radius + avg_minor_radius) / 2

        print(f"  Transformed axis radii:")
        print(f"    Major axis: {avg_major_radius:.1f} pixels")
        print(f"    Minor axis: {avg_minor_radius:.1f} pixels")
        print(f"    Difference: {abs(avg_major_radius - avg_minor_radius):.1f} pixels")

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
                # Apply the improved transformation with visualization
                transformed_image, _, circle_radius = self.apply_ellipse_to_circle_transform(
                    original_frame, detected_ellipse, (w, h))

                if transformed_image is not None:
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

    def detect_ellipse_perspective_transform(self, frame, debug_mode=None):
        """
        Detect perspective transformation based on outer circle appearing as ellipse
        Uses the improved detection from perspective.py module

        Args:
            frame: Input frame to analyze
            debug_mode: Whether to create debug visualizations

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

            # Load ellipse data
            if 'ellipse_data' in calibration_data:
                self.saved_ellipse_data = calibration_data['ellipse_data']

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
        # Always enable debug mode during calibration to generate visualization
        ellipse_matrix, detected_ellipse, debug_frame = self.detect_ellipse_perspective_transform(frame, debug_mode=True)
        if ellipse_matrix is not None:
            self.saved_perspective_matrix = ellipse_matrix
            # Store ellipse data from the debug info if available
            self.saved_ellipse_data = {
                'detection_method': 'ellipse',
                'calibration_timestamp': time.time(),
            }
            return True, "Perspective calibration successful"
        else:
            return False, "No suitable ellipse found for calibration"

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