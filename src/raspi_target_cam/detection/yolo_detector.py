#!/usr/bin/env python3
"""
YOLO11 Detector using Ultralytics
Uses a trained YOLO11 model (NCNN or other formats) to detect bullet holes
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
from pathlib import Path

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: uv pip install ultralytics")


class YoloDetector:
    """
    YOLO11-based detector using Ultralytics API

    Detects bullet holes using a trained YOLO11 model.
    Supports multiple formats: PyTorch (.pt), ONNX, NCNN, etc.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        target_class: int = 0,
        target_detector=None,
    ):
        """
        Initialize YOLO detector

        Args:
            model_path: Path to YOLO model file (.pt) or directory (for NCNN)
                       If None, uses default path: data/models/train21-640/
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            iou_threshold: IoU threshold for NMS (0.0-1.0)
            target_class: Class ID to detect (0=bullet_hole, 1=target_pistol). Default: 0
            target_detector: Optional TargetDetector instance for target-centered cropping
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics is required for YoloDetector. Install with: uv pip install ultralytics")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_class = target_class
        self.target_detector = target_detector

        # Set default model path if not provided
        if model_path is None:
            # Get project root (assuming this file is in src/raspi_target_cam/detection/)
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent

            # Try .pt model first, then NCNN model
            pt_path = project_root / "data" / "models" / "gen21-640.pt"
            ncnn_path = project_root / "data" / "models" / "n_480px3_1cls_ncnn_model"

            if ncnn_path.exists():
                model_path = ncnn_path
            elif pt_path.exists():
                model_path = pt_path
            else:
                raise FileNotFoundError(f"No model found at {pt_path} or {ncnn_path}")
        else:
            model_path = Path(model_path)

        self.model_path = model_path

        # Load model using ultralytics
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(str(model_path))

        print(f"✅ Loaded YOLO11 model from {model_path}")

    def detect_bullet_holes(self, before_frame: np.ndarray, after_frame: np.ndarray) -> List[Tuple]:
        """
        Detect bullet holes in frames

        This method maintains compatibility with other detectors in the project.
        It runs inference on the after_frame to detect bullet holes.

        Args:
            before_frame: Reference frame (not used by YOLO, kept for API compatibility)
            after_frame: Current frame to detect bullet holes in

        Returns:
            List of holes: [(x, y, radius, confidence, area, circularity), ...]
        """
        return self.detect(after_frame)

    def _mask_outside_target_area(self, frame: np.ndarray, center_x: int, center_y: int, outer_radius: int, target_type: str, debug: bool = False) -> np.ndarray:
        """
        Mask areas outside the target detection region with white.

        This masks everything outside a square centered on the target to reduce
        false positives from background artifacts. Works for both pistol and rifle targets.

        Physical dimensions:
        - Rifle targets: Outer ring 8.7cm, detection square 9.5cm (ratio: 1.092)
        - Pistol targets: Outer ring 20cm, paper width 21.5cm (ratio: 1.075)

        For both types, we use outer_radius * 2.2 to give some padding beyond the outer ring.

        Args:
            frame: Input frame
            center_x, center_y: Target center coordinates
            outer_radius: Radius of outer ring (detected or estimated by target detector)
            target_type: 'rifle' or 'pistol' (for debugging)
            debug: Print debug info

        Returns:
            Masked frame with areas outside target region set to white
        """
        h, w = frame.shape[:2]
        masked_frame = frame.copy()

        # Calculate detection square size with padding beyond outer ring
        # Use 2.2x outer radius to give some buffer (about 10% beyond outer ring diameter)
        detection_square_size = int(outer_radius * 2.2)

        # Make it even for easier centering
        if detection_square_size % 2 != 0:
            detection_square_size += 1

        if debug:
            print(f"      🎯 Masking outside {detection_square_size}x{detection_square_size}px square ({target_type} target, outer_r={outer_radius}px)")

        # Create a mask for the detection area
        mask = np.zeros((h, w), dtype=np.uint8)

        # Calculate square boundaries centered on target
        half_size = detection_square_size // 2
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(w, center_x + half_size)
        y2 = min(h, center_y + half_size)

        # Mark detection area in mask
        mask[y1:y2, x1:x2] = 255

        # Create boolean mask (True = background, False = target area)
        background_mask = mask == 0

        # Fill background with white
        masked_frame[background_mask] = [255, 255, 255]

        return masked_frame

    def detect(self, frame: np.ndarray, debug: bool = True) -> List[Tuple]:
        """
        Detect bullet holes in a single frame

        Args:
            frame: Input BGR frame
            debug: If True, print debug information

        Returns:
            List of holes: [(x, y, radius, confidence, area, circularity), ...]
        """
        if debug:
            print(f"      🔍 Input frame shape: {frame.shape}")

        # Crop frame to 1440x1440 centered on target (or frame center if no target detected)
        h, w = frame.shape[:2]
        crop_size = 1440

        # Try to get target center and type from target detector
        center_x = w // 2
        center_y = h // 2
        target_type = None
        outer_radius = None

        if self.target_detector and self.target_detector.target_center:
            # Use detected target center for cropping
            center_x, center_y = self.target_detector.target_center
            target_type = self.target_detector.target_type

            # Get outer ring (detected or estimated by target detector)
            if self.target_detector.outer_circle is not None:
                _, _, outer_radius = self.target_detector.outer_circle

            if debug:
                print(f"      🎯 Using target center: ({center_x}, {center_y}), type: {target_type}, outer_r: {outer_radius}")
        else:
            if debug:
                print(f"      📐 Using frame center: ({center_x}, {center_y})")

        # Apply masking (before cropping)
        # This masks areas outside the target detection zone to reduce false positives
        # Works for both pistol and rifle targets
        masked_frame = frame
        if target_type is not None and outer_radius is not None:
            masked_frame = self._mask_outside_target_area(frame, center_x, center_y, outer_radius, target_type, debug)

        # Calculate center crop coordinates
        x1_crop = center_x - crop_size // 2
        y1_crop = center_y - crop_size // 2
        x2_crop = x1_crop + crop_size
        y2_crop = y1_crop + crop_size

        # Ensure crop is within frame bounds
        x1_crop = max(0, x1_crop)
        y1_crop = max(0, y1_crop)
        x2_crop = min(w, x2_crop)
        y2_crop = min(h, y2_crop)

        cropped_frame = masked_frame[y1_crop:y2_crop, x1_crop:x2_crop]

        if debug:
            print(f"      🔍 Cropped to: {cropped_frame.shape} (offset: x={x1_crop}, y={y1_crop})")

        # Run inference on cropped frame
        results = self.model.predict(
            source=cropped_frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=debug,
            classes=[self.target_class],  # Filter by target class
        )

        detections = []

        # Process results
        for result in results:
            boxes = result.boxes

            if debug:
                print(f"      🔍 Found {len(boxes)} detections")

            for box in boxes:
                # Get box coordinates (xyxy format) - these are relative to cropped frame
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Calculate center and dimensions in cropped frame
                x_center_crop = (x1 + x2) / 2
                y_center_crop = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                # Translate coordinates back to original frame
                x_center = x_center_crop + x1_crop
                y_center = y_center_crop + y1_crop

                # Calculate radius (average of width/height divided by 2)
                radius = (width + height) / 4

                # Calculate area
                area = width * height

                # Estimate circularity based on aspect ratio
                aspect_ratio = width / height if height > 0 else 1.0
                circularity = min(aspect_ratio, 1.0 / aspect_ratio) if aspect_ratio > 0 else 0.0

                if debug:
                    print(
                        f"         Detection: pos=({x_center:.0f},{y_center:.0f}), "
                        f"size=({width:.0f}x{height:.0f}), conf={conf:.3f}, class={cls}"
                    )

                detections.append((x_center, y_center, radius, conf, area, circularity))

        return detections

    def visualize_detections(
        self, frame: np.ndarray, detections: List[Tuple], color=(0, 255, 0), thickness=2
    ) -> np.ndarray:
        """
        Draw detections on frame

        Args:
            frame: Input frame
            detections: List of detections from detect()
            color: BGR color for drawing
            thickness: Line thickness

        Returns:
            Frame with detections drawn
        """
        result = frame.copy()

        for i, det in enumerate(detections):
            x, y, r, conf, area, circ = det
            x, y, r = int(x), int(y), int(r)

            # Draw circle
            cv2.circle(result, (x, y), r, color, thickness)
            cv2.circle(result, (x, y), 2, color, -1)

            # Draw label
            label = f"#{i + 1} {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
            label_y = y - r - 5

            # Draw label background
            cv2.rectangle(
                result,
                (x - label_size[0] // 2 - 2, label_y - label_size[1] - 2),
                (x + label_size[0] // 2 + 2, label_y + 2),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                result, label, (x - label_size[0] // 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness
            )

        return result


def test_yolo_detector():
    """Test YOLO detector on sample frames"""
    print("🎯 Testing YOLO11 Detector")
    print("=" * 60)

    # Check if test frames exist
    before_path = "data/test_frames/frame_0000_clean_target_corrected.jpg"
    after_path = "data/test_frames/frame_0930_all_10_shots_corrected.jpg"

    if not os.path.exists(before_path) or not os.path.exists(after_path):
        print(f"⚠️  Test frames not found. Please ensure these exist:")
        print(f"   - {before_path}")
        print(f"   - {after_path}")
        return

    # Load frames
    before = cv2.imread(before_path)
    after = cv2.imread(after_path)

    print(f"✅ Loaded test frames: {after.shape}")

    # Create detector
    try:
        # Use lower confidence threshold for testing
        detector = YoloDetector(conf_threshold=0.10, iou_threshold=0.45, target_class=0)
    except Exception as e:
        print(f"❌ Failed to create detector: {e}")
        import traceback

        traceback.print_exc()
        return

    # Detect
    print(f"\n🔍 Running YOLO inference...")
    detections = detector.detect(after, debug=True)

    print(f"✅ Found {len(detections)} bullet holes")

    # Print details
    if detections:
        print(f"\n📊 Detection Details:")
        for i, det in enumerate(detections):
            x, y, r, conf, area, circ = det
            print(
                f"   #{i + 1}: pos=({x:.0f},{y:.0f}), radius={r:.0f}px, "
                f"conf={conf:.3f}, area={area:.0f}px, circ={circ:.2f}"
            )

    # Visualize
    result = detector.visualize_detections(after, detections)

    # Add info overlay
    info_height = 80
    info = np.zeros((info_height, result.shape[1], 3), dtype=np.uint8)

    y_off = 25
    cv2.putText(
        info,
        f"YOLO11 Detector - {len(detections)} detections",
        (20, y_off),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    y_off += 30
    cv2.putText(
        info,
        f"Confidence threshold: {detector.conf_threshold:.2f} | "
        f"IoU threshold: {detector.iou_threshold:.2f} | "
        f"Target class: {detector.target_class} (bullet_hole)",
        (20, y_off),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    final = np.vstack([info, result])

    # Save
    output_dir = "test_outputs/yolo_detection"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/yolo_detections.jpg"
    cv2.imwrite(output_path, final)

    print(f"\n💾 Saved visualization to: {output_path}")
    print(f"✅ Test complete!")

    return detections


if __name__ == "__main__":
    test_yolo_detector()
