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
    ):
        """
        Initialize YOLO detector

        Args:
            model_path: Path to YOLO model file (.pt) or directory (for NCNN)
                       If None, uses default path: data/models/train21-640/
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            iou_threshold: IoU threshold for NMS (0.0-1.0)
            target_class: Class ID to detect (0=bullet_hole, 1=target_pistol). Default: 0
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics is required for YoloDetector. Install with: uv pip install ultralytics")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_class = target_class

        # Set default model path if not provided
        if model_path is None:
            # Get project root (assuming this file is in src/raspi_target_cam/detection/)
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent

            # Try .pt model first, then NCNN model
            pt_path = project_root / "data" / "models" / "gen21-640.pt"
            ncnn_path = project_root / "data" / "models" / "train21-640" / "model.ncnn.param"

            if pt_path.exists():
                model_path = pt_path
            elif ncnn_path.exists():
                model_path = ncnn_path
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

        # Crop frame to 1944x1944 centered
        h, w = frame.shape[:2]
        crop_size = 1944

        # Calculate center crop coordinates
        center_x = w // 2
        center_y = h // 2
        x1_crop = center_x - crop_size // 2
        y1_crop = center_y - crop_size // 2
        x2_crop = x1_crop + crop_size
        y2_crop = y1_crop + crop_size

        # Ensure crop is within frame bounds
        x1_crop = max(0, x1_crop)
        y1_crop = max(0, y1_crop)
        x2_crop = min(w, x2_crop)
        y2_crop = min(h, y2_crop)

        cropped_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop]

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
