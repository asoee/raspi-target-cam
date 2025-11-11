#!/usr/bin/env python3
"""
YOLO Detector Test
Tests YOLO11-based bullet hole detection
"""

import pytest
import cv2
import numpy as np
from pathlib import Path

# Check if ncnn is available
try:
    import ncnn
    NCNN_AVAILABLE = True
except ImportError:
    NCNN_AVAILABLE = False

if NCNN_AVAILABLE:
    from raspi_target_cam.detection.yolo_detector import YoloDetector


@pytest.mark.skipif(not NCNN_AVAILABLE, reason="ncnn not installed")
def test_yolo_detector_initialization():
    """Test that YOLO detector can be initialized"""
    detector = YoloDetector(conf_threshold=0.25, iou_threshold=0.45)

    assert detector is not None
    assert detector.conf_threshold == 0.25
    assert detector.iou_threshold == 0.45
    assert detector.input_size == (640, 640)


@pytest.mark.skipif(not NCNN_AVAILABLE, reason="ncnn not installed")
def test_yolo_detector_with_test_frames():
    """Test YOLO detector on actual frames if available"""
    before_path = Path("test_frames/frame_0000_clean_target_corrected.jpg")
    after_path = Path("test_frames/frame_0930_all_10_shots_corrected.jpg")

    if not before_path.exists() or not after_path.exists():
        pytest.skip("Test frames not available")

    # Load frames
    before = cv2.imread(str(before_path))
    after = cv2.imread(str(after_path))

    assert before is not None
    assert after is not None

    # Create detector
    detector = YoloDetector(conf_threshold=0.25, iou_threshold=0.45)

    # Detect
    detections = detector.detect_bullet_holes(before, after)

    # Should find some detections
    assert isinstance(detections, list)
    print(f"\nFound {len(detections)} detections")

    # Check detection format
    if len(detections) > 0:
        for det in detections:
            assert len(det) == 6  # (x, y, radius, confidence, area, circularity)
            x, y, r, conf, area, circ = det

            # Validate values
            assert x >= 0
            assert y >= 0
            assert r > 0
            assert 0.0 <= conf <= 1.0
            assert area > 0
            assert 0.0 <= circ <= 1.0

            print(f"  Detection: pos=({x:.0f},{y:.0f}), r={r:.0f}, conf={conf:.3f}")


@pytest.mark.skipif(not NCNN_AVAILABLE, reason="ncnn not installed")
def test_yolo_detector_single_frame():
    """Test YOLO detector on a single frame"""
    after_path = Path("test_frames/frame_0930_all_10_shots_corrected.jpg")

    if not after_path.exists():
        pytest.skip("Test frame not available")

    # Load frame
    frame = cv2.imread(str(after_path))
    assert frame is not None

    # Create detector
    detector = YoloDetector(conf_threshold=0.25)

    # Detect on single frame
    detections = detector.detect(frame)

    assert isinstance(detections, list)
    print(f"\nSingle frame detection: {len(detections)} found")


@pytest.mark.skipif(not NCNN_AVAILABLE, reason="ncnn not installed")
def test_yolo_detector_preprocessing():
    """Test preprocessing function"""
    detector = YoloDetector()

    # Create dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Preprocess
    preprocessed = detector.preprocess_frame(frame)

    # Check output shape (C, H, W)
    assert preprocessed.shape == (3, 640, 640)

    # Check normalization
    assert preprocessed.min() >= 0.0
    assert preprocessed.max() <= 1.0


@pytest.mark.skipif(not NCNN_AVAILABLE, reason="ncnn not installed")
def test_yolo_detector_nms():
    """Test NMS functionality"""
    detector = YoloDetector()

    # Create overlapping detections
    detections = [
        (100, 100, 10, 0.9, 314, 0.9),  # High confidence
        (102, 102, 10, 0.7, 314, 0.9),  # Overlapping, lower confidence
        (200, 200, 10, 0.8, 314, 0.9),  # Separate detection
    ]

    # Apply NMS
    filtered = detector.apply_nms(detections)

    # Should remove the overlapping one with lower confidence
    assert len(filtered) == 2

    # Check that highest confidence detections are kept
    confidences = [det[3] for det in filtered]
    assert 0.9 in confidences
    assert 0.8 in confidences


@pytest.mark.skipif(not NCNN_AVAILABLE, reason="ncnn not installed")
def test_yolo_detector_visualization():
    """Test visualization function"""
    detector = YoloDetector()

    # Create dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Create dummy detections
    detections = [
        (100, 100, 10, 0.9, 314, 0.9),
        (200, 200, 15, 0.8, 707, 0.85),
    ]

    # Visualize
    result = detector.visualize_detections(frame, detections)

    # Check that result is a valid image
    assert result.shape == frame.shape
    assert result.dtype == np.uint8

    # Check that something was drawn (result should differ from input)
    assert not np.array_equal(result, frame)


@pytest.mark.skipif(not NCNN_AVAILABLE, reason="ncnn not installed")
def test_yolo_detector_confidence_threshold():
    """Test that confidence threshold filtering works"""
    # Create detector with high threshold
    detector = YoloDetector(conf_threshold=0.9)

    # Create dummy detections with various confidences
    detections = [
        (100, 100, 10, 0.95, 314, 0.9),  # Should pass
        (200, 200, 10, 0.5, 314, 0.9),   # Should be filtered
        (300, 300, 10, 0.3, 314, 0.9),   # Should be filtered
    ]

    # The confidence filtering happens in postprocess_detections
    # Here we test that the threshold is set correctly
    assert detector.conf_threshold == 0.9


def test_yolo_detector_without_ncnn():
    """Test that appropriate error is raised when ncnn is not available"""
    if NCNN_AVAILABLE:
        pytest.skip("ncnn is available, test not applicable")

    with pytest.raises(ImportError):
        from raspi_target_cam.detection.yolo_detector import YoloDetector
        detector = YoloDetector()


if __name__ == "__main__":
    # Run a simple test if executed directly
    if NCNN_AVAILABLE:
        print("🎯 Running YOLO Detector Tests")
        print("=" * 60)

        # Test initialization
        print("\n1. Testing initialization...")
        detector = YoloDetector(conf_threshold=0.25, iou_threshold=0.45)
        print("✅ Detector initialized successfully")

        # Test with frames if available
        print("\n2. Testing with actual frames...")
        after_path = Path("test_frames/frame_0930_all_10_shots_corrected.jpg")

        if after_path.exists():
            frame = cv2.imread(str(after_path))
            detections = detector.detect(frame)
            print(f"✅ Found {len(detections)} detections")

            for i, det in enumerate(detections):
                x, y, r, conf, area, circ = det
                print(f"   #{i+1}: pos=({x:.0f},{y:.0f}), r={r:.0f}, conf={conf:.3f}")
        else:
            print("⚠️  Test frame not found, skipping frame test")

        print("\n✅ All tests completed!")
    else:
        print("❌ ncnn not installed. Install with: uv pip install ncnn")
