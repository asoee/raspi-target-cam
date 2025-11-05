#!/usr/bin/env python3
"""
Enhanced Bullet Hole Detection System
Integrates bullet hole detection with scoring system and auto-detection
"""

import cv2
import numpy as np
import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from bullet_hole_detection import BulletHoleDetector
from target_scoring import TargetScoringSystem
from target_detection import TargetDetector
import perspective


class EnhancedBulletDetector:
    """
    Enhanced bullet hole detector with:
    - Auto-detection mode (monitors frame changes)
    - Session management
    - Scoring integration
    - Perspective correction support
    """

    def __init__(self, scoring_system: TargetScoringSystem = None,
                 perspective_calibration_file: str = "perspective_calibration.yaml"):
        # Core detectors
        self.bullet_detector = BulletHoleDetector()
        self.target_detector = TargetDetector()
        self.scoring_system = scoring_system or TargetScoringSystem()

        # Auto-detection settings
        self.auto_detect_enabled = False
        self.frame_diff_threshold = 0.005  # 0.5% frame change triggers detection
        self.min_frames_between_shots = 15  # Minimum frames between detections (0.5s at 30fps)

        # Reference frame and tracking
        self.reference_frame = None
        self.last_frame = None
        self.frames_since_last_detection = 0

        # Perspective correction
        self.perspective_enabled = True
        self.perspective_matrix = None
        self.inverse_perspective_matrix = None
        self.perspective_calibration_file = perspective_calibration_file

        # Load perspective calibration if available
        self.load_perspective_calibration()

        # Target information
        self.target_center = None
        self.outer_ring_radius = None
        self.inner_black_radius = None

        # Detection parameters (lowered for better sensitivity)
        self.bullet_detector.min_hole_area = 50
        self.bullet_detector.max_hole_area = 2000
        self.bullet_detector.circularity_threshold = 0.2  # Lowered to accept irregular holes
        self.bullet_detector.difference_threshold = 15  # More sensitive

        # Debug and visualization
        self.debug_mode = True
        self.last_detection_result = None

    def load_perspective_calibration(self):
        """Load perspective calibration from YAML file"""
        import yaml

        if not os.path.exists(self.perspective_calibration_file):
            print(f"⚠️  No perspective calibration file found: {self.perspective_calibration_file}")
            self.perspective_enabled = False
            return False

        try:
            with open(self.perspective_calibration_file, 'r') as f:
                calibration = yaml.safe_load(f)

            matrix_data = calibration.get('perspective_matrix')
            if matrix_data:
                self.perspective_matrix = np.array(matrix_data, dtype=np.float32)
                self.inverse_perspective_matrix = np.linalg.inv(self.perspective_matrix)
                print(f"✅ Loaded perspective calibration from {self.perspective_calibration_file}")
                return True
            else:
                print(f"⚠️  No perspective matrix in calibration file")
                self.perspective_enabled = False
                return False

        except Exception as e:
            print(f"⚠️  Error loading perspective calibration: {e}")
            self.perspective_enabled = False
            return False

    def set_perspective_correction(self, matrix, inverse_matrix=None):
        """Set perspective correction matrices"""
        self.perspective_matrix = matrix
        self.inverse_perspective_matrix = inverse_matrix

    def calibrate_from_frame(self, frame, profile_name: str = 'large'):
        """
        Calibrate target detection and scoring from a clean target frame

        Args:
            frame: Clean target image (before perspective correction if applicable)
            profile_name: Target profile to calibrate
        """
        # Apply perspective correction if enabled
        if self.perspective_enabled and self.perspective_matrix is not None:
            corrected_frame = cv2.warpPerspective(
                frame,
                self.perspective_matrix,
                (frame.shape[1], frame.shape[0])
            )
        else:
            corrected_frame = frame

        # Detect target circles
        inner_circle = self.target_detector.detect_black_circle_improved(corrected_frame)
        outer_circle = self.target_detector.detect_outer_circle(corrected_frame)

        if inner_circle:
            self.target_center = (int(inner_circle[0]), int(inner_circle[1]))
            self.inner_black_radius = int(inner_circle[2])
        else:
            # Fallback to frame center
            h, w = corrected_frame.shape[:2]
            self.target_center = (w // 2, h // 2)
            self.inner_black_radius = 300  # Default guess

        if outer_circle:
            self.outer_ring_radius = int(outer_circle[2])
        else:
            # Estimate as 2x inner radius
            self.outer_ring_radius = self.inner_black_radius * 2

        # Calibrate scoring system
        self.scoring_system.calibrate_profile(
            profile_name,
            self.outer_ring_radius,
            self.inner_black_radius
        )

        print(f"✅ Calibrated target:")
        print(f"   Center: {self.target_center}")
        print(f"   Inner black radius: {self.inner_black_radius}px")
        print(f"   Outer ring radius: {self.outer_ring_radius}px")

        return True

    def start_session(self, reference_frame, profile_name: str = 'large',
                     auto_calibrate: bool = True):
        """
        Start a new scoring session

        Args:
            reference_frame: Clean target frame (raw, before perspective correction)
            profile_name: Target profile to use
            auto_calibrate: Automatically calibrate from reference frame
        """
        # Store reference frame
        self.reference_frame = reference_frame.copy()
        self.last_frame = reference_frame.copy()

        # Auto-calibrate if requested
        if auto_calibrate:
            self.calibrate_from_frame(reference_frame, profile_name)

        # Start scoring session
        session = self.scoring_system.start_session(
            profile_name,
            self.target_center,
            reference_frame
        )

        print(f"🎯 Started scoring session: {session.session_id}")
        print(f"   Profile: {profile_name}")
        print(f"   Auto-detect: {self.auto_detect_enabled}")

        return session

    def stop_session(self) -> str:
        """Stop current session and save"""
        session_file = self.scoring_system.end_session()
        self.reference_frame = None
        self.auto_detect_enabled = False

        print(f"💾 Session saved to: {session_file}")
        return session_file

    def enable_auto_detection(self, enabled: bool = True):
        """Enable or disable auto-detection mode"""
        self.auto_detect_enabled = enabled
        print(f"🤖 Auto-detection: {'ENABLED' if enabled else 'DISABLED'}")

    def calculate_frame_difference(self, frame1, frame2) -> float:
        """
        Calculate percentage of pixels that changed between frames

        Returns:
            Float between 0 and 1 representing fraction of pixels changed
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        changed_pixels = np.count_nonzero(thresh)
        total_pixels = thresh.shape[0] * thresh.shape[1]

        return changed_pixels / total_pixels

    def process_frame(self, current_frame) -> Tuple[Optional[List[Dict]], np.ndarray]:
        """
        Process a frame with auto-detection

        Args:
            current_frame: Current frame (raw, before perspective correction)

        Returns:
            (new_shots, overlay_frame) where new_shots is list of shot data or None
        """
        if self.reference_frame is None:
            return None, current_frame

        self.frames_since_last_detection += 1
        new_shots = None

        # Check if we should trigger detection (auto-detect mode)
        should_detect = False

        if self.auto_detect_enabled and self.last_frame is not None:
            # Calculate frame difference
            if self.frames_since_last_detection >= self.min_frames_between_shots:
                diff_ratio = self.calculate_frame_difference(self.last_frame, current_frame)

                if diff_ratio > self.frame_diff_threshold:
                    should_detect = True
                    print(f"🔍 Auto-detect triggered: {diff_ratio:.3%} frame change")

        # Manual detection can also trigger this
        if should_detect:
            new_shots = self.detect_and_score(current_frame)

        # Update last frame
        self.last_frame = current_frame.copy()

        # Draw overlay
        overlay_frame = self.draw_overlay(current_frame)

        return new_shots, overlay_frame

    def detect_and_score(self, current_frame) -> List[Dict]:
        """
        Manually trigger detection and scoring on current frame

        Args:
            current_frame: Current frame (raw)

        Returns:
            List of new shot data
        """
        if self.reference_frame is None:
            print("⚠️  No reference frame set. Start a session first.")
            return []

        # Apply perspective correction if enabled
        if self.perspective_enabled and self.perspective_matrix is not None:
            ref_corrected = cv2.warpPerspective(
                self.reference_frame,
                self.perspective_matrix,
                (self.reference_frame.shape[1], self.reference_frame.shape[0])
            )
            current_corrected = cv2.warpPerspective(
                current_frame,
                self.perspective_matrix,
                (current_frame.shape[1], current_frame.shape[0])
            )
        else:
            ref_corrected = self.reference_frame
            current_corrected = current_frame

        # Detect bullet holes (returns coordinates in corrected space)
        holes = self.bullet_detector.detect_bullet_holes(ref_corrected, current_corrected)

        # Score each hole (coordinates are already in corrected space, which is what we want)
        new_shots = []
        for hole_data in holes:
            x, y, radius, confidence = hole_data[:4]

            # Coordinates are in corrected space, which matches our target center
            # No transformation needed - add directly to scoring system

            # Add to scoring system
            shot_data = self.scoring_system.add_shot_to_current_session(
                x, y, radius, confidence
            )

            if shot_data:
                new_shots.append(shot_data)
                print(f"   🎯 Shot #{shot_data['shot_number']}: "
                      f"Score {shot_data['score']} "
                      f"({shot_data['distance_from_center']:.1f}px from center)")

        self.last_detection_result = holes
        self.frames_since_last_detection = 0

        print(f"✅ Detected {len(holes)} holes, scored {len(new_shots)} new shots")

        return new_shots

    def draw_overlay(self, frame) -> np.ndarray:
        """Draw scoring overlay on frame"""
        # Apply perspective correction if enabled
        if self.perspective_enabled and self.perspective_matrix is not None:
            corrected_frame = cv2.warpPerspective(
                frame,
                self.perspective_matrix,
                (frame.shape[1], frame.shape[0])
            )
        else:
            corrected_frame = frame

        # Draw scoring overlay (on corrected frame)
        overlay = self.scoring_system.draw_scoring_overlay(
            corrected_frame,
            show_rings=True,
            show_shots=True
        )

        # Add auto-detect indicator
        if self.auto_detect_enabled:
            cv2.putText(overlay, "AUTO-DETECT ON", (corrected_frame.shape[1] - 250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return overlay

    def reset_reference(self, new_reference_frame):
        """Reset reference frame (for target change)"""
        self.reference_frame = new_reference_frame.copy()
        self.last_frame = new_reference_frame.copy()
        print("🔄 Reference frame reset")

    def get_session_status(self) -> Optional[Dict]:
        """Get current session status"""
        return self.scoring_system.get_session_status()


def test_enhanced_detector():
    """Test enhanced detector with video file"""
    import cv2

    print("🎯 Testing Enhanced Bullet Detector")
    print("=" * 50)

    # Create detector
    detector = EnhancedBulletDetector()

    # Load test video
    video_path = "samples/10-shot-1.mkv"
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    # Read first frame as reference
    ret, reference_frame = cap.read()
    if not ret:
        print("❌ Could not read video")
        return

    print(f"📹 Loaded video: {video_path}")
    print(f"   Resolution: {reference_frame.shape[1]}x{reference_frame.shape[0]}")

    # Start session
    detector.start_session(reference_frame, profile_name='large')

    # Enable auto-detection
    detector.enable_auto_detection(True)

    # Process frames
    frame_count = 0
    total_shots = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process frame with auto-detection
        new_shots, overlay = detector.process_frame(frame)

        if new_shots:
            total_shots += len(new_shots)
            print(f"   Frame {frame_count}: {len(new_shots)} new shot(s)")

        # Show every 30th frame to avoid spam
        if frame_count % 30 == 0:
            print(f"   Processed {frame_count} frames...")

    cap.release()

    # End session
    print(f"\n📊 Session complete:")
    status = detector.get_session_status()
    print(f"   Total frames processed: {frame_count}")
    print(f"   Total shots detected: {status['shot_count']}")
    print(f"   Total score: {status['total_score']}")

    detector.stop_session()

    print("\n✅ Test complete!")


if __name__ == "__main__":
    test_enhanced_detector()
