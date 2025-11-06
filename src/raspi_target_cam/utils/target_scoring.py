#!/usr/bin/env python3
"""
Target Scoring System
Manages scoring sessions, ring definitions, and score calculations for bullet hole detection
"""

import cv2
import numpy as np
import yaml
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class TargetProfile:
    """Defines a target profile with ring specifications"""

    def __init__(self, name: str, profile_data: dict = None):
        self.name = name

        if profile_data:
            self.outer_radius = profile_data.get('outer_radius', 0)
            self.inner_black_radius = profile_data.get('inner_black_radius', 0)
            self.ring_radii = profile_data.get('ring_radii', {})
        else:
            # Default values (will be calibrated)
            self.outer_radius = 0  # Ring 3 outer boundary
            self.inner_black_radius = 0  # Ring 7 boundary (black circle edge)
            self.ring_radii = {}  # Ring number -> radius in pixels

    def calibrate_from_detection(self, outer_radius: float, inner_black_radius: float):
        """
        Calibrate ring sizes based on detected outer ring and inner black circle

        Standard bullseye target proportions:
        - Rings 3-6: Light colored outer rings
        - Rings 7-10: Black inner circle
        """
        self.outer_radius = outer_radius
        self.inner_black_radius = inner_black_radius

        # Calculate ring radii proportionally
        # Outer rings (3-6): divide the space between inner black and outer edge
        outer_ring_range = outer_radius - inner_black_radius
        ring_step = outer_ring_range / 4  # 4 rings in outer area

        self.ring_radii = {
            10: 0,  # Center point (10.9 points for exact center)
            9: inner_black_radius * 0.33,   # Inner third of black circle
            8: inner_black_radius * 0.67,   # Middle third of black circle
            7: inner_black_radius,          # Edge of black circle
            6: inner_black_radius + ring_step * 1,
            5: inner_black_radius + ring_step * 2,
            4: inner_black_radius + ring_step * 3,
            3: inner_black_radius + ring_step * 4,  # Outermost ring
        }

    def get_score_for_distance(self, distance: float) -> int:
        """
        Calculate integer score based on distance from center

        Standard bullseye scoring:
        - Center (10-ring): 10 points
        - 9-ring: 9 points
        - ... down to ...
        - 3-ring: 3 points
        - Outside all rings: 0 points
        """
        # Check which ring the distance falls into (from innermost to outermost)
        for ring in [10, 9, 8, 7, 6, 5, 4, 3]:
            if distance <= self.ring_radii[ring]:
                return ring

        # Outside all rings
        return 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'outer_radius': float(self.outer_radius),
            'inner_black_radius': float(self.inner_black_radius),
            'ring_radii': {k: float(v) for k, v in self.ring_radii.items()}
        }

    @staticmethod
    def from_dict(data: dict) -> 'TargetProfile':
        """Create from dictionary"""
        return TargetProfile(data['name'], data)


class ScoringSession:
    """Manages a scoring session with shot tracking and scoring"""

    def __init__(self, session_id: str, target_profile: TargetProfile,
                 target_center: Tuple[int, int], reference_frame=None):
        self.session_id = session_id
        self.target_profile = target_profile
        self.target_center = target_center  # (x, y) in pixels
        self.reference_frame = reference_frame

        self.shots = []  # List of shot data
        self.total_score = 0
        self.start_time = datetime.now()
        self.end_time = None
        self.active = True

        # Auto-detection tracking
        self.detected_holes = []  # List of (x, y) to avoid duplicate scoring
        self.duplicate_threshold = 30  # pixels - holes closer than this are considered duplicates

    def add_shot(self, hole_x: int, hole_y: int, hole_radius: int,
                 detection_confidence: float) -> Optional[Dict]:
        """
        Add a new shot and calculate its score

        Returns:
            Shot data dict if added, None if duplicate
        """
        # Check for duplicates
        for prev_x, prev_y in self.detected_holes:
            distance = np.sqrt((hole_x - prev_x)**2 + (hole_y - prev_y)**2)
            if distance < self.duplicate_threshold:
                return None  # Duplicate shot, ignore

        # Calculate distance from target center
        dx = hole_x - self.target_center[0]
        dy = hole_y - self.target_center[1]
        distance_from_center = np.sqrt(dx**2 + dy**2)

        # Get score for this distance
        score = self.target_profile.get_score_for_distance(distance_from_center)

        # Create shot data
        shot_data = {
            'shot_number': len(self.shots) + 1,
            'timestamp': datetime.now().isoformat(),
            'x': int(hole_x),
            'y': int(hole_y),
            'radius': int(hole_radius),
            'distance_from_center': float(distance_from_center),
            'score': int(score),
            'detection_confidence': float(detection_confidence)
        }

        self.shots.append(shot_data)
        self.total_score += score
        self.detected_holes.append((hole_x, hole_y))

        return shot_data

    def end_session(self):
        """Mark session as ended"""
        self.active = False
        self.end_time = datetime.now()

    def get_summary(self) -> Dict:
        """Get session summary"""
        return {
            'session_id': self.session_id,
            'active': self.active,
            'target_profile': self.target_profile.name,
            'shot_count': len(self.shots),
            'total_score': self.total_score,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': (
                (self.end_time or datetime.now()) - self.start_time
            ).total_seconds()
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'target_profile': self.target_profile.to_dict(),
            'target_center': list(self.target_center),
            'shots': self.shots,
            'total_score': self.total_score,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'active': self.active
        }

    @staticmethod
    def from_dict(data: Dict) -> 'ScoringSession':
        """Create from dictionary"""
        profile = TargetProfile.from_dict(data['target_profile'])
        session = ScoringSession(
            data['session_id'],
            profile,
            tuple(data['target_center']),
            None  # Reference frame not saved
        )
        session.shots = data['shots']
        session.total_score = data['total_score']
        session.start_time = datetime.fromisoformat(data['start_time'])
        if data['end_time']:
            session.end_time = datetime.fromisoformat(data['end_time'])
        session.active = data['active']

        # Rebuild detected holes list
        session.detected_holes = [(shot['x'], shot['y']) for shot in session.shots]

        return session


class TargetScoringSystem:
    """Main scoring system that manages profiles and sessions"""

    def __init__(self, config_file: str = "scoring_config.yaml"):
        self.config_file = config_file
        self.profiles = {}
        self.current_session = None
        self.sessions_dir = "data/scoring_sessions"

        # Create sessions directory
        os.makedirs(self.sessions_dir, exist_ok=True)

        # Load configuration
        self.load_config()

    def load_config(self):
        """Load target profiles from config file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Load profiles
            for profile_data in config.get('profiles', []):
                profile = TargetProfile.from_dict(profile_data)
                self.profiles[profile.name] = profile
        else:
            # Create default profiles
            self.profiles['large'] = TargetProfile('large')
            self.profiles['small'] = TargetProfile('small')
            self.save_config()

    def save_config(self):
        """Save target profiles to config file"""
        config = {
            'profiles': [profile.to_dict() for profile in self.profiles.values()]
        }

        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def calibrate_profile(self, profile_name: str, outer_radius: float,
                         inner_black_radius: float):
        """Calibrate a target profile with detected ring sizes"""
        if profile_name not in self.profiles:
            self.profiles[profile_name] = TargetProfile(profile_name)

        self.profiles[profile_name].calibrate_from_detection(
            outer_radius, inner_black_radius
        )
        self.save_config()

    def start_session(self, profile_name: str, target_center: Tuple[int, int],
                     reference_frame=None) -> ScoringSession:
        """Start a new scoring session"""
        # End current session if active
        if self.current_session and self.current_session.active:
            self.end_session()

        # Get profile
        if profile_name not in self.profiles:
            raise ValueError(f"Unknown target profile: {profile_name}")

        profile = self.profiles[profile_name]

        # Create session
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session = ScoringSession(
            session_id, profile, target_center, reference_frame
        )

        return self.current_session

    def add_shot_to_current_session(self, hole_x: int, hole_y: int,
                                    hole_radius: int,
                                    detection_confidence: float) -> Optional[Dict]:
        """Add a shot to the current active session"""
        if not self.current_session or not self.current_session.active:
            raise RuntimeError("No active scoring session")

        return self.current_session.add_shot(
            hole_x, hole_y, hole_radius, detection_confidence
        )

    def end_session(self) -> str:
        """End the current session and save to disk"""
        if not self.current_session:
            return None

        self.current_session.end_session()

        # Save session to disk
        session_file = os.path.join(
            self.sessions_dir,
            f"session_{self.current_session.session_id}.json"
        )

        with open(session_file, 'w') as f:
            json.dump(self.current_session.to_dict(), f, indent=2)

        return session_file

    def get_session_status(self) -> Optional[Dict]:
        """Get current session status"""
        if not self.current_session:
            return None

        status = self.current_session.get_summary()
        status['shots'] = self.current_session.shots  # Include shot details
        return status

    def load_session(self, session_file: str) -> ScoringSession:
        """Load a session from disk"""
        with open(session_file, 'r') as f:
            data = json.load(f)

        return ScoringSession.from_dict(data)

    def draw_scoring_overlay(self, frame, show_rings: bool = True,
                            show_shots: bool = True) -> np.ndarray:
        """
        Draw scoring overlay on frame

        Args:
            frame: Frame to draw on
            show_rings: Show ring boundaries
            show_shots: Show shot markers and scores
        """
        if not self.current_session or not self.current_session.active:
            return frame

        overlay = frame.copy()
        profile = self.current_session.target_profile
        center = self.current_session.target_center

        # Draw ring boundaries
        if show_rings and profile.ring_radii:
            # Draw rings from outer to inner
            for ring_num in [3, 4, 5, 6, 7, 8, 9, 10]:
                radius = int(profile.ring_radii.get(ring_num, 0))
                if radius > 0:
                    # Color coding: outer rings in blue, inner in red
                    if ring_num >= 7:
                        color = (0, 0, 255)  # Red for high-value rings
                        thickness = 2
                    else:
                        color = (255, 200, 0)  # Cyan for outer rings
                        thickness = 1

                    cv2.circle(overlay, center, radius, color, thickness)

                    # Add ring number label
                    if ring_num in [10, 9, 8, 7, 5, 3]:  # Show labels for some rings
                        label_pos = (center[0] + radius - 30, center[1] - 5)
                        cv2.putText(overlay, str(ring_num), label_pos,
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw target center crosshair
        cross_size = 20
        cv2.line(overlay,
                (center[0] - cross_size, center[1]),
                (center[0] + cross_size, center[1]),
                (0, 255, 0), 2)
        cv2.line(overlay,
                (center[0], center[1] - cross_size),
                (center[0], center[1] + cross_size),
                (0, 255, 0), 2)

        # Draw shots
        if show_shots:
            for shot in self.current_session.shots:
                x, y = shot['x'], shot['y']
                score = shot['score']

                # Color based on score
                if score >= 9:
                    color = (0, 255, 0)  # Green for excellent shots
                elif score >= 7:
                    color = (0, 255, 255)  # Yellow for good shots
                elif score >= 5:
                    color = (0, 165, 255)  # Orange for okay shots
                else:
                    color = (0, 0, 255)  # Red for poor shots

                # Draw shot marker
                cv2.circle(overlay, (x, y), 5, color, -1)
                cv2.circle(overlay, (x, y), 15, color, 2)

                # Draw score label
                score_text = str(score)
                cv2.putText(overlay, score_text, (x + 20, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw session info overlay
        info_y = 30
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)  # Background

        cv2.putText(overlay, f"Session: {self.current_session.session_id}",
                   (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30

        cv2.putText(overlay, f"Shots: {len(self.current_session.shots)}",
                   (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 30

        cv2.putText(overlay, f"Score: {self.current_session.total_score}",
                   (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return overlay


if __name__ == "__main__":
    # Test the scoring system
    print("🎯 Testing Target Scoring System")
    print("=" * 50)

    # Create scoring system
    scorer = TargetScoringSystem()

    # Calibrate large target profile (example values)
    print("\n📏 Calibrating 'large' target profile...")
    scorer.calibrate_profile('large', outer_radius=700, inner_black_radius=350)

    # Start a session
    print("\n🎮 Starting scoring session...")
    session = scorer.start_session('large', target_center=(1000, 1000))

    # Simulate some shots
    print("\n🎯 Simulating shots...")
    shots = [
        (1005, 1003, 13, 0.95),  # Center hit (10-ring)
        (1050, 1020, 14, 0.90),  # 9-ring
        (1100, 1050, 12, 0.85),  # 8-ring
        (1200, 1000, 13, 0.92),  # 7-ring
        (1350, 1100, 14, 0.88),  # 6-ring
    ]

    for i, (x, y, r, conf) in enumerate(shots):
        shot_data = scorer.add_shot_to_current_session(x, y, r, conf)
        if shot_data:
            print(f"  Shot #{shot_data['shot_number']}: "
                  f"Score {shot_data['score']}, "
                  f"Distance {shot_data['distance_from_center']:.1f}px")

    # Get session status
    print("\n📊 Session Status:")
    status = scorer.get_session_status()
    print(f"  Total Score: {status['total_score']}")
    print(f"  Shots Fired: {status['shot_count']}")

    # End session
    print("\n💾 Ending session...")
    session_file = scorer.end_session()
    print(f"  Session saved to: {session_file}")

    print("\n✅ Test complete!")
