#!/usr/bin/env python3
"""
Bullet Hole Tracker
Tracks bullet holes across frames with temporal filtering and position averaging
"""

import numpy as np
from typing import List, Tuple, Optional
import time


class TrackedHole:
    """Represents a tracked bullet hole across multiple frames"""

    def __init__(
        self, x: float, y: float, radius: float, confidence: float, area: float, circularity: float, hole_id: int
    ):
        """Initialize a tracked hole

        Args:
            x, y: Initial position
            radius: Initial radius
            confidence: Initial confidence
            area: Initial area
            circularity: Initial circularity
            hole_id: Unique ID for this hole
        """
        self.hole_id = hole_id

        # Position tracking with running average
        self.positions = [(x, y)]  # History of positions
        self.avg_x = x
        self.avg_y = y

        # Size tracking with running average
        self.radii = [radius]
        self.avg_radius = radius

        # Confidence tracking
        self.confidences = [confidence]
        self.avg_confidence = confidence

        # Additional properties
        self.area = area
        self.circularity = circularity

        # Stability tracking
        self.detection_count = 1  # Number of frames this hole has been detected
        self.frames_since_last_seen = 0  # Frames since last detection
        self.first_seen_time = time.time()
        self.is_stable = False  # Only show stable holes

    def update(self, x: float, y: float, radius: float, confidence: float, area: float, circularity: float):
        """Update hole with new detection

        Args:
            x, y: New position
            radius: New radius
            confidence: New confidence
            area: New area
            circularity: New circularity
        """
        # Add to history
        self.positions.append((x, y))
        self.radii.append(radius)
        self.confidences.append(confidence)

        # Keep only recent history (last 10 frames)
        max_history = 30
        if len(self.positions) > max_history:
            self.positions = self.positions[-max_history:]
            self.radii = self.radii[-max_history:]
            self.confidences = self.confidences[-max_history:]

        # Update running averages
        self.avg_x = np.mean([p[0] for p in self.positions])
        self.avg_y = np.mean([p[1] for p in self.positions])
        self.avg_radius = np.mean(self.radii)
        self.avg_confidence = np.mean(self.confidences)

        # Update other properties with latest values
        self.area = area
        self.circularity = circularity

        # Update stability tracking
        self.detection_count += 1
        self.frames_since_last_seen = 0

        # Consider stable after being detected in at least 3 frames
        if self.detection_count >= 3:
            self.is_stable = True

    def increment_frames_since_seen(self):
        """Increment counter for frames without detection"""
        self.frames_since_last_seen += 1

    def get_detection_tuple(self) -> Tuple:
        """Get detection as tuple (x, y, radius, confidence, area, circularity)"""
        return (self.avg_x, self.avg_y, self.avg_radius, self.avg_confidence, self.area, self.circularity)

    def distance_to(self, x: float, y: float) -> float:
        """Calculate distance to given position"""
        return np.sqrt((self.avg_x - x) ** 2 + (self.avg_y - y) ** 2)


class BulletHoleTracker:
    """Tracks bullet holes across frames with temporal filtering"""

    def __init__(
        self, match_distance_threshold: float = 30.0, max_frames_missing: int = 5, min_detections_for_stability: int = 3
    ):
        """Initialize tracker

        Args:
            match_distance_threshold: Max distance (pixels) to consider same hole
            max_frames_missing: Max frames a hole can be missing before removal
            min_detections_for_stability: Min detections before showing hole
        """
        self.match_distance_threshold = match_distance_threshold
        self.max_frames_missing = max_frames_missing
        self.min_detections_for_stability = min_detections_for_stability

        self.tracked_holes: List[TrackedHole] = []
        self.next_hole_id = 1

    def update(self, detections: List[Tuple]) -> List[Tuple]:
        """Update tracker with new detections

        Args:
            detections: List of (x, y, radius, confidence, area, circularity)

        Returns:
            List of stable tracked holes as tuples (x, y, radius, confidence, area, circularity)
        """
        # Mark all existing holes as not seen this frame
        for hole in self.tracked_holes:
            hole.increment_frames_since_seen()

        # Match new detections to existing holes
        unmatched_detections = []

        for detection in detections:
            x, y, radius, confidence, area, circularity = detection

            # Find closest existing hole
            best_match = None
            best_distance = float("inf")

            for hole in self.tracked_holes:
                distance = hole.distance_to(x, y)
                if distance < self.match_distance_threshold and distance < best_distance:
                    best_match = hole
                    best_distance = distance

            if best_match:
                # Update existing hole
                best_match.update(x, y, radius, confidence, area, circularity)
            else:
                # New hole
                unmatched_detections.append(detection)

        # Add new holes
        for detection in unmatched_detections:
            x, y, radius, confidence, area, circularity = detection
            new_hole = TrackedHole(x, y, radius, confidence, area, circularity, self.next_hole_id)
            self.tracked_holes.append(new_hole)
            self.next_hole_id += 1

        # Remove holes that haven't been seen for too long
        self.tracked_holes = [
            hole for hole in self.tracked_holes if hole.frames_since_last_seen <= self.max_frames_missing
        ]

        # Return only stable holes (detected in multiple frames)
        stable_holes = [hole.get_detection_tuple() for hole in self.tracked_holes if hole.is_stable]

        return stable_holes

    def reset(self):
        """Reset tracker (clear all tracked holes)"""
        self.tracked_holes = []
        self.next_hole_id = 1

    def get_hole_count(self) -> int:
        """Get number of stable tracked holes"""
        return sum(1 for hole in self.tracked_holes if hole.is_stable)

    def get_all_holes(self, include_unstable: bool = False) -> List[Tuple]:
        """Get all tracked holes

        Args:
            include_unstable: If True, include holes that aren't stable yet

        Returns:
            List of tracked holes as tuples
        """
        if include_unstable:
            return [hole.get_detection_tuple() for hole in self.tracked_holes]
        else:
            return [hole.get_detection_tuple() for hole in self.tracked_holes if hole.is_stable]
