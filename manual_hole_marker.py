#!/usr/bin/env python3
"""
Manual Hole Marker Tool
Interactive UI for manually marking the exact positions of bullet holes
This creates ground truth data for tuning the detector
"""

import cv2
import numpy as np
import json
import os


class ManualHoleMarker:
    """Interactive tool for marking bullet holes"""

    def __init__(self, image_path, output_file="ground_truth_holes.json"):
        self.image_path = image_path
        self.output_file = output_file

        # Load image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Scale for display (make it larger for easier clicking)
        self.display_scale = 0.7
        self.display_image = cv2.resize(self.original_image, None,
                                       fx=self.display_scale,
                                       fy=self.display_scale)

        # Marked holes
        self.holes = []

        # UI state
        self.window_name = "Manual Hole Marker - Click on each bullet hole"
        self.temp_image = None

    def _scale_to_original(self, x, y):
        """Convert display coordinates to original image coordinates"""
        return int(x / self.display_scale), int(y / self.display_scale)

    def _scale_to_display(self, x, y):
        """Convert original coordinates to display coordinates"""
        return int(x * self.display_scale), int(y * self.display_scale)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - add hole
            orig_x, orig_y = self._scale_to_original(x, y)

            self.holes.append({
                'hole_number': len(self.holes) + 1,
                'x': orig_x,
                'y': orig_y,
                'display_x': x,
                'display_y': y
            })

            print(f"✅ Marked hole #{len(self.holes)} at ({orig_x}, {orig_y})")
            self._redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - remove last hole
            if self.holes:
                removed = self.holes.pop()
                print(f"❌ Removed hole #{removed['hole_number']}")
                self._redraw()

    def _redraw(self):
        """Redraw the image with all marked holes"""
        self.temp_image = self.display_image.copy()

        # Draw all marked holes
        for hole in self.holes:
            x, y = hole['display_x'], hole['display_y']
            num = hole['hole_number']

            # Draw circle
            cv2.circle(self.temp_image, (x, y), 10, (0, 255, 0), 2)
            cv2.circle(self.temp_image, (x, y), 3, (0, 255, 0), -1)

            # Draw number label
            label = str(num)
            label_pos = (x + 15, y - 10)

            # Background for label
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(self.temp_image,
                         (label_pos[0] - 5, label_pos[1] - text_height - 5),
                         (label_pos[0] + text_width + 5, label_pos[1] + 5),
                         (0, 0, 0), -1)

            # Label text
            cv2.putText(self.temp_image, label, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add instructions overlay
        self._add_instructions()

        cv2.imshow(self.window_name, self.temp_image)

    def _add_instructions(self):
        """Add instruction text overlay"""
        instructions = [
            f"Holes marked: {len(self.holes)}/10",
            "",
            "LEFT CLICK: Mark hole",
            "RIGHT CLICK: Remove last",
            "'s': Save and exit",
            "'q': Quit without saving",
            "'c': Clear all"
        ]

        # Draw semi-transparent background
        overlay = self.temp_image.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.temp_image, 0.3, 0, self.temp_image)

        y = 35
        for line in instructions:
            color = (255, 255, 255) if line else (150, 150, 150)
            cv2.putText(self.temp_image, line, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y += 25

    def run(self):
        """Run the interactive marking tool"""
        print("🎯 Manual Hole Marker Tool")
        print("=" * 60)
        print(f"Image: {self.image_path}")
        print(f"Resolution: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
        print()
        print("Instructions:")
        print("  - LEFT CLICK on each bullet hole to mark it")
        print("  - RIGHT CLICK to remove the last marked hole")
        print("  - Press 's' to save and exit")
        print("  - Press 'c' to clear all marks")
        print("  - Press 'q' to quit without saving")
        print()
        print("Please mark all 10 bullet holes...")
        print()

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # Initial draw
        self._redraw()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # Save
                if len(self.holes) == 0:
                    print("⚠️  No holes marked! Not saving.")
                    continue

                self.save()
                print(f"\n✅ Saved {len(self.holes)} holes to {self.output_file}")
                break

            elif key == ord('q'):
                # Quit without saving
                print("\n❌ Quit without saving")
                break

            elif key == ord('c'):
                # Clear all
                self.holes = []
                print("🗑️  Cleared all marks")
                self._redraw()

        cv2.destroyAllWindows()
        return self.holes

    def save(self):
        """Save marked holes to JSON file"""
        data = {
            'image_path': self.image_path,
            'image_resolution': {
                'width': self.original_image.shape[1],
                'height': self.original_image.shape[0]
            },
            'total_holes': len(self.holes),
            'holes': [
                {
                    'hole_number': hole['hole_number'],
                    'x': hole['x'],
                    'y': hole['y']
                }
                for hole in self.holes
            ]
        }

        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n💾 Ground truth saved to: {self.output_file}")
        print(f"   Total holes marked: {len(self.holes)}")

        # Also create a visualization
        self._create_visualization()

    def _create_visualization(self):
        """Create and save visualization of marked holes"""
        viz = self.original_image.copy()

        for hole in self.holes:
            x, y = hole['x'], hole['y']
            num = hole['hole_number']

            # Draw circle
            cv2.circle(viz, (x, y), 15, (0, 255, 0), 3)
            cv2.circle(viz, (x, y), 4, (0, 255, 0), -1)

            # Draw number
            label = str(num)
            cv2.putText(viz, label, (x + 20, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # Add info
        info_height = 100
        info_panel = np.zeros((info_height, viz.shape[1], 3), dtype=np.uint8)

        cv2.putText(info_panel, f"Ground Truth: {len(self.holes)} holes manually marked",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info_panel, f"Green circles and numbers show exact hole positions",
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        final = np.vstack([info_panel, viz])

        # Save
        output_dir = "test_outputs/ground_truth"
        os.makedirs(output_dir, exist_ok=True)

        viz_path = os.path.join(output_dir, "ground_truth_visualization.jpg")
        cv2.imwrite(viz_path, final)

        print(f"   Visualization saved to: {viz_path}")


def load_ground_truth(json_file="ground_truth_holes.json"):
    """Load ground truth holes from JSON file"""
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r') as f:
        data = json.load(f)

    return data


if __name__ == "__main__":
    # Use the frame with all 10 holes
    image_path = "test_frames/frame_0930_all_10_shots_corrected.jpg"

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("   Please ensure the test frame exists")
        exit(1)

    marker = ManualHoleMarker(image_path)
    holes = marker.run()

    if holes:
        print(f"\n📋 Marked Holes:")
        for hole in holes:
            print(f"   #{hole['hole_number']}: ({hole['x']}, {hole['y']})")
