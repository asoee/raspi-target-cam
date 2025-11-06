#!/usr/bin/env python3
"""
Find the exact frame where the first shot appears (between frames 50-100)
"""

import cv2
import numpy as np
from raspi_target_cam.core.perspective import Perspective


def scan_frame_range(video_path, start_frame, end_frame, reference_frame_num=50):
    """
    Scan a range of frames to find where the first shot appears
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    perspective = Perspective()

    # Get reference frame (clean target)
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_num)
    ret, ref_frame = cap.read()
    if not ret:
        print(f"❌ Could not read reference frame {reference_frame_num}")
        return None

    ref_frame = cv2.rotate(ref_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ref_frame = perspective.apply_perspective_correction(ref_frame)
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    print(f"🔍 Scanning frames {start_frame} to {end_frame} (reference: frame {reference_frame_num})")
    print(f"=" * 70)

    results = []

    for frame_num in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        # Apply transformations
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = perspective.apply_perspective_correction(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate difference
        diff = cv2.absdiff(ref_gray, gray)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)

        # Count pixels with significant change (> 10 intensity units)
        significant_change = np.sum(diff > 10)

        time_sec = frame_num / fps

        results.append({
            'frame': frame_num,
            'time': time_sec,
            'mean_diff': mean_diff,
            'max_diff': max_diff,
            'changed_pixels': significant_change
        })

        # Print every frame in this range
        marker = ""
        if mean_diff > 2.0:
            marker = "  ⚠️  HIGH DIFF"
        elif mean_diff > 1.0:
            marker = "  ⚡ MEDIUM"

        print(f"Frame {frame_num:3d} ({time_sec:5.2f}s): mean={mean_diff:5.2f}, "
              f"max={max_diff:3.0f}, changed_px={significant_change:6d}{marker}")

    cap.release()

    # Find the frame with biggest jump
    if results:
        print(f"\n📊 Analysis:")
        print(f"=" * 70)

        # Sort by mean difference
        by_mean = sorted(results, key=lambda x: x['mean_diff'], reverse=True)
        print(f"\nTop 5 by mean difference:")
        for i, r in enumerate(by_mean[:5]):
            print(f"  {i+1}. Frame {r['frame']} ({r['time']:.2f}s): mean={r['mean_diff']:.2f}")

        # Look for sudden jumps
        print(f"\nLooking for sudden jumps in difference:")
        for i in range(1, len(results)):
            prev_mean = results[i-1]['mean_diff']
            curr_mean = results[i]['mean_diff']
            jump = curr_mean - prev_mean

            if jump > 0.5:  # Significant jump
                print(f"  Frame {results[i]['frame']} ({results[i]['time']:.2f}s): "
                      f"jump={jump:+.2f} (from {prev_mean:.2f} to {curr_mean:.2f})")

        # Return the frame with highest mean diff
        best_frame = by_mean[0]['frame']
        print(f"\n✅ Most likely first shot frame: {best_frame} ({by_mean[0]['time']:.2f}s)")

        return best_frame

    return None


def extract_and_compare(video_path, ref_frame_num, shot_frame_num):
    """
    Extract two frames and save them for visual comparison
    """
    print(f"\n📸 Extracting frames for comparison...")

    cap = cv2.VideoCapture(video_path)
    perspective = Perspective()

    # Reference frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_num)
    ret, ref = cap.read()
    ref = cv2.rotate(ref, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ref = perspective.apply_perspective_correction(ref)

    # Shot frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, shot_frame_num)
    ret, shot = cap.read()
    shot = cv2.rotate(shot, cv2.ROTATE_90_COUNTERCLOCKWISE)
    shot = perspective.apply_perspective_correction(shot)

    cap.release()

    # Create difference visualization
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    shot_gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(ref_gray, shot_gray)

    # Enhance difference for visualization
    diff_enhanced = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff_colored = cv2.applyColorMap(diff_enhanced, cv2.COLORMAP_JET)

    # Create side-by-side comparison
    h, w = ref.shape[:2]

    # Add labels
    ref_labeled = ref.copy()
    shot_labeled = shot.copy()

    cv2.putText(ref_labeled, f"Frame {ref_frame_num} (REFERENCE)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.putText(shot_labeled, f"Frame {shot_frame_num} (FIRST SHOT)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.putText(diff_colored, "DIFFERENCE (enhanced)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Stack vertically for comparison
    comparison = np.vstack([ref_labeled, shot_labeled, diff_colored])

    # Save
    import os
    output_dir = "test_outputs/first_shot"
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(f"{output_dir}/reference_frame_{ref_frame_num}.jpg", ref)
    cv2.imwrite(f"{output_dir}/first_shot_frame_{shot_frame_num}.jpg", shot)
    cv2.imwrite(f"{output_dir}/difference_map.jpg", diff_colored)
    cv2.imwrite(f"{output_dir}/comparison.jpg", comparison)

    print(f"   ✅ Saved to {output_dir}/")
    print(f"      - reference_frame_{ref_frame_num}.jpg")
    print(f"      - first_shot_frame_{shot_frame_num}.jpg")
    print(f"      - difference_map.jpg")
    print(f"      - comparison.jpg")


if __name__ == "__main__":
    video_path = "samples/10-shot-1.mkv"

    # Scan frames 50-100 to find first shot
    first_shot_frame = scan_frame_range(video_path, start_frame=50, end_frame=100, reference_frame_num=50)

    if first_shot_frame:
        # Extract and compare
        extract_and_compare(video_path, ref_frame_num=50, shot_frame_num=first_shot_frame)

        print(f"\n✅ First shot analysis complete!")
        print(f"   Reference: Frame 50")
        print(f"   First shot: Frame {first_shot_frame}")
