"""
GestureFlow - CLI Hand Landmark Dataset Recorder.
Allows capturing real-time normalized 63-D landmark vectors from webcam frames
and appending them with corresponding pose labels into data/landmarks_dataset.csv.
"""

import sys
import os
import csv
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.hand_tracker import HandTracker
from src.vision.normalizer import LandmarkNormalizer

POSE_CHOICES = {
    "1": "POSE_OPEN_PALM",
    "2": "POSE_FIST",
    "3": "POSE_PEACE",
    "4": "POSE_THUMBS_UP",
    "5": "POSE_POINTING",
    "6": "POSE_OK",
}


def main():
    csv_path = Path("data/landmarks_dataset.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize CSV header if file does not exist
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"coord_{i}" for i in range(63)]
            writer.writerow(header)

    print("==================================================")
    print("      GestureFlow Landmark Dataset Recorder       ")
    print("==================================================")
    print("Commands:")
    for k, v in POSE_CHOICES.items():
        print(f"  [{k}] - Record sample for {v}")
    print("  [SPACE] - Burst record (30 frames)")
    print("  [q] - Quit")
    print("==================================================")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = HandTracker()
    normalizer = LandmarkNormalizer()

    active_pose = POSE_CHOICES["1"]
    recording_burst = 0
    total_recorded = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            # Mirror for natural user interaction
            frame = cv2.flip(frame, 1)

            result = tracker.process_frame(frame)
            hand_present = result is not None and result.has_landmarks

            status_text = f"Active Pose: {active_pose} (Hand: {'YES' if hand_present else 'NO'})"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f"Total Samples: {total_recorded}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if hand_present:
                landmarks_3d = result.world_landmarks
                _, feat_63d, _ = normalizer.normalize(landmarks_3d)

                if recording_burst > 0:
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([active_pose] + list(feat_63d))
                    recording_burst -= 1
                    total_recorded += 1
                    cv2.putText(frame, f"RECORDING BURST... {recording_burst} left", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("GestureFlow Dataset Recorder", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif chr(key) in POSE_CHOICES:
                active_pose = POSE_CHOICES[chr(key)]
                print(f"Switched active pose to: {active_pose}")
                # Record single sample
                if hand_present:
                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([active_pose] + list(feat_63d))
                    total_recorded += 1
                    print(f"Recorded 1 sample for {active_pose} (total: {total_recorded})")
            elif key == ord(" "):
                recording_burst = 30
                print(f"Recording burst of 30 frames for {active_pose}...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print(f"\nRecording session finished. Total samples recorded: {total_recorded}")


if __name__ == "__main__":
    main()
