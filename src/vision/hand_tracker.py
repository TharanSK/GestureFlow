"""
GestureFlow - MediaPipe 3D Hand Tracker Module.
Wraps MediaPipe HandLandmarker Tasks API with automatic model asset provisioning,
RGB color conversion, and extraction of normalized, world, and pixel landmarks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import urllib.request
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEFAULT_MODEL_PATH = "data/models/hand_landmarker.task"


@dataclass
class SingleHandResult:
    landmarks_norm: np.ndarray    # (21, 3) in [0, 1] normalized image space
    world_landmarks: np.ndarray   # (21, 3) metric 3D space
    pixel_landmarks: np.ndarray   # (21, 2) pixel coordinates (x, y)
    handedness: str               # "Left" or "Right"
    confidence: float             # score in [0.0, 1.0]


@dataclass
class HandTrackingResult:
    has_landmarks: bool
    primary_hand: Optional[SingleHandResult] = None
    all_hands: Optional[List[SingleHandResult]] = None

    @property
    def world_landmarks(self) -> Optional[np.ndarray]:
        return self.primary_hand.world_landmarks if self.primary_hand else None

    @property
    def landmarks_norm(self) -> Optional[np.ndarray]:
        return self.primary_hand.landmarks_norm if self.primary_hand else None

    @property
    def pixel_landmarks(self) -> Optional[np.ndarray]:
        return self.primary_hand.pixel_landmarks if self.primary_hand else None


class HandTracker:
    """
    Worker tracking 21 3D joint coordinates alongside handedness labels.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.70,
        min_tracking_confidence: float = 0.70,
        max_num_hands: int = 2,
    ):
        self.model_path = Path(model_path)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_hands = max_num_hands

        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        if not self.model_path.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading MediaPipe HandLandmarker model to {self.model_path}...")
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, self.model_path)
            print("Download completed.")

    def process_frame(self, bgr_frame: np.ndarray) -> HandTrackingResult:
        """
        Converts BGR frame to RGB and runs hand landmark detection.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return HandTrackingResult(has_landmarks=False)

        h, w = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)

        if not detection_result or not detection_result.hand_landmarks:
            return HandTrackingResult(has_landmarks=False, primary_hand=None, all_hands=[])

        all_hands: List[SingleHandResult] = []

        for i, lm_list in enumerate(detection_result.hand_landmarks):
            norm_pts = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)

            # Metric 3D world landmarks (or fallback to normalized if not available)
            if detection_result.hand_world_landmarks and i < len(detection_result.hand_world_landmarks):
                world_pts = np.array([[lm.x, lm.y, lm.z] for lm in detection_result.hand_world_landmarks[i]], dtype=np.float32)
            else:
                world_pts = norm_pts.copy()

            pixel_pts = np.array([[lm.x * w, lm.y * h] for lm in lm_list], dtype=np.float32)

            handedness_label = "Right"
            score = 0.90
            if detection_result.handedness and i < len(detection_result.handedness):
                cat = detection_result.handedness[i][0]
                handedness_label = cat.category_name
                score = float(cat.score)

            hand = SingleHandResult(
                landmarks_norm=norm_pts,
                world_landmarks=world_pts,
                pixel_landmarks=pixel_pts,
                handedness=handedness_label,
                confidence=score,
            )
            all_hands.append(hand)

        # Primary hand is the first detected hand (or highest confidence)
        primary = max(all_hands, key=lambda h: h.confidence) if all_hands else None

        return HandTrackingResult(
            has_landmarks=True,
            primary_hand=primary,
            all_hands=all_hands,
        )

    def close(self):
        if hasattr(self, "detector") and self.detector:
            self.detector.close()
