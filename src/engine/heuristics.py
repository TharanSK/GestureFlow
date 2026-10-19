"""
GestureFlow - Swipe Detection & Region-of-Interest (ROI) Heuristics.
Implements 12-frame circular deque swipe detection via velocity norm & cosine similarity,
and central 60% ROI gating to eliminate the 'Midas Touch' accidental gesticulation problem.
"""

from collections import deque
from typing import Deque, Optional, Tuple
import numpy as np
import time


class SwipeDetector:
    """
    Evaluates dynamic directional trajectories over a 12-frame circular buffer of wrist centroids:
      (x_t, y_t, t).
    Triggers when:
      1. Overall displacement velocity ||v||_2 > v_min
      2. Cosine similarity between velocity vector and cardinal direction > 0.85
    """

    CARDINALS = {
        "SWIPE_LEFT": np.array([-1.0, 0.0], dtype=np.float32),
        "SWIPE_RIGHT": np.array([1.0, 0.0], dtype=np.float32),
        "SWIPE_UP": np.array([0.0, -1.0], dtype=np.float32),
        "SWIPE_DOWN": np.array([0.0, 1.0], dtype=np.float32),
    }

    def __init__(self, buffer_size: int = 12, min_velocity: float = 0.35, min_cosine: float = 0.85):
        self.buffer_size = buffer_size
        self.min_velocity = min_velocity
        self.min_cosine = min_cosine
        self.history: Deque[Tuple[float, float, float]] = deque(maxlen=buffer_size)

    def add_point(self, x: float, y: float, timestamp: Optional[float] = None) -> Optional[str]:
        """
        Adds a new wrist centroid point (in normalized image coordinates [0, 1]) and checks for swipe.
        Returns gesture token ('SWIPE_LEFT', 'SWIPE_RIGHT', etc.) if detected, else None.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        self.history.append((float(x), float(y), float(timestamp)))

        if len(self.history) < max(4, self.buffer_size // 2):
            return None

        # Compare oldest and newest point in the window
        x0, y0, t0 = self.history[0]
        x1, y1, t1 = self.history[-1]

        dt = t1 - t0
        if dt <= 1e-4:
            return None

        dx = x1 - x0
        dy = y1 - y0

        # Velocity in normalized units / second
        vx = dx / dt
        vy = dy / dt
        v_vec = np.array([vx, vy], dtype=np.float32)
        v_norm = float(np.linalg.norm(v_vec))

        if v_norm < self.min_velocity:
            return None

        v_unit = v_vec / v_norm

        best_token = None
        best_similarity = -1.0

        for token, cardinal_vec in self.CARDINALS.items():
            cosine = float(np.dot(v_unit, cardinal_vec))
            if cosine > self.min_cosine and cosine > best_similarity:
                best_similarity = cosine
                best_token = token

        if best_token is not None:
            # Clear history to prevent duplicate triggers from the same motion
            self.history.clear()
            return best_token

        return None

    def reset(self):
        self.history.clear()


class ROIGate:
    """
    Central Region-of-Interest (ROI) bounding box filter.
    Default: central 60% box (x in [0.20, 0.80], y in [0.20, 0.80]).
    Ignores gesticulation outside this active zone.
    Also maps normalized ROI coordinates to normalized screen coordinates [0, 1].
    """

    def __init__(self, x_min: float = 0.20, x_max: float = 0.80, y_min: float = 0.20, y_max: float = 0.80):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def is_inside(self, x: float, y: float) -> bool:
        """Returns True if (x, y) is inside the central ROI box."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def map_to_screen_norm(self, x: float, y: float, mirror_x: bool = True) -> Tuple[float, float]:
        """
        Maps (x, y) from image coords into clamped [0, 1] screen normalized coordinates.
        If mirror_x is True, horizontal direction is inverted so moving hand right moves cursor right.
        """
        # Clamp to ROI box
        clamped_x = max(self.x_min, min(self.x_max, x))
        clamped_y = max(self.y_min, min(self.y_max, y))

        norm_x = (clamped_x - self.x_min) / (self.x_max - self.x_min)
        norm_y = (clamped_y - self.y_min) / (self.y_max - self.y_min)

        if mirror_x:
            norm_x = 1.0 - norm_x

        return float(norm_x), float(norm_y)
