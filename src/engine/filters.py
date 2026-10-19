"""
GestureFlow - Adaptive EMA Filter & Pinch Hysteresis Module.
Provides dual-threshold hysteresis for stable click/drag without chatter,
and velocity-adaptive Exponential Moving Average for smooth jitter-free cursor tracking.
"""

from typing import Tuple, Optional
import numpy as np


class PinchHysteresis:
    """
    Dual-threshold stateful hysteresis filter for pinch detection:
      - Engage (pinch down) when d_pinch < tau_engage (default 0.22)
      - Release (pinch up) when d_pinch > tau_release (default 0.30)
      - Holds previous state when between thresholds to eliminate click chatter.
    """

    def __init__(self, tau_engage: float = 0.35, tau_release: float = 0.45):
        self.tau_engage = tau_engage
        self.tau_release = tau_release
        self.is_pinched = False

    def update(self, d_pinch: float, is_palm_valid: bool = True) -> bool:
        """
        Updates hysteresis state:
          - Engages when d_pinch < tau_engage (default 0.35)
          - Releases when d_pinch > tau_release (default 0.45)
          - Maintains state between thresholds to eliminate click chatter.
          - Pauses evaluation when is_palm_valid is False (severe edge-on self-occlusion).
        """
        if not is_palm_valid:
            return self.is_pinched

        if not self.is_pinched:
            if d_pinch < self.tau_engage:
                self.is_pinched = True
        else:
            if d_pinch > self.tau_release:
                self.is_pinched = False

        return self.is_pinched

    def reset(self):
        self.is_pinched = False


class AdaptiveEMACursorFilter:
    """
    Speed-adaptive Exponential Moving Average (EMA) cursor filter:
      S_t = alpha_t * P_8,t + (1 - alpha_t) * S_{t-1}
      alpha_t = min(1.0, alpha_min + beta * ||P_8,t - P_8,t-1||_2)
    When stationary (||delta_P|| ~ 0), alpha is low (~0.15), providing high smoothing.
    When moving fast, alpha increases up to 1.0, giving near-zero latency tracking.
    """

    def __init__(self, alpha_min: float = 0.15, beta: float = 4.5, deadzone_pixels: float = 1.0):
        self.alpha_min = alpha_min
        self.beta = beta
        self.deadzone_pixels = deadzone_pixels
        self.prev_raw: Optional[np.ndarray] = None
        self.smoothed: Optional[np.ndarray] = None

    def filter(self, current_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Args:
            current_pos: (x, y) coordinates (normalized ROI or screen pixel coords)
        Returns:
            smoothed_pos: (x, y) filtered coordinates
        """
        curr = np.array(current_pos, dtype=np.float32)

        if self.smoothed is None or self.prev_raw is None:
            self.prev_raw = curr.copy()
            self.smoothed = curr.copy()
            return float(self.smoothed[0]), float(self.smoothed[1])

        # Velocity norm: ||P_t - P_{t-1}||_2
        velocity = float(np.linalg.norm(curr - self.prev_raw))

        # Deadzone suppression for sub-pixel jitter
        if velocity < self.deadzone_pixels:
            self.prev_raw = curr.copy()
            return float(self.smoothed[0]), float(self.smoothed[1])

        # Adaptive alpha
        alpha = min(1.0, self.alpha_min + self.beta * velocity)

        # EMA update: S_t = alpha * P_t + (1 - alpha) * S_{t-1}
        self.smoothed = alpha * curr + (1.0 - alpha) * self.smoothed
        self.prev_raw = curr.copy()

        return float(self.smoothed[0]), float(self.smoothed[1])

    def reset(self):
        self.prev_raw = None
        self.smoothed = None
