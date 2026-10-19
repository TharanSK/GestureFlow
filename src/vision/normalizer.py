"""
GestureFlow - Scale-Invariant Feature Engineering and Normalization Module.
Implements coordinate translation relative to wrist origin (P0), scale normalization
by palm span (||P9 - P0||), and palm surface normal computation.
"""

from typing import Tuple, Optional
import numpy as np


class LandmarkNormalizer:
    """
    Normalizes 21 3D hand landmarks according to GestureFlow specification:
      P_hat_i = (P_i - P_0) / (||P_9 - P_0||_2 + eps), for all i in {0, ..., 20}
    Also computes the unit palm normal vector:
      n_hat = (P_5 - P_0) x (P_17 - P_0) / ||(P_5 - P_0) x (P_17 - P_0)||_2
    """

    WRIST_IDX = 0
    INDEX_MCP_IDX = 5
    MIDDLE_MCP_IDX = 9
    PINKY_MCP_IDX = 17
    THUMB_TIP_IDX = 4
    INDEX_TIP_IDX = 8

    def __init__(self, epsilon: float = 1e-6, min_palm_normal_z: float = 0.35):
        self.epsilon = epsilon
        self.min_palm_normal_z = min_palm_normal_z

    def normalize(self, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Args:
            landmarks: Array of shape (21, 3) representing (x, y, z) for 21 joints.
        Returns:
            normalized_landmarks: Array of shape (21, 3)
            feature_vector_63d: Flattened 1D array of length 63
            scale_factor: Palm span distance ||P_9 - P_0||
        """
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.shape != (21, 3):
            raise ValueError(f"Expected landmarks of shape (21, 3), got {landmarks.shape}")

        wrist = landmarks[self.WRIST_IDX]
        middle_mcp = landmarks[self.MIDDLE_MCP_IDX]

        # Translate relative to wrist P0
        translated = landmarks - wrist

        # Compute Euclidean palm span ||P_9 - P_0||_2
        palm_span = float(np.linalg.norm(middle_mcp - wrist))
        scale = palm_span + self.epsilon

        normalized = translated / scale
        feature_vector = normalized.flatten()

        return normalized, feature_vector, palm_span

    def compute_palm_normal(self, landmarks: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Computes unit surface normal n_hat = (P5 - P0) x (P17 - P0) / ||...||
        Returns:
            normal: Unit normal vector (nx, ny, nz)
            is_valid_facing: True if |nz| >= min_palm_normal_z (not edge-on / self-occluded)
        """
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.shape != (21, 3):
            raise ValueError(f"Expected landmarks of shape (21, 3), got {landmarks.shape}")

        p0 = landmarks[self.WRIST_IDX]
        p5 = landmarks[self.INDEX_MCP_IDX]
        p17 = landmarks[self.PINKY_MCP_IDX]

        v1 = p5 - p0
        v2 = p17 - p0

        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < self.epsilon:
            unit_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            unit_normal = (normal / norm).astype(np.float32)

        # Self-occlusion and edge-on filter: check |nz|
        is_valid_facing = abs(float(unit_normal[2])) >= self.min_palm_normal_z
        return unit_normal, is_valid_facing

    def compute_pinch_distance(self, normalized_landmarks: np.ndarray) -> float:
        """
        Computes normalized Euclidean distance between thumb tip (P4) and index tip (P8):
        d_pinch = ||P_hat_4 - P_hat_8||_2
        """
        p4 = normalized_landmarks[self.THUMB_TIP_IDX]
        p8 = normalized_landmarks[self.INDEX_TIP_IDX]
        return float(np.linalg.norm(p4 - p8))
