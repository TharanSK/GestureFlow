"""
Unit tests for LandmarkNormalizer.
Verifies scale and translation invariance, palm surface normal vector computation,
and self-occlusion / edge-on rotation filtering.
"""

import pytest
import numpy as np
from src.vision.normalizer import LandmarkNormalizer


@pytest.fixture
def base_landmarks():
    # 21 landmarks representing a standard hand
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.0, 0.0, 0.0]     # Wrist P0
    lm[5] = [-0.2, 0.8, 0.0]    # Index MCP P5
    lm[9] = [0.0, 1.0, 0.0]     # Middle MCP P9
    lm[17] = [0.3, 0.7, 0.0]    # Pinky MCP P17
    lm[4] = [-0.15, 0.4, 0.0]   # Thumb tip P4
    lm[8] = [-0.15, 0.6, 0.0]   # Index tip P8
    return lm


def test_translation_invariance(base_landmarks):
    normalizer = LandmarkNormalizer()
    _, feat_orig, _ = normalizer.normalize(base_landmarks)

    # Shift by arbitrary (dx, dy, dz)
    shift = np.array([5.5, -12.3, 8.1], dtype=np.float32)
    translated_lm = base_landmarks + shift

    _, feat_trans, _ = normalizer.normalize(translated_lm)

    # Vectors must be virtually identical
    np.testing.assert_allclose(feat_orig, feat_trans, atol=1e-5)


def test_scale_invariance(base_landmarks):
    normalizer = LandmarkNormalizer()
    _, feat_orig, _ = normalizer.normalize(base_landmarks)

    # Scale coordinates by 2.5x and 0.4x
    scaled_up = base_landmarks * 2.5
    scaled_down = base_landmarks * 0.4

    _, feat_up, _ = normalizer.normalize(scaled_up)
    _, feat_down, _ = normalizer.normalize(scaled_down)

    np.testing.assert_allclose(feat_orig, feat_up, atol=1e-4)
    np.testing.assert_allclose(feat_orig, feat_down, atol=1e-4)


def test_palm_surface_normal_and_edge_on_filtering():
    normalizer = LandmarkNormalizer(min_palm_normal_z=0.35)

    # Hand facing camera directly (normal along Z axis)
    facing_lm = np.zeros((21, 3), dtype=np.float32)
    facing_lm[0] = [0.0, 0.0, 0.0]    # P0
    facing_lm[5] = [1.0, 0.0, 0.0]    # P5 (along X)
    facing_lm[17] = [0.0, 1.0, 0.0]   # P17 (along Y)
    # Cross product (1,0,0) x (0,1,0) = (0,0,1)
    normal, is_valid = normalizer.compute_palm_normal(facing_lm)
    assert abs(normal[2]) > 0.99
    assert is_valid is True

    # Hand turned edge-on to the camera (normal along X or Y, nz ~ 0)
    edge_on_lm = np.zeros((21, 3), dtype=np.float32)
    edge_on_lm[0] = [0.0, 0.0, 0.0]
    edge_on_lm[5] = [0.0, 0.0, 1.0]   # along Z
    edge_on_lm[17] = [0.0, 1.0, 0.0]  # along Y
    # Cross product (0,0,1) x (0,1,0) = (-1,0,0) -> nz = 0
    normal_edge, is_valid_edge = normalizer.compute_palm_normal(edge_on_lm)
    assert abs(normal_edge[2]) < 0.35
    assert is_valid_edge is False


def test_pinch_distance():
    normalizer = LandmarkNormalizer()
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[4] = [0.1, 0.2, 0.0]
    lm[8] = [0.1, 0.5, 0.0]
    dist = normalizer.compute_pinch_distance(lm)
    assert pytest.approx(dist, 1e-4) == 0.3
