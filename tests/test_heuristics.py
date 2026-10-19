"""
Unit tests for SwipeDetector and ROIGate.
"""

import pytest
import numpy as np
from src.engine.heuristics import SwipeDetector, ROIGate


def test_swipe_detector_cardinal_directions():
    detector = SwipeDetector(buffer_size=12, min_velocity=0.35, min_cosine=0.85)

    # 1. Simulate fast Left Swipe: x decreases from 0.8 to 0.2 over 0.2 seconds
    t0 = 100.0
    detected = None
    for i in range(8):
        x = 0.8 - (0.6 * (i / 7))
        t = t0 + (0.2 * (i / 7))
        res = detector.add_point(x=x, y=0.5, timestamp=t)
        if res:
            detected = res

    assert detected == "SWIPE_LEFT"

    # Immediately checking again without new motion must return None (zero duplicates)
    assert detector.add_point(0.2, 0.5, timestamp=t0 + 0.25) is None

    # 2. Simulate fast Right Swipe: x increases from 0.2 to 0.8
    detector.reset()
    detected = None
    t0 = 200.0
    for i in range(8):
        x = 0.2 + (0.6 * (i / 7))
        t = t0 + (0.2 * (i / 7))
        res = detector.add_point(x=x, y=0.5, timestamp=t)
        if res:
            detected = res

    assert detected == "SWIPE_RIGHT"

    # 3. Simulate slow movement (below velocity threshold)
    detector.reset()
    detected = None
    t0 = 300.0
    for i in range(8):
        x = 0.8 - (0.05 * (i / 7))  # very small displacement
        t = t0 + (1.0 * (i / 7))
        res = detector.add_point(x=x, y=0.5, timestamp=t)
        if res:
            detected = res

    assert detected is None


def test_roi_gate():
    roi = ROIGate(x_min=0.20, x_max=0.80, y_min=0.20, y_max=0.80)

    # Inside
    assert roi.is_inside(0.5, 0.5) is True
    assert roi.is_inside(0.25, 0.75) is True

    # Border/Outside
    assert roi.is_inside(0.10, 0.5) is False
    assert roi.is_inside(0.5, 0.95) is False
    assert roi.is_inside(0.05, 0.05) is False

    # Mapping to normalized screen coordinates
    # Center (0.5, 0.5) maps to center (0.5, 0.5)
    sx, sy = roi.map_to_screen_norm(0.5, 0.5, mirror_x=False)
    assert pytest.approx(sx, 1e-3) == 0.5
    assert pytest.approx(sy, 1e-3) == 0.5

    # Corner (0.2, 0.2) maps to (0.0, 0.0)
    sx, sy = roi.map_to_screen_norm(0.2, 0.2, mirror_x=False)
    assert pytest.approx(sx, 1e-3) == 0.0
    assert pytest.approx(sy, 1e-3) == 0.0
