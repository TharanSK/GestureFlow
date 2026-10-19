"""
Unit tests for PinchHysteresis and AdaptiveEMACursorFilter.
"""

import pytest
from src.engine.filters import PinchHysteresis, AdaptiveEMACursorFilter


def test_pinch_hysteresis_dual_threshold():
    hys = PinchHysteresis(tau_engage=0.22, tau_release=0.30)
    assert hys.is_pinched is False

    # 1. Above engage threshold (0.25): should NOT engage
    state = hys.update(0.25)
    assert state is False
    assert hys.is_pinched is False

    # 2. Drops below engage threshold (0.20 < 0.22): should ENGAGE
    state = hys.update(0.20)
    assert state is True
    assert hys.is_pinched is True

    # 3. Rises into hysteresis deadband (0.26, between 0.22 and 0.30): must STAY ENGAGED
    state = hys.update(0.26)
    assert state is True
    assert hys.is_pinched is True

    # 4. Rises above release threshold (0.32 > 0.30): should RELEASE
    state = hys.update(0.32)
    assert state is False
    assert hys.is_pinched is False

    # 5. Drops back into hysteresis deadband (0.26): must STAY RELEASED
    state = hys.update(0.26)
    assert state is False
    assert hys.is_pinched is False


def test_pinch_hysteresis_pauses_on_invalid_palm():
    hys = PinchHysteresis(tau_engage=0.22, tau_release=0.30)

    # Hand edge-on: is_palm_valid = False
    state = hys.update(0.15, is_palm_valid=False)
    # Should NOT engage even though d_pinch < 0.22
    assert state is False
    assert hys.is_pinched is False


def test_adaptive_ema_cursor_filter():
    ema = AdaptiveEMACursorFilter(alpha_min=0.15, beta=4.5, deadzone_pixels=0.01)

    # First point sets baseline
    p0 = (0.5, 0.5)
    s0 = ema.filter(p0)
    assert pytest.approx(s0[0]) == 0.5
    assert pytest.approx(s0[1]) == 0.5

    # Sub-pixel movement (velocity < deadzone)
    p_tiny = (0.5002, 0.5001)
    s_tiny = ema.filter(p_tiny)
    assert pytest.approx(s_tiny[0]) == 0.5
    assert pytest.approx(s_tiny[1]) == 0.5

    # High velocity leap: alpha scales towards 1.0
    p_fast = (0.9, 0.9)
    s_fast = ema.filter(p_fast)
    # Filter output moves significantly towards target
    assert s_fast[0] > 0.8
    assert s_fast[1] > 0.8
