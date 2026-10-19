"""
Unit tests for DebounceFSM.
Verifies the 4-state lifecycle:
  [IDLE] -> [ARMED (400 ms Clutch)] -> [TRIGGERED (3 Frames)] -> [COOLDOWN (350 ms Lock)]
"""

import pytest
from src.engine.state_machine import DebounceFSM, FSMState


def test_fsm_clutch_activation():
    fsm = DebounceFSM(clutch_time_ms=400.0, trigger_frames=3, cooldown_ms=350.0)
    assert fsm.state == FSMState.IDLE

    # 1. Holding clutch pose for 200 ms: remains IDLE
    state, event = fsm.update("POSE_OPEN_PALM", timestamp=1.0)
    assert state == FSMState.IDLE
    assert event is None

    state, event = fsm.update("POSE_OPEN_PALM", timestamp=1.2)
    assert state == FSMState.IDLE
    assert event is None

    # 2. Reaching 400 ms (t=1.4): transitions to ARMED
    state, event = fsm.update("POSE_OPEN_PALM", timestamp=1.45)
    assert state == FSMState.ARMED
    assert event is None


def test_fsm_consecutive_trigger_frames_and_cooldown():
    fsm = DebounceFSM(clutch_time_ms=400.0, trigger_frames=3, cooldown_ms=350.0)

    # Force ARMED for direct gesture testing
    fsm.state = FSMState.ARMED

    # Frame 1: Candidate POSE_FIST
    state, event = fsm.update("POSE_FIST", timestamp=2.0)
    assert state == FSMState.ARMED
    assert event is None

    # Frame 2: Candidate POSE_FIST
    state, event = fsm.update("POSE_FIST", timestamp=2.033)
    assert state == FSMState.ARMED
    assert event is None

    # Frame 3: Candidate POSE_FIST reaches threshold (3 frames) -> TRIGGERED & emits event
    state, event = fsm.update("POSE_FIST", timestamp=2.066)
    assert state == FSMState.TRIGGERED
    assert event == "POSE_FIST"

    # Immediately after trigger, FSM is in COOLDOWN
    # Frame 4 at t=2.100 (elapsed cooldown: 34ms < 350ms): must REJECT everything
    state, event = fsm.update("POSE_FIST", timestamp=2.100)
    assert state == FSMState.COOLDOWN
    assert event is None

    # Frame 5 at t=2.300 (elapsed: 234ms < 350ms): still locked in COOLDOWN
    state, event = fsm.update("POSE_PEACE", timestamp=2.300)
    assert state == FSMState.COOLDOWN
    assert event is None

    # Frame 6 at t=2.450 (elapsed: 384ms > 350ms): COOLDOWN expired.
    # If clutch is held, returns to ARMED
    state, event = fsm.update("POSE_OPEN_PALM", timestamp=2.450)
    assert state == FSMState.ARMED
    assert event is None


def test_fsm_noise_rejection():
    fsm = DebounceFSM(clutch_time_ms=400.0, trigger_frames=3, cooldown_ms=350.0)
    fsm.state = FSMState.ARMED

    # Single frame noise
    fsm.update("POSE_THUMBS_UP", timestamp=3.0)
    # Then switches immediately to another pose
    fsm.update("POSE_PEACE", timestamp=3.033)
    # Then back to None
    state, event = fsm.update(None, timestamp=3.066)

    assert state == FSMState.ARMED
    assert event is None
