"""
Unit tests for ActionDispatcher.
Tests keymap resolution, mock dispatch, and state tracking.
"""

import pytest
from src.os_bridge.action_dispatcher import ActionDispatcher


def test_action_dispatcher_mock():
    dispatcher = ActionDispatcher(keymaps_path="config/default_keymaps.json", mock_mode=True)
    assert len(dispatcher.keymaps) > 0

    # Dispatch hotkey
    res = dispatcher.dispatch_gesture("SWIPE_LEFT")
    assert res is True
    assert len(dispatcher.dispatched_actions) == 1
    assert dispatcher.dispatched_actions[0][0] == "SWIPE_LEFT"

    # Dispatch media key
    res = dispatcher.dispatch_gesture("POSE_FIST")
    assert res is True

    # Pinch down/up
    dispatcher.set_pinch_state(True)
    assert dispatcher.is_mouse_down is True
    dispatcher.set_pinch_state(False)
    assert dispatcher.is_mouse_down is False

    # Unknown gesture
    res = dispatcher.dispatch_gesture("UNKNOWN_TOKEN_XYZ")
    assert res is False
