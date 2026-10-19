"""
GestureFlow - OS Input Bridge Module.
Translates validated gesture events and cursor coordinates into synthetic OS inputs
using pynput and native platform hooks, driven by config/default_keymaps.json.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key


KEY_NAME_MAP = {
    "ctrl": Key.ctrl_l,
    "ctrl_l": Key.ctrl_l,
    "ctrl_r": Key.ctrl_r,
    "shift": Key.shift,
    "alt": Key.alt_l,
    "alt_l": Key.alt_l,
    "alt_r": Key.alt_r,
    "win": Key.cmd,
    "cmd": Key.cmd,
    "tab": Key.tab,
    "enter": Key.enter,
    "esc": Key.esc,
    "space": Key.space,
    "left": Key.left,
    "right": Key.right,
    "up": Key.up,
    "down": Key.down,
    "volume_up": Key.media_volume_up,
    "volume_down": Key.media_volume_down,
    "volume_mute": Key.media_volume_mute,
    "play_pause": Key.media_play_pause,
}


class ActionDispatcher:
    """
    Synthesizes operating system input events:
      - Cursor movement
      - Mouse clicks and drags (with stateful press/release tracking)
      - Hotkeys and key combinations
      - Media control keys
    """

    def __init__(self, keymaps_path: str = "config/default_keymaps.json", mock_mode: bool = False):
        self.keymaps_path = Path(keymaps_path)
        self.mock_mode = mock_mode
        self.keymaps: Dict[str, Any] = {}

        self.mouse = MouseController() if not mock_mode else None
        self.keyboard = KeyboardController() if not mock_mode else None

        self.is_mouse_down = False
        self.screen_width = 1920
        self.screen_height = 1080
        self._init_screen_metrics()
        self.load_keymaps()

        # Telemetry / action history for logging and testing
        self.dispatched_actions: List[Tuple[str, float]] = []

    def _init_screen_metrics(self):
        if sys.platform == "win32":
            try:
                import ctypes
                self.screen_width = ctypes.windll.user32.GetSystemMetrics(0)
                self.screen_height = ctypes.windll.user32.GetSystemMetrics(1)
            except Exception:
                pass

    def load_keymaps(self):
        if self.keymaps_path.exists():
            try:
                with open(self.keymaps_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.keymaps = data.get("gestures", {})
                print(f"[ActionDispatcher] Loaded {len(self.keymaps)} gesture mappings from {self.keymaps_path}")
            except Exception as e:
                print(f"[ActionDispatcher] Failed to load keymaps: {e}")
                self.keymaps = {}
        else:
            print(f"[ActionDispatcher] Keymaps file not found at {self.keymaps_path}")

    def dispatch_gesture(self, gesture_token: str) -> bool:
        """
        Dispatches the action mapped to the given gesture token.
        """
        mapping = self.keymaps.get(gesture_token)
        if not mapping:
            return False

        action_type = mapping.get("action")
        timestamp = time.perf_counter()
        self.dispatched_actions.append((gesture_token, timestamp))

        if self.mock_mode:
            print(f"[MOCK DISPATCH] Gesture: {gesture_token} -> {mapping}")
            return True

        try:
            if action_type == "hotkey":
                keys = mapping.get("keys", [])
                self._send_hotkey(keys)
                return True
            elif action_type == "media_key":
                key_name = mapping.get("key")
                self._send_media_key(key_name)
                return True
            elif action_type == "mouse_click":
                button_name = mapping.get("button", "left")
                self._send_mouse_click(button_name)
                return True
            elif action_type in ("clutch", "pointer_mode"):
                # Internal state transitions, no OS input needed
                return True
            else:
                print(f"[ActionDispatcher] Unknown action type: {action_type}")
                return False
        except Exception as e:
            print(f"[ActionDispatcher] Dispatch error for {gesture_token}: {e}")
            return False

    def set_pinch_state(self, is_pinched: bool):
        """
        Maintains mouse button down / up state for click and drag.
        """
        if self.mock_mode:
            self.is_mouse_down = is_pinched
            return

        if is_pinched and not self.is_mouse_down:
            self.mouse.press(Button.left)
            self.is_mouse_down = True
        elif not is_pinched and self.is_mouse_down:
            self.mouse.release(Button.left)
            self.is_mouse_down = False

    def move_cursor_norm(self, norm_x: float, norm_y: float):
        """
        Moves cursor to absolute coordinates mapped from [0, 1] screen normalized coordinates.
        """
        target_x = int(norm_x * (self.screen_width - 1))
        target_y = int(norm_y * (self.screen_height - 1))

        if self.mock_mode:
            return

        self.mouse.position = (target_x, target_y)

    def _resolve_key(self, key_name: str):
        key_lower = key_name.lower()
        if key_lower in KEY_NAME_MAP:
            return KEY_NAME_MAP[key_lower]
        if len(key_name) == 1:
            return key_name
        return None

    def _send_hotkey(self, keys: List[str]):
        resolved = []
        for k in keys:
            resolved_key = self._resolve_key(k)
            if resolved_key:
                resolved.append(resolved_key)

        if not resolved:
            return

        # Press keys in forward order
        for rk in resolved:
            self.keyboard.press(rk)

        time.sleep(0.02)  # Short debounce

        # Release keys in reverse order
        for rk in reversed(resolved):
            self.keyboard.release(rk)

    def _send_media_key(self, key_name: str):
        rk = self._resolve_key(key_name)
        if rk:
            self.keyboard.press(rk)
            time.sleep(0.01)
            self.keyboard.release(rk)

    def _send_mouse_click(self, button_name: str = "left"):
        btn = Button.right if button_name.lower() == "right" else Button.left
        self.mouse.click(btn, 1)

    def cleanup(self):
        """Release held mouse buttons upon shutdown."""
        if not self.mock_mode and self.is_mouse_down:
            try:
                self.mouse.release(Button.left)
            except Exception:
                pass
            self.is_mouse_down = False
