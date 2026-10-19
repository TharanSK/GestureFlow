"""
GestureFlow - Debounce Finite State Machine (FSM).
Implements the 4-state flow:
  [IDLE] -> [ARMED (400 ms Clutch)] -> [TRIGGERED (3 Frames)] -> [COOLDOWN (350 ms Lock)]
Guarantees zero duplicate triggers, noise suppression, and return-stroke protection.
"""

from enum import Enum, auto
from typing import Optional, Tuple
import time


class FSMState(Enum):
    IDLE = auto()
    ARMED = auto()
    TRIGGERED = auto()
    COOLDOWN = auto()


class DebounceFSM:
    """
    Debounce Finite State Machine for reliable gesture activation:
      - Requires clutch gesture (e.g. POSE_OPEN_PALM) held for clutch_time_ms to become ARMED.
      - In ARMED state, gesture candidate must persist for consecutive_frames (default 3) to TRIGGER.
      - Once TRIGGERED, emits the validated token and enters COOLDOWN for cooldown_ms (default 350 ms)
        to prevent reverse-motion recoil triggers.
    """

    def __init__(
        self,
        clutch_time_ms: float = 400.0,
        trigger_frames: int = 3,
        cooldown_ms: float = 350.0,
        clutch_token: str = "POSE_OPEN_PALM",
    ):
        self.clutch_time_ms = clutch_time_ms
        self.trigger_frames = trigger_frames
        self.cooldown_ms = cooldown_ms
        self.clutch_token = clutch_token

        self.state = FSMState.IDLE
        self.clutch_start_time: Optional[float] = None
        self.candidate_token: Optional[str] = None
        self.candidate_frame_count: int = 0
        self.cooldown_start_time: Optional[float] = None
        self.last_triggered_token: Optional[str] = None

    def update(
        self,
        detected_token: Optional[str],
        timestamp: Optional[float] = None,
        force_armed: bool = False,
    ) -> Tuple[FSMState, Optional[str]]:
        """
        Updates FSM state given the detected token in the current frame.

        Args:
            detected_token: Currently observed pose/swipe token or None
            timestamp: Monotonic timestamp in seconds
            force_armed: If True, bypasses clutch check (useful for testing or direct mode)

        Returns:
            (current_state, event_to_dispatch)
            event_to_dispatch is non-None ONLY in the single frame where TRIGGERED occurs.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        now_ms = timestamp * 1000.0
        dispatched_event: Optional[str] = None

        # -------------------------------------------------------------
        # STATE: COOLDOWN
        # -------------------------------------------------------------
        if self.state == FSMState.COOLDOWN:
            elapsed = now_ms - (self.cooldown_start_time or now_ms)
            if elapsed >= self.cooldown_ms:
                # Cooldown expired. Check if clutch is currently held to stay ARMED
                if force_armed or detected_token == self.clutch_token:
                    self.state = FSMState.ARMED
                else:
                    self.state = FSMState.IDLE
                    self.clutch_start_time = None
                self.cooldown_start_time = None
                self.candidate_token = None
                self.candidate_frame_count = 0
            else:
                # Still locked in cooldown: suppress everything
                return FSMState.COOLDOWN, None

        # -------------------------------------------------------------
        # STATE: IDLE
        # -------------------------------------------------------------
        if self.state == FSMState.IDLE:
            if force_armed:
                self.state = FSMState.ARMED
            elif detected_token == self.clutch_token:
                if self.clutch_start_time is None:
                    self.clutch_start_time = now_ms
                elif (now_ms - self.clutch_start_time) >= self.clutch_time_ms:
                    self.state = FSMState.ARMED
                    self.candidate_token = None
                    self.candidate_frame_count = 0
            else:
                # Reset clutch timer if clutch pose is broken
                self.clutch_start_time = None

            return self.state, None

        # -------------------------------------------------------------
        # STATE: ARMED
        # -------------------------------------------------------------
        if self.state == FSMState.ARMED:
            # Swipe tokens are dynamic and already debounced via 12-frame deque
            is_swipe = detected_token is not None and detected_token.startswith("SWIPE_")

            if is_swipe:
                self.state = FSMState.TRIGGERED
                self.last_triggered_token = detected_token
                dispatched_event = detected_token

                # Immediately enter cooldown
                self.state = FSMState.COOLDOWN
                self.cooldown_start_time = now_ms
                self.candidate_token = None
                self.candidate_frame_count = 0
                return FSMState.TRIGGERED, dispatched_event

            # For static poses (other than clutch itself)
            if detected_token is not None and detected_token != self.clutch_token:
                if detected_token == self.candidate_token:
                    self.candidate_frame_count += 1
                else:
                    self.candidate_token = detected_token
                    self.candidate_frame_count = 1

                if self.candidate_frame_count >= self.trigger_frames:
                    self.state = FSMState.TRIGGERED
                    self.last_triggered_token = detected_token
                    dispatched_event = detected_token

                    # Lock into cooldown
                    self.state = FSMState.COOLDOWN
                    self.cooldown_start_time = now_ms
                    self.candidate_token = None
                    self.candidate_frame_count = 0
                    return FSMState.TRIGGERED, dispatched_event
            else:
                # Reset candidate count if candidate was lost
                self.candidate_token = None
                self.candidate_frame_count = 0

            return FSMState.ARMED, None

        return self.state, None

    def is_active(self) -> bool:
        """Returns True if the system is currently ARMED or in active clutch."""
        return self.state in (FSMState.ARMED, FSMState.TRIGGERED)

    def reset(self):
        self.state = FSMState.IDLE
        self.clutch_start_time = None
        self.candidate_token = None
        self.candidate_frame_count = 0
        self.cooldown_start_time = None
        self.last_triggered_token = None
