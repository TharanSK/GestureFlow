"""
GestureFlow - Main Application Coordinator.
Orchestrates the 4-stage pipeline:
  Stage 1: CameraProducer Thread
  Stage 2: Vision Landmarker Worker
  Stage 3: Gesture Engine & Debounce FSM Evaluator
  Stage 4: OS Input Dispatcher & PyQt6 System Tray Consumer
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Optional
import yaml
import numpy as np
import cv2

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

# Import GestureFlow modules
from src.capture.camera_stream import CameraStream
from src.vision.hand_tracker import HandTracker
from src.vision.normalizer import LandmarkNormalizer
from src.engine.filters import PinchHysteresis, AdaptiveEMACursorFilter
from src.engine.heuristics import SwipeDetector, ROIGate
from src.engine.classifier import PoseClassifier
from src.engine.state_machine import DebounceFSM, FSMState
from src.os_bridge.action_dispatcher import ActionDispatcher
from src.ui.hud_overlay import HUDOverlayWindow, SystemTrayManager


def load_settings(settings_path: str = "config/settings.yaml") -> dict:
    p = Path(settings_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class GestureFlowApp:
    """
    Main application engine running the pipeline loop and integrating with PyQt6.
    """

    def __init__(self, settings_path: str = "config/settings.yaml", keymaps_path: str = "config/default_keymaps.json", headless: bool = False):
        self.settings = load_settings(settings_path)
        self.headless = headless
        self.is_paused = False

        # Configuration parameters
        cam_cfg = self.settings.get("camera", {})
        self.camera_device = cam_cfg.get("device_index", 0)
        self.cam_width = cam_cfg.get("width", 640)
        self.cam_height = cam_cfg.get("height", 480)
        self.target_fps = cam_cfg.get("target_fps", 60)
        self.idle_fps = cam_cfg.get("idle_fps", 2)
        self.idle_timeout = cam_cfg.get("idle_timeout_seconds", 5.0)

        roi_cfg = self.settings.get("vision", {}).get("roi", {})
        self.roi_x_min = roi_cfg.get("x_min", 0.20)
        self.roi_x_max = roi_cfg.get("x_max", 0.80)
        self.roi_y_min = roi_cfg.get("y_min", 0.20)
        self.roi_y_max = roi_cfg.get("y_max", 0.80)

        norm_cfg = self.settings.get("normalizer", {})
        self.min_palm_normal_z = norm_cfg.get("min_palm_normal_z", 0.35)

        hys_cfg = self.settings.get("hysteresis", {})
        self.pinch_engage = hys_cfg.get("pinch_engage_threshold", 0.22)
        self.pinch_release = hys_cfg.get("pinch_release_threshold", 0.30)

        cur_cfg = self.settings.get("cursor", {})
        self.alpha_min = cur_cfg.get("alpha_min", 0.15)
        self.beta = cur_cfg.get("beta", 4.5)

        swipe_cfg = self.settings.get("swipe", {})
        self.swipe_buf = swipe_cfg.get("buffer_size", 12)
        self.swipe_v_min = swipe_cfg.get("min_velocity", 0.35)
        self.swipe_cos_min = swipe_cfg.get("min_cosine_similarity", 0.85)

        fsm_cfg = self.settings.get("fsm", {})
        self.clutch_time_ms = fsm_cfg.get("clutch_hold_time_ms", 400.0)
        self.trigger_frames = fsm_cfg.get("trigger_consecutive_frames", 3)
        self.cooldown_ms = fsm_cfg.get("cooldown_lock_ms", 350.0)

        clf_cfg = self.settings.get("classifier", {})
        self.model_path = clf_cfg.get("model_path", "data/models/pose_classifier.onnx")
        self.min_probability = clf_cfg.get("min_probability", 0.85)

        # -------------------------------------------------------------
        # Initialize Pipeline Components
        # -------------------------------------------------------------
        # Stage 1: Producer
        self.camera_stream = CameraStream(
            device_index=self.camera_device,
            width=self.cam_width,
            height=self.cam_height,
            target_fps=self.target_fps,
            idle_fps=self.idle_fps,
            idle_timeout_seconds=self.idle_timeout,
        )

        # Stage 2: Worker
        self.hand_tracker = HandTracker()

        # Stage 3: Evaluator
        self.normalizer = LandmarkNormalizer(min_palm_normal_z=self.min_palm_normal_z)
        self.roi_gate = ROIGate(x_min=self.roi_x_min, x_max=self.roi_x_max, y_min=self.roi_y_min, y_max=self.roi_y_max)
        self.pinch_filter = PinchHysteresis(tau_engage=self.pinch_engage, tau_release=self.pinch_release)
        self.cursor_filter = AdaptiveEMACursorFilter(alpha_min=self.alpha_min, beta=self.beta)
        self.swipe_detector = SwipeDetector(buffer_size=self.swipe_buf, min_velocity=self.swipe_v_min, min_cosine=self.swipe_cos_min)
        self.classifier = PoseClassifier(model_path=self.model_path, min_probability=self.min_probability)
        self.fsm = DebounceFSM(clutch_time_ms=self.clutch_time_ms, trigger_frames=self.trigger_frames, cooldown_ms=self.cooldown_ms)

        # Stage 4: Consumer
        self.action_dispatcher = ActionDispatcher(keymaps_path=keymaps_path)

        # UI
        self.hud_window: Optional[HUDOverlayWindow] = None
        self.tray_manager: Optional[SystemTrayManager] = None
        self.timer: Optional[QTimer] = None

    def start(self):
        print("[GestureFlow] Starting Camera Producer Stage 1...")
        self.camera_stream.start()

        if not self.headless:
            self.hud_window = HUDOverlayWindow()
            self.tray_manager = SystemTrayManager()

            # Wire tray signals
            self.tray_manager.toggle_hud_requested.connect(self._toggle_hud)
            self.tray_manager.pause_resume_requested.connect(self._toggle_pause)
            self.tray_manager.reload_keymaps_requested.connect(self.action_dispatcher.load_keymaps)
            self.tray_manager.exit_requested.connect(self.shutdown)

            self.hud_window.show()

            # Qt high-frequency tick timer (1 ms interval for lowest latency)
            self.timer = QTimer()
            self.timer.timeout.connect(self.process_pipeline_tick)
            self.timer.start(1)

    def _toggle_hud(self):
        if self.hud_window:
            if self.hud_window.isVisible():
                self.hud_window.hide()
            else:
                self.hud_window.show()

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.tray_manager:
            self.tray_manager.set_paused_status(self.is_paused)
        print(f"[GestureFlow] Gestures {'PAUSED' if self.is_paused else 'RESUMED'}")

    def process_pipeline_tick(self):
        """
        Single tick of the pipeline: receives frame from bounded queue, runs vision inference,
        evaluates heuristics/FSM, dispatches OS actions, and updates HUD display.
        """
        frame, capture_timestamp = self.camera_stream.get_frame(timeout=0.005)
        if frame is None:
            return

        now = time.perf_counter()
        # Flip horizontally for natural mirror behavior
        frame_mirrored = cv2.flip(frame, 1)

        # Stage 2: Hand Landmark Tracking
        tracking_result = self.hand_tracker.process_frame(frame_mirrored)
        hand_present = tracking_result.has_landmarks and tracking_result.primary_hand is not None

        # Signal presence to CameraStream for idle sleep throttling
        self.camera_stream.signal_hand_presence(hand_present)

        detected_pose: Optional[str] = None
        current_fsm_state = self.fsm.state.name
        is_pinched = self.pinch_filter.is_pinched
        pixel_landmarks = None

        if hand_present and not self.is_paused:
            hand = tracking_result.primary_hand
            pixel_landmarks = hand.pixel_landmarks
            wrist_norm = hand.landmarks_norm[0]  # Wrist (x, y, z)
            index_norm = hand.landmarks_norm[8]  # Index tip (x, y, z)

            # Central 60% ROI Check ("Midas Touch" defense)
            in_roi = self.roi_gate.is_inside(wrist_norm[0], wrist_norm[1]) or self.roi_gate.is_inside(index_norm[0], index_norm[1])

            if in_roi:
                # Stage 3: Feature Engineering & Normalization
                landmarks_3d = hand.world_landmarks
                norm_lm, feat_63d, _ = self.normalizer.normalize(landmarks_3d)

                # Palm normal & self-occlusion check
                _, is_palm_valid = self.normalizer.compute_palm_normal(landmarks_3d)

                # Static Pinch Distance & Hysteresis
                d_pinch = self.normalizer.compute_pinch_distance(norm_lm)
                new_pinch = self.pinch_filter.update(d_pinch, is_palm_valid=is_palm_valid)

                if new_pinch != is_pinched:
                    self.action_dispatcher.set_pinch_state(new_pinch)
                    is_pinched = new_pinch

                # Static Pose Classification (ONNX MLP)
                classified_pose, confidence = self.classifier.predict(feat_63d)

                # Dynamic Swipe Trajectory (12-Frame Deque)
                swipe_token = self.swipe_detector.add_point(wrist_norm[0], wrist_norm[1], capture_timestamp)

                # Prioritize dynamic swipe over static pose
                candidate_token = swipe_token or classified_pose
                detected_pose = candidate_token

                # Debounce Finite State Machine Evaluation
                fsm_state, event_to_dispatch = self.fsm.update(candidate_token, capture_timestamp)
                current_fsm_state = fsm_state.name

                if event_to_dispatch:
                    print(f"[GestureFlow] Dispatching Validated Gesture: {event_to_dispatch}")
                    self.action_dispatcher.dispatch_gesture(event_to_dispatch)

                # Precision Pointer Cursor Tracking
                # Map index fingertip to screen
                screen_norm_x, screen_norm_y = self.roi_gate.map_to_screen_norm(index_norm[0], index_norm[1], mirror_x=False)
                smooth_x, smooth_y = self.cursor_filter.filter((screen_norm_x, screen_norm_y))

                # Allow cursor positioning when system is ARMED or when pointing pose is active
                if self.fsm.is_active() or classified_pose == "POSE_POINTING":
                    self.action_dispatcher.move_cursor_norm(smooth_x, smooth_y)

            else:
                # Outside active ROI: reset transient filters
                self.cursor_filter.reset()
                self.swipe_detector.reset()
                self.fsm.update(None, capture_timestamp)

        else:
            # No hand in frame
            self.cursor_filter.reset()
            self.swipe_detector.reset()
            self.fsm.update(None, capture_timestamp)

        # Stage 4: HUD Overlay Visualization
        if self.hud_window and self.hud_window.isVisible():
            e2e_latency_ms = (time.perf_counter() - capture_timestamp) * 1000.0
            roi_box = (self.roi_x_min, self.roi_x_max, self.roi_y_min, self.roi_y_max)
            annotated = self.hud_window.draw_telemetry_on_frame(
                frame=frame_mirrored,
                pixel_landmarks=pixel_landmarks,
                roi_box=roi_box,
                fps=self.camera_stream.current_fps,
                latency_ms=e2e_latency_ms,
                fsm_state=current_fsm_state,
                current_gesture=detected_pose,
                is_pinched=is_pinched,
                is_sleeping=self.camera_stream.is_sleeping,
            )
            self.hud_window.update_display(annotated)

    def shutdown(self):
        print("[GestureFlow] Shutting down...")
        if self.timer:
            self.timer.stop()
        self.camera_stream.stop()
        self.action_dispatcher.cleanup()
        self.hand_tracker.close()
        QApplication.quit()


def main():
    parser = argparse.ArgumentParser(description="GestureFlow: Camera-Based OS Gesture Controller")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to settings.yaml")
    parser.add_argument("--keymaps", default="config/default_keymaps.json", help="Path to default_keymaps.json")
    parser.add_argument("--headless", action="store_true", help="Run without PyQt6 GUI/HUD")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    gf = GestureFlowApp(settings_path=args.settings, keymaps_path=args.keymaps, headless=args.headless)
    gf.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
