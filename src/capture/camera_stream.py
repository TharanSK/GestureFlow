"""
GestureFlow - Asynchronous Threaded Camera Capture Module.
Implements Stage 1 Producer:
  - Polls OpenCV VideoCapture at 640x480 resolution (attempts 60 FPS).
  - Thread-safe bounded queue (maxsize=1) with zero-lag stale frame eviction.
  - Idle power throttling: throttles polling down to 2 FPS when signaled idle for > 5.0s,
    and restores full 60 FPS instantly upon palm detection.
"""

import threading
import queue
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class CameraStream:
    """
    Producer thread polling cv2.VideoCapture into a bounded Queue(maxsize=1).
    Evicts unread frames to guarantee zero buffer lag.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 60,
        idle_fps: int = 2,
        idle_timeout_seconds: float = 5.0,
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.idle_fps = idle_fps
        self.idle_timeout_seconds = idle_timeout_seconds

        # Bounded 1-frame queue for zero latency
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        # Idle auto-sleep management
        self._last_hand_detected_time = time.perf_counter()
        self._is_sleeping = False
        self._lock = threading.Lock()

        # Telemetry
        self.current_fps: float = 0.0

    def start(self):
        """Starts the producer thread."""
        if self._running:
            return

        self._running = True
        self._last_hand_detected_time = time.perf_counter()
        self._is_sleeping = False

        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraCaptureThread")
        self._thread.start()

    def signal_hand_presence(self, hand_present: bool):
        """Called by downstream worker to signal palm detection or lack thereof."""
        with self._lock:
            now = time.perf_counter()
            if hand_present:
                self._last_hand_detected_time = now
                if self._is_sleeping:
                    self._is_sleeping = False
            else:
                if not self._is_sleeping and (now - self._last_hand_detected_time) >= self.idle_timeout_seconds:
                    self._is_sleeping = True

    @property
    def is_sleeping(self) -> bool:
        with self._lock:
            return self._is_sleeping

    def _init_camera(self) -> bool:
        # Use DirectShow backend on Windows for lowest latency
        self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback to default backend
            self._cap = cv2.VideoCapture(self.device_index)

        if not self._cap.isOpened():
            print(f"[CameraStream] Error: Unable to open video device {self.device_index}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        # Minimize buffer size in hardware driver
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[CameraStream] Camera initialized: {actual_w:.0f}x{actual_h:.0f} @ {actual_fps:.0f} FPS")
        return True

    def _capture_loop(self):
        if not self._init_camera():
            self._running = False
            return

        fps_counter = 0
        fps_start = time.perf_counter()

        while self._running:
            loop_start = time.perf_counter()

            with self._lock:
                sleeping = self._is_sleeping

            # Rate limit when in idle sleep mode
            if sleeping:
                time.sleep(1.0 / self.idle_fps)

            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Evict stale frame if queue is full to enforce zero buffer lag
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait((frame, time.perf_counter()))
            except queue.Full:
                pass

            fps_counter += 1
            if time.perf_counter() - fps_start >= 1.0:
                self.current_fps = fps_counter / (time.perf_counter() - fps_start)
                fps_counter = 0
                fps_start = time.perf_counter()

            # Maintain target FPS in active mode
            if not sleeping and self.target_fps > 0:
                elapsed = time.perf_counter() - loop_start
                delay = (1.0 / self.target_fps) - elapsed
                if delay > 0:
                    time.sleep(delay)

        if self._cap:
            self._cap.release()
            self._cap = None

    def get_frame(self, timeout: float = 0.1) -> Tuple[Optional[np.ndarray], float]:
        """
        Retrieves the latest available frame from the queue.
        Returns: (frame, capture_timestamp)
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, 0.0

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
