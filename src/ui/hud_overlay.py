"""
GestureFlow - HUD Overlay & PyQt6 System Tray Integration.
Provides real-time skeletal visualization, 60% ROI bounding box rendering,
pipeline telemetry (FPS, end-to-end latency, FSM state, pinch status),
and non-intrusive system tray control.
"""

import sys
import time
from typing import Optional, List, Tuple
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QIcon, QPen, QAction
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QSystemTrayIcon,
    QMenu,
)

# Hand skeleton connections (21 joints)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (5, 9), (9, 13), (13, 17),
]


class VideoHUDWidget(QWidget):
    """
    Renders the live video feed along with hand skeleton, ROI box, and telemetry overlay.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.current_pixmap: Optional[QPixmap] = None

    def update_frame(self, bgr_image: np.ndarray):
        h, w, ch = bgr_image.shape
        bytes_per_line = ch * w
        # Convert BGR to RGB for Qt
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(qt_img)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.current_pixmap:
            painter.drawPixmap(0, 0, self.width(), self.height(), self.current_pixmap)
        else:
            painter.fillRect(self.rect(), QColor(20, 20, 25))
            painter.setPen(QColor(180, 180, 180))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "GestureFlow: Waiting for camera...")


class HUDOverlayWindow(QMainWindow):
    """
    HUD Window displaying live skeleton debug overlay and HUD telemetry.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GestureFlow - HUD Skeleton Monitor")
        self.resize(640, 480)
        self.video_widget = VideoHUDWidget(self)
        self.setCentralWidget(self.video_widget)

    def draw_telemetry_on_frame(
        self,
        frame: np.ndarray,
        pixel_landmarks: Optional[np.ndarray],
        roi_box: Tuple[float, float, float, float],
        fps: float,
        latency_ms: float,
        fsm_state: str,
        current_gesture: Optional[str],
        is_pinched: bool,
        is_sleeping: bool,
    ) -> np.ndarray:
        """
        Draws skeleton bones, joints, ROI bounding box, and metrics directly onto frame.
        """
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # 1. Draw central 60% Region of Interest (ROI) bounding box
        x_min, x_max, y_min, y_max = roi_box
        rx1, rx2 = int(x_min * w), int(x_max * w)
        ry1, ry2 = int(y_min * h), int(y_max * h)

        # Draw ROI dashed or translucent box
        roi_color = (0, 165, 255)  # Orange
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), roi_color, 2)
        cv2.putText(canvas, "ACTIVE 60% ROI", (rx1 + 5, ry1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)

        # 2. Draw Skeleton Joints and Bones
        if pixel_landmarks is not None and len(pixel_landmarks) == 21:
            # Draw bones
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                pt1 = (int(pixel_landmarks[p1_idx][0]), int(pixel_landmarks[p1_idx][1]))
                pt2 = (int(pixel_landmarks[p2_idx][0]), int(pixel_landmarks[p2_idx][1]))
                cv2.line(canvas, pt1, pt2, (0, 255, 128), 2, cv2.LINE_AA)

            # Draw joints
            for idx, pt in enumerate(pixel_landmarks):
                center = (int(pt[0]), int(pt[1]))
                # Highlight index tip (P8) and thumb tip (P4)
                if idx in (4, 8):
                    color = (0, 0, 255) if is_pinched else (255, 255, 0)
                    radius = 6 if is_pinched else 5
                else:
                    color = (0, 200, 255)
                    radius = 3
                cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)

        # 3. Draw Telemetry HUD Header
        # Semi-transparent dark banner at top
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        fps_text = f"FPS: {fps:.1f}"
        lat_text = f"Latency: {latency_ms:.1f}ms"
        state_color = (0, 255, 0) if fsm_state == "ARMED" else ((0, 140, 255) if fsm_state == "TRIGGERED" else (200, 200, 200))
        fsm_text = f"FSM: {fsm_state}"
        gesture_text = f"Pose: {current_gesture or 'NONE'}"
        pinch_text = f"Pinch: {'DOWN' if is_pinched else 'UP'}"

        cv2.putText(canvas, fps_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(canvas, lat_text, (120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(canvas, fsm_text, (280, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)
        cv2.putText(canvas, gesture_text, (430, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 50), 1)

        if is_sleeping:
            cv2.putText(canvas, "[IDLE SLEEP 2 FPS]", (w - 200, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        return canvas

    def update_display(self, frame_annotated: np.ndarray):
        self.video_widget.update_frame(frame_annotated)


def create_tray_icon() -> QIcon:
    """Generates an in-memory crisp icon pixmap for the system tray."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Dark background circle
    painter.setBrush(QColor(30, 144, 255))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(2, 2, 28, 28)

    # Stylized hand / gesture symbol
    painter.setPen(QPen(QColor(255, 255, 255), 2))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "GF")
    painter.end()
    return QIcon(pixmap)


class SystemTrayManager(QObject):
    """
    Manages the PyQt6 system tray icon and its context menu.
    """
    toggle_hud_requested = pyqtSignal()
    pause_resume_requested = pyqtSignal()
    reload_keymaps_requested = pyqtSignal()
    exit_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tray_icon = QSystemTrayIcon(create_tray_icon(), parent)
        self.tray_menu = QMenu()

        self.status_action = QAction("GestureFlow: Active", self.tray_menu)
        self.status_action.setEnabled(False)
        self.tray_menu.addAction(self.status_action)
        self.tray_menu.addSeparator()

        self.hud_action = QAction("Toggle Skeleton HUD", self.tray_menu)
        self.hud_action.triggered.connect(self.toggle_hud_requested.emit)
        self.tray_menu.addAction(self.hud_action)

        self.pause_action = QAction("Pause Gestures", self.tray_menu)
        self.pause_action.triggered.connect(self.pause_resume_requested.emit)
        self.tray_menu.addAction(self.pause_action)

        self.reload_action = QAction("Reload Keymaps", self.tray_menu)
        self.reload_action.triggered.connect(self.reload_keymaps_requested.emit)
        self.tray_menu.addAction(self.reload_action)

        self.tray_menu.addSeparator()
        self.exit_action = QAction("Exit GestureFlow", self.tray_menu)
        self.exit_action.triggered.connect(self.exit_requested.emit)
        self.tray_menu.addAction(self.exit_action)

        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.setToolTip("GestureFlow OS Controller")
        self.tray_icon.show()

    def set_paused_status(self, is_paused: bool):
        if is_paused:
            self.status_action.setText("GestureFlow: PAUSED")
            self.pause_action.setText("Resume Gestures")
        else:
            self.status_action.setText("GestureFlow: Active")
            self.pause_action.setText("Pause Gestures")
