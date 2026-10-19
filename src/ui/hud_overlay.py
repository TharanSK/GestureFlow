"""
GestureFlow - Real-Time Gesture Display & HUD Overlay.
Provides:
  1. Main Dashboard Window:
     - Left: Real-time Camera & Hand Skeleton Canvas (with 60% ROI box & joint telemetry)
     - Right: Live Gesture Panel with Active Gesture Spotlight, Clutch Progress Bar,
              FSM State Badge, Gesture Reference Cheat Sheet with real-time active highlighting,
              and Dispatched Actions Event Log.
  2. Floating Mini-HUD (Desktop Widget):
     - An always-on-top, draggable, compact glassmorphic pill widget showing active gesture
       and FSM state across other desktop applications.
  3. System Tray Manager:
     - Tray icon, context menu, and pause/resume control.
"""

import sys
import time
from datetime import datetime
from collections import deque
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint
from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QColor,
    QFont,
    QIcon,
    QPen,
    QAction,
    QLinearGradient,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFrame,
    QScrollArea,
    QPushButton,
    QProgressBar,
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

# Gesture catalog with visual emojis and descriptions
GESTURE_CATALOG = [
    {"token": "POSE_OPEN_PALM", "icon": "🖐️", "name": "Open Palm", "action": "Wake / Clutch (Hold 400ms)"},
    {"token": "POSE_POINTING", "icon": "👆", "name": "Pointing", "action": "Precision Cursor Control"},
    {"token": "PINCH", "icon": "👌", "name": "Pinch & Drag", "action": "Left Click / Click & Drag"},
    {"token": "POSE_FIST", "icon": "✊", "name": "Fist", "action": "Play / Pause Media"},
    {"token": "POSE_PEACE", "icon": "✌️", "name": "Peace / V", "action": "Mute / Unmute Audio"},
    {"token": "POSE_THUMBS_UP", "icon": "👍", "name": "Thumbs Up", "action": "Snipping Tool (Win+Shift+S)"},
    {"token": "POSE_OK", "icon": "🆗", "name": "OK Sign", "action": "App Switcher (Alt+Tab)"},
    {"token": "SWIPE_LEFT", "icon": "⬅️", "name": "Swipe Left", "action": "Next Desktop / Next Slide"},
    {"token": "SWIPE_RIGHT", "icon": "➡️", "name": "Swipe Right", "action": "Prev Desktop / Prev Slide"},
    {"token": "SWIPE_UP", "icon": "⬆️", "name": "Swipe Up", "action": "Volume Up"},
    {"token": "SWIPE_DOWN", "icon": "⬇️", "name": "Swipe Down", "action": "Volume Down"},
]


class VideoHUDWidget(QWidget):
    """Renders the video feed with skeleton bones and ROI box."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.current_pixmap: Optional[QPixmap] = None

    def update_frame(self, bgr_image: np.ndarray):
        if bgr_image is None or bgr_image.size == 0:
            return
        h, w, ch = bgr_image.shape
        bytes_per_line = ch * w
        rgb_image = np.ascontiguousarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self.current_pixmap = QPixmap.fromImage(qt_img)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.current_pixmap:
            painter.drawPixmap(0, 0, self.width(), self.height(), self.current_pixmap)
        else:
            painter.fillRect(self.rect(), QColor(18, 18, 24))
            painter.setPen(QColor(160, 160, 180))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for camera stream...")


class GestureItemCard(QFrame):
    """An individual gesture card in the sidebar cheat sheet that lights up when active."""

    def __init__(self, icon: str, name: str, action: str, token: str, parent=None):
        super().__init__(parent)
        self.token = token
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedHeight(50)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(10)

        self.icon_label = QLabel(icon)
        self.icon_label.setFont(QFont("Segoe UI Emoji", 16))
        self.icon_label.setFixedWidth(28)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(1)
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet("font-weight: bold; color: #E0E0E0; font-size: 12px;")
        self.action_label = QLabel(action)
        self.action_label.setStyleSheet("color: #888899; font-size: 11px;")
        text_layout.addWidget(self.name_label)
        text_layout.addWidget(self.action_label)

        self.badge_label = QLabel("")
        self.badge_label.setStyleSheet("font-size: 10px; font-weight: bold; padding: 2px 6px; border-radius: 4px;")
        self.badge_label.setVisible(False)

        layout.addWidget(self.icon_label)
        layout.addLayout(text_layout)
        layout.addStretch()
        layout.addWidget(self.badge_label)

        self.set_active(False)

    def set_active(self, is_active: bool, is_triggered: bool = False):
        if is_triggered:
            self.setStyleSheet(
                "GestureItemCard { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF9100, stop:1 #FF5722); "
                "border: 1px solid #FFA726; border-radius: 8px; }"
            )
            self.name_label.setStyleSheet("font-weight: bold; color: #FFFFFF; font-size: 12px;")
            self.action_label.setStyleSheet("color: #FFF3E0; font-size: 11px;")
            self.badge_label.setText("FIRED!")
            self.badge_label.setStyleSheet("background: #BF360C; color: white; border-radius: 4px; font-weight: bold;")
            self.badge_label.setVisible(True)
        elif is_active:
            self.setStyleSheet(
                "GestureItemCard { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1E3C72, stop:1 #2A5298); "
                "border: 1px solid #4FC3F7; border-radius: 8px; }"
            )
            self.name_label.setStyleSheet("font-weight: bold; color: #FFFFFF; font-size: 12px;")
            self.action_label.setStyleSheet("color: #B3E5FC; font-size: 11px;")
            self.badge_label.setText("ACTIVE")
            self.badge_label.setStyleSheet("background: #0288D1; color: white; border-radius: 4px; font-weight: bold;")
            self.badge_label.setVisible(True)
        else:
            self.setStyleSheet(
                "GestureItemCard { background: #1C1C24; border: 1px solid #2B2B38; border-radius: 8px; }"
            )
            self.name_label.setStyleSheet("font-weight: bold; color: #D0D0DD; font-size: 12px;")
            self.action_label.setStyleSheet("color: #888899; font-size: 11px;")
            self.badge_label.setVisible(False)


class FloatingGestureWidget(QWidget):
    """
    Compact, draggable, semi-transparent pill overlay that floats on top of all desktop apps.
    Shows the active gesture and status while you work in other software.
    """

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(260, 68)
        self._drag_pos = QPoint()

        # Layout
        container = QFrame(self)
        container.setObjectName("pillContainer")
        container.setStyleSheet(
            "#pillContainer { background-color: rgba(18, 20, 28, 225); "
            "border: 1px solid rgba(80, 140, 255, 120); border-radius: 14px; }"
        )
        layout = QHBoxLayout(container)
        layout.setContentsMargins(12, 8, 12, 8)

        self.icon_label = QLabel("🖐️")
        self.icon_label.setFont(QFont("Segoe UI Emoji", 22))

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        self.title_label = QLabel("IDLE")
        self.title_label.setStyleSheet("color: #00E676; font-weight: bold; font-size: 13px;")
        self.desc_label = QLabel("Waiting for hand...")
        self.desc_label.setStyleSheet("color: #AAAAAA; font-size: 11px;")
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.desc_label)

        self.state_badge = QLabel("IDLE")
        self.state_badge.setStyleSheet(
            "background: #333344; color: #DDD; font-size: 10px; font-weight: bold; "
            "padding: 3px 6px; border-radius: 6px;"
        )

        layout.addWidget(self.icon_label)
        layout.addLayout(text_layout)
        layout.addStretch()
        layout.addWidget(self.state_badge)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(container)

        # Position at bottom-right of screen
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - 290, screen.height() - 120)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def update_status(self, gesture_token: Optional[str], fsm_state: str, is_pinched: bool):
        # Determine icon & label
        icon = "🖐️"
        title = "NO HAND"
        desc = "Move hand into active zone"

        if is_pinched:
            icon = "👌"
            title = "PINCH DOWN"
            desc = "Click & Drag active"
        elif gesture_token:
            for g in GESTURE_CATALOG:
                if g["token"] == gesture_token:
                    icon = g["icon"]
                    title = g["name"]
                    desc = g["action"]
                    break
        elif fsm_state == "ARMED":
            icon = "⚡"
            title = "SYSTEM ARMED"
            desc = "Ready for gesture"

        self.icon_label.setText(icon)
        self.title_label.setText(title)
        self.desc_label.setText(desc)
        self.state_badge.setText(fsm_state)

        # Badge colors
        if fsm_state == "ARMED":
            self.state_badge.setStyleSheet("background: #00E676; color: #000; font-size: 10px; font-weight: bold; padding: 3px 6px; border-radius: 6px;")
        elif fsm_state == "TRIGGERED":
            self.state_badge.setStyleSheet("background: #FF9100; color: #000; font-size: 10px; font-weight: bold; padding: 3px 6px; border-radius: 6px;")
        elif fsm_state == "COOLDOWN":
            self.state_badge.setStyleSheet("background: #FF5252; color: #FFF; font-size: 10px; font-weight: bold; padding: 3px 6px; border-radius: 6px;")
        else:
            self.state_badge.setStyleSheet("background: #333344; color: #AAA; font-size: 10px; font-weight: bold; padding: 3px 6px; border-radius: 6px;")


class HUDOverlayWindow(QMainWindow):
    """
    Main GestureFlow Dashboard:
      - Video & Skeleton preview on the left
      - Real-Time Gesture status, clutch gauge, active cards & event history on the right.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GestureFlow - Gesture Monitor & Telemetry Dashboard")
        self.resize(1020, 560)
        self.setStyleSheet("QMainWindow { background-color: #121218; }")

        # Center on primary screen
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            x = max(20, (geom.width() - 1020) // 2)
            y = max(20, (geom.height() - 560) // 2)
            self.move(x, y)

        # Floating Mini Widget
        self.floating_widget = FloatingGestureWidget()

        # Build UI Layout
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(14)

        # ---------------- Left: Video HUD ----------------
        video_col = QVBoxLayout()
        video_col.setSpacing(8)

        self.video_widget = VideoHUDWidget(self)
        video_col.addWidget(self.video_widget, stretch=1)

        # Under-camera controls
        controls_bar = QHBoxLayout()
        self.toggle_mini_btn = QPushButton("📌 Show Floating Mini-Display")
        self.toggle_mini_btn.setStyleSheet(
            "QPushButton { background: #262635; color: #D0D0FF; border: 1px solid #3E3E55; "
            "padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 500; }"
            "QPushButton:hover { background: #35354A; border-color: #58587A; }"
        )
        self.toggle_mini_btn.clicked.connect(self._toggle_floating_widget)
        controls_bar.addWidget(self.toggle_mini_btn)
        controls_bar.addStretch()

        self.status_mini_label = QLabel("FPS: -- | Latency: --")
        self.status_mini_label.setStyleSheet("color: #888899; font-size: 12px;")
        controls_bar.addWidget(self.status_mini_label)

        video_col.addLayout(controls_bar)
        main_layout.addLayout(video_col, stretch=6)

        # ---------------- Right: Gesture Monitor Sidebar ----------------
        sidebar = QVBoxLayout()
        sidebar.setSpacing(10)

        # 1. Active Gesture Spotlight Card
        self.spotlight_card = QFrame()
        self.spotlight_card.setStyleSheet(
            "QFrame { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1A1A26, stop:1 #14141E); "
            "border: 1px solid #2B2B3E; border-radius: 12px; padding: 10px; }"
        )
        spot_layout = QVBoxLayout(self.spotlight_card)
        spot_layout.setContentsMargins(12, 10, 12, 10)
        spot_layout.setSpacing(6)

        spot_header = QHBoxLayout()
        spot_header_lbl = QLabel("CURRENT GESTURE")
        spot_header_lbl.setStyleSheet("color: #8888AA; font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        self.fsm_badge = QLabel("IDLE")
        self.fsm_badge.setStyleSheet(
            "background: #252538; color: #AAA; font-size: 11px; font-weight: bold; "
            "padding: 2px 8px; border-radius: 6px;"
        )
        spot_header.addWidget(spot_header_lbl)
        spot_header.addStretch()
        spot_header.addWidget(self.fsm_badge)
        spot_layout.addLayout(spot_header)

        # Big active pose row
        pose_row = QHBoxLayout()
        self.spot_icon = QLabel("🖐️")
        self.spot_icon.setFont(QFont("Segoe UI Emoji", 26))
        self.spot_name = QLabel("No Hand Detected")
        self.spot_name.setStyleSheet("color: #FFFFFF; font-size: 18px; font-weight: bold;")
        pose_row.addWidget(self.spot_icon)
        pose_row.addWidget(self.spot_name)
        pose_row.addStretch()
        spot_layout.addLayout(pose_row)

        self.spot_action = QLabel("Enter central 60% ROI box to begin")
        self.spot_action.setStyleSheet("color: #00E676; font-size: 12px;")
        spot_layout.addWidget(self.spot_action)

        # Clutch hold gauge
        clutch_row = QHBoxLayout()
        clutch_row.setSpacing(6)
        clutch_lbl = QLabel("Clutch Hold:")
        clutch_lbl.setStyleSheet("color: #888899; font-size: 11px;")
        self.clutch_bar = QProgressBar()
        self.clutch_bar.setFixedHeight(8)
        self.clutch_bar.setTextVisible(False)
        self.clutch_bar.setRange(0, 100)
        self.clutch_bar.setValue(0)
        self.clutch_bar.setStyleSheet(
            "QProgressBar { background: #222230; border-radius: 4px; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00B0FF, stop:1 #00E676); border-radius: 4px; }"
        )
        clutch_row.addWidget(clutch_lbl)
        clutch_row.addWidget(self.clutch_bar)
        spot_layout.addLayout(clutch_row)

        sidebar.addWidget(self.spotlight_card)

        # 2. Gesture Cheat Sheet / Live Catalog
        cat_header = QLabel("GESTURE REFERENCE GUIDE")
        cat_header.setStyleSheet("color: #8888AA; font-size: 11px; font-weight: 700; letter-spacing: 1px; margin-top: 4px;")
        sidebar.addWidget(cat_header)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        self.card_layout = QVBoxLayout(scroll_content)
        self.card_layout.setContentsMargins(0, 0, 4, 0)
        self.card_layout.setSpacing(6)

        self.gesture_cards: Dict[str, GestureItemCard] = {}
        for g in GESTURE_CATALOG:
            card = GestureItemCard(icon=g["icon"], name=g["name"], action=g["action"], token=g["token"])
            self.card_layout.addWidget(card)
            self.gesture_cards[g["token"]] = card

        scroll_area.setWidget(scroll_content)
        sidebar.addWidget(scroll_area, stretch=1)

        # 3. Recent Activity Log
        log_header = QLabel("RECENT ACTION DISPATCHES")
        log_header.setStyleSheet("color: #8888AA; font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        sidebar.addWidget(log_header)

        self.log_container = QFrame()
        self.log_container.setStyleSheet("background: #161620; border: 1px solid #242432; border-radius: 8px; padding: 6px;")
        self.log_layout = QVBoxLayout(self.log_container)
        self.log_layout.setContentsMargins(8, 6, 8, 6)
        self.log_layout.setSpacing(3)
        self.log_labels = []
        for _ in range(3):
            lbl = QLabel("• System initialized")
            lbl.setStyleSheet("color: #777788; font-size: 11px;")
            self.log_layout.addWidget(lbl)
            self.log_labels.append(lbl)

        sidebar.addWidget(self.log_container)
        main_layout.addLayout(sidebar, stretch=4)

        self.setCentralWidget(main_widget)
        self.recent_events: deque = deque(maxlen=3)

    def _toggle_floating_widget(self):
        if self.floating_widget.isVisible():
            self.floating_widget.hide()
            self.toggle_mini_btn.setText("📌 Show Floating Mini-Display")
        else:
            self.floating_widget.show()
            self.toggle_mini_btn.setText("📌 Hide Floating Mini-Display")

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
        """Draws skeleton, ROI box, and camera HUD onto frame."""
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # Central 60% ROI Box
        x_min, x_max, y_min, y_max = roi_box
        rx1, rx2 = int(x_min * w), int(x_max * w)
        ry1, ry2 = int(y_min * h), int(y_max * h)

        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
        cv2.putText(canvas, "ACTIVE 60% ROI", (rx1 + 5, ry1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        # Skeleton Joints & Bones
        if pixel_landmarks is not None and len(pixel_landmarks) == 21:
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                pt1 = (int(pixel_landmarks[p1_idx][0]), int(pixel_landmarks[p1_idx][1]))
                pt2 = (int(pixel_landmarks[p2_idx][0]), int(pixel_landmarks[p2_idx][1]))
                cv2.line(canvas, pt1, pt2, (0, 255, 128), 2, cv2.LINE_AA)

            for idx, pt in enumerate(pixel_landmarks):
                center = (int(pt[0]), int(pt[1]))
                if idx in (4, 8):
                    color = (0, 0, 255) if is_pinched else (255, 255, 0)
                    radius = 6 if is_pinched else 5
                else:
                    color = (0, 200, 255)
                    radius = 3
                cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)

        # Telemetry Banner Top
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        fps_text = f"FPS: {fps:.0f}"
        lat_text = f"Latency: {latency_ms:.1f}ms"
        state_color = (0, 255, 0) if fsm_state == "ARMED" else ((0, 140, 255) if fsm_state == "TRIGGERED" else (200, 200, 200))
        fsm_text = f"FSM: {fsm_state}"

        cv2.putText(canvas, fps_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, lat_text, (100, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, fsm_text, (240, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)

        if is_sleeping:
            cv2.putText(canvas, "[STANDBY 2 FPS]", (w - 180, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        return canvas

    def update_gesture_dashboard(
        self,
        current_gesture: Optional[str],
        fsm_state: str,
        clutch_progress: float,
        is_pinched: bool,
        fps: float,
        latency_ms: float,
        latest_event: Optional[str] = None,
    ):
        """Updates the sidebar spotlight, cards, clutch progress, and floating mini-display."""
        # 1. Update Spotlight Card
        active_token = "PINCH" if is_pinched else current_gesture
        matching_meta = None
        for g in GESTURE_CATALOG:
            if g["token"] == active_token:
                matching_meta = g
                break

        if matching_meta:
            self.spot_icon.setText(matching_meta["icon"])
            self.spot_name.setText(matching_meta["name"])
            self.spot_action.setText(matching_meta["action"])
        elif fsm_state == "ARMED":
            self.spot_icon.setText("⚡")
            self.spot_name.setText("System Armed")
            self.spot_action.setText("Ready: Perform any static pose or swipe")
        else:
            self.spot_icon.setText("🖐️")
            self.spot_name.setText("Idle / Ready")
            self.spot_action.setText("Hold Open Palm for 400ms to Wake/Clutch")

        # FSM Badge
        self.fsm_badge.setText(fsm_state)
        if fsm_state == "ARMED":
            self.fsm_badge.setStyleSheet("background: #00E676; color: #000; font-size: 11px; font-weight: bold; padding: 2px 8px; border-radius: 6px;")
        elif fsm_state == "TRIGGERED":
            self.fsm_badge.setStyleSheet("background: #FF9100; color: #000; font-size: 11px; font-weight: bold; padding: 2px 8px; border-radius: 6px;")
        elif fsm_state == "COOLDOWN":
            self.fsm_badge.setStyleSheet("background: #FF5252; color: #FFF; font-size: 11px; font-weight: bold; padding: 2px 8px; border-radius: 6px;")
        else:
            self.fsm_badge.setStyleSheet("background: #252538; color: #AAA; font-size: 11px; font-weight: bold; padding: 2px 8px; border-radius: 6px;")

        # Clutch progress bar
        self.clutch_bar.setValue(int(clutch_progress * 100))

        # 2. Highlight matching card in the cheat sheet
        for token, card in self.gesture_cards.items():
            is_active = (token == active_token)
            is_trig = is_active and (fsm_state == "TRIGGERED")
            card.set_active(is_active, is_triggered=is_trig)

        # 3. Update Activity Log
        if latest_event:
            ts = datetime.now().strftime("%H:%M:%S")
            self.recent_events.appendleft(f"{ts} - {latest_event}")
            for i, text in enumerate(self.recent_events):
                if i < len(self.log_labels):
                    self.log_labels[i].setText(f"• {text}")
                    self.log_labels[i].setStyleSheet("color: #00E676; font-size: 11px;" if i == 0 else "color: #777788; font-size: 11px;")

        # 4. Status footer
        self.status_mini_label.setText(f"FPS: {fps:.0f} | Latency: {latency_ms:.1f} ms")

        # 5. Floating widget sync
        if self.floating_widget.isVisible():
            self.floating_widget.update_status(active_token, fsm_state, is_pinched)

    def update_display(self, frame_annotated: np.ndarray):
        self.video_widget.update_frame(frame_annotated)

    def closeEvent(self, event):
        if hasattr(self, "floating_widget") and self.floating_widget:
            self.floating_widget.close()
        event.accept()


def create_tray_icon() -> QIcon:
    """Generates an in-memory crisp icon pixmap for the system tray."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    painter.setBrush(QColor(30, 144, 255))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(2, 2, 28, 28)

    painter.setPen(QPen(QColor(255, 255, 255), 2))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "GF")
    painter.end()
    return QIcon(pixmap)


class SystemTrayManager(QObject):
    """Manages the PyQt6 system tray icon and its context menu."""
    toggle_hud_requested = pyqtSignal()
    toggle_floating_requested = pyqtSignal()
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

        self.hud_action = QAction("Toggle Gesture Dashboard", self.tray_menu)
        self.hud_action.triggered.connect(self.toggle_hud_requested.emit)
        self.tray_menu.addAction(self.hud_action)

        self.floating_action = QAction("Toggle Floating Mini-Display", self.tray_menu)
        self.floating_action.triggered.connect(self.toggle_floating_requested.emit)
        self.tray_menu.addAction(self.floating_action)

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
