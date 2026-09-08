"""
ASTRA - Dashboard Starter Skeleton
--------------------------------------------------------
Run this to see a working window with:
  - Live webcam feed (left)
  - Step checklist panel (right) - currently using FAKE data
  - Next-step prompt banner
  - Alert log panel
  - Footer status bar (REC / STREAM indicators)

Next steps for your team:
  1. Replace `FAKE_SEQUENCE` logic with real StepTracker output
     from the Logic Engineer's module.
  2. Replace `draw_dummy_overlay()` with real YOLO + MediaPipe
     detection/pose overlays.
  3. Wire actual video recording (cv2.VideoWriter) and streaming
     (socket/RTSP) into `toggle_recording()` / `toggle_streaming()`.
"""

"""
ASTRA - Autonomous Spaceflight Tactical Real-time Assistant
Experiment: Liquid Mixing Detection & Microgravity Fluid Monitoring
Mission Control HUD / EVA AI Payload Operations
"""

import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QFrame, QTextEdit, QProgressBar
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor

MODEL_PATH = "../hazard-triage-experiment/runs_hazard_triage/v1/weights/best.pt"

# ---------------------------------------------------------------------
# EXPERIMENT SEQUENCE: 6-State Liquid Mixing Architecture
# ---------------------------------------------------------------------
EXPERIMENT_SEQUENCE = [
    {"step_id": "STATE 01", "label": "Empty Test Tube Baseline Verification", "signal": "YOLO + HSV"},
    {"step_id": "STATE 02", "label": "Bottle A Extraction & Target Acquired", "signal": "YOLO"},
    {"step_id": "STATE 03", "label": "Reagent A Dispensed (First Liquid In)", "signal": "HSV (Colour A)"},
    {"step_id": "STATE 04", "label": "Bottle B Extraction & Target Acquired", "signal": "YOLO"},
    {"step_id": "STATE 05", "label": "Reagent B Dispense & Pouring Action", "signal": "Geometry Overlap"},
    {"step_id": "STATE 06", "label": "Reagent Interaction & Mixture Complete", "signal": "HSV (Mixture)"},
]

HUD_ICONS = {
    "done": "[ VERIFIED ]",
    "warning": "[ ! WARNING ! ]",
    "current": "[ >> ACTIVE << ]",
    "pending": "[ STANDBY ]",
}

HUD_STYLESHEET = """
QWidget {
    background-color: #070B12;
    color: #E0F7FA;
    font-family: 'Consolas', 'Courier New', monospace;
}
QFrame.Card {
    background-color: #0E1626;
    border: 1px solid #00E5FF;
    border-radius: 4px;
}
QLabel {
    color: #80DEEA;
}
QListWidget {
    background-color: #0B111D;
    border: 1px solid #1A365D;
    border-radius: 4px;
    padding: 6px;
    color: #B0BEC5;
    font-size: 11px;
}
QListWidget::item {
    padding: 8px;
    border-bottom: 1px solid #132238;
}
QListWidget::item:selected {
    background-color: #003B46;
    color: #00E5FF;
}
QTextEdit {
    background-color: #05080E;
    border: 1px solid #00E5FF;
    color: #00FF66;
    font-size: 11px;
    font-family: 'Consolas', monospace;
}
QProgressBar {
    border: 1px solid #00E5FF;
    text-align: center;
    background-color: #0A0F1D;
    color: #FFFFFF;
    font-weight: bold;
    height: 16px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0052D4, stop:1 #00E5FF);
}
"""


class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        # Renamed window title as requested
        self.setWindowTitle("ASTRA - Astronaut AI assistance")
        self.resize(1280, 780)
        self.setStyleSheet(HUD_STYLESHEET)

        self.current_step_index = 0
        self.step_status = ["pending"] * len(EXPERIMENT_SEQUENCE)
        self.step_status[0] = "current"
        self.start_epoch = time.time()
        self.inference_fps = 0.0

        self._build_ui()

        # Load YOLO model
        try:
            self.model = YOLO(MODEL_PATH)
            self.log_event("SYSTEM", "YOLO neural weights initialized successfully.")
        except Exception as e:
            self.model = None
            self.log_event("SYS_ERR", f"Model weight loading bypassed: {e}")

        # Optical Stream Setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log_event("SENSOR_ERR", "Primary optical feed offline.")

        # Real-time Telemetry & Stream Loops
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_frame)
        self.video_timer.start(33)

        # Simulation Step Progressor (swap with real FSM hook)
        self.fake_progress_timer = QTimer()
        self.fake_progress_timer.timeout.connect(self.fake_advance_step)
        self.fake_progress_timer.start(4500)

    def _build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Top Header Bar
        top_bar = QHBoxLayout()
        header_title = QLabel("ASTRA // ASTRONAUT AI ASSISTANCE - FLUID MIXING HUD")
        header_title.setStyleSheet("font-size: 17px; font-weight: bold; color: #00E5FF; letter-spacing: 2px;")

        self.met_label = QLabel("MET 00:00:00 | UTC 00:00:00")
        self.met_label.setStyleSheet("color: #FFB300; font-weight: bold; font-size: 13px;")

        top_bar.addWidget(header_title)
        top_bar.addStretch()
        top_bar.addWidget(self.met_label)
        main_layout.addLayout(top_bar)

        # Center Workstation Layout
        center_layout = QHBoxLayout()

        # Left Column: Optical HUD Stream
        left_box = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(760, 480)
        self.video_label.setStyleSheet("border: 2px solid #00E5FF; background-color: #030508;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_box.addWidget(self.video_label)

        # Active Step Dynamic HUD Banner
        self.next_step_banner = QLabel("SYSTEM IDLE")
        self.next_step_banner.setStyleSheet(
            "background-color: #09203F; border-left: 4px solid #00E5FF; "
            "color: #FFFFFF; font-size: 13px; font-weight: bold; padding: 10px;"
        )
        left_box.addWidget(self.next_step_banner)
        center_layout.addLayout(left_box)

        # Right Column: Procedural Checklist & Diagnostics
        right_box = QVBoxLayout()

        lbl_task = QLabel("PROCEDURAL EXECUTION: 6-STATE PIPELINE")
        lbl_task.setStyleSheet("font-weight: bold; color: #00E5FF; font-size: 12px; letter-spacing: 1px;")
        right_box.addWidget(lbl_task)

        self.checklist = QListWidget()
        self._refresh_checklist()
        right_box.addWidget(self.checklist)

        # Mission Progress Indicator
        right_box.addWidget(QLabel("SEQUENCE VALIDATION PROGRESS:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(EXPERIMENT_SEQUENCE))
        self.progress_bar.setValue(0)
        right_box.addWidget(self.progress_bar)

        # Event Console
        lbl_alert = QLabel("EVENT TELEMETRY & MULTI-MODAL LOG")
        lbl_alert.setStyleSheet("font-weight: bold; color: #00E5FF; font-size: 12px; margin-top: 6px;")
        right_box.addWidget(lbl_alert)

        self.alert_log = QTextEdit()
        self.alert_log.setReadOnly(True)
        self.alert_log.setFixedHeight(120)
        right_box.addWidget(self.alert_log)

        center_layout.addLayout(right_box)
        main_layout.addLayout(center_layout)

        # Footer Status Matrix
        footer = QHBoxLayout()
        self.hud_optics_status = QLabel("[ SENSOR: OPTICAL_HD (RGB+HSV) ]")
        self.hud_optics_status.setStyleSheet("color: #00E5FF; font-weight: bold;")
        self.hud_fsm_status = QLabel("[ FSM: DETERMINISTIC MONOTONIC ]")
        self.hud_fsm_status.setStyleSheet("color: #00FF66; font-weight: bold;")
        self.hud_telemetry = QLabel("[ FPS: -- | LATENCY: nominal ]")
        self.hud_telemetry.setStyleSheet("color: #B0BEC5;")

        footer.addWidget(self.hud_optics_status)
        footer.addWidget(self.hud_fsm_status)
        footer.addWidget(self.hud_telemetry)
        footer.addStretch()
        main_layout.addLayout(footer)

        self.setLayout(main_layout)
        self._update_next_step_banner()

    def draw_sci_fi_hud(self, frame):
        """Renders aerospace reticles, corner brackets, and on-feed metrics."""
        h, w, _ = frame.shape
        color_hud = (255, 229, 0)  # Cyan in BGR

        # Reticle Center Crosshair
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), color_hud, 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), color_hud, 1)
        cv2.circle(frame, (cx, cy), 35, color_hud, 1)

        # Corner Brackets
        margin = 15
        length = 25
        # Top-Left
        cv2.line(frame, (margin, margin), (margin + length, margin), color_hud, 2)
        cv2.line(frame, (margin, margin), (margin, margin + length), color_hud, 2)
        # Top-Right
        cv2.line(frame, (w - margin, margin), (w - margin - length, margin), color_hud, 2)
        cv2.line(frame, (w - margin, margin), (w - margin, margin + length), color_hud, 2)
        # Bottom-Left
        cv2.line(frame, (margin, h - margin), (margin + length, h - margin), color_hud, 2)
        cv2.line(frame, (margin, h - margin), (margin, h - margin - length), color_hud, 2)
        # Bottom-Right
        cv2.line(frame, (w - margin, h - margin), (w - margin - length, h - margin), color_hud, 2)
        cv2.line(frame, (w - margin, h - margin), (w - margin, h - margin - length), color_hud, 2)

        # Live Telemetry Overlays
        cv2.putText(frame, "TRACKING: TUBE_ROI + REAGENT BOTTLES", (margin + 10, margin + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_hud, 1)
        cv2.putText(frame, f"INFERENCE: {self.inference_fps:.1f} FPS", (margin + 10, h - margin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 100), 1)

        return frame

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        t_start = time.time()
        if self.model:
            results = self.model(frame, verbose=False)
            frame = results[0].plot()

        t_delta = time.time() - t_start
        self.inference_fps = (1.0 / t_delta) if t_delta > 0 else 30.0

        # Apply HUD overlay
        frame = self.draw_sci_fi_hud(frame)

        # Convert frame to QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)

        # Update telemetry stamps
        met_seconds = int(time.time() - self.start_epoch)
        met_str = time.strftime('%H:%M:%S', time.gmtime(met_seconds))
        utc_str = time.strftime('%H:%M:%S UTC', time.gmtime())
        self.met_label.setText(f"MET: {met_str} | SYS: {utc_str}")
        self.hud_telemetry.setText(f"[ FPS: {self.inference_fps:.1f} | SENSOR: CALIBRATED ]")

    def fake_advance_step(self):
        if self.current_step_index >= len(EXPERIMENT_SEQUENCE):
            return

        self.step_status[self.current_step_index] = "done"
        curr = EXPERIMENT_SEQUENCE[self.current_step_index]
        self.log_event("FSM_COMMIT", f"{curr['step_id']} VERIFIED [{curr['signal']}]")

        self.current_step_index += 1
        if self.current_step_index < len(EXPERIMENT_SEQUENCE):
            self.step_status[self.current_step_index] = "current"

        self.progress_bar.setValue(self.current_step_index)
        self._refresh_checklist()
        self._update_next_step_banner()

    def _refresh_checklist(self):
        self.checklist.clear()
        for i, step in enumerate(EXPERIMENT_SEQUENCE):
            status = self.step_status[i]
            prefix = HUD_ICONS.get(status, "[   ]")
            item = QListWidgetItem(f"{prefix} {step['step_id']}: {step['label']}")

            if status == "current":
                item.setForeground(QColor("#00E5FF"))
                item.setBackground(QColor("#0F2238"))
            elif status == "done":
                item.setForeground(QColor("#00FF66"))
            elif status == "warning":
                item.setForeground(QColor("#FF9100"))
            else:
                item.setForeground(QColor("#546E7A"))

            self.checklist.addItem(item)

    def _update_next_step_banner(self):
        if self.current_step_index < len(EXPERIMENT_SEQUENCE):
            step = EXPERIMENT_SEQUENCE[self.current_step_index]
            self.next_step_banner.setText(
                f"CURRENT DIRECTIVE >> {step['step_id']}: {step['label']}  |  SIGNAL: {step['signal']}"
            )
        else:
            self.next_step_banner.setText("PROCEDURE COMPLETE: REACTION STABILIZED & LOGGED")
            self.next_step_banner.setStyleSheet(
                "background-color: #003B26; border-left: 4px solid #00FF66; "
                "color: #00FF66; font-size: 13px; font-weight: bold; padding: 10px;"
            )

    def log_event(self, tag, message):
        timestamp = time.strftime("%H:%M:%S")
        self.alert_log.append(f"[{timestamp}] [{tag}] {message}")

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    sys.exit(app.exec_())