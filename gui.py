from __future__ import annotations
from llm_tools import LLMClient
from vlm import get_bbox_from_targets
from pid import PIDY, PIDXZ
from roarm import create_controller, RoArmPoseController
from vision import (
    open_camera,
    resize_for_preview,
    resize_for_inference,
    get_aruco_tcp_point,
    draw_aruco_marker_point,
    draw_detections_on_preview,
    draw_fps_overlay,
    update_smoothed_fps,
    anchor_point_from_mask,
    scale_bbox_from_inference,
    PREVIEW_SIZE,
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QTextCursor, QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QObject
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter, QScrollArea,
    QFrame, QSizePolicy, QComboBox, QDialog, QFormLayout,
    QDoubleSpinBox, QSpinBox, QTabWidget, QGroupBox, QDialogButtonBox,
)
from dataclasses import dataclass, field, fields
import glob
import json
from pathlib import Path
import numpy as np
import cv2
import signal
import time
import threading
import math
import sys

import os
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""


@dataclass
class Settings:
    camera_index: int = 2
    serial_port: str = "/dev/ttyUSB0"

    llm_model: str = ""

    control_period: float = 0.03
    y_sign: float = 1.0
    y_dead_zone: float = 12.0
    xz_dead_zone: float = 5.0

    open_gripper_reaching_px: float = 120.0
    close_gripper_reached_px: float = 65.0
    grasp_lift_clearance_mm: float = 80.0
    grasp_settle_sec: float = 3.0
    lift_settle_sec: float = 1.0
    gripper_open_t: float = 2.0
    gripper_close_t: float = 3.14

    reaching_timeout_sec: float = 40.0
    grasping_timeout_sec: float = 5.0
    lifting_timeout_sec: float = 5.0
    placing_timeout_sec: float = 30.0

    place_x_engage_px: float = 300.0
    place_stable_delta_px: float = 5.0
    place_stable_sec: float = 0.4
    place_settle_sec: float = 1.0

    _SETTINGS_PATH = Path(__file__).parent / "settings.json"

    def save(self) -> None:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        self._SETTINGS_PATH.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls) -> "Settings":
        if cls._SETTINGS_PATH.exists():
            try:
                data = json.loads(cls._SETTINGS_PATH.read_text())
                valid = {f.name for f in fields(cls)}
                return cls(**{k: v for k, v in data.items() if k in valid})
            except Exception:
                pass
        return cls()


def _list_serial_ports() -> list[str]:
    """List available serial ports on the system."""
    ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    if not ports:
        ports = ["/dev/ttyUSB0"]
    return ports


def _list_cameras() -> list[tuple[int, str]]:
    """List all /dev/video* devices with their names (includes virtual cams)."""
    devices: list[tuple[int, str]] = []
    for path in sorted(glob.glob("/dev/video*")):
        try:
            idx = int(path.replace("/dev/video", ""))
        except ValueError:
            continue
        name_path = f"/sys/class/video4linux/video{idx}/name"
        try:
            with open(name_path) as f:
                name = f.read().strip()
        except OSError:
            name = "Unknown"
        devices.append((idx, name))
    if not devices:
        devices = [(0, "Default")]
    return devices


PHASE_IDLE = "idle"
PHASE_REACHING = "reaching"
PHASE_GRASPING = "grasping"
PHASE_LIFTING = "lifting"
PHASE_LIFT_SETTLE = "lift_settle"
PHASE_PLACING = "placing"
PHASE_PLACE_SETTLE = "place_settle"


def _bboxes_intersect(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _filter_place_detections(detections: dict, vlm_targets: dict) -> dict:
    return detections


def _anchor_point_from_bbox(bbox: tuple, anchor: str) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if anchor == "top":
        return int(cx), int(y1)
    elif anchor == "bottom":
        return int(cx), int(y2)
    elif anchor == "left":
        return int(x1), int(cy)
    elif anchor == "right":
        return int(x2), int(cy)
    elif anchor == "top_left":
        return int(x1), int(y1)
    elif anchor == "top_right":
        return int(x2), int(y1)
    elif anchor == "bottom_left":
        return int(x1), int(y2)
    elif anchor == "bottom_right":
        return int(x2), int(y2)
    return int(cx), int(cy)


def _best_object_point_px(
    detections: dict,
    targets: dict[str, str],
) -> tuple[int, int] | None:
    for name, anchor in targets.items():
        item = detections.get(name)
        if item is None:
            continue
        bboxes, masks, scores = item
        if not scores:
            continue

        if bboxes:
            areas = [
                (b[2] - b[0]) * (b[3] - b[1]) for b in bboxes
            ]
            best_idx = int(np.argmin(areas))
        else:
            best_idx = 0

        if masks and best_idx < len(masks) and masks[best_idx] is not None:
            pt = anchor_point_from_mask(masks[best_idx], anchor, PREVIEW_SIZE)
            if pt is not None:
                return pt

        if bboxes and best_idx < len(bboxes):
            bbox = scale_bbox_from_inference(bboxes[best_idx], PREVIEW_SIZE)
            return _anchor_point_from_bbox(bbox, anchor)
    return None


_SETTING_LABELS = {
    "camera_index": "Camera Index",
    "serial_port": "Serial Port",
    "control_period": "Control Period (s)",
    "y_sign": "Y Sign",
    "y_dead_zone": "Y Dead Zone (px)",
    "xz_dead_zone": "XZ Dead Zone (px)",
    "open_gripper_reaching_px": "Open Gripper Reach (px)",
    "close_gripper_reached_px": "Close Gripper Reach (px)",
    "grasp_lift_clearance_mm": "Grasp Lift Clearance (mm)",
    "grasp_settle_sec": "Grasp Settle (s)",
    "lift_settle_sec": "Lift Settle (s)",
    "gripper_open_t": "Gripper Open (rad)",
    "gripper_close_t": "Gripper Close (rad)",
    "reaching_timeout_sec": "Reaching Timeout (s)",
    "grasping_timeout_sec": "Grasping Timeout (s)",
    "lifting_timeout_sec": "Lifting Timeout (s)",
    "placing_timeout_sec": "Placing Timeout (s)",
    "place_x_engage_px": "Place X Engage (px)",
    "place_stable_delta_px": "Place Stable Delta (px)",
    "place_stable_sec": "Place Stable (s)",
    "place_settle_sec": "Place Settle (s)",
}

_INPUT_STYLE = """
    QDoubleSpinBox, QSpinBox, QComboBox {
        background-color: #1a1b26;
        border: 1px solid #3b4261;
        border-radius: 4px;
        padding: 4px 8px;
        color: #c0caf5;
        font-size: 12px;
    }
    QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
        border-color: #7aa2f7;
    }
"""


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, llm_client=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedWidth(420)
        self.setStyleSheet("""
            QDialog { background-color: #1f2335; }
            QLabel { color: #c0caf5; font-size: 12px; }
            QTabWidget::pane { border: 1px solid #3b4261; border-radius: 4px; }
            QTabBar::tab {
                background: #292e42; color: #a9b1d6; padding: 6px 16px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background: #1f2335; color: #c0caf5; }
        """)

        self._settings = settings
        self._widgets: dict[str, QWidget] = {}

        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        device_widget = QWidget()
        device_form = QFormLayout(device_widget)
        device_form.setContentsMargins(12, 12, 12, 12)
        device_form.setSpacing(8)

        self._cam_combo = QComboBox()
        self._cam_combo.setStyleSheet(_INPUT_STYLE)
        cameras = _list_cameras()
        for idx, name in cameras:
            self._cam_combo.addItem(f"{idx}: {name}", idx)
        current_cam_idx = self._cam_combo.findData(settings.camera_index)
        if current_cam_idx >= 0:
            self._cam_combo.setCurrentIndex(current_cam_idx)
        device_form.addRow("Camera:", self._cam_combo)

        self._serial_combo = QComboBox()
        self._serial_combo.setStyleSheet(_INPUT_STYLE)
        self._serial_combo.setEditable(True)
        ports = _list_serial_ports()
        for p in ports:
            self._serial_combo.addItem(p)
        idx = self._serial_combo.findText(settings.serial_port)
        if idx >= 0:
            self._serial_combo.setCurrentIndex(idx)
        else:
            self._serial_combo.setEditText(settings.serial_port)
        device_form.addRow("Serial Port:", self._serial_combo)

        self._model_combo = QComboBox()
        self._model_combo.setStyleSheet(_INPUT_STYLE)
        self._model_combo.setEditable(True)
        if llm_client is not None:
            models = llm_client.list_models()
            for m in models:
                self._model_combo.addItem(m)
        current_model = settings.llm_model or (
            llm_client.model if llm_client else "")
        midx = self._model_combo.findText(current_model)
        if midx >= 0:
            self._model_combo.setCurrentIndex(midx)
        else:
            self._model_combo.setEditText(current_model)
        device_form.addRow("LLM Model:", self._model_combo)

        tabs.addTab(device_widget, "Device")

        control_widget = QWidget()
        control_form = QFormLayout(control_widget)
        control_form.setContentsMargins(12, 12, 12, 12)
        control_form.setSpacing(6)

        control_fields = [
            "control_period", "y_sign", "y_dead_zone", "xz_dead_zone",
            "open_gripper_reaching_px", "close_gripper_reached_px",
            "grasp_lift_clearance_mm", "grasp_settle_sec", "lift_settle_sec",
            "gripper_open_t", "gripper_close_t",
        ]
        for name in control_fields:
            spin = QDoubleSpinBox()
            spin.setStyleSheet(_INPUT_STYLE)
            spin.setDecimals(3)
            spin.setRange(-9999.0, 9999.0)
            spin.setSingleStep(0.1)
            spin.setValue(getattr(settings, name))
            control_form.addRow(_SETTING_LABELS.get(name, name) + ":", spin)
            self._widgets[name] = spin

        tabs.addTab(control_widget, "Control")

        timeout_widget = QWidget()
        timeout_form = QFormLayout(timeout_widget)
        timeout_form.setContentsMargins(12, 12, 12, 12)
        timeout_form.setSpacing(6)

        timeout_fields = [
            "reaching_timeout_sec", "grasping_timeout_sec",
            "lifting_timeout_sec", "placing_timeout_sec",
            "place_x_engage_px", "place_stable_delta_px",
            "place_stable_sec", "place_settle_sec",
        ]
        for name in timeout_fields:
            spin = QDoubleSpinBox()
            spin.setStyleSheet(_INPUT_STYLE)
            spin.setDecimals(2)
            spin.setRange(0.0, 9999.0)
            spin.setSingleStep(1.0)
            spin.setValue(getattr(settings, name))
            timeout_form.addRow(_SETTING_LABELS.get(name, name) + ":", spin)
            self._widgets[name] = spin

        tabs.addTab(timeout_widget, "Timeouts")

        layout.addWidget(tabs)

        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.setStyleSheet("""
            QPushButton {
                background-color: #7aa2f7; color: #1a1b26;
                font-weight: bold; border: none; border-radius: 4px;
                padding: 6px 16px; font-size: 12px;
            }
            QPushButton:hover { background-color: #89b4fa; }
        """)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_settings(self) -> Settings:
        """Return a new Settings with values from the dialog."""
        s = Settings()
        s.camera_index = self._cam_combo.currentData() or 0
        s.serial_port = self._serial_combo.currentText().strip() or "/dev/ttyUSB0"
        s.llm_model = self._model_combo.currentText().strip()
        for name, widget in self._widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                setattr(s, name, widget.value())
        return s


class RobotEngine(QObject):
    """Manages arm, camera, VLM detection, and PID control in a background thread."""

    frame_ready = pyqtSignal()
    phase_changed = pyqtSignal(str)
    execution_finished = pyqtSignal(str)
    log_message = pyqtSignal(str)
    no_video = pyqtSignal()
    no_serial = pyqtSignal(str)

    def __init__(self, settings: Settings | None = None, parent=None):
        super().__init__(parent)

        self.settings = settings or Settings()
        self._running = True
        self._executing = False
        self._stop_requested = False
        self._start_requested = False
        self._reset_requested = False
        self._camera_switch_requested = False

        self._vlm_targets: dict[str, str] = {}
        self._vlm_place_targets: dict[str, str] = {}
        self._targets_lock = threading.Lock()

        self._frame_lock = threading.Lock()
        self._latest_infer_frame: np.ndarray | None = None
        self._detection_lock = threading.Lock()
        self._latest_detections: dict = {}

        self._phase = PHASE_IDLE
        self._arm: RoArmPoseController | None = None

        self._preview_lock = threading.Lock()
        self._latest_preview: np.ndarray | None = None

    @property
    def phase(self) -> str:
        return self._phase

    def set_vlm_targets(self, targets: dict[str, str]):
        with self._targets_lock:
            self._vlm_targets = dict(targets)
        self.log_message.emit(f"Grasp targets set: {targets}")

    def set_vlm_place_targets(self, targets: dict[str, str]):
        with self._targets_lock:
            self._vlm_place_targets = dict(targets)
        self.log_message.emit(f"Place targets set: {targets}")

    def _clear_all_targets(self):
        """Reset all VLM targets and cached detections (called on idle)."""
        with self._targets_lock:
            self._vlm_targets.clear()
            self._vlm_place_targets.clear()
        with self._detection_lock:
            self._latest_detections.clear()

    def start_execution(self):
        if self._arm is None:
            self.log_message.emit("Cannot execute — no robot arm connected.")
            return
        if self._executing:
            self.log_message.emit("Already executing!")
            return
        self._start_requested = True

    def request_stop(self):
        self._stop_requested = True
        self.log_message.emit("Stop requested — returning home...")

    def reset_to_home(self):
        self._reset_requested = True

    def shutdown(self):
        self._running = False

    def _vlm_loop(self):
        while self._running:
            with self._frame_lock:
                frame = self._latest_infer_frame

            if frame is None:
                time.sleep(0.05)
                continue

            with self._targets_lock:
                all_names = list(set(
                    list(self._vlm_targets.keys()) +
                    list(self._vlm_place_targets.keys())
                ))

            if not all_names:
                time.sleep(0.1)
                continue

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = get_bbox_from_targets(rgb, all_names)
                with self._detection_lock:
                    self._latest_detections = detections
            except ConnectionRefusedError:
                time.sleep(1.0)
            except Exception as e:
                self.log_message.emit(f"[VLM] {e}")
                time.sleep(0.5)

    def run_loop(self):
        """Call from a QThread. Runs camera + PID + state machine."""
        s = self.settings

        try:
            self._arm = create_controller(serial_port=s.serial_port)
            self._arm.reset_to_home()
            self.log_message.emit(
                f"Arm initialized on {s.serial_port} and homed.")
        except RuntimeError as e:
            self.no_serial.emit(str(e))
            self.log_message.emit(f"[Serial] {e}")
            self._arm = None

        vlm_thread = threading.Thread(
            target=self._vlm_loop, daemon=True, name="vlm-loop")
        vlm_thread.start()

        try:
            cap = open_camera(camera_index=s.camera_index)
            self.log_message.emit(f"Camera {s.camera_index} opened.")
        except RuntimeError as e:
            self.no_video.emit()
            self.log_message.emit(f"[Camera] {e}")
            while self._running:
                time.sleep(0.1)
            return

        pid_y = PIDY()
        pid_xz = PIDXZ()

        target_x = PREVIEW_SIZE[0] // 2
        target_y = PREVIEW_SIZE[1] // 2

        phase_start = time.monotonic()
        grasp_settle_start: float | None = None
        lift_target_z: float | None = None
        lift_settle_start: float = 0.0
        place_settle_start: float = 0.0
        place_anchor_dist: float | None = None
        place_stable_start: float | None = None

        fps = 0.0
        prev_time = time.monotonic()

        try:
            while self._running:
                if self._camera_switch_requested:
                    self._camera_switch_requested = False
                    new_idx = self.settings.camera_index
                    self.log_message.emit(f"Switching to camera {new_idx}...")
                    cap.release()
                    try:
                        cap = open_camera(camera_index=new_idx)
                        self.log_message.emit(f"Camera {new_idx} opened.")
                    except RuntimeError as e:
                        self.log_message.emit(f"[Camera] {e}")
                        self.no_video.emit()
                        while self._running and not self._camera_switch_requested:
                            time.sleep(0.1)
                        continue

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                now = time.monotonic()
                dt = now - prev_time
                prev_time = now
                fps = update_smoothed_fps(fps, dt)

                preview = resize_for_preview(frame)
                tcp_point = get_aruco_tcp_point(preview)

                infer_frame = resize_for_inference(frame)
                with self._frame_lock:
                    self._latest_infer_frame = infer_frame.copy()

                with self._detection_lock:
                    detections = dict(self._latest_detections)

                with self._targets_lock:
                    vlm_targets = dict(self._vlm_targets)
                    vlm_place_targets = dict(self._vlm_place_targets)

                detections = _filter_place_detections(detections, vlm_targets)
                obj_point = _best_object_point_px(detections, vlm_targets)
                place_point = _best_object_point_px(
                    detections, vlm_place_targets)

                phase = self._phase

                if self._start_requested and not self._executing:
                    self._start_requested = False
                    self._stop_requested = False
                    self._executing = True
                    phase = PHASE_REACHING
                    phase_start = time.monotonic()
                    self.phase_changed.emit(phase)
                    self.log_message.emit(
                        "Execution started \u2014 reaching for object...")

                if self._reset_requested:
                    self._reset_requested = False
                    if self._arm:
                        self._arm.reset_to_home()
                    pid_y = PIDY()
                    pid_xz = PIDXZ()
                    phase = PHASE_IDLE
                    self._executing = False
                    self._stop_requested = False
                    self._start_requested = False
                    self._clear_all_targets()
                    self.phase_changed.emit(phase)
                    self.log_message.emit("Arm reset to home position.")

                if self._stop_requested and self._executing:
                    if self._arm:
                        self._arm.reset_to_home()
                    pid_y = PIDY()
                    pid_xz = PIDXZ()
                    phase = PHASE_IDLE
                    self._executing = False
                    self._stop_requested = False
                    self._clear_all_targets()
                    self.phase_changed.emit(phase)
                    self.execution_finished.emit("stopped")
                    self.log_message.emit("Stopped — arm returned home.")

                if phase in (PHASE_PLACING,):
                    goal_x, goal_y = (
                        place_point if place_point else (None, None))
                else:
                    goal_x, goal_y = (obj_point if obj_point else (None, None))

                if isinstance((goal_x, goal_y), tuple) and goal_x is None:
                    goal_x, goal_y = None, None

                dist_px: float | None = None
                if tcp_point is not None and goal_x is not None and dt > 0:
                    error_x = goal_x - tcp_point[0]
                    error_y = goal_y - tcp_point[1]
                    dist_px = math.hypot(error_x, error_y)

                    if phase in (PHASE_REACHING, PHASE_PLACING) and self._arm:
                        moved = False
                        if abs(error_x) > s.y_dead_zone:
                            dy = pid_y.update(error_x, dt) * s.y_sign
                            self._arm.step("y", dy)
                            moved = True
                        if abs(error_y) > s.xz_dead_zone:
                            dx, dz = pid_xz.update(error_y, dt)
                            if phase == PHASE_PLACING:
                                if abs(error_x) < s.place_x_engage_px:
                                    self._arm.step("x", dx)
                                    moved = True
                            else:
                                self._arm.step("x", dx)
                                self._arm.step("z", dz)
                                moved = True
                        if moved:
                            self._arm.send_pose()

                    cv2.drawMarker(preview, (goal_x, goal_y),
                                   (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                    cv2.line(preview, tcp_point,
                             (goal_x, goal_y), (255, 255, 0), 1)
                    cv2.putText(preview, f"{dist_px:.0f}px",
                                (tcp_point[0] + 10, tcp_point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

                elapsed = time.monotonic() - phase_start

                if phase == PHASE_REACHING and self._arm:
                    if elapsed > s.reaching_timeout_sec:
                        self._arm.reset_to_home()
                        pid_y = PIDY()
                        pid_xz = PIDXZ()
                        phase = PHASE_REACHING
                        phase_start = time.monotonic()
                    elif dist_px is not None and dist_px < s.close_gripper_reached_px:
                        self._arm.step_gripper(s.gripper_close_t)
                        self._arm.send_pose()
                        grasp_settle_start = time.monotonic()
                        phase = PHASE_GRASPING
                        phase_start = time.monotonic()
                    elif dist_px is not None and dist_px < s.open_gripper_reaching_px:
                        self._arm.step_gripper(s.gripper_open_t)
                        self._arm.send_pose()

                elif phase == PHASE_GRASPING and self._arm:
                    if elapsed > s.grasping_timeout_sec:
                        self._arm.step_gripper(s.gripper_open_t)
                        self._arm.send_pose()
                        self._arm.reset_to_home()
                        pid_y = PIDY()
                        pid_xz = PIDXZ()
                        phase = PHASE_IDLE
                        phase_start = time.monotonic()
                        self._executing = False
                        self._clear_all_targets()
                        self.execution_finished.emit("failed: grasp timed out")
                    elif grasp_settle_start and time.monotonic() - grasp_settle_start >= s.grasp_settle_sec:
                        _, _, cur_z, _ = self._arm.current_pose()
                        lift_target_z = cur_z + s.grasp_lift_clearance_mm
                        phase = PHASE_LIFTING
                        phase_start = time.monotonic()

                elif phase == PHASE_LIFTING and self._arm:
                    if elapsed > s.lifting_timeout_sec:
                        phase = PHASE_REACHING
                        phase_start = time.monotonic()
                    else:
                        _, _, cur_z, _ = self._arm.current_pose()
                        if lift_target_z is not None and cur_z < lift_target_z:
                            self._arm.step(
                                "z", min(2.0, lift_target_z - cur_z))
                            self._arm.send_pose()
                        else:
                            lift_target_z = None
                            lift_settle_start = time.monotonic()
                            phase = PHASE_LIFT_SETTLE
                            phase_start = time.monotonic()

                elif phase == PHASE_LIFT_SETTLE:
                    if time.monotonic() - lift_settle_start >= s.lift_settle_sec:
                        pid_y = PIDY()
                        pid_xz = PIDXZ()
                        place_anchor_dist = None
                        place_stable_start = None
                        phase = PHASE_PLACING
                        phase_start = time.monotonic()

                elif phase == PHASE_PLACING and self._arm:
                    if elapsed > s.placing_timeout_sec:
                        self._arm.step_gripper(s.gripper_open_t)
                        self._arm.send_pose()
                        self._arm.reset_to_home()
                        pid_y = PIDY()
                        pid_xz = PIDXZ()
                        phase = PHASE_REACHING
                        phase_start = time.monotonic()
                    elif dist_px is not None and dist_px < s.place_x_engage_px:
                        if place_anchor_dist is None:
                            place_anchor_dist = dist_px
                            place_stable_start = time.monotonic()
                        elif abs(dist_px - place_anchor_dist) < s.place_stable_delta_px:
                            if time.monotonic() - place_stable_start >= s.place_stable_sec:
                                self._arm.step_gripper(s.gripper_open_t)
                                self._arm.send_pose()
                                place_settle_start = time.monotonic()
                                place_stable_start = None
                                place_anchor_dist = None
                                phase = PHASE_PLACE_SETTLE
                                phase_start = time.monotonic()
                        else:
                            place_anchor_dist = dist_px
                            place_stable_start = time.monotonic()

                elif phase == PHASE_PLACE_SETTLE and self._arm:
                    if time.monotonic() - place_settle_start >= s.place_settle_sec:
                        self._arm.reset_to_home()
                        pid_y = PIDY()
                        pid_xz = PIDXZ()
                        phase = PHASE_IDLE
                        phase_start = time.monotonic()
                        self._executing = False
                        self._clear_all_targets()
                        self.execution_finished.emit("success")
                        self.log_message.emit("Pick-and-place complete.")

                self._phase = phase
                self.phase_changed.emit(phase)

                if detections:
                    preview = draw_detections_on_preview(preview, detections)
                if obj_point is not None:
                    cv2.circle(preview, obj_point, 6, (0, 200, 255), -1)
                    cv2.circle(preview, obj_point, 6, (0, 0, 0), 1)
                preview = draw_aruco_marker_point(preview)
                if phase in (PHASE_PLACING,) and place_point is not None:
                    cv2.circle(preview, place_point, 8, (0, 0, 255), -1)
                    cv2.circle(preview, place_point, 8, (255, 255, 255), 1)

                phase_color = {
                    PHASE_IDLE: (200, 200, 200), PHASE_REACHING: (0, 255, 255),
                    PHASE_GRASPING: (0, 165, 255), PHASE_LIFTING: (0, 255, 0),
                    PHASE_LIFT_SETTLE: (180, 180, 0), PHASE_PLACING: (255, 0, 255),
                    PHASE_PLACE_SETTLE: (128, 0, 128),
                }.get(phase, (200, 200, 200))
                cv2.putText(preview, f"Phase: {phase}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2, cv2.LINE_AA)
                preview = draw_fps_overlay(preview, fps)

                with self._preview_lock:
                    self._latest_preview = preview
                self.frame_ready.emit()

                time.sleep(0.001)

        finally:
            cap.release()


class EngineThread(QThread):
    def __init__(self, engine: RobotEngine):
        super().__init__()
        self.engine = engine

    def run(self):
        self.engine.run_loop()


class LLMWorker(QThread):
    text_delta = pyqtSignal(str)
    tool_call_signal = pyqtSignal(str, dict, str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    awaiting_tools = pyqtSignal()

    def __init__(self, llm: LLMClient, user_text: str = "", is_continuation: bool = False):
        super().__init__()
        self.llm = llm
        self.user_text = user_text
        self.is_continuation = is_continuation

    def run(self):
        try:
            if self.is_continuation:
                gen = self.llm.continue_after_tools()
            else:
                gen = self.llm.send_message(self.user_text)

            for event in gen:
                if event["type"] == "text_delta":
                    self.text_delta.emit(event["content"])
                elif event["type"] == "tool_call":
                    self.tool_call_signal.emit(
                        event["name"], event["arguments"], event["tool_call_id"])
                elif event["type"] == "awaiting_tool_results":
                    self.awaiting_tools.emit()
                    return
                elif event["type"] == "error":
                    self.error.emit(event["content"])
                elif event["type"] == "done":
                    pass
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class ChatBubble(QFrame):
    def __init__(self, text: str, role: str, parent=None):
        super().__init__(parent)
        self.role = role

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)

        role_label = QLabel("You" if role == "user" else "Assistant")
        role_label.setFont(QFont("Segoe UI", 8, QFont.Bold))

        if role == "user":
            role_label.setStyleSheet("color: #7aa2f7;")
            self.setStyleSheet("""
                ChatBubble {
                    background-color: #1a1b26;
                    border: 1px solid #292e42;
                    border-radius: 12px;
                }
            """)
        else:
            role_label.setStyleSheet("color: #9ece6a;")
            self.setStyleSheet("""
                ChatBubble {
                    background-color: #1f2335;
                    border: 1px solid #292e42;
                    border-radius: 12px;
                }
            """)

        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setTextFormat(Qt.PlainText)
        self.text_label.setStyleSheet("color: #c0caf5; font-size: 13px;")
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        layout.addWidget(role_label)
        layout.addWidget(self.text_label)

    def append_text(self, text: str):
        self.text_label.setText(self.text_label.text() + text)


class ToolCallBubble(QFrame):
    def __init__(self, name: str, args: dict, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            ToolCallBubble {
                background-color: #292e42;
                border: 1px solid #3b4261;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        header = QLabel(f"Tool: {name}")
        header.setFont(QFont("Segoe UI", 9, QFont.Bold))
        header.setStyleSheet("color: #bb9af7;")

        if args:
            import json
            args_text = json.dumps(args, indent=2)
        else:
            args_text = "(no arguments)"
        body = QLabel(args_text)
        body.setWordWrap(True)
        body.setStyleSheet(
            "color: #a9b1d6; font-size: 11px; font-family: monospace;")

        layout.addWidget(header)
        layout.addWidget(body)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RoArm PD — Controller")
        self.setMinimumSize(1400, 750)
        self.setStyleSheet("""
            QMainWindow { background-color: #16161e; }
            QWidget { background-color: #16161e; color: #c0caf5; }
        """)

        self.llm = LLMClient()
        self._llm_worker: LLMWorker | None = None
        self._pending_tool_calls: list[tuple[str, dict, str]] = []
        self._current_assistant_bubble: ChatBubble | None = None

        self._is_busy = False
        self._settings = Settings.load()
        if not self._settings.llm_model:
            self._settings.llm_model = self.llm.model

        self.engine = RobotEngine(settings=self._settings)
        self.engine.frame_ready.connect(self._update_frame)
        self.engine.phase_changed.connect(self._update_phase)
        self.engine.execution_finished.connect(self._on_execution_finished)
        self.engine.log_message.connect(self._add_system_message)
        self.engine.no_video.connect(self._on_no_video)
        self.engine.no_serial.connect(self._on_no_serial)

        self.engine_thread = EngineThread(self.engine)
        self.engine_thread.start()

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(4)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1a1b26;
                border: 1px solid #292e42;
                border-radius: 8px;
                color: #565f89;
                font-size: 18px;
                qproperty-alignment: 'AlignCenter';
            }
        """)
        self.video_label.setText("No Video Available")

        controls_bar = QWidget()
        controls_layout = QHBoxLayout(controls_bar)
        controls_layout.setContentsMargins(4, 4, 4, 4)

        self.phase_label = QLabel("Phase: idle")
        self.phase_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.phase_label.setStyleSheet("color: #7aa2f7;")

        self.home_btn = QPushButton("HOME")
        self.home_btn.setFixedSize(90, 32)
        self.home_btn.setStyleSheet("""
            QPushButton {
                background-color: #7aa2f7;
                color: #1a1b26;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #89b4fa; }
            QPushButton:pressed { background-color: #5d7ec7; }
        """)
        self.home_btn.clicked.connect(self._on_home)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setFixedSize(90, 32)
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #292e42;
                color: #a9b1d6;
                font-weight: bold;
                border: 1px solid #3b4261;
                border-radius: 6px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #3b4261; }
            QPushButton:pressed { background-color: #1a1b26; }
        """)
        self.settings_btn.clicked.connect(self._on_settings)

        controls_layout.addWidget(self.phase_label)
        controls_layout.addStretch()
        controls_layout.addWidget(self.settings_btn)
        controls_layout.addWidget(self.home_btn)

        video_layout.addWidget(self.video_label, stretch=1)
        video_layout.addWidget(controls_bar)

        chat_container = QWidget()
        chat_container.setFixedWidth(380)
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(4)

        chat_header = QWidget()
        header_layout = QHBoxLayout(chat_header)
        header_layout.setContentsMargins(8, 4, 8, 4)

        title = QLabel("Chat")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: #c0caf5;")

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedSize(60, 26)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #292e42;
                color: #a9b1d6;
                border: 1px solid #3b4261;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #3b4261; }
        """)
        self.clear_btn.clicked.connect(self._clear_chat)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.clear_btn)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #1a1b26;
                border: 1px solid #292e42;
                border-radius: 8px;
            }
            QScrollBar:vertical {
                background: #1a1b26;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #3b4261;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.setContentsMargins(8, 8, 8, 8)
        self.chat_layout.setSpacing(8)

        self._empty_stretch_top = self.chat_layout.count()
        self.chat_layout.addStretch(1)

        self._empty_label = QLabel(
            "No messages yet.\nChat a command to get started.")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #565f89; font-size: 13px;")
        self.chat_layout.addWidget(self._empty_label)

        self.chat_layout.addStretch(1)
        self.scroll_area.setWidget(self.chat_widget)

        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(4)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(
            "Chat your command... e.g. 'pick the cube and place into box'")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #1a1b26;
                border: 1px solid #3b4261;
                border-radius: 8px;
                padding: 8px 12px;
                color: #c0caf5;
                font-size: 13px;
            }
            QLineEdit:focus { border-color: #7aa2f7; }
        """)
        self.input_field.returnPressed.connect(self._on_send_stop_clicked)

        self._send_style = """
            QPushButton {
                background-color: #7aa2f7;
                color: #1a1b26;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #89b4fa; }
        """
        self._stop_style = """
            QPushButton {
                background-color: #f7768e;
                color: #1a1b26;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #ff9e64; }
        """

        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedSize(60, 36)
        self.send_btn.setStyleSheet(self._send_style)
        self.send_btn.clicked.connect(self._on_send_stop_clicked)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)

        chat_layout.addWidget(chat_header)
        chat_layout.addWidget(self.scroll_area, stretch=1)
        chat_layout.addWidget(input_container)

        main_layout.addWidget(video_container, stretch=1)
        main_layout.addWidget(chat_container)

    def _update_frame(self):
        with self.engine._preview_lock:
            frame = self.engine._latest_preview
        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        scaled = qimg.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def _update_phase(self, phase: str):
        colors = {
            PHASE_IDLE: "#a9b1d6", PHASE_REACHING: "#e0af68",
            PHASE_GRASPING: "#ff9e64", PHASE_LIFTING: "#9ece6a",
            PHASE_LIFT_SETTLE: "#73daca", PHASE_PLACING: "#bb9af7",
            PHASE_PLACE_SETTLE: "#7dcfff",
        }
        color = colors.get(phase, "#a9b1d6")
        self.phase_label.setText(f"Phase: {phase}")
        self.phase_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _scroll_to_bottom(self):
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()))

    def _add_bubble(self, text: str, role: str) -> ChatBubble:
        self._empty_label.hide()
        bubble = ChatBubble(text, role)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        self._scroll_to_bottom()
        return bubble

    def _add_tool_bubble(self, name: str, args: dict):
        bubble = ToolCallBubble(name, args)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def _add_system_message(self, text: str):
        self._empty_label.hide()
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("""
            color: #565f89;
            font-size: 11px;
            font-style: italic;
            padding: 2px 8px;
        """)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, label)
        self._scroll_to_bottom()

    def _clear_chat(self):
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            w = item.widget()
            if w and w is not self._empty_label:
                w.deleteLater()
        self.chat_layout.insertWidget(0, self._empty_label)
        self._empty_label.show()
        self.llm.reset_history()

    def _on_send_stop_clicked(self):
        if self._is_busy:
            self._do_stop()
        else:
            self._do_send()

    def _do_send(self):
        text = self.input_field.text().strip()
        if not text:
            return
        self.input_field.clear()
        self._set_busy(True)

        if not self.llm.get_history():
            self._clear_chat()

        self._add_bubble(text, "user")
        self._current_assistant_bubble = None
        self._pending_tool_calls = []

        self._llm_worker = LLMWorker(self.llm, user_text=text)
        self._connect_llm_worker(self._llm_worker)
        self._llm_worker.start()

    def _do_stop(self):
        """Stop everything: halt LLM, stop execution, reset arm home."""
        self.engine.request_stop()
        self._add_system_message("Stopped by user.")
        self._set_busy(False)

    def _set_busy(self, busy: bool):
        self._is_busy = busy
        if busy:
            self.send_btn.setText("Stop")
            self.send_btn.setStyleSheet(self._stop_style)
            self.input_field.setEnabled(False)
        else:
            self.send_btn.setText("Send")
            self.send_btn.setStyleSheet(self._send_style)
            self.input_field.setEnabled(True)
            self.input_field.setFocus()

    def _connect_llm_worker(self, worker: LLMWorker):
        worker.text_delta.connect(self._on_text_delta)
        worker.tool_call_signal.connect(self._on_tool_call)
        worker.awaiting_tools.connect(self._on_awaiting_tools)
        worker.finished.connect(self._on_llm_finished)
        worker.error.connect(self._on_llm_error)

    def _on_text_delta(self, text: str):
        if self._current_assistant_bubble is None:
            self._current_assistant_bubble = self._add_bubble("", "assistant")
        self._current_assistant_bubble.append_text(text)
        self._scroll_to_bottom()

    def _on_tool_call(self, name: str, args: dict, tool_call_id: str):
        self._add_tool_bubble(name, args)
        self._pending_tool_calls.append((name, args, tool_call_id))

    def _on_awaiting_tools(self):
        """Execute pending tool calls, inject results, then continue LLM."""
        for name, args, tool_call_id in self._pending_tool_calls:
            result = self._execute_tool(name, args)
            self.llm.inject_tool_result(tool_call_id, result)
            self._add_system_message(f"Result: {result}")

        self._pending_tool_calls = []
        self._current_assistant_bubble = None

        worker = LLMWorker(self.llm, is_continuation=True)
        self._connect_llm_worker(worker)
        self._llm_worker = worker
        worker.start()

    def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool call and return result string."""
        try:
            if name == "set_vlm_targets":
                targets = args.get("targets", {})
                self.engine.set_vlm_targets(targets)
                return f"Grasp targets set: {targets}"

            elif name == "set_vlm_place_targets":
                targets = args.get("targets", {})
                self.engine.set_vlm_place_targets(targets)
                return f"Place targets set: {targets}"

            elif name == "start_execution":
                self.engine.start_execution()
                return "Execution started. The arm is now reaching for the object."

            elif name == "reset_to_home":
                self.engine.reset_to_home()
                return "Arm has been reset to home position."

            elif name == "get_execution_status":
                phase = self.engine.phase
                executing = self.engine._executing
                return (
                    f"Phase: {phase}, Executing: {executing}"
                )

            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error executing {name}: {e}"

    def _on_llm_finished(self):
        if not self.engine._executing:
            self._set_busy(False)
        self._current_assistant_bubble = None

    def _on_llm_error(self, error: str):
        self._add_system_message(f"Error: {error}")
        self._set_busy(False)

    def _on_execution_finished(self, result: str):
        self._add_system_message(f"Execution finished: {result}")

        if result == "stopped":
            feedback = "Execution was stopped by the user. Arm returned home."
        elif result == "success":
            feedback = (
                "Pick-and-place completed successfully. "
                "The object was grasped, lifted, moved to the target, and released. "
                "Arm has returned home. "
                "If there are remaining tasks from the user's request, "
                "proceed with the next task now."
            )
        else:
            feedback = f"Execution ended with problem: {result}. Arm returned home."

        self.llm.messages.append({"role": "user", "content": feedback})
        self._current_assistant_bubble = None

        worker = LLMWorker(self.llm, is_continuation=False)
        worker.user_text = ""
        worker.is_continuation = True
        self._connect_llm_worker(worker)
        self._llm_worker = worker
        worker.start()

    def _on_no_video(self):
        self.video_label.setText("No Video Available\n\nCamera not detected.")

    def _on_no_serial(self, msg: str):
        self._add_system_message(f"No robot arm connected: {msg}")

    def _on_home(self):
        self.engine.reset_to_home()

    def _on_settings(self):
        dlg = SettingsDialog(self._settings, llm_client=self.llm, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            new = dlg.get_settings()
            if new.llm_model and new.llm_model != self.llm.model:
                self.llm.model = new.llm_model
                self._add_system_message(f"Model changed to: {new.llm_model}")

            camera_changed = new.camera_index != self._settings.camera_index
            self._settings = new
            self.engine.settings = new
            self._add_system_message("Settings updated.")
            new.save()
            if camera_changed:
                self.engine._camera_switch_requested = True

    def closeEvent(self, event):
        self.engine.shutdown()
        self.engine_thread.quit()
        self.engine_thread.wait(5000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _sig_timer = QTimer()
    _sig_timer.start(200)
    _sig_timer.timeout.connect(lambda: None)

    from PyQt5.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(22, 22, 30))
    palette.setColor(QPalette.WindowText, QColor(192, 202, 245))
    palette.setColor(QPalette.Base, QColor(26, 27, 38))
    palette.setColor(QPalette.Text, QColor(192, 202, 245))
    palette.setColor(QPalette.Button, QColor(41, 46, 66))
    palette.setColor(QPalette.ButtonText, QColor(192, 202, 245))
    palette.setColor(QPalette.Highlight, QColor(122, 162, 247))
    app.setPalette(palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
