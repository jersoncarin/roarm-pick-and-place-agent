from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass
from typing import Any, Literal
import serial as pyserial

CMD_XYZT_GOAL_CTRL = 104
CMD_XYZT_DIRECT_CTRL = 1041
CMD_SERVO_RAD_FEEDBACK = 105
CMD_SERVO_RAD_FEEDBACK_RESPONSE = 1051
DEFAULT_MOVE_SPEED = 0.25


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def parse_json_message(message: Any) -> dict[str, Any] | None:
    if isinstance(message, dict):
        return message

    if not isinstance(message, str):
        return None

    text = message.strip()
    if not text:
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    return data if isinstance(data, dict) else None


def feedback_to_pose(message: Any) -> dict[str, float] | None:
    data = parse_json_message(message)
    if data is None or data.get("T") != CMD_SERVO_RAD_FEEDBACK_RESPONSE:
        return None

    try:
        return {
            "x": float(data["x"]),
            "y": float(data["y"]),
            "z": float(data["z"]),
            "t": float(data["t"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def build_xyzt_goal_payload(
    x: float,
    y: float,
    z: float,
    t: float,
    spd: float = DEFAULT_MOVE_SPEED,
) -> dict[str, float]:
    return {
        "T": CMD_XYZT_GOAL_CTRL,
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "z": round(float(z), 3),
        "t": round(float(t), 4),
        "spd": round(max(float(spd), 0.01), 3),
    }


@dataclass(frozen=True)
class RoArmTeleopConfig:
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout: float = 0.2

    move_speed: float = 0.25
    step_xz: float = 5.0
    step_y: float = -5.0
    step_t: float = 0.1

    x_limit = (120.0, 480.0)
    y_limit = (-180.0, 180.0)
    z_limit = (-125.0, 260.0)
    t_limit = (1.5, 3.14)

    base_x: float = 201.0
    base_y: float = -9.0
    base_z: float = -40.0
    base_t: float = 3.14


@dataclass
class Pose:
    x: float
    y: float
    z: float
    t: float


class RoArmPoseController:
    def __init__(self, config: RoArmTeleopConfig | None = None) -> None:
        self.config = config or RoArmTeleopConfig()
        self.pose = Pose(
            x=self.config.base_x,
            y=self.config.base_y,
            z=self.config.base_z,
            t=self.config.base_t
        )

        self._serial_lock = threading.Lock()

        try:
            self.serial = pyserial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
        except pyserial.SerialException as e:
            raise RuntimeError(
                f"Failed to open serial port {self.serial_port}: {e}"
            ) from e

    @property
    def serial_port(self) -> str:
        return self.config.serial_port

    @property
    def baudrate(self) -> int:
        return self.config.baudrate

    @property
    def timeout(self) -> float:
        return self.config.timeout

    @property
    def move_speed(self) -> float:
        return self.config.move_speed

    def current_pose(self) -> tuple[float, float, float, float]:
        return (
            self.pose.x,
            self.pose.y,
            self.pose.z,
            self.pose.t
        )

    def reset_to_home(self, settle: float = 1.0) -> dict[str, float]:
        self.pose = Pose(
            x=self.config.base_x,
            y=self.config.base_y,
            z=self.config.base_z,
            t=self.config.base_t,
        )
        result = self.send_pose()
        time.sleep(settle)
        self.pose.t = self.config.t_limit[0]
        self.send_pose()
        time.sleep(0.4)
        self.pose.t = self.config.base_t
        self.send_pose()
        time.sleep(0.3)
        return result

    def send_pose(self):
        payload = build_xyzt_goal_payload(
            x=self.pose.x,
            y=self.pose.y,
            z=self.pose.z,
            t=self.pose.t,
            spd=self.move_speed
        )

        with self._serial_lock:
            self.serial.write(json.dumps(
                payload, separators=(",", ":")).encode("utf-8") + b"\n")
            self.serial.flush()

        return payload

    def step_gripper(self, rad: float):
        self.pose.t = clamp(rad, *self.config.t_limit)

    def step(self, axis: Literal["x", "y", "z", "t"], delta: float) -> Pose:
        if axis == "x":
            self.pose.x = clamp(
                self.pose.x + delta, *self.config.x_limit)
        elif axis == "y":
            self.pose.y = clamp(
                self.pose.y + delta, *self.config.y_limit)
        elif axis == "z":
            self.pose.z = clamp(
                self.pose.z + delta, *self.config.z_limit)
        elif axis == "t":
            raise NotImplementedError(
                "Direct control of gripper angle is not supported. Use step_gripper() instead.")
        else:
            raise ValueError(f"Invalid axis: {axis}")

        return self.pose


def create_controller(serial_port: str | None = None) -> RoArmPoseController:
    config = RoArmTeleopConfig(
        serial_port=serial_port) if serial_port else RoArmTeleopConfig()
    return RoArmPoseController(config)
