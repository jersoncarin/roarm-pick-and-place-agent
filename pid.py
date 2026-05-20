from __future__ import annotations
from roarm import clamp

PID_Y = (0.020, 0.003, 0.002, 1.2)
PID_XZ = (0.018, 0.003, 0.002, 1.0)
PID_I_LIMIT = 40.0
X_ALPHA = 0.55
Z_ALPHA = -0.85


class PID:
    def __init__(self, Kp: float, Ki: float, Kd: float, out_limit: float) -> None:
        self.p = Kp
        self.i = Ki
        self.d = Kd
        self.out_limit = out_limit
        self.i_limit = PID_I_LIMIT

        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        self.integral = clamp(self.integral, -self.i_limit, self.i_limit)

        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.p * error + self.i * self.integral + self.d * derivative

        self.prev_error = error

        return clamp(output, -self.out_limit, self.out_limit)


class PIDY(PID):
    def __init__(self) -> None:
        super().__init__(*PID_Y)


class PIDXZ(PID):
    def __init__(self) -> None:
        super().__init__(*PID_XZ)

    def update(self, error: float, dt: float) -> float:
        output = super().update(error, dt)
        return (output * X_ALPHA, output * Z_ALPHA)
