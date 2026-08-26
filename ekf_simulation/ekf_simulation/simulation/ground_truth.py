from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from config import Config
from ekf.models import (
    angular_velocity_from_quats,
    attitude_from_specific_force_yaw,
    quat_to_dcm,
)
from simulation.dynamics import UAVDynamics


class GroundTruth:
    """
    Generates the true UAV trajectory, attitude, body-specific force,
    and body angular rates.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dt = 1.0 / cfg.SIM_RATE
        self.n = int(round(cfg.SIM_DURATION * cfg.SIM_RATE)) + 1
        self.time = np.arange(self.n, dtype=float) * self.dt

        self.pos = np.zeros((self.n, 3), dtype=float)
        self.vel = np.zeros((self.n, 3), dtype=float)
        self.acc_cmd = np.zeros((self.n, 3), dtype=float)
        self.q = np.zeros((self.n, 4), dtype=float)
        self.gyro = np.zeros((self.n, 3), dtype=float)
        self.accel_body = np.zeros((self.n, 3), dtype=float)
        self.v_body = np.zeros((self.n, 3), dtype=float)

        self._generate()

    def _waypoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Normalized waypoints for a gentle 60 m leg."""
        frac = np.array([0.00, 0.08, 0.25, 0.42, 0.53, 0.67, 0.83, 0.92, 1.00])
        x = np.array([0.0, 0.8, 10.0, 25.0, 38.0, 48.0, 57.0, 59.3, 60.0])
        y = np.array([0.0, 0.0, 0.5, 1.0, 3.0, 2.0, 0.5, 0.1, 0.0])
        z = np.array([2.0, 2.2, 2.5, 2.4, 2.6, 2.5, 2.2, 2.1, 2.0])

        T = max(self.cfg.SIM_DURATION, 1.0e-3)
        times = frac * T
        points = np.column_stack([x, y, z])
        return times, points

    def _yaw_profile(self, t: np.ndarray) -> np.ndarray:
        T = max(self.cfg.SIM_DURATION, 1.0e-3)
        return (
            0.08 * np.sin(2.0 * np.pi * t / T)
            + 0.03 * np.sin(2.0 * np.pi * 3.0 * t / T)
        )

    def _generate(self) -> None:
        times, points = self._waypoints()

        splines = [
            CubicSpline(times, points[:, i], bc_type="clamped")
            for i in range(3)
        ]

        p_ref = np.column_stack([s(self.time) for s in splines])
        v_ref = np.column_stack([s.derivative()(self.time) for s in splines])
        a_ref = np.column_stack([s.derivative(2)(self.time) for s in splines])

        dynamics = UAVDynamics(self.cfg)
        yaw = self._yaw_profile(self.time)

        p = p_ref[0].copy()
        v = v_ref[0].copy()

        for i in range(self.n):
            if i == 0:
                a_cmd = a_ref[0].copy()
            else:
                a_cmd = dynamics.command_accel(
                    p_ref[i],
                    v_ref[i],
                    a_ref[i],
                    p,
                    v,
                )
                p, v = dynamics.integrate(p, v, a_cmd, self.dt)

            self.pos[i] = p
            self.vel[i] = v
            self.acc_cmd[i] = a_cmd

            # Specific force in nav frame: a - g, with g = [0,0,-9.81]
            f_nav = a_cmd - dynamics.gravity

            q_i = attitude_from_specific_force_yaw(f_nav, yaw[i])
            self.q[i] = q_i

            R = quat_to_dcm(q_i)
            self.accel_body[i] = R.T @ f_nav
            self.v_body[i] = R.T @ v

        # True angular velocity from quaternion sequence.
        if self.n > 1:
            for i in range(1, self.n):
                self.gyro[i] = angular_velocity_from_quats(
                    self.q[i - 1],
                    self.q[i],
                    self.dt,
                )
            self.gyro[0] = self.gyro[1]

    def index(self, t: float) -> int:
        idx = int(round(t / self.dt))
        return int(np.clip(idx, 0, self.n - 1))

    def get_state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return true position and velocity at time t."""
        idx = self.index(t)
        return self.pos[idx].copy(), self.vel[idx].copy()