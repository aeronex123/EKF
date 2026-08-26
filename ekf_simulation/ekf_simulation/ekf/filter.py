from __future__ import annotations

import numpy as np

from config import Config
from ekf.jacobians import H_baro, H_camera, H_flow, H_lidar
from ekf.models import quat_to_dcm, skew
from ekf.prediction import predict_nominal
from ekf.state import make_initial_covariance, make_initial_state
from ekf.update import UpdateResult, measurement_update


class EKF:
    """ArduPilot-inspired asynchronous GPS-denied navigation EKF."""

    SENSOR_KEYS = ("FLOW", "LIDAR", "BARO", "CAMERA")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = make_initial_state(cfg)
        self.P = make_initial_covariance(cfg)

        self.g_nav = np.array([0.0, 0.0, -cfg.GRAVITY], dtype=float)

        self.last_accel = np.array([0.0, 0.0, cfg.GRAVITY], dtype=float)
        self.last_gyro = np.zeros(3, dtype=float)

        self.counters = {
            key: {"accepted": 0, "rejected": 0, "total": 0}
            for key in self.SENSOR_KEYS
        }

        self.last_updates = {
            key: {
                "innovation": np.array([]),
                "nis": np.nan,
                "accepted": False,
                "time": np.nan,
            }
            for key in self.SENSOR_KEYS
        }

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    def _predict_to(self, t: float) -> None:
        if t <= self.state.time + 1.0e-12:
            return

        # Substep for numerical stability when delayed events arrive.
        while t > self.state.time + 1.0e-12:
            dt = min(0.02, t - self.state.time)
            self.P = predict_nominal(
                self.state,
                self.P,
                self.last_accel,
                self.last_gyro,
                dt,
                self.cfg,
            )

    # ------------------------------------------------------------
    # Delay compensation helpers
    # ------------------------------------------------------------
    def _est_nav_accel(self) -> np.ndarray:
        R = quat_to_dcm(self.state.q)
        return R @ (self.last_accel - self.state.ba) + self.g_nav

    def _compensate_camera(self, z: np.ndarray, delay: float) -> np.ndarray:
        if delay <= 0.0:
            return z
        a = self._est_nav_accel()
        return z + self.state.v[:2] * delay + 0.5 * a[:2] * delay * delay

    def _compensate_alt(self, z: float, delay: float) -> np.ndarray:
        if delay <= 0.0:
            return np.array([z], dtype=float)
        a = self._est_nav_accel()
        z_new = z + self.state.v[2] * delay + 0.5 * a[2] * delay * delay
        return np.array([z_new], dtype=float)

    def _compensate_flow(self, z: np.ndarray, delay: float) -> np.ndarray:
        if delay <= 0.0:
            return z

        R = quat_to_dcm(self.state.q)
        a_nav = self._est_nav_accel()
        omega = self.last_gyro - self.state.bg
        v_body = R.T @ self.state.v

        dv_body = R.T @ a_nav - skew(omega) @ v_body
        return z + dv_body[:2] * delay

    # ------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------
    def process_event(self, event) -> UpdateResult | None:
        kind = event.sensor_type.name

        if kind == "IMU":
            self.last_accel = np.asarray(event.data["accel"], dtype=float)
            self.last_gyro = np.asarray(event.data["gyro"], dtype=float)
            self._predict_to(event.arrival_time)
            return None

        if kind == "LOG":
            self._predict_to(event.arrival_time)
            return None

        self._predict_to(event.arrival_time)

        if kind == "FLOW":
            return self._update_flow(event)
        if kind == "LIDAR":
            return self._update_lidar(event)
        if kind == "BARO":
            return self._update_baro(event)
        if kind == "CAMERA":
            return self._update_camera(event)

        return None

    # ------------------------------------------------------------
    # Sensor updates
    # ------------------------------------------------------------
    def _record_update(self, key: str, result: UpdateResult) -> None:
        self.counters[key]["total"] += 1
        if result.accepted:
            self.counters[key]["accepted"] += 1
        else:
            self.counters[key]["rejected"] += 1

        self.last_updates[key] = {
            "innovation": np.asarray(result.innovation, dtype=float).copy(),
            "nis": float(result.nis),
            "accepted": bool(result.accepted),
            "time": float(result.time),
        }

    def _update_flow(self, event) -> UpdateResult:
        data = event.data
        delay = max(0.0, event.arrival_time - event.timestamp)

        z = np.array([data["vx_body"], data["vy_body"]], dtype=float)
        z = self._compensate_flow(z, delay)

        sigma = max(float(data["sigma"]), 1.0e-4)
        R = np.eye(2, dtype=float) * sigma * sigma

        R_dcm = quat_to_dcm(self.state.q)
        v_body = R_dcm.T @ self.state.v
        h = v_body[:2].copy()

        H = H_flow(R_dcm, v_body)

        pre = None if bool(data.get("valid", True)) else "quality"

        P_new, y, nis, accepted, reason = measurement_update(
            self.state,
            self.P,
            z,
            h,
            H,
            R,
            self.cfg.NIS_GATE_FLOW,
            pre,
        )
        self.P = P_new

        result = UpdateResult(
            sensor="FLOW",
            time=event.arrival_time,
            timestamp=event.timestamp,
            innovation=y,
            nis=nis,
            accepted=accepted,
            reason=reason,
            measurement=z,
        )
        self._record_update("FLOW", result)
        return result

    def _update_lidar(self, event) -> UpdateResult:
        data = event.data
        delay = max(0.0, event.arrival_time - event.timestamp)

        z = self._compensate_alt(float(data["range"]), delay)

        sigma = max(float(data["sigma"]), 1.0e-4)
        R = np.array([[sigma * sigma]], dtype=float)

        h = np.array([self.state.p[2]], dtype=float)
        H = H_lidar()

        pre = None if bool(data.get("valid", True)) else "invalid"

        P_new, y, nis, accepted, reason = measurement_update(
            self.state,
            self.P,
            z,
            h,
            H,
            R,
            self.cfg.NIS_GATE_LIDAR,
            pre,
        )
        self.P = P_new

        result = UpdateResult(
            sensor="LIDAR",
            time=event.arrival_time,
            timestamp=event.timestamp,
            innovation=y,
            nis=nis,
            accepted=accepted,
            reason=reason,
            measurement=z,
        )
        self._record_update("LIDAR", result)
        return result

    def _update_baro(self, event) -> UpdateResult:
        data = event.data
        delay = max(0.0, event.arrival_time - event.timestamp)

        z = self._compensate_alt(float(data["alt"]), delay)

        sigma = max(float(data["sigma"]), 1.0e-4)
        R = np.array([[sigma * sigma]], dtype=float)

        h = np.array([self.state.p[2]], dtype=float)
        H = H_baro()

        pre = None if bool(data.get("valid", True)) else "invalid"

        P_new, y, nis, accepted, reason = measurement_update(
            self.state,
            self.P,
            z,
            h,
            H,
            R,
            self.cfg.NIS_GATE_BARO,
            pre,
        )
        self.P = P_new

        result = UpdateResult(
            sensor="BARO",
            time=event.arrival_time,
            timestamp=event.timestamp,
            innovation=y,
            nis=nis,
            accepted=accepted,
            reason=reason,
            measurement=z,
        )
        self._record_update("BARO", result)
        return result

    def _update_camera(self, event) -> UpdateResult:
        data = event.data
        delay = max(0.0, event.arrival_time - event.timestamp)

        z = np.array([data["px"], data["py"]], dtype=float)
        z = self._compensate_camera(z, delay)

        sigma = max(float(data["sigma"]), 1.0e-4)
        R = np.eye(2, dtype=float) * sigma * sigma

        h = self.state.p[:2].copy()
        H = H_camera()

        pre = None if bool(data.get("valid", True)) else "invalid"

        P_new, y, nis, accepted, reason = measurement_update(
            self.state,
            self.P,
            z,
            h,
            H,
            R,
            self.cfg.NIS_GATE_CAMERA,
            pre,
        )
        self.P = P_new

        result = UpdateResult(
            sensor="CAMERA",
            time=event.arrival_time,
            timestamp=event.timestamp,
            innovation=y,
            nis=nis,
            accepted=accepted,
            reason=reason,
            measurement=z,
        )
        self._record_update("CAMERA", result)
        return result