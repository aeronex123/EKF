from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


class SimulationLogger:
    """Logs navigation state and sensor update information."""

    def __init__(self) -> None:
        self.nav_records: List[Dict[str, Any]] = []
        self.meas_records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # Navigation log
    # ------------------------------------------------------------
    def record_navigation(
        self,
        t: float,
        true_pos: np.ndarray,
        true_vel: np.ndarray,
        ekf,
        camera_available: bool,
    ) -> None:
        p = ekf.state.p
        v = ekf.state.v
        P = ekf.P

        pos_err = float(np.linalg.norm(p[:2] - true_pos[:2]))
        vel_err = float(np.linalg.norm(v - true_vel))

        def last(key: str) -> Dict[str, Any]:
            return ekf.last_updates.get(key, {})

        def inn_component(key: str, idx: int) -> float:
            inn = last(key).get("innovation", np.array([]))
            if inn is None or idx >= len(inn):
                return float("nan")
            return float(inn[idx])

        def nis(key: str) -> float:
            val = last(key).get("nis", np.nan)
            return float(val) if np.isfinite(val) else float("nan")

        def accepted(key: str) -> bool:
            return bool(last(key).get("accepted", False))

        row = {
            "timestamp": float(t),
            "true_x": float(true_pos[0]),
            "true_y": float(true_pos[1]),
            "true_z": float(true_pos[2]),
            "estimated_x": float(p[0]),
            "estimated_y": float(p[1]),
            "estimated_z": float(p[2]),
            "true_vx": float(true_vel[0]),
            "true_vy": float(true_vel[1]),
            "true_vz": float(true_vel[2]),
            "estimated_vx": float(v[0]),
            "estimated_vy": float(v[1]),
            "estimated_vz": float(v[2]),
            "position_error": pos_err,
            "velocity_error": vel_err,
            "P_x": float(P[0, 0]),
            "P_y": float(P[1, 1]),
            "P_z": float(P[2, 2]),
            "innovation_flow_x": inn_component("FLOW", 0),
            "innovation_flow_y": inn_component("FLOW", 1),
            "innovation_lidar": inn_component("LIDAR", 0),
            "innovation_baro": inn_component("BARO", 0),
            "innovation_camera_x": inn_component("CAMERA", 0),
            "innovation_camera_y": inn_component("CAMERA", 1),
            "NIS_flow": nis("FLOW"),
            "NIS_lidar": nis("LIDAR"),
            "NIS_baro": nis("BARO"),
            "NIS_camera": nis("CAMERA"),
            "flow_accepted": accepted("FLOW"),
            "lidar_accepted": accepted("LIDAR"),
            "baro_accepted": accepted("BARO"),
            "camera_accepted": accepted("CAMERA"),
            "camera_available": bool(camera_available),
        }

        self.nav_records.append(row)

    # ------------------------------------------------------------
    # Measurement log
    # ------------------------------------------------------------
    def record_measurement(self, result) -> None:
        inn = np.asarray(result.innovation, dtype=float)

        row = {
            "timestamp": float(result.time),
            "sensor": result.sensor,
            "innovation_x": float(inn[0]) if len(inn) > 0 else float("nan"),
            "innovation_y": float(inn[1]) if len(inn) > 1 else float("nan"),
            "nis": float(result.nis),
            "accepted": bool(result.accepted),
            "reason": result.reason,
        }
        self.meas_records.append(row)

    # ------------------------------------------------------------
    # DataFrames
    # ------------------------------------------------------------
    def to_nav_dataframe(self) -> pd.DataFrame:
        if not self.nav_records:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "true_x",
                    "true_y",
                    "true_z",
                    "estimated_x",
                    "estimated_y",
                    "estimated_z",
                    "true_vx",
                    "true_vy",
                    "true_vz",
                    "estimated_vx",
                    "estimated_vy",
                    "estimated_vz",
                    "position_error",
                    "velocity_error",
                    "P_x",
                    "P_y",
                    "P_z",
                    "innovation_flow_x",
                    "innovation_flow_y",
                    "innovation_lidar",
                    "innovation_baro",
                    "innovation_camera_x",
                    "innovation_camera_y",
                    "NIS_flow",
                    "NIS_lidar",
                    "NIS_baro",
                    "NIS_camera",
                    "flow_accepted",
                    "lidar_accepted",
                    "baro_accepted",
                    "camera_accepted",
                    "camera_available",
                ]
            )
        return pd.DataFrame(self.nav_records)

    def to_measurement_dataframe(self) -> pd.DataFrame:
        if not self.meas_records:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "sensor",
                    "innovation_x",
                    "innovation_y",
                    "nis",
                    "accepted",
                    "reason",
                ]
            )
        return pd.DataFrame(self.meas_records)