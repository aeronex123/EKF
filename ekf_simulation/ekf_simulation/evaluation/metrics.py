from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute horizontal position error metrics."""
    if df.empty or "position_error" not in df.columns:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "max_error": float("nan"),
            "p95_error": float("nan"),
            "flight_distance": float("nan"),
        }

    err = df["position_error"].to_numpy(dtype=float)
    err = err[np.isfinite(err)]

    if err.size == 0:
        rmse = mae = max_err = p95 = float("nan")
    else:
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(err))
        max_err = float(np.max(err))
        p95 = float(np.percentile(err, 95))

    if {"true_x", "true_y"}.issubset(df.columns) and len(df) > 1:
        dx = np.diff(df["true_x"].to_numpy(dtype=float))
        dy = np.diff(df["true_y"].to_numpy(dtype=float))
        flight_distance = float(np.sum(np.hypot(dx, dy)))
    else:
        flight_distance = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_err,
        "p95_error": p95,
        "flight_distance": flight_distance,
    }


def format_report(
    metrics: Dict[str, float],
    counters: Dict[str, Dict[str, int]],
    cfg,
    exec_time: float,
) -> str:
    status = "PASS" if metrics["rmse"] < cfg.TARGET_RMSE else "FAIL"

    camera_updates = counters["CAMERA"]["accepted"]
    flow_updates = counters["FLOW"]["accepted"]
    lidar_updates = counters["LIDAR"]["accepted"]
    baro_updates = counters["BARO"]["accepted"]

    rejected_flow = counters["FLOW"]["rejected"]
    rejected_lidar = counters["LIDAR"]["rejected"]
    rejected_camera = counters["CAMERA"]["rejected"]
    rejected_baro = counters["BARO"]["rejected"]

    real_time_factor = cfg.SIM_DURATION / max(exec_time, 1.0e-9)

    lines = [
        "========================================",
        "GPS-DENIED EKF PERFORMANCE",
        "========================================",
        "",
        f"Flight distance : {metrics['flight_distance']:.1f} m",
        "",
        f"Horizontal RMSE : {metrics['rmse']:.3f} m",
        f"Mean error      : {metrics['mae']:.3f} m",
        f"Maximum error   : {metrics['max_error']:.3f} m",
        f"95% error       : {metrics['p95_error']:.3f} m",
        "",
        f"Camera updates  : {camera_updates}",
        f"Flow updates    : {flow_updates}",
        f"LiDAR updates   : {lidar_updates}",
        f"Baro updates    : {baro_updates}",
        "",
        f"Rejected flow   : {rejected_flow}",
        f"Rejected LiDAR  : {rejected_lidar}",
        f"Rejected baro   : {rejected_baro}",
        f"Rejected camera : {rejected_camera}",
        "",
        f"Simulation time : {cfg.SIM_DURATION:.1f} s",
        f"Execution time  : {exec_time:.2f} s",
        f"Real-time factor: {real_time_factor:.1f}x",
        "",
        f"TARGET RMSE     : < {cfg.TARGET_RMSE:.2f} m",
        f"STATUS          : {status}",
        "========================================",
    ]

    return "\n".join(lines)