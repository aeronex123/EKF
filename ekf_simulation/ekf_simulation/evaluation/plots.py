from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.6,
    }
)


def _camera_correction_points(meas_df: pd.DataFrame, nav_df: pd.DataFrame):
    if meas_df.empty or nav_df.empty:
        return np.array([]), np.array([])

    cam = meas_df[(meas_df["sensor"] == "CAMERA") & meas_df["accepted"]]
    if cam.empty:
        return np.array([]), np.array([])

    times = cam["timestamp"].to_numpy(dtype=float)
    x = np.interp(times, nav_df["timestamp"], nav_df["estimated_x"])
    y = np.interp(times, nav_df["timestamp"], nav_df["estimated_y"])
    return x, y


def plot_xy(nav_df: pd.DataFrame, meas_df: pd.DataFrame, cfg, metrics, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.plot(nav_df["true_x"], nav_df["true_y"], color="dimgray", label="Ground Truth")
    ax.plot(nav_df["estimated_x"], nav_df["estimated_y"], color="tab:blue", label="EKF Estimate")

    ax.scatter(nav_df["true_x"].iloc[0], nav_df["true_y"].iloc[0], color="green", marker="o", label="Start")
    ax.scatter(nav_df["true_x"].iloc[-1], nav_df["true_y"].iloc[-1], color="red", marker="s", label="End")

    cx, cy = _camera_correction_points(meas_df, nav_df)
    if len(cx) > 0:
        ax.scatter(cx, cy, color="tab:orange", marker="x", s=28, label="Camera Corrections")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("GPS-Denied EKF Localization — 60 m Flight")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_xy_trajectory.png"))
    plt.close(fig)


def plot_position_error(nav_df: pd.DataFrame, metrics, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    ax.plot(nav_df["timestamp"], nav_df["position_error"], color="tab:blue", label="Horizontal Position Error")
    ax.axhline(0.5, color="red", linestyle="--", label="0.5 m Target")

    rmse = metrics["rmse"]
    mae = metrics["mae"]
    max_err = metrics["max_error"]

    txt = f"RMSE = {rmse:.3f} m\nMAE = {mae:.3f} m\nMax = {max_err:.3f} m"
    ax.text(
        0.98,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [m]")
    ax.set_title("EKF Horizontal Position Error")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_position_error.png"))
    plt.close(fig)


def plot_covariance(nav_df: pd.DataFrame, meas_df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    t = nav_df["timestamp"]
    ax.plot(t, np.sqrt(nav_df["P_x"]), label="sqrt(Pxx)")
    ax.plot(t, np.sqrt(nav_df["P_y"]), label="sqrt(Pyy)")
    ax.plot(t, np.sqrt(nav_df["P_z"]), label="sqrt(Pzz)")

    if not meas_df.empty:
        cam = meas_df[(meas_df["sensor"] == "CAMERA") & meas_df["accepted"]]
        for ct in cam["timestamp"]:
            ax.axvline(ct, color="tab:orange", alpha=0.12, linewidth=1.0)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("1σ Position Uncertainty [m]")
    ax.set_title("Position Covariance")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_covariance.png"))
    plt.close(fig)


def _plot_innovation_axis(ax, meas_df, sensor, columns, gate_threshold):
    rows = meas_df[meas_df["sensor"] == sensor]
    if rows.empty:
        ax.set_title(f"{sensor} innovations (no data)")
        return

    for col, label in columns:
        ax.plot(rows["timestamp"], rows[col], label=label)

    ax.axhline(gate_threshold, color="red", linestyle="--", alpha=0.6)
    ax.axhline(-gate_threshold, color="red", linestyle="--", alpha=0.6)
    ax.set_title(f"{sensor} Innovations")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Innovation")
    ax.legend()


def plot_innovations(meas_df: pd.DataFrame, cfg, out_dir: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 11.0), sharex=True)

    flow_gate = cfg.FLOW_NOISE_BASE * np.sqrt(cfg.NIS_GATE_FLOW)
    lidar_gate = cfg.LIDAR_NOISE * np.sqrt(cfg.NIS_GATE_LIDAR)
    baro_gate = cfg.BARO_NOISE * np.sqrt(cfg.NIS_GATE_BARO)
    cam_gate = cfg.CAMERA_NOISE * np.sqrt(cfg.NIS_GATE_CAMERA)

    _plot_innovation_axis(
        axes[0],
        meas_df,
        "FLOW",
        [("innovation_x", "x"), ("innovation_y", "y")],
        flow_gate,
    )
    _plot_innovation_axis(
        axes[1],
        meas_df,
        "LIDAR",
        [("innovation_x", "altitude")],
        lidar_gate,
    )
    _plot_innovation_axis(
        axes[2],
        meas_df,
        "BARO",
        [("innovation_x", "altitude")],
        baro_gate,
    )
    _plot_innovation_axis(
        axes[3],
        meas_df,
        "CAMERA",
        [("innovation_x", "x"), ("innovation_y", "y")],
        cam_gate,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_innovations.png"))
    plt.close(fig)


def plot_availability(nav_df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 8.0), sharex=True)

    keys = [
        ("camera_available", "Camera Available"),
        ("flow_accepted", "Flow Accepted"),
        ("lidar_accepted", "LiDAR Accepted"),
        ("baro_accepted", "Baro Accepted"),
    ]

    for ax, (key, label) in zip(axes, keys):
        ax.step(nav_df["timestamp"], nav_df[key].astype(int), where="post")
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_ylabel(label)

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Sensor Availability / Acceptance")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_sensor_availability.png"))
    plt.close(fig)


def plot_3d(nav_df: pd.DataFrame, out_dir: str) -> None:
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(nav_df["true_x"], nav_df["true_y"], nav_df["true_z"], color="dimgray", label="Ground Truth")
    ax.plot(nav_df["estimated_x"], nav_df["estimated_y"], nav_df["estimated_z"], color="tab:blue", label="EKF Estimate")

    ax.scatter(nav_df["true_x"].iloc[0], nav_df["true_y"].iloc[0], nav_df["true_z"].iloc[0], color="green", label="Start")
    ax.scatter(nav_df["true_x"].iloc[-1], nav_df["true_y"].iloc[-1], nav_df["true_z"].iloc[-1], color="red", label="End")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("3D GPS-Denied Trajectory")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_3d_trajectory.png"))
    plt.close(fig)


def plot_all(
    nav_df: pd.DataFrame,
    meas_df: pd.DataFrame,
    metrics,
    cfg,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    plot_xy(nav_df, meas_df, cfg, metrics, out_dir)
    plot_position_error(nav_df, metrics, out_dir)
    plot_covariance(nav_df, meas_df, out_dir)
    plot_innovations(meas_df, cfg, out_dir)
    plot_availability(nav_df, out_dir)
    plot_3d(nav_df, out_dir)