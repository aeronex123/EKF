from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt_num(x, fmt: str = "{:.3f}", unit: str = "") -> str:
    """Format a numeric value safely for the report."""
    if x is None:
        return "Not available"
    if isinstance(x, float) and (np.isnan(x) or not np.isfinite(x)):
        return "Not available"
    return fmt.format(x) + unit


def fmt_int(x) -> str:
    if x is None:
        return "Not available"
    return str(int(x))


def truth_status(rmse: float) -> str:
    if rmse is None or np.isnan(rmse) or not np.isfinite(rmse):
        return "RMSE not available. Run the simulation first."
    if rmse < 0.5:
        return f"The horizontal position RMSE is {rmse:.3f} m, which is below the 0.5 m target."
    return (
        f"The horizontal position RMSE is {rmse:.3f} m. "
        "Do not claim that the target was met unless the actual logged result is below 0.5 m."
    )


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
def load_nav(output_dir: Path) -> pd.DataFrame | None:
    path = output_dir / "ekf_log.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_meas(output_dir: Path) -> pd.DataFrame | None:
    path = output_dir / "sensor_updates.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def compute_metrics(nav: pd.DataFrame | None) -> dict:
    metrics = {
        "rmse": float("nan"),
        "max_error": float("nan"),
        "flight_distance": float("nan"),
        "camera_lost_tested": False,
    }

    if nav is None or nav.empty:
        return metrics

    if "position_error" in nav.columns:
        err = nav["position_error"].to_numpy(dtype=float)
        err = err[np.isfinite(err)]
        if err.size > 0:
            metrics["rmse"] = float(np.sqrt(np.mean(err**2)))
            metrics["max_error"] = float(np.max(err))

    if {"true_x", "true_y"}.issubset(nav.columns) and len(nav) > 1:
        dx = np.diff(nav["true_x"].to_numpy(dtype=float))
        dy = np.diff(nav["true_y"].to_numpy(dtype=float))
        metrics["flight_distance"] = float(np.nansum(np.hypot(dx, dy)))

    if "camera_available" in nav.columns:
        metrics["camera_lost_tested"] = bool((nav["camera_available"] == False).any())

    return metrics


def compute_counts(meas: pd.DataFrame | None) -> dict | None:
    if meas is None or meas.empty:
        return None

    counts = {
        "CAMERA": 0,
        "FLOW": 0,
        "LIDAR": 0,
        "BARO": 0,
    }

    if "accepted" not in meas.columns or "sensor" not in meas.columns:
        return counts

    accepted_mask = (
        meas["accepted"]
        .astype(str)
        .str.strip()
        .str.str.lower()
        .isin(["true", "1"])
    )

    accepted = meas[accepted_mask]

    grouped = accepted.groupby("sensor").size()

    for key in counts:
        counts[key] = int(grouped.get(key, 0))

    return counts


# ------------------------------------------------------------
# Figure: main result
# ------------------------------------------------------------
def make_main_figure(nav: pd.DataFrame | None, metrics: dict, out_path: Path) -> bool:
    if nav is None or nav.empty:
        return False

    fig = plt.figure(figsize=(7.2, 6.0), dpi=300)

    # Top: XY trajectory
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(nav["true_x"], nav["true_y"], color="dimgray", linewidth=1.8, label="Ground Truth")
    ax1.plot(nav["estimated_x"], nav["estimated_y"], color="tab:blue", linewidth=1.4, label="EKF Estimate")

    ax1.scatter(nav["true_x"].iloc[0], nav["true_y"].iloc[0], color="green", s=18, label="Start")
    ax1.scatter(nav["true_x"].iloc[-1], nav["true_y"].iloc[-1], color="red", s=18, label="End")

    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_title("GPS-Denied EKF Localization — Estimated vs Reference Path")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="datalim")

    # Bottom: position error
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.plot(nav["timestamp"], nav["position_error"], color="tab:blue", linewidth=1.2)
    ax2.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="0.5 m Target")

    rmse = metrics.get("rmse", float("nan"))
    max_err = metrics.get("max_error", float("nan"))

    txt = f"RMSE = {fmt_num(rmse, '{:.3f}', ' m')}\nMax = {fmt_num(max_err, '{:.3f}', ' m')}"
    ax2.text(
        0.98,
        0.98,
        txt,
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Horizontal Position Error [m]")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


# ------------------------------------------------------------
# Figure: covariance
# ------------------------------------------------------------
def make_covariance_figure(
    nav: pd.DataFrame | None,
    meas: pd.DataFrame | None,
    out_path: Path,
) -> bool:
    if nav is None or nav.empty:
        return False

    fig, ax = plt.subplots(figsize=(7.2, 2.2), dpi=300)

    t = nav["timestamp"]

    ax.plot(t, np.sqrt(nav["P_x"]), label="sqrt(Pxx)", linewidth=1.2)
    ax.plot(t, np.sqrt(nav["P_y"]), label="sqrt(Pyy)", linewidth=1.2)
    ax.plot(t, np.sqrt(nav["P_z"]), label="sqrt(Pzz)", linewidth=1.2)

    # Mark accepted camera corrections if available.
    if meas is not None and not meas.empty:
        accepted_mask = (
            meas["accepted"]
            .astype(str)
            .str.strip()
            .str.str.lower()
            .isin(["true", "1"])
        )
        cam = meas[(meas["sensor"] == "CAMERA") & accepted_mask]

        for ct in cam["timestamp"]:
            ax.axvline(ct, color="tab:orange", alpha=0.14, linewidth=1.0)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("1σ Position Uncertainty [m]")
    ax.set_title("Position Covariance")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


# ------------------------------------------------------------
# Figure: architecture diagram
# ------------------------------------------------------------
def make_architecture_figure(out_path: Path) -> bool:
    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=300)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.02",
                fc=fc,
                ec="black",
                lw=1.0,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=6.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.1),
            annotation_clip=False,
        )

    # Main vertical flow
    box(5.0, 9.3, 2.5, 0.75, "IMU\nFlight Controller\n200 Hz", "#dbeafe")
    box(5.0, 7.9, 2.8, 0.75, "EKF PREDICT\nState + P", "#dcfce7")
    box(5.0, 6.4, 3.4, 0.75, "SENSOR CORRECTION\nmeasurement update + gating", "#fef9c3")
    box(5.0, 2.3, 3.3, 0.9, "FUSED NAVIGATION\nX,Y,Z + velocity + covariance", "#bfdbfe")
    box(5.0, 0.8, 3.6, 0.75, "PATH PLANNING / SWARM NAVIGATION", "#ddd6fe")

    # Sensors
    box(1.5, 4.2, 2.0, 1.0, "PMW3901\nvelocity\n50 Hz", "#e5e7eb")
    box(3.8, 4.2, 1.8, 1.0, "TF-Luna\nheight\n~100 Hz", "#e5e7eb")
    box(6.2, 4.2, 1.8, 1.0, "Barometer\nheight\n25 Hz", "#e5e7eb")
    box(8.5, 4.2, 2.0, 1.0, "Camera\nXY\n5–10 Hz", "#e5e7eb")

    # Arrows
    arrow(5.0, 8.92, 5.0, 8.32)
    arrow(5.0, 7.52, 5.0, 6.82)

    arrow(1.5, 4.75, 4.2, 6.0)
    arrow(3.8, 4.75, 4.6, 6.0)
    arrow(6.2, 4.75, 5.4, 6.0)
    arrow(8.5, 4.75, 5.8, 6.0)

    arrow(5.0, 5.98, 5.0, 2.80)
    arrow(5.0, 1.85, 5.0, 1.22)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


# ------------------------------------------------------------
# QR code
# ------------------------------------------------------------
def make_qr_code(url: str, out_path: Path) -> bool:
    if not url:
        return False

    try:
        import qrcode
    except Exception:
        return False

    img = qrcode.make(url)
    img.save(out_path)
    return True


# ------------------------------------------------------------
# Extract actual code snippets
# ------------------------------------------------------------
def extract_code_block(path: Path, start_substring: str, max_lines: int = 12) -> str | None:
    if not path.exists():
        return None

    lines = path.read_text(encoding="utf-8").splitlines()

    for i, line in enumerate(lines):
        if start_substring in line:
            block = []
            for j in range(i, min(len(lines), i + max_lines * 2)):
                if len(block) >= max_lines:
                    break
                s = lines[j].rstrip()
                if not block and not s.strip():
                    continue
                block.append(s)

            while block and not block[-1].strip():
                block.pop()

            return "\n".join(block[:max_lines])

    return None


def get_code_snippets(project_root: Path) -> tuple[str | None, str | None]:
    pred = extract_code_block(
        project_root / "ekf" / "prediction.py",
        "F = F_matrix(R, a_corr, omega, dt, cfg)",
        max_lines=10,
    )

    upd = extract_code_block(
        project_root / "ekf" / "update.py",
        "if pre_reject_reason is not None:",
        max_lines=12,
    )

    return pred, upd


# ------------------------------------------------------------
# Report writer
# ------------------------------------------------------------
def write_report(
    report_dir: Path,
    nav: pd.DataFrame | None,
    meas: pd.DataFrame | None,
    metrics: dict,
    counts: dict | None,
    github_url: str,
    pred_snippet: str | None,
    upd_snippet: str | None,
) -> Path:
    rmse = metrics.get("rmse", float("nan"))
    max_err = metrics.get("max_error", float("nan"))
    flight_distance = metrics.get("flight_distance", float("nan"))
    camera_lost_tested = metrics.get("camera_lost_tested", False)

    camera_updates = counts.get("CAMERA") if counts is not None else None
    flow_updates = counts.get("FLOW") if counts is not None else None
    lidar_updates = counts.get("LIDAR") if counts is not None else None
    baro_updates = counts.get("BARO") if counts is not None else None

    status = truth_status(rmse)

    if camera_lost_tested:
        failure_text = (
            "Sensor-loss test: executed in the logged simulation. "
            "The camera availability trace contains a dropout period. "
            "During the dropout, the estimator continues using IMU prediction plus optical-flow, "
            "LiDAR and barometer constraints, while position uncertainty increases until camera corrections return."
        )
    else:
        failure_text = (
            "Sensor-loss test: planned validation scenario. "
            "The currently loaded log does not contain a camera dropout. "
            "Run the camera_dropout scenario to generate actual failure evidence."
        )

    pred_code = pred_snippet or "Code snippet not found. Copy the corresponding lines from ekf/prediction.py."
    upd_code = upd_snippet or "Code snippet not found. Copy the corresponding lines from ekf/update.py."

    fence = chr(96) * 3

    lines = []

    lines.append("# PAGE 1 — EKF CONCEPT + OUR ARCHITECTURE")
    lines.append("")
    lines.append("## GPS-Denied EKF Sensor Fusion and Localization")
    lines.append("")
    lines.append(
        "Our UAV cannot rely on GPS/GNSS in the competition environment. "
        "Therefore, onboard sensors must be fused to estimate position, velocity and altitude. "
        "An Extended Kalman Filter (EKF) combines high-rate IMU prediction with complementary "
        "sensor corrections while maintaining an estimate of navigation uncertainty."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 1. What is an EKF?")
    lines.append("")
    lines.append(
        "The EKF is a recursive estimator for nonlinear systems. It predicts the UAV state using the IMU, "
        "then corrects that prediction when sensor measurements arrive. The innovation is the difference "
        "between the measured value and the value predicted by the current state. The Kalman gain determines "
        "how strongly the measurement should correct the state, based on the relative uncertainty of the "
        "prediction and the sensor. The covariance matrix represents estimator uncertainty. It usually "
        "increases during IMU-only prediction and decreases when reliable measurements are fused."
    )
    lines.append("")
    lines.append("Prediction:")
    lines.append("")
    lines.append("xₖ₊₁ = f(xₖ,uₖ) + wₖ")
    lines.append("Pₖ₊₁ = FₖPₖFₖᵀ + Qₖ")
    lines.append("")
    lines.append("Measurement innovation:")
    lines.append("")
    lines.append("rₖ = zₖ − h(xₖ)")
    lines.append("")
    lines.append("Innovation covariance:")
    lines.append("")
    lines.append("Sₖ = HₖPₖHₖᵀ + Rₖ")
    lines.append("")
    lines.append("Kalman gain:")
    lines.append("")
    lines.append("Kₖ = PₖHₖᵀSₖ⁻¹")
    lines.append("")
    lines.append("Correction:")
    lines.append("")
    lines.append("xₖ ← xₖ + Kₖrₖ")
    lines.append("Pₖ ← (I − KₖHₖ)Pₖ")
    lines.append("")
    lines.append(
        "Prediction estimates how the UAV moves using the IMU. Sensor measurements then correct the prediction. "
        "The covariance represents estimator uncertainty and normally increases during prediction and decreases "
        "when reliable measurements are fused."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 2. Sensor-Fusion Architecture")
    lines.append("")
    lines.append("![Architecture](fig_architecture.png)")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 3. Role of Each Sensor")
    lines.append("")
    lines.append("| Sensor    | Information                 | EKF Role                         |")
    lines.append("| --------- | --------------------------- | -------------------------------- |")
    lines.append("| IMU       | Acceleration + angular rate | High-rate prediction             |")
    lines.append("| PMW3901   | Horizontal velocity         | XY velocity correction           |")
    lines.append("| TF-Luna   | Height above ground         | Altitude correction              |")
    lines.append("| Barometer | Relative altitude           | Long-term altitude stabilization |")
    lines.append("| Camera    | Visual XY position          | Position correction              |")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 4. Why Sensor Fusion?")
    lines.append("")
    lines.append(
        "The IMU alone drifts rapidly. Optical flow provides velocity but depends on texture, altitude and "
        "surface conditions. LiDAR provides accurate local height but has range limitations. The barometer "
        "provides useful altitude trend information but drifts slowly. The camera provides positional "
        "corrections but can temporarily fail. Therefore, the EKF combines complementary measurements so that "
        "the failure or degradation of one sensor does not necessarily cause immediate loss of localization."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("# PAGE 2 — PYTHON IMPLEMENTATION + SIMULATION EVIDENCE")
    lines.append("")
    lines.append("## Python EKF Implementation and Simulation Validation")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 5. Implementation")
    lines.append("")
    lines.append(
        "The EKF was implemented as a modular Python system. Prediction, measurement models, Jacobians, "
        "gating, logging and evaluation are separated so that each sensor and filter operation can be tested "
        "and replaced independently."
    )
    lines.append("")
    lines.append("| Function           | Implementation    |")
    lines.append("| ------------------ | ----------------- |")
    lines.append("| State definition   | ekf/state.py      |")
    lines.append("| Motion/prediction  | ekf/prediction.py |")
    lines.append("| Measurement models | ekf/models.py     |")
    lines.append("| Jacobians          | ekf/jacobians.py  |")
    lines.append("| Sensor update      | ekf/update.py     |")
    lines.append("| Innovation gating  | ekf/gating.py     |")
    lines.append("| EKF manager        | ekf/filter.py     |")
    lines.append("| Sensor simulation  | sensors/          |")
    lines.append("| Ground truth       | simulation/       |")
    lines.append("| Logging            | datalog/          |")
    lines.append("| Evaluation         | evaluation/       |")
    lines.append("| Tests              | tests/            |")
    lines.append("")
    lines.append(
        "The complete implementation is modular rather than being contained in a single script. "
        "Each sensor and EKF operation can therefore be tested and replaced independently."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 6. Implementation Excerpts")
    lines.append("")
    lines.append("Snippet A — EKF prediction / covariance propagation:")
    lines.append("")
    lines.append(fence + "python")
    lines.extend(pred_code.splitlines())
    lines.append(fence)
    lines.append("")
    lines.append("Implementation excerpt from our Python EKF.")
    lines.append("")
    lines.append("Snippet B — Measurement update / innovation gating:")
    lines.append("")
    lines.append(fence + "python")
    lines.extend(upd_code.splitlines())
    lines.append(fence)
    lines.append("")
    lines.append("Implementation excerpt from our Python EKF.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 7. Simulation Result")
    lines.append("")
    lines.append("![Main Result](fig_results.png)")
    lines.append("")
    lines.append(
        "Figure 3. Python simulation of GPS-denied localization. Estimated path (blue) vs. reference path "
        "(grey); horizontal position RMSE is evaluated against the 0.5 m target."
    )
    lines.append("")
    lines.append(status)
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 8. Covariance / Sensor Fusion Evidence")
    lines.append("")
    lines.append("![Covariance](fig_covariance.png)")
    lines.append("")
    lines.append(
        "The covariance trace provides an estimate of localization confidence. During periods without strong "
        "position observations, uncertainty increases because the estimator relies primarily on prediction. "
        "When reliable sensor measurements become available, the EKF reduces the corresponding uncertainty "
        "through measurement updates."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 9. Sensor-Loss Evidence")
    lines.append("")
    lines.append(failure_text)
    lines.append("")
    lines.append("Conceptual behavior:")
    lines.append("")
    lines.append("CAMERA AVAILABLE")
    lines.append("→ EKF correction")
    lines.append("")
    lines.append("CAMERA LOST")
    lines.append("→ IMU + Optical Flow + LiDAR + Barometer")
    lines.append("→ continued localization")
    lines.append("→ increased uncertainty")
    lines.append("")
    lines.append("CAMERA REACQUIRED")
    lines.append("→ position correction")
    lines.append("→ reduced uncertainty")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 10. Results Table")
    lines.append("")
    lines.append("| Metric                 | Result |")
    lines.append("| ---------------------- | -----: |")
    lines.append(f"| Simulation distance    | {fmt_num(flight_distance, '{:.1f}', ' m')} |")
    lines.append(f"| Position RMSE          | {fmt_num(rmse, '{:.3f}', ' m')} |")
    lines.append(f"| Maximum position error | {fmt_num(max_err, '{:.3f}', ' m')} |")
    lines.append(f"| Camera updates         | {fmt_int(camera_updates)} |")
    lines.append(f"| Optical-flow updates   | {fmt_int(flow_updates)} |")
    lines.append(f"| LiDAR updates          | {fmt_int(lidar_updates)} |")
    lines.append(f"| Barometer updates      | {fmt_int(baro_updates)} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 11. Connection to Real Drone")
    lines.append("")
    lines.append("Real UAV architecture:")
    lines.append("")
    lines.append("Kakute H7 / Flight Controller")
    lines.append("│")
    lines.append("IMU data")
    lines.append("│")
    lines.append("MAVLink")
    lines.append("▼")
    lines.append("Raspberry Pi 5")
    lines.append("│")
    lines.append("┌────────┼─────────┐")
    lines.append("│        │         │")
    lines.append("PMW3901 TF-Luna Camera")
    lines.append("│        │         │")
    lines.append("└────────┼─────────┘")
    lines.append("▼")
    lines.append("EKF / Sensor Fusion")
    lines.append("▼")
    lines.append("GPS-DENIED LOCALIZATION")
    lines.append("▼")
    lines.append("Path Planning / Swarm")
    lines.append("")
    lines.append(
        "The current Python implementation provides the estimator development and validation environment. "
        "The intended onboard architecture places the flight controller IMU/attitude source and additional "
        "sensors on the Raspberry Pi-based autonomy layer. ArduPilot/SITL and MAVLink can subsequently be "
        "used to validate the interface before real-flight deployment. Simulation, SITL and real hardware "
        "are treated as separate validation stages."
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 12. Complete Source Code and Reproducibility")
    lines.append("")
    lines.append(
        "The complete Python implementation, sensor models, EKF modules, simulation configuration, logging, "
        "evaluation scripts and unit tests are maintained in our GitHub repository."
    )
    lines.append("")

    if github_url:
        lines.append(f"GitHub: {github_url}")
        lines.append("")
        if (report_dir / "github_qr.png").exists():
            lines.append("QR code:")
            lines.append("")
            lines.append("![QR](github_qr.png)")
        else:
            lines.append("QR code: not generated. Install qrcode[pil] and rerun with --github-url.")
    else:
        lines.append("GitHub: [INSERT ACTUAL GITHUB URL]")
        lines.append("")
        lines.append("QR code: [GENERATE QR CODE FROM THE ACTUAL GITHUB URL]")

    lines.append("")
    lines.append("Repository structure:")
    lines.append("")
    lines.append("main.py → sensors/ → ekf/ → evaluation/ → tests/")
    lines.append("")
    lines.append(
        "Third-party libraries are used for numerical computation and visualization; the EKF architecture, "
        "sensor models, simulation logic and evaluation pipeline are implemented as project code."
    )
    lines.append("")

    out_path = report_dir / "report_content.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path