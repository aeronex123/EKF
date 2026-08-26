from __future__ import annotations

import argparse
import os
import time

import numpy as np

from config import make_config
from ekf.filter import EKF
from evaluation.metrics import compute_metrics, format_report
from evaluation.plots import plot_all
from sim_logging.logger import SimulationLogger
from simulation.ground_truth import GroundTruth
from simulation.scheduler import SensorScheduler


def run_simulation(
    cfg,
    make_plots: bool = True,
    save: bool = True,
    output_dir: str = "output",
):
    rng = np.random.default_rng(cfg.seed)

    t_start = time.perf_counter()

    gt = GroundTruth(cfg)
    scheduler = SensorScheduler()
    events = scheduler.build_events(gt, cfg, rng)

    ekf = EKF(cfg)
    logger = SimulationLogger()

    for event in events:
        if event.arrival_time > cfg.SIM_DURATION + 0.25:
            break

        result = ekf.process_event(event)

        if result is not None:
            logger.record_measurement(result)

        if event.sensor_type.name == "LOG" and event.arrival_time <= cfg.SIM_DURATION + 1.0e-9:
            true_pos, true_vel = gt.get_state(event.arrival_time)
            logger.record_navigation(
                event.arrival_time,
                true_pos,
                true_vel,
                ekf,
                cfg.camera_available(event.arrival_time),
            )

    nav_df = logger.to_nav_dataframe()
    meas_df = logger.to_measurement_dataframe()

    metrics = compute_metrics(nav_df)
    exec_time = time.perf_counter() - t_start

    if save:
        os.makedirs(output_dir, exist_ok=True)
        nav_df.to_csv(os.path.join(output_dir, "ekf_log.csv"), index=False)
        meas_df.to_csv(os.path.join(output_dir, "sensor_updates.csv"), index=False)

        if make_plots:
            plot_all(nav_df, meas_df, metrics, cfg, output_dir)

    return nav_df, meas_df, metrics, ekf.counters, exec_time


def run_monte_carlo(scenario: str, runs: int, duration, seed: int, output_dir: str) -> None:
    rms = []

    for i in range(runs):
        cfg = make_config(scenario=scenario, duration=duration, seed=seed + i)
        _, _, metrics, _, _ = run_simulation(
            cfg,
            make_plots=False,
            save=False,
            output_dir=output_dir,
        )
        rms.append(metrics["rmse"])

    rms_arr = np.array(rms, dtype=float)
    rms_arr = rms_arr[np.isfinite(rms_arr)]

    below = int(np.sum(rms_arr < 0.5))

    print("\n## Monte Carlo Results")
    print(f"\nRuns: {runs}")
    print(f"\nMean RMSE: {np.mean(rms_arr):.3f} m")
    print(f"Std RMSE : {np.std(rms_arr):.3f} m")
    print(f"Best     : {np.min(rms_arr):.3f} m")
    print(f"Worst    : {np.max(rms_arr):.3f} m")
    print(f"\nRuns below 0.5 m: {below} / {runs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPS-denied EKF localization simulation")
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal",
        choices=[
            "normal",
            "camera_dropout",
            "flow_failure",
            "lidar_failure",
            "baro_drift",
            "combined",
        ],
    )
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monte-carlo", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    if args.monte_carlo > 0:
        run_monte_carlo(
            scenario=args.scenario,
            runs=args.monte_carlo,
            duration=args.duration,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        return

    cfg = make_config(scenario=args.scenario, duration=args.duration, seed=args.seed)

    nav_df, meas_df, metrics, counters, exec_time = run_simulation(
        cfg,
        make_plots=not args.no_plots,
        save=True,
        output_dir=args.output_dir,
    )

    print(format_report(metrics, counters, cfg, exec_time))


if __name__ == "__main__":
    main()