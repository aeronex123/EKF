from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Scenario:
    """Scenario-level fault injection configuration."""

    name: str = "normal"

    camera_dropout_start: float = 0.0
    camera_dropout_duration: float = 0.0

    flow_degradation_start: float = 0.0
    flow_degradation_duration: float = 0.0

    lidar_failure_start: float = 0.0
    lidar_failure_duration: float = 0.0

    baro_drift_rate: float = 0.0
    camera_outlier_time: Optional[float] = None


@dataclass
class Config:
    """Central simulation and filter configuration."""

    # ------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------
    SIM_DURATION: float = 60.0
    SIM_RATE: float = 1000.0       # ground truth integration rate [Hz]
    LOG_RATE: float = 50.0         # navigation log rate [Hz]

    IMU_RATE: float = 200.0
    FLOW_RATE: float = 50.0
    LIDAR_RATE: float = 100.0
    BARO_RATE: float = 25.0
    CAMERA_RATE: float = 10.0

    # ------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------
    INITIAL_POSITION: Tuple[float, float, float] = (0.0, 0.0, 2.0)
    INITIAL_VELOCITY: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    GRAVITY: float = 9.81

    # ------------------------------------------------------------
    # IMU noise / bias
    # ------------------------------------------------------------
    ACCEL_NOISE: float = 0.05              # m/s^2 per prediction sample
    GYRO_NOISE: float = 0.003              # rad/s per prediction sample
    ACCEL_BIAS_RW: float = 5.0e-4          # accel bias random walk
    GYRO_BIAS_RW: float = 2.0e-5           # gyro bias random walk

    INITIAL_ACCEL_BIAS: Tuple[float, float, float] = (0.04, -0.03, 0.06)
    INITIAL_GYRO_BIAS: Tuple[float, float, float] = (0.003, -0.002, 0.001)

    # ------------------------------------------------------------
    # EKF process noise / initial covariance
    # ------------------------------------------------------------
    POS_PROCESS_NOISE: float = 1.0e-3
    VEL_PROCESS_NOISE: float = 5.0e-3
    ATT_PROCESS_NOISE: float = 1.0e-4

    P_INIT_POS: float = 0.25
    P_INIT_VEL: float = 0.09
    P_INIT_ATT: float = 0.01
    P_INIT_AB: float = 0.01
    P_INIT_GB: float = 1.0e-4

    # ------------------------------------------------------------
    # Optical flow
    # ------------------------------------------------------------
    FLOW_NOISE_BASE: float = 0.04
    FLOW_NOISE_ALT_GAIN: float = 0.008
    FLOW_MAX_ALT: float = 10.0
    FLOW_MIN_QUALITY: float = 0.35
    FLOW_QUALITY_BASE: float = 0.95

    # ------------------------------------------------------------
    # TF-Luna LiDAR
    # ------------------------------------------------------------
    LIDAR_NOISE: float = 0.03
    LIDAR_OUTLIER_PROB: float = 0.002
    LIDAR_INVALID_PROB: float = 0.001
    LIDAR_MIN_RANGE: float = 0.10
    LIDAR_MAX_RANGE: float = 8.00

    # ------------------------------------------------------------
    # Barometer
    # ------------------------------------------------------------
    BARO_NOISE: float = 0.35
    BARO_BIAS_RW: float = 1.0e-4

    # ------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------
    CAMERA_NOISE: float = 0.12

    # ------------------------------------------------------------
    # Innovation gates
    # ------------------------------------------------------------
    NIS_GATE_FLOW: float = 9.0
    NIS_GATE_LIDAR: float = 9.0
    NIS_GATE_BARO: float = 9.0
    NIS_GATE_CAMERA: float = 9.0

    # ------------------------------------------------------------
    # Sensor delays: (min, max) seconds
    # ------------------------------------------------------------
    IMU_DELAY: Tuple[float, float] = (0.0, 0.002)
    FLOW_DELAY: Tuple[float, float] = (0.010, 0.030)
    LIDAR_DELAY: Tuple[float, float] = (0.005, 0.015)
    BARO_DELAY: Tuple[float, float] = (0.020, 0.050)
    CAMERA_DELAY: Tuple[float, float] = (0.050, 0.150)

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    TARGET_RMSE: float = 0.50

    # ------------------------------------------------------------
    # Scenario / reproducibility
    # ------------------------------------------------------------
    scenario: Scenario = field(default_factory=Scenario)
    seed: int = 42

    # ------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------
    @staticmethod
    def _in_interval(t: float, start: float, duration: float) -> bool:
        return duration > 0.0 and start <= t < start + duration

    def camera_available(self, t: float) -> bool:
        return not self._in_interval(
            t,
            self.scenario.camera_dropout_start,
            self.scenario.camera_dropout_duration,
        )

    def flow_degraded(self, t: float) -> bool:
        return self._in_interval(
            t,
            self.scenario.flow_degradation_start,
            self.scenario.flow_degradation_duration,
        )

    def lidar_failed(self, t: float) -> bool:
        return self._in_interval(
            t,
            self.scenario.lidar_failure_start,
            self.scenario.lidar_failure_duration,
        )


def make_config(
    scenario: str = "normal",
    duration: Optional[float] = None,
    seed: int = 42,
) -> Config:
    """Create a scenario-specific configuration."""

    cfg = Config(seed=seed)
    if duration is not None:
        cfg.SIM_DURATION = float(duration)

    cfg.scenario.name = scenario

    if scenario == "camera_dropout":
        cfg.scenario.camera_dropout_start = 15.0
        cfg.scenario.camera_dropout_duration = 4.0

    elif scenario == "flow_failure":
        cfg.scenario.flow_degradation_start = 20.0
        cfg.scenario.flow_degradation_duration = 6.0

    elif scenario == "lidar_failure":
        cfg.scenario.lidar_failure_start = 20.0
        cfg.scenario.lidar_failure_duration = 5.0

    elif scenario == "baro_drift":
        cfg.scenario.baro_drift_rate = 0.08

    elif scenario == "combined":
        cfg.scenario.camera_dropout_start = 12.0
        cfg.scenario.camera_dropout_duration = 3.0

        cfg.scenario.flow_degradation_start = 25.0
        cfg.scenario.flow_degradation_duration = 4.0

        cfg.scenario.lidar_failure_start = 35.0
        cfg.scenario.lidar_failure_duration = 3.0

        cfg.scenario.baro_drift_rate = 0.04
        cfg.scenario.camera_outlier_time = 45.0

    else:
        # normal: no forced faults
        pass

    return cfg