from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List

import numpy as np

from config import Config


class SensorType(Enum):
    IMU = auto()
    FLOW = auto()
    LIDAR = auto()
    BARO = auto()
    CAMERA = auto()
    LOG = auto()


@dataclass
class SensorEvent:
    """
    Time-stamped sensor event.

    timestamp: physical measurement validity time
    arrival_time: time when the measurement is available to the filter
    """

    arrival_time: float
    timestamp: float
    sensor_type: SensorType
    data: Dict[str, Any]


# Import after SensorType/SensorEvent are defined to avoid circular imports.
from sensors.imu import IMUSensor
from sensors.optical_flow import OpticalFlowSensor
from sensors.tf_luna import TFLunaSensor
from sensors.barometer import BarometerSensor
from sensors.camera import CameraSensor


class SensorScheduler:
    """Builds a chronologically sorted asynchronous sensor event stream."""

    def build_events(self, gt, cfg: Config, rng: np.random.Generator) -> List[SensorEvent]:
        events: List[SensorEvent] = []

        events.extend(IMUSensor().generate_events(gt, cfg, rng))
        events.extend(OpticalFlowSensor().generate_events(gt, cfg, rng))
        events.extend(TFLunaSensor().generate_events(gt, cfg, rng))
        events.extend(BarometerSensor().generate_events(gt, cfg, rng))
        events.extend(CameraSensor().generate_events(gt, cfg, rng))

        # Periodic navigation logging events.
        log_step = max(1, int(round(cfg.SIM_RATE / cfg.LOG_RATE)))
        for idx in range(0, gt.time.size, log_step):
            t = float(gt.time[idx])
            if t <= cfg.SIM_DURATION + 1.0e-9:
                events.append(
                    SensorEvent(
                        arrival_time=t,
                        timestamp=t,
                        sensor_type=SensorType.LOG,
                        data={},
                    )
                )

        events.sort(key=lambda e: e.arrival_time)
        return events