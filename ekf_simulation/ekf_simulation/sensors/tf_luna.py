from __future__ import annotations

from typing import List

import numpy as np

from config import Config
from simulation.scheduler import SensorEvent, SensorType


class TFLunaSensor:
    """Simulated TF-Luna downward-facing LiDAR rangefinder."""

    def generate_events(
        self,
        gt,
        cfg: Config,
        rng: np.random.Generator,
    ) -> List[SensorEvent]:
        events: List[SensorEvent] = []
        step = max(1, int(round(cfg.SIM_RATE / cfg.LIDAR_RATE)))

        for idx in range(0, gt.time.size, step):
            t = float(gt.time[idx])
            alt = float(gt.pos[idx, 2])

            failure_active = cfg.lidar_failed(t)

            invalid = False
            if failure_active and rng.random() < 0.70:
                range_meas = -1.0
                invalid = True
            elif rng.random() < cfg.LIDAR_INVALID_PROB:
                range_meas = -1.0
                invalid = True
            elif rng.random() < cfg.LIDAR_OUTLIER_PROB:
                sign = float(rng.choice([-1.0, 1.0]))
                range_meas = alt + sign * float(rng.uniform(0.7, 2.5))
            else:
                range_meas = alt + float(rng.normal(0.0, cfg.LIDAR_NOISE))

            if range_meas < cfg.LIDAR_MIN_RANGE or range_meas > cfg.LIDAR_MAX_RANGE:
                invalid = True

            delay = float(rng.uniform(*cfg.LIDAR_DELAY))
            arrival = t + delay

            if arrival <= cfg.SIM_DURATION + 0.25:
                events.append(
                    SensorEvent(
                        arrival_time=arrival,
                        timestamp=t,
                        sensor_type=SensorType.LIDAR,
                        data={
                            "range": float(range_meas),
                            "valid": bool(not invalid),
                            "sigma": float(cfg.LIDAR_NOISE),
                        },
                    )
                )

        return events