from __future__ import annotations

from typing import List

import numpy as np

from config import Config
from simulation.scheduler import SensorEvent, SensorType


class CameraSensor:
    """
    Simulated camera / visual odometry / landmark position fix.

    Provides [px, py] observations with dropout and optional outlier.
    """

    def generate_events(
        self,
        gt,
        cfg: Config,
        rng: np.random.Generator,
    ) -> List[SensorEvent]:
        events: List[SensorEvent] = []
        step = max(1, int(round(cfg.SIM_RATE / cfg.CAMERA_RATE)))

        for idx in range(0, gt.time.size, step):
            t = float(gt.time[idx])

            if not cfg.camera_available(t):
                continue

            xy = gt.pos[idx, 0:2] + rng.normal(0.0, cfg.CAMERA_NOISE, size=2)

            if (
                cfg.scenario.camera_outlier_time is not None
                and abs(t - cfg.scenario.camera_outlier_time)
                <= 0.5 / cfg.CAMERA_RATE
            ):
                xy += np.array([1.8, -1.2], dtype=float)

            delay = float(rng.uniform(*cfg.CAMERA_DELAY))
            arrival = t + delay

            if arrival <= cfg.SIM_DURATION + 0.25:
                events.append(
                    SensorEvent(
                        arrival_time=arrival,
                        timestamp=t,
                        sensor_type=SensorType.CAMERA,
                        data={
                            "px": float(xy[0]),
                            "py": float(xy[1]),
                            "valid": True,
                            "sigma": float(cfg.CAMERA_NOISE),
                        },
                    )
                )

        return events