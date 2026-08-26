from __future__ import annotations

from typing import List

import numpy as np

from config import Config
from simulation.scheduler import SensorEvent, SensorType


class BarometerSensor:
    """Simulated barometer with slow drift and white noise."""

    def generate_events(
        self,
        gt,
        cfg: Config,
        rng: np.random.Generator,
    ) -> List[SensorEvent]:
        events: List[SensorEvent] = []
        step = max(1, int(round(cfg.SIM_RATE / cfg.BARO_RATE)))
        dt = 1.0 / cfg.BARO_RATE

        bias = 0.0

        for idx in range(0, gt.time.size, step):
            t = float(gt.time[idx])
            alt = float(gt.pos[idx, 2])

            if t > 5.0:
                bias += cfg.scenario.baro_drift_rate * dt

            bias += float(rng.normal(0.0, cfg.BARO_BIAS_RW * np.sqrt(dt)))

            meas = alt + bias + float(rng.normal(0.0, cfg.BARO_NOISE))

            delay = float(rng.uniform(*cfg.BARO_DELAY))
            arrival = t + delay

            if arrival <= cfg.SIM_DURATION + 0.25:
                events.append(
                    SensorEvent(
                        arrival_time=arrival,
                        timestamp=t,
                        sensor_type=SensorType.BARO,
                        data={
                            "alt": float(meas),
                            "valid": True,
                            "sigma": float(cfg.BARO_NOISE),
                        },
                    )
                )

        return events