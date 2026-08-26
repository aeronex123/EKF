from __future__ import annotations

from typing import List

import numpy as np

from config import Config
from simulation.scheduler import SensorEvent, SensorType


class IMUSensor:
    """Simulated IMU with white noise and random-walk biases."""

    def generate_events(
        self,
        gt,
        cfg: Config,
        rng: np.random.Generator,
    ) -> List[SensorEvent]:
        events: List[SensorEvent] = []

        step = max(1, int(round(cfg.SIM_RATE / cfg.IMU_RATE)))
        dt = 1.0 / cfg.IMU_RATE

        ba = np.array(cfg.INITIAL_ACCEL_BIAS, dtype=float)
        bg = np.array(cfg.INITIAL_GYRO_BIAS, dtype=float)

        for idx in range(0, gt.time.size, step):
            t = float(gt.time[idx])

            ba += rng.normal(0.0, cfg.ACCEL_BIAS_RW * np.sqrt(dt), size=3)
            bg += rng.normal(0.0, cfg.GYRO_BIAS_RW * np.sqrt(dt), size=3)

            accel = (
                gt.accel_body[idx]
                + ba
                + rng.normal(0.0, cfg.ACCEL_NOISE, size=3)
            )
            gyro = (
                gt.gyro[idx]
                + bg
                + rng.normal(0.0, cfg.GYRO_NOISE, size=3)
            )

            delay = float(rng.uniform(*cfg.IMU_DELAY))
            arrival = t + delay

            if arrival <= cfg.SIM_DURATION + 0.25:
                events.append(
                    SensorEvent(
                        arrival_time=arrival,
                        timestamp=t,
                        sensor_type=SensorType.IMU,
                        data={
                            "accel": accel.astype(float),
                            "gyro": gyro.astype(float),
                        },
                    )
                )

        return events