from __future__ import annotations

from typing import List

import numpy as np

from config import Config
from simulation.scheduler import SensorEvent, SensorType


class OpticalFlowSensor:
    """
    Simulated PMW3901-style optical flow sensor.

    The sensor measures body-frame horizontal velocity.
    """

    def generate_events(
        self,
        gt,
        cfg: Config,
        rng: np.random.Generator,
    ) -> List[SensorEvent]:
        events: List[SensorEvent] = []
        step = max(1, int(round(cfg.SIM_RATE / cfg.FLOW_RATE)))

        for idx in range(0, gt.time.size, step):
            t = float(gt.time[idx])
            alt = float(gt.pos[idx, 2])

            degraded = cfg.flow_degraded(t)

            quality = (
                cfg.FLOW_QUALITY_BASE
                - 0.05 * max(0.0, alt - 2.0)
                - (0.70 if degraded else 0.0)
                + rng.normal(0.0, 0.03)
            )
            quality = float(np.clip(quality, 0.0, 1.0))

            sigma = cfg.FLOW_NOISE_BASE + cfg.FLOW_NOISE_ALT_GAIN * max(0.0, alt - 2.0)
            if degraded:
                sigma *= 3.0

            v_body_true = gt.v_body[idx, 0:2]
            meas = v_body_true + rng.normal(0.0, sigma, size=2)

            valid = (
                alt >= 0.15
                and alt <= cfg.FLOW_MAX_ALT
                and quality >= cfg.FLOW_MIN_QUALITY
            )

            delay = float(rng.uniform(*cfg.FLOW_DELAY))
            arrival = t + delay

            if arrival <= cfg.SIM_DURATION + 0.25:
                events.append(
                    SensorEvent(
                        arrival_time=arrival,
                        timestamp=t,
                        sensor_type=SensorType.FLOW,
                        data={
                            "vx_body": float(meas[0]),
                            "vy_body": float(meas[1]),
                            "quality": quality,
                            "valid": bool(valid),
                            "sigma": float(sigma),
                        },
                    )
                )

        return events