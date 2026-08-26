from __future__ import annotations

import numpy as np

from ekf.jacobians import F_matrix
from ekf.models import quat_integrate, quat_to_dcm
from ekf.state import IDX_BA, IDX_BG, IDX_P, IDX_TH, IDX_V, N_ERROR, NavigationState


def process_noise(cfg, dt: float) -> np.ndarray:
    Q = np.zeros((N_ERROR, N_ERROR), dtype=float)
    I3 = np.eye(3, dtype=float)

    qa = cfg.ACCEL_NOISE**2
    qg = cfg.GYRO_NOISE**2

    # Position/velocity process noise due to accelerometer noise.
    Q[IDX_P, IDX_P] = (dt**3 / 3.0) * qa * I3 + cfg.POS_PROCESS_NOISE * dt * I3
    Q[IDX_V, IDX_V] = dt * qa * I3 + cfg.VEL_PROCESS_NOISE * dt * I3

    cross = 0.5 * dt * dt * qa * I3
    Q[IDX_P, IDX_V] = cross
    Q[IDX_V, IDX_P] = cross.T

    # Attitude process noise due to gyro noise/tuning.
    Q[IDX_TH, IDX_TH] = dt * qg * I3 + cfg.ATT_PROCESS_NOISE * dt * I3

    # Bias random walks.
    Q[IDX_BA, IDX_BA] = (cfg.ACCEL_BIAS_RW**2) * dt * I3
    Q[IDX_BG, IDX_BG] = (cfg.GYRO_BIAS_RW**2) * dt * I3

    return Q


def predict_nominal(
    state: NavigationState,
    P: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    dt: float,
    cfg,
) -> np.ndarray:
    """
    EKF IMU prediction step.

    Returns propagated covariance.
    Mutates nominal state.
    """
    if dt <= 0.0:
        return P

    a_corr = accel - state.ba
    omega = gyro - state.bg

    R = quat_to_dcm(state.q)

    F = F_matrix(R, a_corr, omega, dt, cfg)
    Q = process_noise(cfg, dt)

    P_new = F @ P @ F.T + Q
    P_new = 0.5 * (P_new + P_new.T)

    g_nav = np.array([0.0, 0.0, -cfg.GRAVITY], dtype=float)
    a_nav = R @ a_corr + g_nav

    state.p = state.p + state.v * dt + 0.5 * a_nav * dt * dt
    state.v = state.v + a_nav * dt
    state.q = quat_integrate(state.q, omega, dt)
    state.time += dt

    return P_new