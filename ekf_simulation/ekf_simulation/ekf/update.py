from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ekf.gating import accept_nis
from ekf.models import quat_from_small, quat_multiply, quat_normalize
from ekf.state import IDX_BA, IDX_BG, IDX_P, IDX_TH, IDX_V, NavigationState


@dataclass
class UpdateResult:
    sensor: str
    time: float
    timestamp: float
    innovation: np.ndarray
    nis: float
    accepted: bool
    reason: str
    measurement: np.ndarray


def joseph_update(
    P: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Joseph-form covariance update for numerical stability."""
    I = np.eye(P.shape[0], dtype=float)
    IKH = I - K @ H
    return IKH @ P @ IKH.T + K @ R @ K.T


def inject_error(state: NavigationState, dx: np.ndarray) -> None:
    """Inject error-state correction into nominal state."""
    state.p += dx[IDX_P]
    state.v += dx[IDX_V]

    dtheta = dx[IDX_TH]
    state.q = quat_normalize(quat_multiply(state.q, quat_from_small(dtheta)))

    state.ba += dx[IDX_BA]
    state.bg += dx[IDX_BG]

    # Sanity bounds to keep the simulation robust.
    state.ba = np.clip(state.ba, -1.0, 1.0)
    state.bg = np.clip(state.bg, -0.2, 0.2)


def measurement_update(
    state: NavigationState,
    P: np.ndarray,
    z: np.ndarray,
    h: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    gate: float,
    pre_reject_reason: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float, bool, str]:
    """
    Generic EKF measurement update with innovation gating.

    Returns:
      P_new, innovation, NIS, accepted, reason
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    h = np.atleast_1d(np.asarray(h, dtype=float))

    y = z - h
    S = H @ P @ H.T + R
    S = 0.5 * (S + S.T)

    try:
        S_inv = np.linalg.inv(S)
        nis = float(y @ S_inv @ y)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        nis = float(y @ S_inv @ y) if np.all(np.isfinite(y)) else float("nan")

    accepted = True
    reason = "ok"

    if pre_reject_reason is not None:
        accepted = False
        reason = pre_reject_reason
    elif not accept_nis(nis, gate):
        accepted = False
        reason = "gate"

    if not accepted:
        return P, y, nis, accepted, reason

    K = P @ H.T @ S_inv
    dx = K @ y

    P_new = joseph_update(P, H, R, K)
    P_new = 0.5 * (P_new + P_new.T)

    # Enforce non-negative diagonal variances.
    diag = np.diag_indices_from(P_new)
    P_new[diag] = np.maximum(P_new[diag], 0.0)

    inject_error(state, dx)

    return P_new, y, nis, accepted, reason