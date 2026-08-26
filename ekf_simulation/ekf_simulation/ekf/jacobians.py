from __future__ import annotations

import numpy as np

from ekf.models import skew
from ekf.state import IDX_BA, IDX_BG, IDX_P, IDX_TH, IDX_V, N_ERROR


def F_matrix(
    R: np.ndarray,
    a_corr: np.ndarray,
    omega: np.ndarray,
    dt: float,
    cfg,
) -> np.ndarray:
    """
    Discrete error-state transition Jacobian.

    Error state:
      [δp, δv, δθ, δba, δbg]

    Nominal acceleration:
      a = R (a_meas - ba) + g

    Attitude perturbation convention:
      q_true = q_nom ⊗ δq
      R_true ≈ R_nom (I + [δθ]x)
    """
    F = np.eye(N_ERROR, dtype=float)
    I3 = np.eye(3, dtype=float)

    # δp <- δv
    F[IDX_P, IDX_V] = dt * I3

    # δv <- δθ
    # a_true ≈ a_nom - R [a_corr]x δθ
    F[IDX_V, IDX_TH] = -R @ skew(a_corr) * dt

    # δv <- δba
    F[IDX_V, IDX_BA] = -R * dt

    # δp <- δθ
    F[IDX_P, IDX_TH] = -0.5 * R @ skew(a_corr) * dt * dt

    # δp <- δba
    F[IDX_P, IDX_BA] = -0.5 * R * dt * dt

    # δθ <- δθ
    F[IDX_TH, IDX_TH] = I3 - skew(omega) * dt

    # δθ <- δbg
    F[IDX_TH, IDX_BG] = -I3 * dt

    return F


def H_camera() -> np.ndarray:
    H = np.zeros((2, N_ERROR), dtype=float)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    return H


def H_lidar() -> np.ndarray:
    H = np.zeros((1, N_ERROR), dtype=float)
    H[0, 2] = 1.0
    return H


def H_baro() -> np.ndarray:
    H = np.zeros((1, N_ERROR), dtype=float)
    H[0, 2] = 1.0
    return H


def H_flow(R: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """
    Optical flow measures body-frame horizontal velocity:

      z_flow = (Rᵀ v_nav)[0:2]

    Attitude perturbation:
      v_b_true ≈ v_b_nom + [v_b_nom]x δθ
    """
    H = np.zeros((2, N_ERROR), dtype=float)

    H[:, IDX_V] = R.T[:2, :]
    H[:, IDX_TH] = skew(v_body)[:2, :]

    return H