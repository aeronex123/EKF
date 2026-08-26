from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config

# Error-state dimension:
# [δp(3), δv(3), δθ(3), δba(3), δbg(3)]
N_ERROR = 15

IDX_P = slice(0, 3)
IDX_V = slice(3, 6)
IDX_TH = slice(6, 9)
IDX_BA = slice(9, 12)
IDX_BG = slice(12, 15)


@dataclass
class NavigationState:
    """Nominal navigation state."""

    p: np.ndarray
    v: np.ndarray
    q: np.ndarray       # quaternion body -> nav, [w, x, y, z]
    ba: np.ndarray
    bg: np.ndarray
    time: float = 0.0


def make_initial_state(cfg: Config) -> NavigationState:
    return NavigationState(
        p=np.array(cfg.INITIAL_POSITION, dtype=float),
        v=np.array(cfg.INITIAL_VELOCITY, dtype=float),
        q=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        ba=np.zeros(3, dtype=float),
        bg=np.zeros(3, dtype=float),
        time=0.0,
    )


def make_initial_covariance(cfg: Config) -> np.ndarray:
    P = np.zeros((N_ERROR, N_ERROR), dtype=float)

    P[IDX_P, IDX_P] = cfg.P_INIT_POS * np.eye(3)
    P[IDX_V, IDX_V] = cfg.P_INIT_VEL * np.eye(3)
    P[IDX_TH, IDX_TH] = cfg.P_INIT_ATT * np.eye(3)
    P[IDX_BA, IDX_BA] = cfg.P_INIT_AB * np.eye(3)
    P[IDX_BG, IDX_BG] = cfg.P_INIT_GB * np.eye(3)

    return P