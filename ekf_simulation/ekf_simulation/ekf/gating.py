from __future__ import annotations

import numpy as np


def accept_nis(nis: float, gate: float) -> bool:
    """Accept measurement if normalized innovation squared is inside gate."""
    return bool(np.isfinite(nis) and nis <= gate)


def chi_square_threshold(dim: int, confidence: float = 0.99) -> float:
    """
    Optional chi-square gate helper.

    Uses scipy only when called.
    """
    from scipy.stats import chi2

    return float(chi2.ppf(confidence, dim))