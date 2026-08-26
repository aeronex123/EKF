import numpy as np
import pandas as pd

from config import make_config
from ekf.gating import accept_nis
from evaluation.metrics import compute_metrics


def test_accept_nis():
    assert accept_nis(1.0, 9.0)
    assert not accept_nis(10.0, 9.0)
    assert not accept_nis(float("nan"), 9.0)


def test_camera_dropout_config():
    cfg = make_config("camera_dropout", duration=30.0, seed=1)

    assert cfg.camera_available(5.0)
    assert not cfg.camera_available(16.0)
    assert cfg.camera_available(25.0)


def test_rmse_calculation():
    df = pd.DataFrame({"position_error": [0.0, 1.0]})
    metrics = compute_metrics(df)

    assert np.isclose(metrics["rmse"], np.sqrt(0.5))
    assert np.isclose(metrics["mae"], 0.5)
    assert np.isclose(metrics["max_error"], 1.0)