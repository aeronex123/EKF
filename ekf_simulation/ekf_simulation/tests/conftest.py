import pytest

from config import Config


@pytest.fixture
def small_cfg():
    cfg = Config()
    cfg.SIM_DURATION = 5.0
    cfg.SIM_RATE = 200.0
    cfg.IMU_RATE = 100.0
    cfg.FLOW_RATE = 50.0
    cfg.LIDAR_RATE = 50.0
    cfg.BARO_RATE = 25.0
    cfg.CAMERA_RATE = 5.0
    cfg.LOG_RATE = 20.0
    return cfg