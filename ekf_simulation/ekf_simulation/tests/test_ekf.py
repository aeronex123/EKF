import numpy as np

from config import Config, make_config
from ekf.filter import EKF
from simulation.scheduler import SensorEvent, SensorType


def make_imu_event(t):
    return SensorEvent(
        arrival_time=t,
        timestamp=t,
        sensor_type=SensorType.IMU,
        data={
            "accel": np.array([0.0, 0.0, 9.81]),
            "gyro": np.zeros(3),
        },
    )


def test_state_initialization():
    cfg = Config()
    ekf = EKF(cfg)

    assert ekf.state.p.shape == (3,)
    assert ekf.state.v.shape == (3,)
    assert ekf.state.q.shape == (4,)
    assert ekf.P.shape == (15, 15)
    assert np.allclose(ekf.P, ekf.P.T)


def test_prediction_step():
    cfg = Config()
    ekf = EKF(cfg)

    p_before = ekf.state.p.copy()
    ekf.process_event(make_imu_event(0.01))

    assert ekf.state.time > 0.0
    assert np.linalg.norm(ekf.state.p - p_before) < 1.0e-6


def test_covariance_propagation():
    cfg = Config()
    ekf = EKF(cfg)
    trace_before = np.trace(ekf.P)

    ekf.process_event(make_imu_event(0.01))

    assert np.allclose(ekf.P, ekf.P.T)
    assert np.trace(ekf.P) > trace_before
    assert np.all(np.diag(ekf.P) >= 0.0)


def test_flow_update_reduces_velocity_covariance():
    cfg = Config()
    ekf = EKF(cfg)

    ekf.state.v[:] = np.array([1.0, 0.0, 0.0])

    event = SensorEvent(
        arrival_time=0.01,
        timestamp=0.01,
        sensor_type=SensorType.FLOW,
        data={
            "vx_body": 1.0,
            "vy_body": 0.0,
            "quality": 0.9,
            "valid": True,
            "sigma": 0.04,
        },
    )

    p_before = ekf.P[3, 3]
    ekf.process_event(event)

    assert ekf.P[3, 3] < p_before


def test_lidar_update():
    cfg = Config()
    ekf = EKF(cfg)
    ekf.state.p[2] = 3.0

    event = SensorEvent(
        arrival_time=0.01,
        timestamp=0.01,
        sensor_type=SensorType.LIDAR,
        data={"range": 2.0, "valid": True, "sigma": 0.03},
    )

    p_before = ekf.P[2, 2]
    ekf.process_event(event)

    assert ekf.P[2, 2] < p_before


def test_baro_update():
    cfg = Config()
    ekf = EKF(cfg)
    ekf.state.p[2] = 3.0

    event = SensorEvent(
        arrival_time=0.01,
        timestamp=0.01,
        sensor_type=SensorType.BARO,
        data={"alt": 2.0, "valid": True, "sigma": 0.35},
    )

    p_before = ekf.P[2, 2]
    ekf.process_event(event)

    assert ekf.P[2, 2] < p_before


def test_camera_update():
    cfg = Config()
    ekf = EKF(cfg)
    ekf.state.p[:2] = np.array([1.0, 2.0])

    event = SensorEvent(
        arrival_time=0.01,
        timestamp=0.01,
        sensor_type=SensorType.CAMERA,
        data={"px": 0.0, "py": 0.0, "valid": True, "sigma": 0.12},
    )

    p_before = ekf.P[0, 0]
    ekf.process_event(event)

    assert ekf.P[0, 0] < p_before


def test_camera_outlier_rejected():
    cfg = Config()
    ekf = EKF(cfg)
    ekf.state.p[:2] = np.array([0.0, 0.0])

    event = SensorEvent(
        arrival_time=0.01,
        timestamp=0.01,
        sensor_type=SensorType.CAMERA,
        data={"px": 20.0, "py": -20.0, "valid": True, "sigma": 0.12},
    )

    result = ekf.process_event(event)

    assert result is not None
    assert not result.accepted
    assert result.reason == "gate"