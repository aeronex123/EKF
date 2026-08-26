import numpy as np

from sensors.imu import IMUSensor
from sensors.optical_flow import OpticalFlowSensor
from sensors.tf_luna import TFLunaSensor
from sensors.barometer import BarometerSensor
from sensors.camera import CameraSensor
from simulation.ground_truth import GroundTruth
from simulation.scheduler import SensorType


def test_imu_events(small_cfg):
    gt = GroundTruth(small_cfg)
    rng = np.random.default_rng(0)
    events = IMUSensor().generate_events(gt, small_cfg, rng)

    assert len(events) > 0
    assert events[0].sensor_type == SensorType.IMU
    assert "accel" in events[0].data
    assert "gyro" in events[0].data


def test_flow_events(small_cfg):
    gt = GroundTruth(small_cfg)
    rng = np.random.default_rng(0)
    events = OpticalFlowSensor().generate_events(gt, small_cfg, rng)

    assert len(events) > 0
    assert events[0].sensor_type == SensorType.FLOW
    assert "vx_body" in events[0].data
    assert "quality" in events[0].data


def test_lidar_events(small_cfg):
    gt = GroundTruth(small_cfg)
    rng = np.random.default_rng(0)
    events = TFLunaSensor().generate_events(gt, small_cfg, rng)

    assert len(events) > 0
    assert events[0].sensor_type == SensorType.LIDAR
    assert "range" in events[0].data


def test_baro_events(small_cfg):
    gt = GroundTruth(small_cfg)
    rng = np.random.default_rng(0)
    events = BarometerSensor().generate_events(gt, small_cfg, rng)

    assert len(events) > 0
    assert events[0].sensor_type == SensorType.BARO


def test_camera_events(small_cfg):
    gt = GroundTruth(small_cfg)
    rng = np.random.default_rng(0)
    events = CameraSensor().generate_events(gt, small_cfg, rng)

    assert len(events) > 0
    assert events[0].sensor_type == SensorType.CAMERA