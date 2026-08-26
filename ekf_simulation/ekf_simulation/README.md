# GPS-Denied EKF Localization Simulation

ArduPilot-inspired EKF simulation for GPS-denied UAV localization over a 60 m leg.

## Sensors

- IMU @ 200 Hz
- PMW3901-style optical flow @ 50 Hz
- TF-Luna-style LiDAR @ 100 Hz
- Barometer @ 25 Hz
- Camera position fixes @ 10 Hz

## Notes

The original specification requested a folder named `logging/`.
This project uses `sim_logging/` instead because a top-level Python package named
`logging` shadows the standard library and can break matplotlib/pytest.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt