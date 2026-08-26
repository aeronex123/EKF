from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r

    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ],
        dtype=float,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Quaternion body -> nav to DCM body -> nav."""
    q = quat_normalize(q)
    w, x, y, z = q

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def quat_from_dcm(R: np.ndarray) -> np.ndarray:
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s

    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return quat_normalize(np.array([w, x, y, z], dtype=float))


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Propagate quaternion by body angular rate."""
    theta = float(np.linalg.norm(omega) * dt)

    if theta < 1.0e-12:
        delta = np.array(
            [1.0, 0.5 * omega[0] * dt, 0.5 * omega[1] * dt, 0.5 * omega[2] * dt],
            dtype=float,
        )
    else:
        axis = omega / np.linalg.norm(omega)
        delta = np.array(
            [
                np.cos(0.5 * theta),
                axis[0] * np.sin(0.5 * theta),
                axis[1] * np.sin(0.5 * theta),
                axis[2] * np.sin(0.5 * theta),
            ],
            dtype=float,
        )

    return quat_normalize(quat_multiply(q, delta))


def quat_from_small(theta: np.ndarray) -> np.ndarray:
    """Quaternion from small attitude error vector."""
    angle = float(np.linalg.norm(theta))
    if angle < 1.0e-12:
        return quat_normalize(
            np.array([1.0, 0.5 * theta[0], 0.5 * theta[1], 0.5 * theta[2]], dtype=float)
        )

    axis = theta / angle
    return quat_normalize(
        np.array(
            [
                np.cos(0.5 * angle),
                axis[0] * np.sin(0.5 * angle),
                axis[1] * np.sin(0.5 * angle),
                axis[2] * np.sin(0.5 * angle),
            ],
            dtype=float,
        )
    )


def attitude_from_specific_force_yaw(f_nav: np.ndarray, yaw: float) -> np.ndarray:
    """
    Construct attitude quaternion from desired specific-force direction and yaw.

    Body frame:
      x: forward
      y: left
      z: up
    """
    f_norm = float(np.linalg.norm(f_nav))
    if f_norm < 1.0e-6:
        z_b = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        z_b = f_nav / f_norm

    x_h = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
    x_b = x_h - np.dot(x_h, z_b) * z_b

    if np.linalg.norm(x_b) < 1.0e-6:
        x_b = np.array([1.0, 0.0, 0.0], dtype=float) - z_b[0] * z_b
        if np.linalg.norm(x_b) < 1.0e-6:
            x_b = np.array([0.0, 1.0, 0.0], dtype=float) - z_b[1] * z_b

    x_b /= np.linalg.norm(x_b)
    y_b = np.cross(z_b, x_b)
    y_b /= np.linalg.norm(y_b)

    R = np.column_stack([x_b, y_b, z_b])
    return quat_from_dcm(R)


def angular_velocity_from_quats(q0: np.ndarray, q1: np.ndarray, dt: float) -> np.ndarray:
    """Body angular velocity from q0 -> q1."""
    if dt <= 0.0:
        return np.zeros(3, dtype=float)

    delta = quat_multiply(quat_conjugate(q0), q1)
    if delta[0] < 0.0:
        delta = -delta

    vec = delta[1:4]
    norm_vec = float(np.linalg.norm(vec))

    if norm_vec < 1.0e-12:
        return 2.0 * vec / dt

    angle = 2.0 * np.arctan2(norm_vec, delta[0])
    axis = vec / norm_vec
    return axis * angle / dt