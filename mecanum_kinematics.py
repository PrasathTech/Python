"""Mecanum-wheel robot kinematics (4 wheels, X-configuration).

Frame: x forward, y left, theta counter-clockwise (rad).
Wheel order: front-left (FL), front-right (FR), rear-left (RL), rear-right (RR).

Parameters:
    r  - wheel radius (m)
    lx - half of wheelbase (front-rear distance / 2) (m)
    ly - half of track width (left-right distance / 2) (m)
"""

from __future__ import annotations

import numpy as np


def _jacobian(r: float, lx: float, ly: float) -> np.ndarray:
    """Matrix J such that wheel_speeds = J @ [vx, vy, wz]."""
    k = lx + ly
    return (1.0 / r) * np.array(
        [
            [1.0, -1.0, -k],  # FL
            [1.0, 1.0, k],  # FR
            [1.0, 1.0, -k],  # RL
            [1.0, -1.0, k],  # RR
        ]
    )


def inverse_kinematics(vx: float, vy: float, wz: float, r: float, lx: float, ly: float) -> np.ndarray:
    """Body velocity (vx, vy, wz) -> wheel angular speeds [FL, FR, RL, RR] in rad/s."""
    return _jacobian(r, lx, ly) @ np.array([vx, vy, wz])


def forward_kinematics(wheels: np.ndarray, r: float, lx: float, ly: float) -> np.ndarray:
    """Wheel angular speeds [FL, FR, RL, RR] -> body velocity [vx, vy, wz] (least squares)."""
    return np.linalg.pinv(_jacobian(r, lx, ly)) @ np.asarray(wheels, dtype=float)


def integrate_pose(pose: np.ndarray, body_vel: np.ndarray, dt: float) -> np.ndarray:
    """Dead-reckon one step. pose = [x, y, theta] in world frame; body_vel in body frame."""
    x, y, th = pose
    vx, vy, wz = body_vel
    c, s = np.cos(th), np.sin(th)
    return np.array(
        [
            x + (vx * c - vy * s) * dt,
            y + (vx * s + vy * c) * dt,
            th + wz * dt,
        ]
    )


if __name__ == "__main__":
    R, LX, LY = 0.05, 0.15, 0.12

    # Round-trip check: IK then FK should recover the commanded velocity.
    cmd = np.array([0.4, -0.2, 0.5])
    w = inverse_kinematics(*cmd, R, LX, LY)
    back = forward_kinematics(w, R, LX, LY)
    assert np.allclose(cmd, back), (cmd, back)
    print("wheel speeds [FL FR RL RR] (rad/s):", np.round(w, 3))
    print("recovered body velocity:", np.round(back, 3))

    # Pure sideways motion: wheels should alternate sign.
    print("strafe left 0.3 m/s:", np.round(inverse_kinematics(0, 0.3, 0, R, LX, LY), 3))

    # Simulate 2 s of motion.
    pose = np.zeros(3)
    for _ in range(200):
        pose = integrate_pose(pose, forward_kinematics(w, R, LX, LY), 0.01)
    print("pose after 2 s [x y theta]:", np.round(pose, 3))
