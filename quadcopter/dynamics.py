from __future__ import annotations

"""dynamics.py – Core quadcopter rigid‑body dynamics.

Defines
-------
Params      – physical constants of the vehicle
QuadState   – convenient container for the 13‑dimensional state

derivative  – right‑hand side f(t, y, u, p) for SciPy `solve_ivp`
"""

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases for readability
# ---------------------------------------------------------------------------
Vec3 = NDArray[np.float64]
Quat = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Physical parameters (all values can be overridden when calling derivative)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    m: float = 0.65                      # mass [kg]
    g: float = 9.81                      # gravity [m/s²]
    l: float = 0.25                      # arm length (CoM → rotor) [m]
    b: float = 3.25e-5                   # thrust constant [N·s²]
    d: float = 7.5e-7                    # drag (yaw) constant [N·m·s²]
    I: NDArray = field(
        default_factory=lambda: np.diag([7.5e-3, 7.5e-3, 1.3e-2])
    )  # inertia matrix [kg·m²]


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------
@dataclass
class QuadState:
    """Continuous‑time state of the quadcopter.

    Attributes
    ----------
    pos : (3,) array – position in world frame  [m]
    vel : (3,) array – linear velocity in world frame  [m/s]
    quat : (4,) array – unit quaternion (w, x, y, z) body → world
    ang_vel : (3,) array – angular velocity in body frame  [rad/s]
    """

    pos: Vec3
    vel: Vec3
    quat: Quat  # (w, x, y, z)
    ang_vel: Vec3

    # -------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------
    def as_vector(self) -> NDArray[np.float64]:
        """Pack to 13‑element 1‑D array: [pos, vel, quat, ang_vel]."""
        return np.hstack((self.pos, self.vel, self.quat, self.ang_vel))

    @staticmethod
    def from_vector(v: NDArray[np.float64]) -> "QuadState":
        assert v.shape == (13,), "State vector must have shape (13,)"
        return QuadState(
            pos=v[0:3],
            vel=v[3:6],
            quat=_normalise_quat(v[6:10]),
            ang_vel=v[10:13],
        )


# ---------------------------------------------------------------------------
# Quaternion utilities (minimal set)
# ---------------------------------------------------------------------------

def _quat_mul(q: Quat, r: Quat) -> Quat:
    """Hamilton product q ⊗ r."""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_rotm(q: Quat) -> NDArray[np.float64]:
    """Convert unit quaternion to 3×3 rotation matrix (body → world)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=np.float64,
    )


def _normalise_quat(q: Quat) -> Quat:
    """Return q / ‖q‖ (safe for zero‑division)."""
    norm = np.linalg.norm(q)
    return q if norm == 0 else q / norm


# ---------------------------------------------------------------------------
# Core ODE – f(t, y, u, p)
# ---------------------------------------------------------------------------

def derivative(
    t: float,
    y: NDArray[np.float64],
    control: NDArray[np.float64],
    p: Params = Params(),
) -> NDArray[np.float64]:
    """Quadcopter 6‑DoF rigid‑body dynamics.

    Parameters
    ----------
    t : float
        Current time [s] (unused but kept for SciPy API).
    y : (13,) ndarray
        Packed state vector [x, y, z,  ẋ, ẏ, ż,  qw, qx, qy, qz,  p, q, r].
    control : (4,) ndarray
        Motor angular speeds [rad/s] ⇒ w1, w2, w3, w4.
    p : Params, optional
        Physical parameters. Supply your own to simulate a different drone.

    Returns
    -------
    (13,) ndarray
        Time derivative ẏ.
    """

    state = QuadState.from_vector(y)
    w1, w2, w3, w4 = control

    # Thrust and body torques ------------------------------------------
    thrust = p.b * (w1**2 + w2**2 + w3**2 + w4**2)
    tau = np.array(
        [
            p.l * p.b * (w3**2 - w1**2),
            p.l * p.b * (w4**2 - w2**2),
            p.d * (w2**2 + w4**2 - w1**2 - w3**2),
        ],
        dtype=np.float64,
    )

    # Translation -------------------------------------------------------
    R = _quat_to_rotm(state.quat)               # body → world
    accel = np.array([0.0, 0.0, -p.g], dtype=np.float64) + R @ np.array(
        [0.0, 0.0, thrust / p.m], dtype=np.float64
    )

    # Rotation ----------------------------------------------------------
    ang_acc = np.linalg.solve(
        p.I,
        tau - np.cross(state.ang_vel, p.I @ state.ang_vel)
    )

    # Quaternion derivative --------------------------------------------
    ang_quat = np.concatenate(([0.0], state.ang_vel))  # (0, p, q, r)
    quat_dot = 0.5 * _quat_mul(state.quat, ang_quat)

    # Pack and return ---------------------------------------------------
    return np.hstack((state.vel, accel, quat_dot, ang_acc))


__all__ = ["Params", "QuadState", "derivative"]
