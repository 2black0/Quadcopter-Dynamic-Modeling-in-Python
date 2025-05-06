"""
env.py – interactive environment wrapper for the quadcopter dynamics core.

Typical loop
------------
>>> from quadcopter.env import QuadcopterEnv
>>> from quadcopter.controllers import AltitudePID
>>> env = QuadcopterEnv(dt=0.02)
>>> ctrl = AltitudePID(setpoint=1.0)
>>> obs = env.reset()                       # dict with state vectors
>>> for _ in range(200):                    # simulate 4 s
...     u = ctrl.update(env.t, env.state)   # external controller
...     obs = env.step(u)                   # advance one step
...     print(obs["pos"][2])                # altitude in metres
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias 

from .dynamics import Params, QuadState, derivative

Vec: TypeAlias = NDArray[np.float64] 


@dataclass
class QuadcopterEnv:
    """Fixed‑step RK4 interactive environment."""

    dt: float = 0.02
    params: Params = field(default_factory=Params)

    # internal simulation state
    t: float = 0.0
    state: QuadState = field(
        default_factory=lambda: QuadState(
            pos=np.zeros(3),
            vel=np.zeros(3),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ang_vel=np.zeros(3),
        )
    )

    def reset(self, *, state: QuadState | None = None) -> dict[str, Vec]:
        """Reset to t=0 and return initial observation."""
        self.t = 0.0
        if state is not None:
            self.state = state
        return self._observation()

    # ------------------------------------------------------------------
    # PUBLIC CORE
    # ------------------------------------------------------------------
    def step(self, motor_omega: Vec) -> dict[str, Vec]:
        """Advance one dt using classic RK4 and return observation dict."""
        y = self.state.as_vector()
        dt = self.dt
        p = self.params
        u = motor_omega

        k1 = derivative(self.t, y, u, p)
        k2 = derivative(self.t + dt / 2, y + k1 * dt / 2, u, p)
        k3 = derivative(self.t + dt / 2, y + k2 * dt / 2, u, p)
        k4 = derivative(self.t + dt, y + k3 * dt, u, p)
        y_next = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        self.t += dt
        self.state = QuadState.from_vector(y_next)
        return self._observation()

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _observation(self) -> dict[str, Vec]:
        """Return current state in a flat dict (easy for downstream code)."""
        s = self.state
        return {
            "pos": s.pos,
            "vel": s.vel,
            "quat": s.quat,
            "ang_vel": s.ang_vel,
            "t": np.array([self.t]),
        }
