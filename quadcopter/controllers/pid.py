from __future__ import annotations

"""pid.py – Basic altitude PID controller for the quadcopter package.

This module implements a *single‑axis* PID that commands total thrust so the
vehicle tracks a desired z‑position (up positive, world frame).  For simplicity
all four motors receive the same speed; attitude is assumed near level.

Public class
------------
AltitudePID
    Subclass of simulation.BaseController with Kp, Ki, Kd gains and a target
    altitude.  Integral wind‑up is limited with a configurable clamp.

Example
-------
>>> from quadcopter.controllers.pid import AltitudePID
>>> ctrl = AltitudePID(setpoint=1.0, kp=2.0, ki=1.0)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..dynamics import Params, QuadState
from ..simulation import BaseController


# ---------------------------------------------------------------------------
# Helper dataclass to hold PID state
# ---------------------------------------------------------------------------

@dataclass
class _PIDState:
    integ: float = 0.0                # integral term
    prev_err: float | None = None     # for derivative calculation
    prev_t: float | None = None


# ---------------------------------------------------------------------------
# Altitude PID controller
# ---------------------------------------------------------------------------

@dataclass
class AltitudePID(BaseController):
    """PID on vertical position (z) + simple anti‑wind‑up.

    Parameters
    ----------
    setpoint : float
        Desired altitude [m].
    kp, ki, kd : float, optional
        PID gains (defaults give a slightly under‑damped 1‑m step for the
        default Params).
    integ_limit : float, optional
        Clamp for integral term (abs value) to avoid wind‑up.
    params : Params, optional
        Physical parameters; must match those used in the simulation.
    """

    setpoint: float = 1.0
    kp: float = 2.0
    ki: float = 1.0
    kd: float = 0.8
    integ_limit: float = 5.0
    params: Params = Params()
    _state: _PIDState = field(default_factory=_PIDState, init=False, repr=False)

    # -------------------------------------------------------------------
    # BaseController API
    # -------------------------------------------------------------------

    def update(self, t: float, state: QuadState) -> NDArray[np.float64]:
        z = state.pos[2]
        err = self.setpoint - z

        # dt for integral and derivative terms --------------------------
        if self._state.prev_t is None:
            dt = 0.0
        else:
            dt = t - self._state.prev_t
        self._state.prev_t = t

        # Integral term (anti‑wind‑up) ----------------------------------
        self._state.integ += err * dt
        self._state.integ = float(
            np.clip(self._state.integ, -self.integ_limit, self.integ_limit)
        )

        # Derivative term ------------------------------------------------
        if self._state.prev_err is None or dt == 0.0:
            derr = 0.0
        else:
            derr = (err - self._state.prev_err) / dt
        self._state.prev_err = err

        # PID control law ----------------------------------------------
        u = (
            self.params.m
            * (
                self.params.g
                + self.kp * err
                + self.ki * self._state.integ
                + self.kd * derr
            )
        )  # required total thrust [N]

        # Convert thrust to motor speed ---------------------------------
        u = np.clip(u, 0.0, np.inf)  # thrust can't be negative
        w = np.sqrt(u / (4 * self.params.b))  # rad/s per motor
        return np.full(4, w, dtype=np.float64)

    # -------------------------------------------------------------------
    # Convenience metrics after a simulation
    # -------------------------------------------------------------------

    def step_metrics(
        self,
        t: NDArray[np.float64],
        z: NDArray[np.float64],
        rise_eps: float = 0.9,
    ) -> dict[str, float]:
        """Return rise‑time, overshoot, steady‑state error, and settling time.

        Uses simple textbook definitions:
        * rise‑time – first time the response crosses *rise_eps*×setpoint.
        * overshoot – max error above setpoint (percentage).
        * steady‑state error – |z(t_end) − setpoint|.
        * settling time – first time response enters ±2 % band and stays.
        """

        sp = self.setpoint
        err = z - sp

        # rise time ----------------------------------------------------
        above = np.where(z >= rise_eps * sp)[0]
        rise_time = t[above[0]] if above.size else np.nan

        # overshoot ----------------------------------------------------
        overshoot = (z.max() - sp) / sp * 100.0

        # steady‑state error ------------------------------------------
        sse = abs(err[-1])

        # settling time (±2 %) ----------------------------------------
        within = np.where(np.abs(err) <= 0.02 * abs(sp))[0]
        if within.size:
            # find first index after which all remaining points are within band
            first_good = within[0]
            if np.all(np.abs(err[first_good:]) <= 0.02 * abs(sp)):
                settling_time = t[first_good]
            else:
                settling_time = np.nan
        else:
            settling_time = np.nan

        return {
            "rise_time": rise_time,
            "overshoot_%": overshoot,
            "steady_state_error": sse,
            "settling_time": settling_time,
        }


__all__ = ["AltitudePID"]
