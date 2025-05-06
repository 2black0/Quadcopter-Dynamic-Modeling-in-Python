from __future__ import annotations

"""simulation.py – High‑level wrapper around SciPy’s ODE integrator.

Fix: construct `t_eval` with `np.linspace` so its last value equals `duration`.
This prevents SciPy’s "Values in t_eval are not within t_span" error when
`duration` is not an integer multiple of `dt`.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from quadcopter.dynamics import Params, QuadState, derivative

# ----------------------------------------------------------------------------
# Base controller interface
# ----------------------------------------------------------------------------

class BaseController:
    def update(self, t: float, state: QuadState) -> NDArray[np.float64]:
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Simple open‑loop hover controller
# ----------------------------------------------------------------------------

@dataclass
class HoverController(BaseController):
    params: Params = Params()

    def __post_init__(self):
        w = np.sqrt(self.params.m * self.params.g / (4 * self.params.b))
        self._command = np.full(4, w, dtype=np.float64)

    def update(self, t: float, state: QuadState) -> NDArray[np.float64]:
        return self._command


# ----------------------------------------------------------------------------
# Helper – robust time vector
# ----------------------------------------------------------------------------

def _make_time_vector(duration: float, dt: float) -> NDArray[np.float64]:
    n_steps = int(round(duration / dt))
    if n_steps <= 0:
        raise ValueError("duration must be > 0 and >= dt")
    return np.linspace(0.0, duration, n_steps + 1, dtype=np.float64)


# ----------------------------------------------------------------------------
# Main simulation function
# ----------------------------------------------------------------------------

def simulate(
    duration: float,
    dt: float,
    controller: BaseController,
    initial_state: QuadState | None = None,
    params: Params = Params(),
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if initial_state is None:
        initial_state = QuadState(
            pos=np.zeros(3),
            vel=np.zeros(3),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            ang_vel=np.zeros(3),
        )

    t_eval = _make_time_vector(duration, dt)
    ctrl_log = np.empty((t_eval.size, 4), dtype=np.float64)

    def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        state = QuadState.from_vector(y)
        u = controller.update(t, state)
        return derivative(t, y, u, params)

    sol = solve_ivp(
        rhs,
        (0.0, duration),
        y0=initial_state.as_vector(),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    for i, (ti, yi) in enumerate(zip(sol.t, sol.y.T)):
        ctrl_log[i] = controller.update(ti, QuadState.from_vector(yi))

    return sol.t, sol.y.T, ctrl_log


__all__ = ["simulate", "HoverController", "BaseController"]
