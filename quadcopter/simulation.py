from __future__ import annotations

"""simulation.py – High‑level wrapper around SciPy’s ODE integrator.

Public API
----------
simulate(...)
    Integrate the quadcopter dynamics with an arbitrary controller.

Example
-------
>>> from quadcopter import QuadState, Params
>>> from quadcopter.simulation import simulate, HoverController
>>> p = Params()
>>> init = QuadState(
...     pos=np.zeros(3), vel=np.zeros(3),
...     quat=np.array([1.0, 0.0, 0.0, 0.0]), ang_vel=np.zeros(3)
... )
>>> t, traj, ctrl = simulate(5.0, 0.01, HoverController(p), init, p)
>>> print(traj.shape)  # (N, 13)
"""

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .dynamics import Params, QuadState, derivative

# ---------------------------------------------------------------------------
# Controller protocol / base class
# ---------------------------------------------------------------------------

class BaseController:
    """Minimal interface every controller must satisfy."""

    def update(self, t: float, state: QuadState) -> NDArray[np.float64]:
        """Return motor speeds (rad/s) at time *t* given the current *state*."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Simple hover controller (equal fixed motor speed)
# ---------------------------------------------------------------------------

@dataclass
class HoverController(BaseController):
    """Open‑loop hover: constant speed that cancels weight."""

    params: Params = Params()

    def __post_init__(self):
        self._w_hover = np.sqrt(self.params.m * self.params.g / (4 * self.params.b))
        self.command = np.full(4, self._w_hover, dtype=np.float64)

    def update(self, t: float, state: QuadState) -> NDArray[np.float64]:  # noqa: D401
        return self.command


# ---------------------------------------------------------------------------
# Simulation routine
# ---------------------------------------------------------------------------

def simulate(
    duration: float,
    dt: float,
    controller: BaseController,
    initial_state: QuadState | None = None,
    params: Params = Params(),
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run a simulation.

    Parameters
    ----------
    duration : float
        Final time [s].
    dt : float
        Output sampling interval [s].  (Integrator step size is adaptive.)
    controller : BaseController
        Object with an *update(t, state) -> control* method.
    initial_state : QuadState, optional
        If omitted, starts at origin, zero velocity/attitude.
    params : Params, optional
        Physical parameters (must match controller if controller relies on them).
    rtol, atol : float, optional
        Tolerances forwarded to `scipy.integrate.solve_ivp`.

    Returns
    -------
    t : (N,) ndarray
        Time stamps.
    traj : (N, 13) ndarray
        State history (packed vectors).
    ctrl : (N, 4) ndarray
        Control history (motor speeds).
    """

    if initial_state is None:
        initial_state = QuadState(
            pos=np.zeros(3),
            vel=np.zeros(3),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            ang_vel=np.zeros(3),
        )

    t_eval = np.arange(0.0, duration + dt, dt)
    control_log = np.empty((t_eval.size, 4), dtype=np.float64)

    # Wrap derivative to include the controller
    def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        state = QuadState.from_vector(y)
        u = controller.update(t, state)
        return derivative(t, y, u, params)

    # Integrate ---------------------------------------------------------
    sol = solve_ivp(
        rhs,
        t_span=(0.0, duration),
        y0=initial_state.as_vector(),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    # Build control log using computed states to avoid drift ------------
    for i, (ti, yi) in enumerate(zip(sol.t, sol.y.T)):
        control_log[i] = controller.update(ti, QuadState.from_vector(yi))

    return sol.t, sol.y.T, control_log


# ---------------------------------------------------------------------------
# Optional CLI for quick experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run a quadcopter hover simulation.")
    parser.add_argument("--time", type=float, default=5.0, help="Duration [s] (default: 5)")
    parser.add_argument("--dt", type=float, default=0.01, help="Output sample interval [s]")
    args = parser.parse_args()

    t, traj, ctrl = simulate(
        duration=args.time,
        dt=args.dt,
        controller=HoverController(),
    )

    # Simple sanity printout -------------------------------------------
    np.set_printoptions(suppress=True, precision=3)
    print("Final state (pos):", traj[-1, 0:3])
    print("Final state (vel):", traj[-1, 3:6])
    print("Average motor speed:", ctrl.mean(axis=0)[0])

    # Exit with success if simulation did not fail
    sys.exit(0)
