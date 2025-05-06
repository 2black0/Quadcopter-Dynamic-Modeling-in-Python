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

    def __post_init__(self) -> None:      # explicit return type
        self._w_hover = np.sqrt(self.params.m * self.params.g / (4 * self.params.b))
        self.command = np.full(4, self._w_hover, dtype=np.float64)

    def update(self, t: float, state: QuadState) -> NDArray[np.float64]:  # noqa: D401
        return self.command


# ---------------------------------------------------------------------------
# Simulation routine
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------
from numpy.typing import NDArray

def simulate(
    duration: float,
    dt: float,
    controller: BaseController,
    initial_state: QuadState | None = None,
    params: Params = Params(),
    *,
    method: str = "rk45",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Integrate the quadcopter dynamics with either adaptive RK45 or fixed‑step RK4."""

    # -----------------------------------------------------------------
    # 0.  Default initial state (pos=0, level, at rest)
    # -----------------------------------------------------------------
    if initial_state is None:
        initial_state = QuadState(
            pos=np.zeros(3),
            vel=np.zeros(3),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            ang_vel=np.zeros(3),
        )

    state_vec0 = initial_state.as_vector()
    state_dim  = state_vec0.size

    # we’ll need this for either integrator
    t_eval = np.arange(0.0, duration + dt, dt)

    # 1. accept callables without '.update' ---------------------------------
    from typing import Protocol

    class _CallableController(Protocol):
        def __call__(self, t: float, state: QuadState) -> NDArray[np.float64]: ...

    if callable(controller) and not hasattr(controller, "update"):
        class _FnController(BaseController):
            """Wrap a plain function (t, state) -> motor speeds to look like a controller."""

            def __init__(self, fn: _CallableController) -> None:
                self._fn = fn

            def update(self, t: float, state: QuadState) -> NDArray[np.float64]:
                return self._fn(t, state)

        controller = _FnController(controller)  # now typed as BaseController

    # -----------------------------------------------------------------
    # 1.  RHS wrapper: dynamics + controller
    # -----------------------------------------------------------------
    def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        s = QuadState.from_vector(y)
        u = controller.update(t, s)
        return derivative(t, y, u, params)

    # -----------------------------------------------------------------
    # 2.  Choose integration method
    # -----------------------------------------------------------------
    if method.lower() == "rk4":
        n_steps = int(np.ceil(duration / dt))
        t = np.linspace(0.0, duration, n_steps + 1)
        y = np.empty((n_steps + 1, state_dim))
        u_log = np.empty((n_steps + 1, 4))
        y[0] = state_vec0

        for i in range(n_steps):
            s_i = QuadState.from_vector(y[i])
            u   = controller.update(t[i], s_i)
            k1 = derivative(t[i],           y[i],               u, params)
            k2 = derivative(t[i] + dt/2.0,  y[i] + k1*dt/2.0,   u, params)
            k3 = derivative(t[i] + dt/2.0,  y[i] + k2*dt/2.0,   u, params)
            k4 = derivative(t[i] + dt,      y[i] + k3*dt,       u, params)
            y[i + 1] = y[i] + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            u_log[i] = u

        u_log[-1] = u_log[-2]          # pad final control
        return t, y, u_log

    elif method.lower() == "rk45":
        u_log = np.empty((t_eval.size, 4))

        ivp_kwargs = dict(rtol=rtol, atol=atol)
        if max_step is not None:
            ivp_kwargs["max_step"] = max_step

        sol = solve_ivp(
            rhs,
            t_span=(0.0, duration),
            y0=state_vec0,
            t_eval=t_eval,
            **ivp_kwargs,
        )

        # Build control log so plotting has u[k] to match t[k]
        for i, (ti, yi) in enumerate(zip(sol.t, sol.y.T)):
            u_log[i] = controller.update(ti, QuadState.from_vector(yi))

        return sol.t, sol.y.T, u_log

    else:
        raise ValueError(f"Unknown integration method '{method}'. "
                         "Choose 'rk45' or 'rk4'.")



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
