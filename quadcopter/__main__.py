"""
quadcopter.__main__
-------------------
Minimal demo: run a 4 s hover with constant motor speed and (optionally)
plot or save the result.

Examples
--------
python -m quadcopter --plot
quadcopter-demo --duration 6 --csv flight.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .dynamics import Params, QuadState
from .plotting import plot_trajectory
from .simulation import simulate, BaseController


# ----------------------------------------------------------------------
# Internal helper controller (constant hover command)
# ----------------------------------------------------------------------
class _HoverController(BaseController):
    """Return a fixed motor speed that balances weight at t=0."""

    def __init__(self, omega_hover: float) -> None:
        self._cmd: NDArray[np.float64] = np.full(4, omega_hover, dtype=np.float64)

    def update(self, t: float, state: QuadState) -> NDArray[np.float64]:  # noqa: D401
        return self._cmd


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:  # noqa: D401
    """Parse CLI arguments, run a simple simulation, optionally plot / save CSV."""
    # ------------------------------------------------------------------ CLI
    p = argparse.ArgumentParser(
        prog="python -m quadcopter",
        description="Quick hover demo for the quadcopter dynamics package.",
    )
    p.add_argument("--duration", type=float, default=4.0, help="simulation time [s]")
    p.add_argument("--dt", type=float, default=0.02, help="integration step [s]")
    p.add_argument(
        "--method",
        choices=["rk45", "rk4"],
        default="rk4",
        help="integration method (adaptive RK45 or fixed‑step RK4)",
    )
    p.add_argument("--rtol", type=float, default=1e-5, help="solver rtol")
    p.add_argument("--atol", type=float, default=1e-7, help="solver atol")
    p.add_argument("--plot", action="store_true", help="show matplotlib figure")
    p.add_argument("--csv", type=Path, help="save (t, state, control) to CSV")
    p.add_argument("--quiet", action="store_true", help="suppress info output")
    args = p.parse_args(argv)

    # ------------------------------------------------------------------ logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("quadcopter.demo")
    if args.quiet:
        log.setLevel(logging.WARNING)

    # ------------------------------------------------------------------ hover controller
    params = Params()
    w_hover = float(np.sqrt(params.m * params.g / (4.0 * params.b)))
    ctrl = _HoverController(w_hover)

    # ------------------------------------------------------------------ simulation
    t, states, controls = simulate(
        duration=args.duration,
        dt=args.dt,
        controller=ctrl,
        rtol=args.rtol,
        atol=args.atol,
        max_step=args.dt,
        method=args.method,
    )

    # ------------------------------------------------------------------ reporting
    final_z: float = float(states[-1, 2])
    log.info("final altitude: %.3f m", final_z)

    if args.csv:
        out = np.column_stack([t, states, controls])
        header = (
            "t,"
            "px,py,pz,vx,vy,vz,"
            "qw,qx,qy,qz,"
            "wx,wy,wz,"
            "u1,u2,u3,u4"
        )
        np.savetxt(args.csv, out, delimiter=",", header=header, comments="")
        log.info("saved data to %s", args.csv)

    if args.plot:
        plot_trajectory(t, states, controls)


# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
