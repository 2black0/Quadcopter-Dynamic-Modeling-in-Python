"""
quadcopter.__main__
-------------------
Command–line demo and quick‑experiment entry point.

Install the package (editable or wheel) and run:

    python -m quadcopter --plot                     # default gains
    python -m quadcopter --kp 6 --kd 4 --plot       # custom PID
    quadcopter-demo --duration 8 --csv flight.csv   # same via console‑script
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .controllers import AltitudePID
from .plotting import plot_trajectory
from .simulation import simulate


def main(argv: list[str] | None = None) -> None:  # noqa: D401
    """Parse CLI arguments, run a simulation, optionally plot / save CSV."""
    # ------------------------------------------------------------------ CLI
    p = argparse.ArgumentParser(
        prog="python -m quadcopter",
        description="Step‑response demo for the quadcopter dynamics package.",
    )
    # simulation
    p.add_argument("--duration", type=float, default=4.0, help="simulation time [s]")
    p.add_argument("--dt", type=float, default=0.02, help="integration step [s]")
    p.add_argument(
        "--method",
        choices=["rk45", "rk4"],
        default="rk45",
        help="integration method (adaptive RK45 or fixed‑step RK4)",
    )
    p.add_argument("--rtol", type=float, default=1e-5, help="solver rtol")
    p.add_argument("--atol", type=float, default=1e-7, help="solver atol")
    # PID gains
    p.add_argument("--kp", type=float, default=3.0, help="proportional gain")
    p.add_argument("--ki", type=float, default=1.5, help="integral gain")
    p.add_argument("--kd", type=float, default=1.0, help="derivative gain")
    # output / visualisation
    p.add_argument("--plot", action="store_true", help="show matplotlib figure")
    p.add_argument("--csv", type=Path, help="save (t, state, control) to CSV")
    args = p.parse_args(argv)

    # ------------------------------------------------------------ controller
    ctrl = AltitudePID(setpoint=1.0, kp=args.kp, ki=args.ki, kd=args.kd)

    # ------------------------------------------------------------------ sim
    t, states, controls = simulate(
        duration=args.duration,
        dt=args.dt,
        controller=ctrl,
        rtol=args.rtol,
        atol=args.atol,
        max_step=args.dt,
        method=args.method,
    )

    # ------------------------------------------------------------ reporting
    for k, v in ctrl.step_metrics(t, states[:, 2]).items():
        print(f"{k:20s}: {v:.3f}")

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
        print(f"Saved data to {args.csv}")

    if args.plot:
        plot_trajectory(t, states, controls)


# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
