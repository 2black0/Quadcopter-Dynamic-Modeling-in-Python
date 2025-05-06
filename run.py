# ─── run.py ─────────────────────────────────────────────────────────────
import argparse
from pathlib import Path
import numpy as np

from quadcopter.simulation import simulate
from quadcopter.controllers import AltitudePID
from quadcopter.plotting import plot_trajectory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Quadcopter altitude step‑response demo"
    )
    # simulation parameters -------------------------------------------------
    parser.add_argument("--duration", type=float, default=4.0, help="simulation time [s]")
    parser.add_argument("--dt",        type=float, default=0.02, help="integration step [s]")
    parser.add_argument("--method", choices=["rk45", "rk4"], default="rk45",
                        help="integration method (adaptive RK45 or fixed‑step RK4)")
    parser.add_argument("--rtol", type=float, default=1e-5, help="solver rtol")
    parser.add_argument("--atol", type=float, default=1e-7, help="solver atol")

    # controller gains ------------------------------------------------------
    parser.add_argument("--kp", type=float, default=3.0, help="PID proportional gain")
    parser.add_argument("--ki", type=float, default=1.5, help="PID integral gain")
    parser.add_argument("--kd", type=float, default=1.0, help="PID derivative gain")

    # output / visualisation ------------------------------------------------
    parser.add_argument("--plot", action="store_true", help="show matplotlib figure")
    parser.add_argument("--csv",  type=Path, help="save (t, states, controls) to CSV")

    args = parser.parse_args(argv)

    # ----------------------------------------------------------------------
    ctrl = AltitudePID(setpoint=1.0, kp=args.kp, ki=args.ki, kd=args.kd)

    t, states, controls = simulate(
        duration=args.duration,
        dt=args.dt,
        controller=ctrl,
        rtol=args.rtol,
        atol=args.atol,
        max_step=args.dt,
        method=args.method,
    )

    # print step response metrics ------------------------------------------
    for k, v in ctrl.step_metrics(t, states[:, 2]).items():
        print(f"{k:20s}: {v:.3f}")

    # optional CSV export ---------------------------------------------------
    if args.csv:
        out = np.column_stack([t, states, controls])
        header = (
            "t "
            "pos_x pos_y pos_z vel_x vel_y vel_z "
            "qw qx qy qz "
            "wx wy wz "
            "u1 u2 u3 u4"
        )
        np.savetxt(args.csv, out, delimiter=",", header=header, comments="")
        print(f"Saved data to {args.csv}")

    # optional plot ---------------------------------------------------------
    if args.plot:
        plot_trajectory(t, states, controls)


if __name__ == "__main__":
    main()
