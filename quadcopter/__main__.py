"""
quadcopter.__main__
-------------------
Comprehensive CLI for quadcopter simulation, control, and analysis.

Examples
--------
python -m quadcopter --plot
python -m quadcopter --controller pid --target-pos 1 1 2 --duration 10 --plot
python -m quadcopter --controller lqr --academic-log results --duration 5
quadcopter-demo --duration 6 --csv flight.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import json

import numpy as np
from numpy.typing import NDArray

from .dynamics import Params, QuadState
from .plotting import plot_trajectory, plot_control_errors, plot_3d_trajectory_comparison
from .simulation import simulate, BaseController, HoverController
from .controllers import PIDController, PositionController, LQRController
from .logging import SimulationLog, simulate_with_logging, AcademicLog, simulate_with_academic_logging
from .evaluation import AcademicEvaluator
from .utils import create_pid_position_controller, create_lqr_controller, create_hover_controller

# ----------------------------------------------------------------------
# Controller implementations
# ----------------------------------------------------------------------

class _HoverController(BaseController):
    """Return a fixed motor speed that balances weight at t=0."""

    def __init__(self, omega_hover: float) -> None:
        self._cmd: NDArray[np.float64] = np.full(4, omega_hover, dtype=np.float64)

    def update(self, t: float, state: QuadState) -> NDArray[np.float64]:  # noqa: D401
        return self._cmd


def _create_pid_controller(args: argparse.Namespace, params: Params) -> BaseController:
    """Create a PID controller based on CLI arguments."""
    if args.controller == "pid":
        if args.target_pos:
            # Use utility function for consistent controller creation
            return create_pid_position_controller(
                target_pos=args.target_pos,
                kp=args.pid_kp,
                ki=args.pid_ki,
                kd=args.pid_kd
            )
        else:
            # Simple hover controller
            return create_hover_controller(params)
    return create_hover_controller(params)


def _create_lqr_controller(args: argparse.Namespace, params: Params) -> BaseController:
    """Create an LQR controller based on CLI arguments."""
    if args.controller == "lqr":
        # Use utility function for consistent controller creation
        return create_lqr_controller(params=params)
    return create_hover_controller(params)


# ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:  # noqa: D401
    """Parse CLI arguments, run simulation with selected controller, and output results."""
    # ------------------------------------------------------------------ CLI
    p = argparse.ArgumentParser(
        prog="python -m quadcopter",
        description="Comprehensive quadcopter simulation and analysis tool.",
    )
    
    # Simulation parameters
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
    
    # Controller selection
    p.add_argument(
        "--controller",
        choices=["hover", "pid", "lqr"],
        default="hover",
        help="controller type to use"
    )
    
    # PID controller parameters
    p.add_argument(
        "--pid-kp",
        type=float,
        nargs=3,
        default=[2.0, 2.0, 4.0],
        help="PID Kp gains for x, y, z axes"
    )
    p.add_argument(
        "--pid-ki",
        type=float,
        nargs=3,
        default=[0.1, 0.1, 0.2],
        help="PID Ki gains for x, y, z axes"
    )
    p.add_argument(
        "--pid-kd",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 1.0],
        help="PID Kd gains for x, y, z axes"
    )
    p.add_argument(
        "--target-pos",
        type=float,
        nargs=3,
        help="target position [x, y, z] for position controller"
    )
    
    # Output options
    p.add_argument("--plot", action="store_true", help="show matplotlib figure")
    p.add_argument("--plot-errors", action="store_true", help="show control error plot")
    p.add_argument("--plot-comparison", action="store_true", help="show trajectory comparison plot")
    p.add_argument("--csv", type=Path, help="save (t, state, control) to CSV")
    p.add_argument("--json", type=Path, help="save simulation log to JSON")
    p.add_argument("--matlab", type=Path, help="save simulation log to MATLAB .mat file")
    
    # Academic logging and evaluation
    p.add_argument("--academic-log", type=Path, help="enable academic logging and save to directory")
    p.add_argument("--controller-type", choices=["pid", "lqr", "rl"], default="pid", 
                   help="controller type for academic logging")
    
    # Initial conditions
    p.add_argument("--init-pos", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help="initial position [x, y, z]")
    p.add_argument("--init-vel", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   help="initial velocity [vx, vy, vz]")
    
    # Verbosity
    p.add_argument("--quiet", action="store_true", help="suppress info output")
    p.add_argument("--verbose", action="store_true", help="enable verbose output")
    
    args = p.parse_args(argv)

    # ------------------------------------------------------------------ logging
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=log_level, format="%(message)s")
    log = logging.getLogger("quadcopter")
    
    # ------------------------------------------------------------------ setup
    params = Params()
    
    # Initial state
    initial_state = QuadState(
        pos=np.array(args.init_pos),
        vel=np.array(args.init_vel),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        ang_vel=np.array([0.0, 0.0, 0.0])
    )
    
    # ------------------------------------------------------------------ controller selection
    if args.controller == "pid":
        controller = _create_pid_controller(args, params)
        log.info("Using PID controller")
    elif args.controller == "lqr":
        controller = _create_lqr_controller(args, params)
        log.info("Using LQR controller")
    else:
        controller = create_hover_controller(params)
        log.info("Using hover controller")
    
    # ------------------------------------------------------------------ simulation
    if args.academic_log:
        log.info("Running simulation with academic logging...")
        # Use academic logging
        log_result = simulate_with_academic_logging(
            duration=args.duration,
            dt=args.dt,
            controller=controller,
            initial_state=initial_state,
            params=params,
            ref_position=np.array(args.target_pos) if args.target_pos else None,
            method=args.method,
            rtol=args.rtol,
            atol=args.atol,
            max_step=args.dt,
            controller_type=args.controller_type
        )
        
        # Save academic log in multiple formats
        args.academic_log.mkdir(parents=True, exist_ok=True)
        log_result.save_csv(args.academic_log / "simulation.csv")
        log_result.save_json(args.academic_log / "simulation.json")
        try:
            log_result.save_matlab(args.academic_log / "simulation.mat")
        except ImportError:
            log.warning("scipy not available, cannot save MATLAB format")
        
        # Generate academic evaluation if requested
        evaluator = AcademicEvaluator(log_result)
        metrics = evaluator.generate_comprehensive_analysis(args.academic_log)
        
        log.info("Academic results saved to %s", args.academic_log)
        
        # Use regular arrays for plotting/CSV
        t = np.array(log_result.times)
        # Reconstruct states array from logged data
        # For simplicity, we'll use identity quaternion [1,0,0,0] since we don't have the raw quaternions
        states = np.array([[
            pos[0], pos[1], pos[2],              # position: px, py, pz
            vel[0], vel[1], vel[2],              # velocity: vx, vy, vz
            1.0, 0.0, 0.0, 0.0,                  # quaternion: qw, qx, qy, qz (identity for simplicity)
            ang_vel[0], ang_vel[1], ang_vel[2]   # angular velocity: wx, wy, wz
        ] for pos, vel, ang_vel in zip(
            log_result.positions, log_result.velocities, 
            log_result.angular_rates
        )])
        controls = np.array(log_result.motor_speeds)
    else:
        log.info("Running simulation...")
        t, states, controls = simulate(
            duration=args.duration,
            dt=args.dt,
            controller=controller,
            initial_state=initial_state,
            params=params,
            method=args.method,
            rtol=args.rtol,
            atol=args.atol,
            max_step=args.dt,
        )
    
    # ------------------------------------------------------------------ reporting
    final_pos = states[-1, 0:3]
    final_z: float = float(final_pos[2])
    log.info("final position: [%.3f, %.3f, %.3f] m", final_pos[0], final_pos[1], final_pos[2])
    
    # Standard logging
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
    
    if args.json:
        if args.academic_log:
            log_result.save_json(args.json)
        else:
            sim_log = SimulationLog(duration=args.duration, dt=args.dt, method=args.method)
            for i in range(len(t)):
                state = QuadState.from_vector(states[i])
                sim_log.add_entry(t[i], state, controls[i])
            sim_log.save_json(args.json)
        log.info("saved JSON log to %s", args.json)
    
    if args.matlab:
        try:
            if args.academic_log:
                log_result.save_matlab(args.matlab)
            else:
                sim_log = SimulationLog(duration=args.duration, dt=args.dt, method=args.method)
                for i in range(len(t)):
                    state = QuadState.from_vector(states[i])
                    sim_log.add_entry(t[i], state, controls[i])
                sim_log.save_matlab(args.matlab)
            log.info("saved MATLAB log to %s", args.matlab)
        except ImportError:
            log.warning("scipy not available, cannot save MATLAB format")
    
    # Enhanced plotting options
    if args.plot:
        plot_trajectory(t, states, controls)
    
    if args.plot_errors and args.target_pos:
        # Create target states for error plotting
        targets = np.zeros_like(states)
        targets[:, 0] = args.target_pos[0]  # Target x position
        targets[:, 1] = args.target_pos[1]  # Target y position
        targets[:, 2] = args.target_pos[2]  # Target z position
        plot_control_errors(t, states, targets)
    
    if args.plot_comparison:
        # For trajectory comparison, we'll run another simulation with different gains
        if args.controller == "pid" and args.target_pos:
            # Create second controller with different gains
            controller2 = create_pid_position_controller(
                target_pos=args.target_pos,
                kp=(args.pid_kp[0] * 0.5, args.pid_kp[1] * 0.5, args.pid_kp[2] * 0.5),
                ki=(args.pid_ki[0] * 0.5, args.pid_ki[1] * 0.5, args.pid_ki[2] * 0.5),
                kd=(args.pid_kd[0] * 0.5, args.pid_kd[1] * 0.5, args.pid_kd[2] * 0.5)
            )
            
            # Run second simulation
            t2, states2, controls2 = simulate(
                duration=args.duration,
                dt=args.dt,
                controller=controller2,
                initial_state=initial_state,
                params=params,
                method=args.method,
                rtol=args.rtol,
                atol=args.atol,
                max_step=args.dt,
            )
            
            # Compare trajectories
            trajectories = [
                (states, f"Controller 1 (Kp={args.pid_kp}, Ki={args.pid_ki}, Kd={args.pid_kd})"),
                (states2, f"Controller 2 (Kp={np.array(args.pid_kp)*0.5}, Ki={np.array(args.pid_ki)*0.5}, Kd={np.array(args.pid_kd)*0.5})")
            ]
            plot_3d_trajectory_comparison(trajectories)
        else:
            log.warning("Trajectory comparison requires PID controller with target position")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
