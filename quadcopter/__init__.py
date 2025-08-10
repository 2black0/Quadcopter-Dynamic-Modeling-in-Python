"""
quadcopter – mini‑package for 6‑DoF quadrotor simulation & control.

Typical usage
-------------
>>> from quadcopter import simulate, create_pid_position_controller, plot_trajectory
>>> controller = create_pid_position_controller([1, 1, 2])
>>> t, states, u = simulate(4.0, 0.02, controller)
>>> plot_trajectory(t, states, u)
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Semantic version (reads from installed package metadata if available)
# ---------------------------------------------------------------------
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__: str = _pkg_version(__name__)
except PackageNotFoundError:              # editable / source checkout
    __version__ = "0.0.0.dev0"

__author__: str = "2black0"
__license__: str = "MIT"

# ---------------------------------------------------------------------
# Public re‑exports – the “one‑stop” API
# ---------------------------------------------------------------------
from .dynamics import Params, QuadState, derivative          # physics core
from .simulation import simulate, HoverController            # integrators
from .plotting import plot_trajectory, animate_trajectory    # visualisation

__all__ = [
    # physics
    "Params", "QuadState", "derivative",
    # simulation
    "simulate", "HoverController",
    # visualisation
    "plot_trajectory", "animate_trajectory",
    # meta
    "__version__",
]

from .env import QuadcopterEnv, RealTimeQuadcopterEnv   # add to earlier export block
__all__.extend(["QuadcopterEnv", "RealTimeQuadcopterEnv"])

# Conditional imports for optional features
try:
    from .controllers import PIDController, PositionController, LQRController
    __all__.extend(["PIDController", "PositionController", "LQRController"])
except ImportError:
    pass

try:
    from .gym_env import QuadcopterGymEnv
    __all__.append("QuadcopterGymEnv")
except ImportError:
    pass

try:
    from .logging import SimulationLog, simulate_with_logging, AcademicLog, simulate_with_academic_logging
    __all__.extend(["SimulationLog", "simulate_with_logging", "AcademicLog", "simulate_with_academic_logging"])
except ImportError:
    pass

try:
    from .evaluation import AcademicEvaluator
    __all__.append("AcademicEvaluator")
except ImportError:
    pass

try:
    from .utils import create_pid_position_controller, create_pid_attitude_controller, create_lqr_controller, create_hover_controller
    __all__.extend(["create_pid_position_controller", "create_pid_attitude_controller", "create_lqr_controller", "create_hover_controller"])
except ImportError:
    pass
