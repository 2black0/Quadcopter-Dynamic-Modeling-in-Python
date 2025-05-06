"""
quadcopter – mini‑package for 6‑DoF quadrotor simulation & control.

Typical usage
-------------
>>> from quadcopter import simulate, AltitudePID, plot_trajectory
>>> t, states, u = simulate(4.0, 0.02, AltitudePID(kp=4, ki=2))
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

from .env import QuadcopterEnv   # add to earlier export block
__all__.append("QuadcopterEnv")
