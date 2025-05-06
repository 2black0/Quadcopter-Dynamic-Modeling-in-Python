"""
quadcopter – Mini‑package for 6‑DoF quadrotor simulation
"""
from importlib.metadata import version, PackageNotFoundError

try:           # optional: safe way to expose package version if you later publish
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

# Re‑export the key symbols so users can write
# >>> from quadcopter import QuadState, derivative
from .dynamics import Params, QuadState, derivative   # noqa: F401
