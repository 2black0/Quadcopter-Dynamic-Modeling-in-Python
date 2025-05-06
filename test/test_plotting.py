"""
test_plotting.py – smoke tests for quadcopter.plotting

These tests run Matplotlib in the 'Agg' backend so they work in a headless
CI environment and do not pop up windows on the developer’s machine.
"""

import matplotlib
matplotlib.use("Agg")              # forces a non‑GUI backend before any pyplot import

import numpy as np
from quadcopter.simulation import simulate, HoverController
from quadcopter.plotting import plot_trajectory, animate_trajectory
from matplotlib import animation

def _dummy_data():
    """Run a tiny 0.5 s hover sim at 50 Hz (25 frames)."""
    return simulate(0.5, 0.02, HoverController())

def test_static_plot_runs(tmp_path):
    t, traj, ctrl = _dummy_data()
    # Save to a temp PNG so we also test the save_path branch
    outfile = tmp_path / "fig.png"
    plot_trajectory(t, traj, ctrl, save_path=outfile, show=False)
    assert outfile.exists() and outfile.stat().st_size > 0

def test_animation_object_type():
    t, traj, _ = _dummy_data()
    anim = animate_trajectory(t, traj, fps=10)   # no save_path → returns object
    assert isinstance(anim, animation.FuncAnimation)
