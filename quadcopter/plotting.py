from __future__ import annotations

"""plotting.py – Utility helpers for visualising quadcopter simulations.

Main public functions
---------------------
plot_trajectory(t, states, controls, *, save_path=None, show=True)
    Static 3‑D plot + time‑series subplots of a simulation result.

animate_trajectory(t, states, *, fps=30, save_path=None)
    Quick Matplotlib animation; saves an MP4 or displays inline (Jupyter).
"""

from pathlib import Path
from typing import Optional, Tuple, Any, List

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – required for 3‑D projection
from matplotlib import animation

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract(trajectory: NDArray[np.float64]) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Return position, velocity, quaternion, angular‑rate views from stacked state array."""
    pos = trajectory[:, 0:3]
    vel = trajectory[:, 3:6]
    quat = trajectory[:, 6:10]
    rates = trajectory[:, 10:13]
    return pos, vel, quat, rates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_trajectory(
    t: NDArray[np.float64],
    states: NDArray[np.float64],
    controls: NDArray[np.float64] | None = None,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Make a quick‑look figure of a simulation.

    Parameters
    ----------
    t : (N,) ndarray
        Time vector.
    states : (N, 13) ndarray
        State history.
    controls : (N, 4) ndarray, optional
        Motor speeds. If None, that subplot is skipped.
    save_path : str or Path, optional
        If provided, saves the figure to this path (PNG / PDF deduced from ext).
    show : bool, default True
        If False, returns after saving without opening a GUI window.
    """

    pos, vel, quat, rates = _extract(states)

    # 3‑D trajectory ----------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(221, projection="3d")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=2)
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")        # type: ignore[attr-defined]
    ax3d.set_title("Trajectory (world frame)")
    ax3d.view_init(elev=20, azim=135)  # type: ignore[attr-defined]
    ax3d.autoscale(enable=True, axis="both", tight=True)

    # position components ----------------------------------------------
    ax_pos = fig.add_subplot(222)
    ax_pos.plot(t, pos)
    ax_pos.set_ylabel("position [m]")
    ax_pos.set_xlabel("time [s]")
    ax_pos.legend(["x", "y", "z"], loc="upper right")
    ax_pos.grid(True, linestyle=":", alpha=0.6)

    # velocity components ----------------------------------------------
    ax_vel = fig.add_subplot(223)
    ax_vel.plot(t, vel)
    ax_vel.set_ylabel("velocity [m/s]")
    ax_vel.set_xlabel("time [s]")
    ax_vel.legend(["vx", "vy", "vz"], loc="upper right")
    ax_vel.grid(True, linestyle=":", alpha=0.6)

    # motor speeds ------------------------------------------------------
    if controls is not None:
        ax_u = fig.add_subplot(224)
        ax_u.plot(t, controls)
        ax_u.set_ylabel("ω [rad/s]")
        ax_u.set_xlabel("time [s]")
        ax_u.legend(["w1", "w2", "w3", "w4"], loc="upper right")
        ax_u.grid(True, linestyle=":", alpha=0.6)
    else:
        ax_rate = fig.add_subplot(224)
        ax_rate.plot(t, rates)
        ax_rate.set_ylabel("angular rate [rad/s]")
        ax_rate.set_xlabel("time [s]")
        ax_rate.legend(["p", "q", "r"], loc="upper right")
        ax_rate.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()

    # Save or show ------------------------------------------------------
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_control_errors(
    t: NDArray[np.float64],
    states: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot control errors over time.
    
    Parameters
    ----------
    t : (N,) ndarray
        Time vector.
    states : (N, 13) ndarray
        State history.
    targets : (N, 13) ndarray
        Target state history.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    show : bool, default True
        If False, returns after saving without opening a GUI window.
    """
    pos, vel, _, _ = _extract(states)
    target_pos, target_vel, _, _ = _extract(targets)
    
    # Calculate errors
    pos_error = pos - target_pos
    vel_error = vel - target_vel
    
    # Create comprehensive error plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Control Errors Over Time", fontsize=16)
    
    # Position errors
    axes[0, 0].plot(t, pos_error[:, 0])
    axes[0, 0].set_title("X Position Error")
    axes[0, 0].set_ylabel("Error [m]")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    
    axes[0, 1].plot(t, pos_error[:, 1])
    axes[0, 1].set_title("Y Position Error")
    axes[0, 1].set_ylabel("Error [m]")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    
    axes[0, 2].plot(t, pos_error[:, 2])
    axes[0, 2].set_title("Z Position Error")
    axes[0, 2].set_ylabel("Error [m]")
    axes[0, 2].set_xlabel("Time [s]")
    axes[0, 2].grid(True, linestyle=":", alpha=0.6)
    
    # Velocity errors
    axes[1, 0].plot(t, vel_error[:, 0])
    axes[1, 0].set_title("X Velocity Error")
    axes[1, 0].set_ylabel("Error [m/s]")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    
    axes[1, 1].plot(t, vel_error[:, 1])
    axes[1, 1].set_title("Y Velocity Error")
    axes[1, 1].set_ylabel("Error [m/s]")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    
    axes[1, 2].plot(t, vel_error[:, 2])
    axes[1, 2].set_title("Z Velocity Error")
    axes[1, 2].set_ylabel("Error [m/s]")
    axes[1, 2].set_xlabel("Time [s]")
    axes[1, 2].grid(True, linestyle=":", alpha=0.6)
    
    fig.tight_layout()
    
    # Save or show ------------------------------------------------------
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_3d_trajectory_comparison(
    trajectories: List[Tuple[NDArray[np.float64], str]],
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot 3D trajectory comparison for multiple simulations.
    
    Parameters
    ----------
    trajectories : list of (states, label) tuples
        List of trajectories to compare.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    show : bool, default True
        If False, returns after saving without opening a GUI window.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (states, label) in enumerate(trajectories):
        pos, _, _, _ = _extract(states)
        color = colors[i % len(colors)]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], lw=2, label=label, color=color)
    
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")        # type: ignore[attr-defined]
    ax.set_title("3D Trajectory Comparison")
    ax.legend()
    ax.view_init(elev=20, azim=135)  # type: ignore[attr-defined]
    
    # Save or show ------------------------------------------------------
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_frequency_analysis(
    t: NDArray[np.float64],
    signals: NDArray[np.float64],
    signal_names: List[str],
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot frequency domain analysis of signals.
    
    Parameters
    ----------
    t : (N,) ndarray
        Time vector.
    signals : (N, M) ndarray
        Signals to analyze (M signals).
    signal_names : list of str
        Names of the signals.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    show : bool, default True
        If False, returns after saving without opening a GUI window.
    """
    dt = t[1] - t[0]
    fs = 1.0 / dt  # Sampling frequency
    
    # Compute FFT
    n = len(t)
    freq = np.fft.fftfreq(n, dt)[:n//2]
    
    fig, axes = plt.subplots(2, len(signal_names), figsize=(5*len(signal_names), 10))
    if len(signal_names) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle("Frequency Domain Analysis", fontsize=16)
    
    for i, (signal, name) in enumerate(zip(signals.T, signal_names)):
        # FFT
        fft_vals = np.fft.fft(signal)
        fft_mag = 2.0/n * np.abs(fft_vals[:n//2])
        
        # Time domain plot
        axes[0, i].plot(t, signal)
        axes[0, i].set_title(f"{name} (Time Domain)")
        axes[0, i].set_xlabel("Time [s]")
        axes[0, i].set_ylabel("Amplitude")
        axes[0, i].grid(True, linestyle=":", alpha=0.6)
        
        # Frequency domain plot
        axes[1, i].semilogy(freq, fft_mag)
        axes[1, i].set_title(f"{name} (Frequency Domain)")
        axes[1, i].set_xlabel("Frequency [Hz]")
        axes[1, i].set_ylabel("Magnitude")
        axes[1, i].grid(True, linestyle=":", alpha=0.6)
        axes[1, i].set_xlim(0, fs/2)
    
    fig.tight_layout()
    
    # Save or show ------------------------------------------------------
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Optional simple animation (Matplotlib FuncAnimation)
# ---------------------------------------------------------------------------


def animate_trajectory(
    t: NDArray[np.float64],
    states: NDArray[np.float64],
    *,
    fps: int = 30,
    save_path: str | Path | None = None,
) -> animation.FuncAnimation:
    """Return a Matplotlib animation of the 3‑D flight path.

    If `save_path` is provided (e.g. "flight.mp4"), the animation is saved using
    ffmpeg. Otherwise it is returned for inline display in Jupyter.
    """

    pos, *_ = _extract(states)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    def _axis_limits(arr: "NDArray[np.float64]", pad: float = 0.1) -> Tuple[float, float]:
        lo, hi = arr.min(), arr.max()
        if np.isclose(lo, hi):
            lo -= pad
            hi += pad
        return lo, hi

    ax.set_xlim(*_axis_limits(pos[:, 0]))
    ax.set_ylim(*_axis_limits(pos[:, 1]))
    ax.set_zlim(*_axis_limits(pos[:, 2]))  # type: ignore[attr-defined]

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")          # type: ignore[attr-defined]

    (traj_line,) = ax.plot([], [], [], lw=2)
    (marker,) = ax.plot([], [], [], "ro", markersize=4)

    def init() -> Tuple[Any, Any]:
        traj_line.set_data([], [])
        traj_line.set_3d_properties([])  # type: ignore[attr-defined]
        marker.set_data([], [])
        marker.set_3d_properties([])  # type: ignore[attr-defined]
        return traj_line, marker

    def update(frame: int) -> Tuple[Any, Any]:
        traj_line.set_data(pos[:frame, 0], pos[:frame, 1])
        traj_line.set_3d_properties(pos[:frame, 2])  # type: ignore[attr-defined]
        marker.set_data(pos[frame - 1 : frame, 0], pos[frame - 1 : frame, 1])
        marker.set_3d_properties(pos[frame - 1 : frame, 2])  # type: ignore[attr-defined]
        return traj_line, marker

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=1000 / fps,
        blit=True,
    )

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, fps=fps, dpi=150)
        plt.close(fig)

    return anim


__all__ = ["plot_trajectory", "animate_trajectory", "plot_control_errors", 
           "plot_3d_trajectory_comparison", "plot_frequency_analysis"]
