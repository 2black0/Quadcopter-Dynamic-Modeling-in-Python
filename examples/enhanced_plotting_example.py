#!/usr/bin/env python3
"""
Example script demonstrating enhanced plotting capabilities.

This script shows how to use the enhanced plotting functions for comprehensive
visualization of quadcopter simulations.
"""

import numpy as np
from quadcopter.dynamics import QuadState
from quadcopter.simulation import simulate
from quadcopter import create_pid_position_controller
from quadcopter.plotting import plot_trajectory, plot_control_errors, plot_3d_trajectory_comparison, plot_frequency_analysis

def run_enhanced_plotting_demo():
    """Run a simulation and demonstrate enhanced plotting capabilities."""
    print("Running enhanced plotting demonstration...")
    
    # Create a position controller using utility function
    position_controller = create_pid_position_controller(
        target_pos=[2.0, 1.0, 3.0],  # Target position: (2, 1, 3)
        kp=(2.0, 2.0, 4.0),
        ki=(0.1, 0.1, 0.2),
        kd=(0.5, 0.5, 1.0)
    )
    
    # Set up initial state (quadcopter at origin, at rest)
    initial_state = QuadState(
        pos=np.array([0.0, 0.0, 0.0]),
        vel=np.array([0.0, 0.0, 0.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        ang_vel=np.array([0.0, 0.0, 0.0])
    )
    
    # Run simulation
    duration = 10.0
    dt = 0.02
    t, states, controls = simulate(
        duration=duration,
        dt=dt,
        controller=position_controller,
        initial_state=initial_state,
        method="rk4"
    )
    
    # Create target states for error plotting
    targets = np.zeros_like(states)
    targets[:, 0] = 2.0  # Target x position
    targets[:, 1] = 1.0  # Target y position
    targets[:, 2] = 3.0  # Target z position
    
    # Generate comprehensive plots
    print("Generating comprehensive plots...")
    
    # 1. Standard trajectory plot
    plot_trajectory(t, states, controls, save_path="trajectory_plot.png", show=False)
    print("  - Saved trajectory plot to trajectory_plot.png")
    
    # 2. Control errors plot
    plot_control_errors(t, states, targets, save_path="control_errors.png", show=False)
    print("  - Saved control errors plot to control_errors.png")
    
    # 3. Frequency analysis
    position_signals = states[:, :3]  # x, y, z positions
    signal_names = ["X Position", "Y Position", "Z Position"]
    plot_frequency_analysis(t, position_signals, signal_names, save_path="frequency_analysis.png", show=False)
    print("  - Saved frequency analysis plot to frequency_analysis.png")
    
    # 4. For trajectory comparison, we'll run another simulation with different gains
    position_controller_2 = create_pid_position_controller(
        target_pos=[2.0, 1.0, 3.0],
        kp=(1.0, 1.0, 2.0),
        ki=(0.05, 0.05, 0.1),
        kd=(0.2, 0.2, 0.5)
    )
    
    t2, states2, controls2 = simulate(
        duration=duration,
        dt=dt,
        controller=position_controller_2,
        initial_state=initial_state,
        method="rk4"
    )
    
    # Compare trajectories
    trajectories = [
        (states, "Controller 1 (Kp=2, Ki=0.1, Kd=0.5)"),
        (states2, "Controller 2 (Kp=1, Ki=0.05, Kd=0.2)")
    ]
    plot_3d_trajectory_comparison(trajectories, save_path="trajectory_comparison.png", show=False)
    print("  - Saved trajectory comparison plot to trajectory_comparison.png")
    
    # Print final results
    final_pos = states[-1, 0:3]
    target_pos = position_controller.target_pos
    error = np.linalg.norm(final_pos - target_pos)
    
    print(f"\nSimulation Results:")
    print(f"Target position: {target_pos}")
    print(f"Final position: {final_pos}")
    print(f"Position error: {error:.4f} meters")
    print(f"Plots saved to current directory.")

if __name__ == "__main__":
    run_enhanced_plotting_demo()