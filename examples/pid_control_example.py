#!/usr/bin/env python3
"""
Example script demonstrating PID controller implementation for quadcopter position control.

This script shows how to use the PID controller to control a quadcopter's position.
"""

import numpy as np
from quadcopter.dynamics import QuadState
from quadcopter.simulation import simulate
from quadcopter import create_pid_position_controller, plot_trajectory

def run_pid_control_demo():
    """Run a simple PID control demo."""
    
    # Create a position controller using utility function
    position_controller = create_pid_position_controller(
        target_pos=[1.0, -1.0, 2.0],  # Target position: (1, -1, 2)
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
    print("Running PID position control simulation...")
    t, states, controls = simulate(
        duration=10.0,
        dt=0.02,
        controller=position_controller,
        initial_state=initial_state,
        method="rk4"
    )
    
    # Print final position
    final_pos = states[-1, 0:3]
    target_pos = position_controller.target_pos
    error = np.linalg.norm(final_pos - target_pos)
    
    print(f"Target position: {target_pos}")
    print(f"Final position: {final_pos}")
    print(f"Position error: {error:.4f} meters")
    
    # Plot results
    plot_trajectory(t, states, controls)

if __name__ == "__main__":
    run_pid_control_demo()