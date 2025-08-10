#!/usr/bin/env python3
"""
Example script demonstrating LQR controller implementation.

This script shows how to use the LQR controller for quadcopter control.
"""

import numpy as np
from quadcopter.dynamics import QuadState
from quadcopter.simulation import simulate
from quadcopter import create_lqr_controller, plot_trajectory

def run_lqr_control_demo():
    """Run a simple LQR control demo."""
    print("Running LQR control simulation...")
    
    # Create an LQR controller using utility function
    lqr_controller = create_lqr_controller()
    
    # Set up initial state (quadcopter at origin, at rest)
    initial_state = QuadState(
        pos=np.array([0.0, 0.0, 0.0]),
        vel=np.array([0.0, 0.0, 0.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        ang_vel=np.array([0.0, 0.0, 0.0])
    )
    
    # Run simulation
    t, states, controls = simulate(
        duration=5.0,
        dt=0.02,
        controller=lqr_controller,
        initial_state=initial_state,
        method="rk4"
    )
    
    # Print final position
    final_pos = states[-1, 0:3]
    print(f"Final position: {final_pos}")
    
    # Plot results
    plot_trajectory(t, states, controls)
    print("LQR control simulation completed.")

if __name__ == "__main__":
    run_lqr_control_demo()