#!/usr/bin/env python3
"""
Example script demonstrating real-time quadcopter simulation.

This script shows how to use the real-time environment for interactive control.
"""

import numpy as np
import time
from quadcopter.env import RealTimeQuadcopterEnv
from quadcopter.dynamics import QuadState

def run_real_time_demo():
    """Run a simple real-time simulation demo."""
    print("Running real-time quadcopter simulation...")
    print("This simulation will run at half real-time speed.")
    print("Press Ctrl+C to stop.\n")
    
    # Create a real-time environment (half speed)
    env = RealTimeQuadcopterEnv(dt=0.02, real_time_factor=0.5)
    
    # Reset environment
    obs = env.reset()
    print(f"Simulation started at t={obs['t'][0]:.3f}s")
    print(f"Initial position: {obs['pos']}")
    
    # Simple controller - just maintain hover
    hover_speed = np.sqrt(0.65 * 9.81 / (4 * 3.25e-5))  # Calculate hover speed
    motor_speeds = np.full(4, hover_speed)
    
    try:
        step_count = 0
        start_time = time.time()
        
        while step_count < 100:  # Run for 2 seconds of simulation time
            # Step the environment
            obs = env.step(motor_speeds)
            step_count += 1
            
            # Print status every 10 steps
            if step_count % 10 == 0:
                print(f"t={obs['t'][0]:.3f}s, pos=[{obs['pos'][0]:.3f}, {obs['pos'][1]:.3f}, {obs['pos'][2]:.3f}]")
        
        end_time = time.time()
        sim_time = obs['t'][0]
        real_time = end_time - start_time
        
        print(f"\nSimulation completed!")
        print(f"Simulation time: {sim_time:.3f}s")
        print(f"Real time elapsed: {real_time:.3f}s")
        print(f"Final position: {obs['pos']}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

if __name__ == "__main__":
    run_real_time_demo()