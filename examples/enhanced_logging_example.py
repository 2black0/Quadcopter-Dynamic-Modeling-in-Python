#!/usr/bin/env python3
"""
Example script demonstrating enhanced logging capabilities.

This script shows how to use the enhanced logging functions for comprehensive
data collection and export in multiple formats.
"""

import numpy as np
from quadcopter.logging import simulate_with_logging
from quadcopter.dynamics import QuadState
from quadcopter import create_pid_position_controller

def run_enhanced_logging_demo():
    """Run a simulation and demonstrate enhanced logging capabilities."""
    print("Running enhanced logging demonstration...")
    
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
    
    # Run simulation with logging
    duration = 5.0
    dt = 0.02
    log = simulate_with_logging(
        duration=duration,
        dt=dt,
        controller=position_controller,
        initial_state=initial_state,
        method="rk4"
    )
    
    # Export data in multiple formats
    print("Exporting data in multiple formats...")
    
    # 1. CSV format
    log.save_csv("simulation_data.csv")
    print("  - Saved data to simulation_data.csv")
    
    # 2. JSON format
    log.save_json("simulation_data.json")
    print("  - Saved data to simulation_data.json")
    
    # 3. MATLAB format (if scipy is available)
    try:
        log.save_matlab("simulation_data.mat")
        print("  - Saved data to simulation_data.mat")
    except:
        print("  - Skipping MATLAB format (scipy not available)")
    
    # Print summary statistics
    print(f"\nSimulation Summary:")
    print(f"  Duration: {log.duration} seconds")
    print(f"  Time step: {log.dt} seconds")
    print(f"  Method: {log.method}")
    print(f"  Data points: {len(log.times)}")
    
    if len(log.times) > 0:
        # Calculate some statistics
        positions = np.array([s['position'] for s in log.states])
        velocities = np.array([s['velocity'] for s in log.states])
        controls = np.array(log.controls)
        
        print(f"\nPosition Statistics:")
        print(f"  Final position: [{positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f}]")
        print(f"  Position range: X[{np.min(positions[:, 0]):.3f}, {np.max(positions[:, 0]):.3f}], "
              f"Y[{np.min(positions[:, 1]):.3f}, {np.max(positions[:, 1]):.3f}], "
              f"Z[{np.min(positions[:, 2]):.3f}, {np.max(positions[:, 2]):.3f}]")
        
        print(f"\nVelocity Statistics:")
        print(f"  Max velocity: {np.max(np.linalg.norm(velocities, axis=1)):.3f} m/s")
        print(f"  Mean velocity: {np.mean(np.linalg.norm(velocities, axis=1)):.3f} m/s")
        
        print(f"\nControl Statistics:")
        print(f"  Motor speed range: [{np.min(controls):.1f}, {np.max(controls):.1f}] rad/s")
        print(f"  Mean motor speed: {np.mean(controls):.1f} rad/s")
    
    # Clean up files
    import os
    os.remove("simulation_data.csv")
    os.remove("simulation_data.json")
    if os.path.exists("simulation_data.mat"):
        os.remove("simulation_data.mat")
    
    print("\nEnhanced logging demonstration completed!")

if __name__ == "__main__":
    run_enhanced_logging_demo()