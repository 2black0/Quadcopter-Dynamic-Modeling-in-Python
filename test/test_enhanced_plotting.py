#!/usr/bin/env python3
"""
Test script for enhanced plotting functions.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quadcopter.plotting import plot_trajectory, plot_control_errors, plot_3d_trajectory_comparison, plot_frequency_analysis
from quadcopter.dynamics import QuadState

def test_enhanced_plotting():
    """Test all enhanced plotting functions."""
    print("Testing enhanced plotting functions...")
    
    # Generate test data
    t = np.linspace(0, 10, 100)
    
    # Create a simple trajectory (spiral going upward)
    states = np.zeros((100, 13))
    targets = np.zeros((100, 13))
    
    for i in range(100):
        # Position: spiral
        states[i, 0] = np.cos(t[i])  # x
        states[i, 1] = np.sin(t[i])  # y
        states[i, 2] = t[i] * 0.1    # z
        
        # Velocity
        states[i, 3] = -np.sin(t[i])  # vx
        states[i, 4] = np.cos(t[i])   # vy
        states[i, 5] = 0.1            # vz
        
        # Quaternion (identity for simplicity)
        states[i, 6] = 1.0  # w
        states[i, 7] = 0.0  # x
        states[i, 8] = 0.0  # y
        states[i, 9] = 0.0  # z
        
        # Angular velocity
        states[i, 10] = 0.0  # wx
        states[i, 11] = 0.0  # wy
        states[i, 12] = 0.1  # wz
        
        # Targets (straight line)
        targets[i, 0] = t[i] * 0.1    # x
        targets[i, 1] = 0.0           # y
        targets[i, 2] = t[i] * 0.05   # z
        
        # Velocity targets
        targets[i, 3] = 0.1   # vx
        targets[i, 4] = 0.0   # vy
        targets[i, 5] = 0.05  # vz
    
    controls = np.random.rand(100, 4) * 400 + 300  # Random motor speeds
    
    # Test plot_trajectory
    print("Testing plot_trajectory...")
    plot_trajectory(t, states, controls, show=False)
    print("plot_trajectory test passed")
    
    # Test plot_control_errors
    print("Testing plot_control_errors...")
    plot_control_errors(t, states, targets, show=False)
    print("plot_control_errors test passed")
    
    # Test plot_3d_trajectory_comparison
    print("Testing plot_3d_trajectory_comparison...")
    trajectories = [
        (states, "Spiral Trajectory"),
        (targets, "Straight Line Target")
    ]
    plot_3d_trajectory_comparison(trajectories, show=False)
    print("plot_3d_trajectory_comparison test passed")
    
    # Test plot_frequency_analysis
    print("Testing plot_frequency_analysis...")
    signals = states[:, :3]  # Position signals
    signal_names = ["X Position", "Y Position", "Z Position"]
    plot_frequency_analysis(t, signals, signal_names, show=False)
    print("plot_frequency_analysis test passed")
    
    print("All enhanced plotting tests passed!")

if __name__ == "__main__":
    test_enhanced_plotting()