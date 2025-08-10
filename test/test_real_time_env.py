#!/usr/bin/env python3
"""
Simple test for the real-time environment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from quadcopter.env import RealTimeQuadcopterEnv
from quadcopter.dynamics import QuadState

def test_real_time_env():
    """Test RealTimeQuadcopterEnv functionality."""
    print("Testing RealTimeQuadcopterEnv...")
    
    # Create a real-time environment
    env = RealTimeQuadcopterEnv(dt=0.02, real_time_factor=0.5)  # Half speed
    
    # Reset environment
    obs = env.reset()
    print(f"Initial time: {obs['t'][0]}")
    
    # Run a few steps
    motor_speeds = np.array([400.0, 400.0, 400.0, 400.0])
    
    start_time = time.time()
    for i in range(10):
        obs = env.step(motor_speeds)
        print(f"Step {i+1}: t={obs['t'][0]:.3f}, pos={obs['pos']}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    sim_time = obs['t'][0]
    
    print(f"Simulation time: {sim_time:.3f}s")
    print(f"Real time elapsed: {elapsed:.3f}s")
    print(f"Expected real time (at half speed): {sim_time / 0.5:.3f}s")
    
    print("RealTimeQuadcopterEnv test passed!")

if __name__ == "__main__":
    test_real_time_env()