#!/usr/bin/env python3
"""
Test script for the Gym environment.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import gymnasium as gym
    from quadcopter.gym_env import QuadcopterGymEnv
    
    def test_gym_env():
        """Test the Gym environment."""
        print("Testing Gym environment...")
        
        # Create environment
        env = QuadcopterGymEnv()
        
        # Check spaces
        assert env.action_space.shape == (4,)
        assert env.observation_space.shape == (13,)
        print("  - Action and observation spaces verified")
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (13,)
        print("  - Reset functionality verified")
        
        # Test step
        action = np.array([400.0, 400.0, 400.0, 400.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (13,)
        print("  - Step functionality verified")
        
        # Check info dict
        assert "time" in info
        assert "position" in info
        assert "velocity" in info
        assert "orientation" in info
        assert "angular_velocity" in info
        print("  - Info dictionary structure verified")
        
        print("All Gym environment tests passed!")
        
    if __name__ == "__main__":
        test_gym_env()
        
except ImportError:
    print("Gymnasium not available, skipping Gym environment tests")