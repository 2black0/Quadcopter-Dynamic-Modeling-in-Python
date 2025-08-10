"""
gym_env.py - Gymnasium compatibility wrapper for quadcopter environment.

This module implements a Gymnasium-compatible environment for the quadcopter
dynamics model, enabling reinforcement learning research and development.

Classes:
    QuadcopterGymEnv: Gymnasium-compatible environment for quadcopter control
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Fallback for environments without gymnasium
    class MockGym:
        def __init__(self):
            class spaces:
                Box = None
            self.spaces = spaces
    gym = MockGym()

import numpy as np
from .env import QuadcopterEnv
from .dynamics import QuadState

class QuadcopterGymEnv(gym.Env):
    """Gymnasium-compatible environment for quadcopter RL."""
    
    metadata = {"render_modes": ["human"], "render_fps": 50}
    
    def __init__(self, dt: float = 0.02, max_steps: int = 2000):
        super().__init__()
        
        self.quad_env = QuadcopterEnv(dt=dt)
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action and observation spaces
        # Action: motor speeds [rad/s] for each of 4 motors
        self.action_space = spaces.Box(
            low=0.0, high=1000.0, shape=(4,), dtype=np.float32
        )
        
        # Observation: position (3), velocity (3), quaternion (4), angular velocity (3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        
        # Define reward parameters
        self.position_target = np.array([0.0, 0.0, 1.0])  # 1m hover height
        self.position_tolerance = 0.1  # meters
        self.orientation_tolerance = 0.1  # radians
        
    def _get_obs(self):
        """Get current observation as a flat array."""
        return self.quad_env.state.as_vector().astype(np.float32)
    
    def _get_info(self):
        """Get additional info."""
        return {
            "time": self.quad_env.t,
            "position": self.quad_env.state.pos,
            "velocity": self.quad_env.state.vel,
            "orientation": self.quad_env.state.quat,
            "angular_velocity": self.quad_env.state.ang_vel
        }
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset to initial state
        initial_state = QuadState(
            pos=np.array([0.0, 0.0, 0.0]),
            vel=np.array([0.0, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ang_vel=np.array([0.0, 0.0, 0.0])
        )
        
        obs = self.quad_env.reset(state=initial_state)
        self.current_step = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """Execute one time step."""
        # Apply action (motor speeds)
        obs_dict = self.quad_env.step(action)
        
        # Get observation
        observation = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self._is_terminated()
        
        # Check truncation
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # Get info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self):
        """Calculate reward based on current state."""
        pos_error = np.linalg.norm(self.quad_env.state.pos - self.position_target)
        vel_error = np.linalg.norm(self.quad_env.state.vel)
        
        # Reward for being close to target position
        position_reward = -pos_error
        
        # Reward for low velocity
        velocity_penalty = -0.1 * vel_error
        
        # Reward for upright orientation (w component of quaternion close to 1)
        orientation_reward = self.quad_env.state.quat[0] - 1.0
        
        # Total reward
        total_reward = position_reward + velocity_penalty + orientation_reward
        
        return float(total_reward)
    
    def _is_terminated(self):
        """Check if episode should be terminated."""
        # Terminate if position is too far from target
        pos_error = np.linalg.norm(self.quad_env.state.pos - self.position_target)
        if pos_error > 5.0:  # More than 5 meters away
            return True
            
        # Terminate if orientation is too extreme
        if self.quad_env.state.quat[0] < 0.5:  # Very tilted
            return True
            
        return False
    
    def render(self):
        """Render the environment (placeholder)."""
        pass

__all__ = ["QuadcopterGymEnv"]