#!/usr/bin/env python3
"""
Example script demonstrating RL training with the quadcopter environment.

This script shows how to use the Gym environment for reinforcement learning.
"""

try:
    import numpy as np
    from quadcopter.gym_env import QuadcopterGymEnv
    
    class SimplePolicy:
        """A simple policy that outputs fixed motor speeds."""
        
        def __init__(self):
            # Base hover speed
            self.hover_speed = np.sqrt(0.65 * 9.81 / (4 * 3.25e-5))
        
        def predict(self, observation):
            """Predict action based on observation."""
            # Simple policy: if too low, increase thrust; if too high, decrease thrust
            z_pos = observation[2]  # Z position
            target_z = 1.0  # Target height
            
            # Adjust motor speeds based on position error
            error = target_z - z_pos
            adjustment = error * 10.0  # Simple proportional control
            
            # Apply adjustment to all motors
            action = np.full(4, self.hover_speed + adjustment)
            
            # Ensure motor speeds are within bounds
            action = np.clip(action, 0.0, 1000.0)
            
            return action, None
    
    def run_rl_demo():
        """Run a simple RL demonstration."""
        print("Running RL training demonstration...")
        
        # Create environment
        env = QuadcopterGymEnv(dt=0.02, max_steps=1000)
        
        # Create a simple policy
        policy = SimplePolicy()
        
        # Training loop
        num_episodes = 5
        episode_rewards = []
        
        for episode in range(num_episodes):
            # Reset environment
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            step_count = 0
            
            print(f"Episode {episode + 1}/{num_episodes}")
            
            while not done:
                # Get action from policy
                action, _ = policy.predict(obs)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Accumulate reward
                total_reward += reward
                step_count += 1
                
                # Print progress every 100 steps
                if step_count % 100 == 0:
                    print(f"  Step {step_count}: Z={obs[2]:.2f}m, Reward={reward:.2f}")
            
            episode_rewards.append(total_reward)
            print(f"  Episode finished after {step_count} steps")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Final Z position: {obs[2]:.2f}m")
            print()
        
        # Print summary
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print("Training Summary:")
        print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Best episode: {np.max(episode_rewards):.2f}")
        print(f"  Worst episode: {np.min(episode_rewards):.2f}")
        
        print("RL demonstration completed!")
    
    if __name__ == "__main__":
        run_rl_demo()
        
except ImportError:
    print("Gymnasium not available, skipping RL demonstration")
    print("To run this example, install gymnasium with: pip install gymnasium")