from quadcopter import QuadcopterEnv
import numpy as np

def my_controller(obs):
    return np.full(4, 400.0)          # replace with PID / MPC / RL

env = QuadcopterEnv(dt=0.01)
obs = env.reset()
for _ in range(400):                  # 4 s @ 100 Hz
    obs = env.step(my_controller(obs))

print("final altitude:", obs["pos"][2])
