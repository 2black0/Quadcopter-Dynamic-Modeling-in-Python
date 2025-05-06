from quadcopter import QuadcopterEnv, AltitudePID

env   = QuadcopterEnv(dt=0.02)
ctrl  = AltitudePID(setpoint=1.0, kp=3.0, ki=1.5, kd=1.0)
obs   = env.reset()

for _ in range(200):            # 4 s
    omega = ctrl.update(env.t, env.state)
    obs   = env.step(omega)

print("final altitude:", obs["pos"][2])
