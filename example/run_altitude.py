"""
run.py – 5 m step with Optuna‑tuned altitude‑PID
================================================

• Altitude gains copied from `optimal.py` best trial  
  (kp=4.0531, ki=0.0186, kd=2.408).  
• Integral wind‑up and ESC limits handled.  
• Attitude loop keeps p, q, r ≈ 0.  
• Uses public QuadcopterEnv API only.
"""

from __future__ import annotations
import numpy as np
from quadcopter import QuadcopterEnv, Params
from quadcopter.plotting import plot_trajectory

# ------------------------------------------------------------------ targets
Z_SP = 5.0
BODY_RATE_SP = np.zeros(3)

# ------------------------------------------------------------------ altitude‑PID gains (Optuna trial 27)
#{'kp': 3.8256366526919017, 'ki': 0.0009449363883856005, 'kd': 2.9946421709435156}
ALT_KP = 3.8256366526919017#4.053078965715498
ALT_KI = 0.0009449363883856005#0.018619736983976974
ALT_KD = 2.9946421709435156#2.407821600106919
INT_CLAMP = 4.0                 # m·s anti‑wind‑up

# ------------------------------------------------------------------ attitude (rate) PID
ATT_KP, ATT_KD = 0.05, 0.001

# ------------------------------------------------------------------ limits
MAX_THRUST_G   = 1.6            # multiple of weight
ESC_MAX_OMEGA  = 900.0          # rad/s

# ------------------------------------------------------------------ sim settings
DT, DUR = 0.01, 8.0
N = int(DUR / DT)

# ------------------------------------------------------------------ constants & mixer
p = Params()
A_INV = np.linalg.inv(
    np.array([
        [ p.b,  p.b,  p.b,  p.b],
        [ 0,    p.b*p.l, 0, -p.b*p.l],
        [-p.b*p.l, 0,  p.b*p.l, 0],
        [ p.d, -p.d, p.d, -p.d]])
)

# ------------------------------------------------------------------ memories
alt_int, prev_alt_err = 0.0, 0.0
prev_rate_err = np.zeros(3)

# ------------------------------------------------------------------ env + logs
env   = QuadcopterEnv(dt=DT)
obs   = env.reset()

t_log, x_log, u_log = [0.0], [env.state.as_vector()], [np.zeros(4)]

for _ in range(N):
    # ---------------- altitude PID (with AWU)
    z_err   = Z_SP - obs["pos"][2]
    d_err   = (z_err - prev_alt_err) / DT
    prev_alt_err = z_err

    # integral only if output not saturated OR would drive opposite way
    can_integrate = True
    if np.sign(z_err) == np.sign(alt_int) and abs(alt_int) >= INT_CLAMP:
        can_integrate = False
    if can_integrate:
        alt_int = np.clip(alt_int + z_err*DT, -INT_CLAMP, INT_CLAMP)

    thrust_cmd = p.m * (p.g + ALT_KP*z_err + ALT_KI*alt_int + ALT_KD*d_err)
    thrust_cmd = np.clip(thrust_cmd, 0.0, MAX_THRUST_G * p.m * p.g)

    # stop integrating while saturated and still climbing
    if thrust_cmd >= MAX_THRUST_G * p.m * p.g and z_err > 0:
        alt_int -= z_err * DT     # back‑calculate AWU

    # ---------------- attitude rate PID
    rate_err = BODY_RATE_SP - obs["ang_vel"]
    rate_der = (rate_err - prev_rate_err) / DT
    prev_rate_err = rate_err
    tau_cmd = ATT_KP*rate_err + ATT_KD*rate_der

    # ---------------- mixer → ω
    omega_sq = A_INV @ np.hstack((thrust_cmd, tau_cmd))
    omega_sq = np.clip(omega_sq, 0.0, ESC_MAX_OMEGA**2)
    omega_cmd = np.sqrt(omega_sq)

    obs = env.step(omega_cmd)

    # ---------------- log
    t_log.append(env.t)
    x_log.append(env.state.as_vector())
    u_log.append(omega_cmd)

# ------------------------------------------------------------------ plot
t_arr      = np.asarray(t_log)
states_arr = np.vstack(x_log)
u_arr      = np.vstack(u_log)

print(f"final altitude: {states_arr[-1,2]:.3f} m")
plot_trajectory(t_arr, states_arr, u_arr)
