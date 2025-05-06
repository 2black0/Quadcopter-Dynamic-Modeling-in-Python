"""
run_attitude.py – simultaneous altitude (5 m) + attitude hold.
Initial roll = 10°; pitch, yaw = 0°; pos = (0,0,5).
Uses:
    • Outer P‑loop on roll/pitch/yaw angles  -> desired body‑rates
    • Inner PID on body‑rates     -> motor torques
    • Previously tuned altitude‑PID for z
"""

from __future__ import annotations
import numpy as np
from quadcopter import QuadcopterEnv, Params, QuadState
from quadcopter.plotting import plot_trajectory

# ------------------------------------------------------------------ set‑points
Z_SP         = 5.0
ANGLE_SP_DEG = np.array([0.0, 0.0, 0.0])     # roll, pitch, yaw in deg

# gains (altitude from Optuna run)
ALT_KP, ALT_KI, ALT_KD = 3.8256366526919017, 0.0009449363883856005, 2.9946421709435156
INT_CLAMP       = 4.0
MAX_THRUST_G    = 1.6
ESC_MAX_OMEGA   = 900.0

# roll/pitch/yaw outer loop (angle -> rate) and inner rate PID (all axes)
#Best gains: {'ang_kp_roll': 3.1885864424801165, 'ang_kp_pitch': 5.461127467602318, 'ang_kp_yaw': 0.6136445657545656, 
# 'rate_kp_roll': 0.07368764730228543, 'rate_kp_pitch': 0.028620827827169307, 'rate_kp_yaw': 0.14814940214664976, 
# 'rate_kd_roll': 0.004471952365169602, 'rate_kd_pitch': 0.009479194005677325, 'rate_kd_yaw': 0.00998569890261953}
ANG_KP = np.array([3.1885864424801165, 5.461127467602318, 0.6136445657545656])            # deg -> deg/s (roll, pitch, yaw)
RATE_KP = np.array([0.07368764730228543, 0.028620827827169307, 0.14814940214664976])         # PID body rate control (p, q, r)
RATE_KD = np.array([0.004471952365169602, 0.009479194005677325, 0.00998569890261953])

# sim parameters
DT, DUR = 0.01, 8.0
N = int(DUR / DT)

# ------------------------------------------------------------------ helpers
def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Return roll, pitch, yaw in **radians** from w,x,y,z quaternion."""
    w, x, y, z = q
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw

# ------------------------------------------------------------------ constants / mixer
p = Params()
A_INV = np.linalg.inv(
    np.array([
        [p.b,  p.b,  p.b,  p.b],
        [0,    p.b*p.l, 0, -p.b*p.l],
        [-p.b*p.l, 0,  p.b*p.l, 0],
        [p.d, -p.d, p.d, -p.d]])
)

# ------------------------------------------------------------------ PID memory
alt_int, prev_alt_err = 0.0, 0.0
prev_rate_err = np.zeros(3)

# ------------------------------------------------------------------ environment with specified initial state
init_state = QuadState(
    pos=np.array([0.0, 0.0, Z_SP]),
    vel=np.zeros(3),
    quat=np.array([
        np.cos(np.deg2rad(10)/2),          # roll = 10 deg
        np.sin(np.deg2rad(10)/2), 0.0, 0.0 # x, y, z
    ]),
    ang_vel=np.zeros(3),
)
env = QuadcopterEnv(dt=DT)
env.reset()
env.state = init_state
obs = env._observation()

# logs
t_log, x_log, u_log = [0.0], [env.state.as_vector()], [np.zeros(4)]

for _ in range(N):
    # ------------------------------------------------ altitude PID
    z_err  = Z_SP - obs["pos"][2]
    d_err  = (z_err - prev_alt_err) / DT
    prev_alt_err = z_err

    alt_int = np.clip(alt_int + z_err*DT, -INT_CLAMP, INT_CLAMP)
    thrust  = p.m * (p.g + ALT_KP*z_err + ALT_KI*alt_int + ALT_KD*d_err)
    thrust  = np.clip(thrust, 0.0, MAX_THRUST_G * p.m * p.g)

    # ------------------------------------------------ attitude: angle -> rate -> torque
    roll, pitch, yaw = quat_to_euler(obs["quat"])
    angles = np.rad2deg(np.array([roll, pitch, yaw]))
    angle_err = np.clip(ANGLE_SP_DEG - angles, -30, 30)
    rate_sp = np.deg2rad(ANG_KP * angle_err)  # desired p, q, r in rad/s

    rate_err = np.clip(rate_sp - obs["ang_vel"], -10, 10)
    rate_der = (rate_err - prev_rate_err) / DT
    prev_rate_err = rate_err

    tau_cmd = np.clip(RATE_KP * rate_err + RATE_KD * rate_der, -1, 1)

    # ------------------------------------------------ mix to motor speeds
    omega_sq = A_INV @ np.hstack((thrust, tau_cmd))
    omega_sq = np.clip(omega_sq, 1e-6, ESC_MAX_OMEGA**2)
    omega_cmd = np.sqrt(omega_sq)

    obs = env.step(omega_cmd)

    # ------------------------------------------------ log
    t_log.append(env.t)
    x_log.append(env.state.as_vector())
    u_log.append(omega_cmd)

# ------------------------------------------------------------------ plot + report
t_arr      = np.asarray(t_log)
states_arr = np.vstack(x_log)
controls   = np.vstack(u_log)

final_rpy = quat_to_euler(states_arr[-1,6:10])
print(f"final altitude: {states_arr[-1,2]:.3f} m, final RPY: {np.rad2deg(final_rpy)} °")
plot_trajectory(t_arr, states_arr, controls)