"""
optimal_attitude.py – Optuna optimization for full attitude PID control
Set initial roll, pitch, yaw to 10°, target is 0°,0°,0°
"""

from __future__ import annotations
import numpy as np, optuna, math, warnings
from quadcopter.simulation import simulate, QuadState, Params, BaseController

Z_SP = 5.0
INIT_ANGLE_DEG = 10.0
DT, DUR = 0.01, 4.0
P = Params()

# ----------------- helper: quaternion → euler
def quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = q
    roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(np.clip(2*(w*y - z*x), -1, 1))
    yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw

# ----------------- cascaded controller class
class FullAttitudeCascade(BaseController):
    def __init__(self, ang_kp: np.ndarray, rate_kp: np.ndarray, rate_kd: np.ndarray) -> None:
        self.ang_kp = ang_kp      # [Kp_roll, Kp_pitch, Kp_yaw]
        self.rate_kp = rate_kp
        self.rate_kd = rate_kd
        self.prev_rate_err = np.zeros(3)

    def update(self, t: float, s: QuadState) -> np.ndarray:
        thrust = P.m * P.g  # maintain altitude (hover)

        # extract current euler angles
        roll, pitch, yaw = quat_to_euler(s.quat)
        angle_err = np.array([0.0 - roll, 0.0 - pitch, 0.0 - yaw])
        pqr_des = self.ang_kp * angle_err

        rate_err = pqr_des - s.ang_vel
        rate_der = (rate_err - self.prev_rate_err) / DT
        self.prev_rate_err = rate_err

        # PID on rate error
        tau_cmd = self.rate_kp * rate_err + self.rate_kd * rate_der
        tau_cmd = np.nan_to_num(tau_cmd)

        # thrust + torque to ω²
        b, l, d = P.b, P.l, P.d
        A_inv = np.linalg.inv([
            [b,  b,  b,  b],
            [0,  b*l, 0, -b*l],
            [-b*l, 0,  b*l, 0],
            [d, -d,  d, -d]
        ])

        u_vec = np.hstack((thrust, tau_cmd))
        omega_sq = A_inv @ u_vec
        omega_sq = np.clip(omega_sq, 1e-6, 900**2)
        return np.sqrt(omega_sq)

# ----------------- Optuna objective
def objective(trial: optuna.Trial) -> float:
    ang_kp = np.array([
        trial.suggest_float("ang_kp_roll", 0.5, 6.0),
        trial.suggest_float("ang_kp_pitch", 0.5, 6.0),
        trial.suggest_float("ang_kp_yaw", 0.5, 6.0),
    ])
    rate_kp = np.array([
        trial.suggest_float("rate_kp_roll", 0.02, 0.15),
        trial.suggest_float("rate_kp_pitch", 0.02, 0.15),
        trial.suggest_float("rate_kp_yaw", 0.02, 0.15),
    ])
    rate_kd = np.array([
        trial.suggest_float("rate_kd_roll", 0.0, 0.01),
        trial.suggest_float("rate_kd_pitch", 0.0, 0.01),
        trial.suggest_float("rate_kd_yaw", 0.0, 0.01),
    ])

    ctrl = FullAttitudeCascade(ang_kp, rate_kp, rate_kd)

    angle_rad = np.deg2rad(INIT_ANGLE_DEG)
    init_quat = np.array([
        math.cos(angle_rad/2),
        math.sin(angle_rad/2),
        math.sin(angle_rad/2),
        math.sin(angle_rad/2)
    ])
    init_quat /= np.linalg.norm(init_quat)

    init = QuadState(
        pos=np.array([0, 0, Z_SP]),
        vel=np.zeros(3),
        quat=init_quat,
        ang_vel=np.zeros(3),
    )

    try:
        t, y, _ = simulate(DUR, DT, ctrl, initial_state=init, method="rk4")
    except Exception:
        return float("inf")
    if np.isnan(y).any() or np.isinf(y).any():
        return float("inf")

    roll_errs  = []
    pitch_errs = []
    yaw_errs   = []

    for q in y[:, 6:10]:
        roll, pitch, yaw = quat_to_euler(q)
        roll_errs.append(roll)
        pitch_errs.append(pitch)
        yaw_errs.append(yaw)

    mse = np.mean(np.square(roll_errs) + np.square(pitch_errs) + np.square(yaw_errs))
    return mse

# ----------------- run optimization
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1000, show_progress_bar=True)
    print("Best gains:", study.best_params)
