"""
optimal.py – Bayesian search for altitude‑PID gains that hit 5 m step
"""

from __future__ import annotations
import numpy as np
import optuna

from quadcopter.simulation import simulate, BaseController, QuadState, Params
from quadcopter.dynamics import derivative

Z_SP = 5.0          # altitude set‑point
DT    = 0.02
DUR   = 6.0
P     = Params()


# ------------------------------------------------------------------ minimal PID controller
class AltPID(BaseController):
    def __init__(self, kp: float, ki: float, kd: float) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.int_err = 0.0
        self.prev_err: float | None = None

    def update(self, t: float, state: QuadState) -> np.ndarray:
        z = state.pos[2]
        err = Z_SP - z

        if self.prev_err is None:
            der = 0.0
        else:
            der = (err - self.prev_err) / DT
        self.prev_err = err

        self.int_err += err * DT
        thrust = P.m * (P.g + self.kp*err + self.ki*self.int_err + self.kd*der)
        thrust = np.clip(thrust, 0.0, 2.0 * P.m * P.g)        # 2 g cap

        # equal share per motor
        w = np.sqrt(thrust / (4 * P.b))
        return np.full(4, w, dtype=np.float64)


# ------------------------------------------------------------------ Optuna objective
def objective(trial: optuna.Trial) -> float:
    kp = trial.suggest_float("kp", 0.5, 5.0)
    ki = trial.suggest_float("ki", 0.0, 3.0)
    kd = trial.suggest_float("kd", 0.0, 3.0)

    ctrl = AltPID(kp, ki, kd)

    t, y, _ = simulate(DUR, DT, ctrl, method="rk4")
    z = y[:, 2]

    rise_time = next((ti for ti, zi in zip(t, z) if zi >= 0.9 * Z_SP), DUR)
    overshoot = max(0.0, (z.max() - Z_SP) / Z_SP)
    settle_idx = next((i for i in range(len(z) - 1, 0, -1)
                       if abs(z[i] - Z_SP) > 0.02 * Z_SP), None)
    settling_time = t[settle_idx] if settle_idx else rise_time

    # objective: we want to minimise time and overshoot
    return rise_time + 3.0 * overshoot + settling_time


# ------------------------------------------------------------------ run search
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1000, show_progress_bar=True)

    print("Best gains:", study.best_params)
