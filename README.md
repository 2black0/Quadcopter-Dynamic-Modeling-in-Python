# Quadcopter‑Dynamics

Light‑weight, strictly‑typed Python toolkit for **6‑DoF quadrotor simulation**, 3‑D plotting and step‑wise control loops — perfect for control‑systems classes, flight‑code prototyping or RL research.

[![CI](https://github.com/your‑repo/actions/workflows/ci.yml/badge.svg)](…) 
[![PyPI](https://img.shields.io/pypi/v/quadcopter-dynamics.svg)](https://pypi.org/project/quadcopter-dynamics)

---

## Installation

```bash
# latest release
pip install quadcopter-dynamics

# dev install
git clone https://github.com/your-repo/quadcopter-dynamics
cd  quadcopter-dynamics
pip install -e .[dev]      # +pytest +mypy +black +ruff …
```

---

## Quick demo

```bash
python -m quadcopter --plot               # 4 s hover + 3‑D figure
python -m quadcopter --duration 6 --csv run.csv --quiet
```

---

## API at a glance

| Function / class                             | Purpose                                                                                                                                                                      | Key arguments                            |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **`quadcopter.simulation.simulate`**         | One‑shot trajectory generator (adaptive RK45 or fixed‑step RK4). Accepts **either** a controller object with `.update(t,state)` **or** a plain function `(t,state)→motor_ω`. | `duration`, `dt`, `controller`, `method` |
| **`quadcopter.env.QuadcopterEnv`**           | Real‑time, fixed‑step RK4 environment – one `reset()`, then `step(motor_omega)`; ideal for PID / MPC / RL loops.                                                             | `dt`, `reset()`, `step()`                |
| **`quadcopter.dynamics.Params`**             | Physical constants (mass, arm length, thrust factor …).                                                                                                                      | edit attributes to match your air‑frame  |
| **`quadcopter.dynamics.QuadState`**          | Minimal dataclass for the 13‑dim state.                                                                                                                                      | `.from_vector(vec)` / `.as_vector()`     |
| **`quadcopter.plotting.plot_trajectory`**    | Static 3‑D + time‑series figure.                                                                                                                                             | `t, states, controls`                    |
| **`quadcopter.plotting.animate_trajectory`** | Matplotlib animation (MP4 / Jupyter).                                                                                                                                        | `t, states`, `fps`, `save_path`          |

---

### Minimal one‑liner

```python
import numpy as np
from quadcopter.simulation import simulate, Params
from quadcopter.plotting   import plot_trajectory

p = Params()
hover_speed = np.sqrt(p.m * p.g / (4 * p.b))          # rad/s

t, s, u = simulate(
    4.0, 0.02,
    controller=lambda *_: np.full(4, hover_speed),
    method="rk4",
)
plot_trajectory(t, s, u)
```

### Real‑time loop (use your own controller)

```python
from quadcopter import QuadcopterEnv
import numpy as np

def my_controller(obs):
    return np.full(4, 400.0)          # replace with PID / MPC / RL

env = QuadcopterEnv(dt=0.01)
obs = env.reset()
for _ in range(400):                  # 4 s @ 100 Hz
    obs = env.step(my_controller(obs))

print("final altitude:", obs["pos"][2])
```

---

## Verification

```bash
pytest -q                # unit + perf tests (should be all dots)
mypy quadcopter          # static typing gate (should be ‘Success’)
python -m quadcopter --quiet   # CLI smoke test
```

All three finish without errors; a 4 s RK4 run takes ≈ 0.05–0.08 s on a 2020‑era laptop.

---

## Road‑map

* Gymnasium‑compatible wrapper for RL training
* Optional aerodynamic drag model
* Notebook benchmark for tuning PID / LQR / MPC / RL policies

---

Released under the **MIT License**. Contributions and issues are very welcome!
