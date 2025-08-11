# Quadcopter‑Sim

Light‑weight, strictly‑typed Python toolkit for **6‑DoF quadrotor simulation**, 3‑D plotting and step‑wise control loops — perfect for control‑systems classes, flight‑code prototyping or RL research.

[![PyPI](https://img.shields.io/pypi/v/quadcopter-sim.svg)](https://pypi.org/project/quadcopter-sim)

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Line Interface](#command-line-interface)
- [Core Features](#core-features)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Examples & Notebooks](#examples--notebooks)
- [Testing & Verification](#testing--verification)
- [Academic Use](#academic-use)
- [Roadmap](#roadmap)

---

## Installation

```bash
# Latest release
pip install quadcopter-sim

# Development install with all optional dependencies
git clone https://github.com/2black0/quadcopter-sim-python
cd quadcopter-sim-python
pip install -e .[all]      # includes RL, control, and data analysis dependencies

# Install specific optional dependencies
pip install -e .[rl]       # Gymnasium for RL
pip install -e .[control]  # SciPy for advanced control
pip install -e .[data]     # SciPy for data export
pip install -e .[dev]      # Development tools
```

---

## Quick Start

### Minimal Example

```python
import numpy as np
from quadcopter.simulation import simulate, Params
from quadcopter.plotting import plot_trajectory

p = Params()
hover_speed = np.sqrt(p.m * p.g / (4 * p.b))          # rad/s

t, s, u = simulate(
    4.0, 0.02,
    controller=lambda *_: np.full(4, hover_speed),
    method="rk4",
)
plot_trajectory(t, s, u)
```

### Command Line Usage

```bash
# Basic hover simulation with 3D plot
python -m quadcopter --plot

# PID position control
python -m quadcopter --controller pid --target-pos 1 1 2 --duration 10 --plot
```

---

## Command Line Interface

The package provides a comprehensive command-line interface for simulation, control, and analysis:

### Basic Usage Examples

```bash
# Basic hover simulation
python -m quadcopter --plot               # 4 s hover + 3‑D figure
python -m quadcopter --duration 6 --csv run.csv --quiet

# PID position control
python -m quadcopter --controller pid --target-pos 1 1 2 --duration 10 --plot

# LQR control with custom parameters
python -m quadcopter --controller lqr --duration 5 --plot

# Academic analysis and logging
python -m quadcopter --controller pid --target-pos 1 1 2 --duration 10 --academic-log results

# Advanced PID tuning
python -m quadcopter --controller pid --target-pos 1 0 1 --duration 5 \
  --pid-kp 3 3 5 --pid-ki 0.2 0.2 0.3 --pid-kd 0.6 0.6 1.2 \
  --plot

# Custom initial conditions
python -m quadcopter --controller pid --target-pos 0 0 2 --duration 5 \
  --init-pos 0 0 1 --init-vel 0 0 0.5 \
  --csv trajectory.csv --json log.json --matlab data.mat

# Enhanced plotting options
python -m quadcopter --controller pid --target-pos 1 1 2 --duration 10 --plot-errors
python -m quadcopter --controller pid --target-pos 1 1 2 --duration 10 --plot-comparison
```

### CLI Options Reference

```
usage: python -m quadcopter [-h] [--duration DURATION] [--dt DT]
                            [--method {rk45,rk4}] [--rtol RTOL] [--atol ATOL]
                            [--controller {hover,pid,lqr}]
                            [--pid-kp PID_KP PID_KP PID_KP]
                            [--pid-ki PID_KI PID_KI PID_KI]
                            [--pid-kd PID_KD PID_KD PID_KD]
                            [--target-pos TARGET_POS TARGET_POS TARGET_POS]
                            [--plot] [--csv CSV] [--json JSON]
                            [--matlab MATLAB] [--academic-log ACADEMIC_LOG]
                            [--controller-type {pid,lqr,rl}]
                            [--init-pos INIT_POS INIT_POS INIT_POS]
                            [--init-vel INIT_VEL INIT_VEL INIT_VEL] [--quiet]
                            [--verbose]

Comprehensive quadcopter simulation and analysis tool.

Simulation Parameters:
  --duration DURATION       simulation time [s]
  --dt DT                   integration step [s]
  --method {rk45,rk4}       integration method (adaptive RK45 or fixed‑step RK4)
  --rtol RTOL               solver rtol
  --atol ATOL               solver atol

Controller Options:
  --controller {hover,pid,lqr}    controller type to use
  --pid-kp PID_KP PID_KP PID_KP   PID Kp gains for x, y, z axes
  --pid-ki PID_KI PID_KI PID_KI   PID Ki gains for x, y, z axes
  --pid-kd PID_KD PID_KD PID_KD   PID Kd gains for x, y, z axes
  --target-pos TARGET_POS TARGET_POS TARGET_POS  target position [x, y, z]

Initial Conditions:
  --init-pos INIT_POS INIT_POS INIT_POS    initial position [x, y, z]
  --init-vel INIT_VEL INIT_VEL INIT_VEL    initial velocity [vx, vy, vz]

Output Options:
  --plot                    show matplotlib figure
  --csv CSV                 save (t, state, control) to CSV
  --json JSON               save simulation log to JSON
  --matlab MATLAB           save simulation log to MATLAB .mat file
  --academic-log ACADEMIC_LOG    enable academic logging and save to directory
  --quiet                   suppress info output
  --verbose                 enable verbose output
```

---

## Core Features

### 1. Advanced Control Systems

Comprehensive control system implementations:

```python
from quadcopter.controllers import PIDController, PositionController, LQRController

# PID Position Control
position_ctrl = PositionController(
    x_pid=PIDController(kp=2.0, ki=0.1, kd=0.5),
    y_pid=PIDController(kp=2.0, ki=0.1, kd=0.5),
    z_pid=PIDController(kp=4.0, ki=0.2, kd=1.0),
    target_pos=np.array([1.0, -1.0, 2.0])
)

# LQR Control
A = np.eye(12)  # State matrix
B = np.eye(12, 4)  # Input matrix
lqr_ctrl = LQRController(A=A, B=B, Q=np.eye(12), R=np.eye(4))
```

### 2. Reinforcement Learning Integration

Gymnasium-compatible environment for RL research:

```python
from quadcopter.gym_env import QuadcopterGymEnv

env = QuadcopterGymEnv()
obs, info = env.reset()
action = env.action_space.sample()  # Random action
obs, reward, terminated, truncated, info = env.step(action)
```

### 3. Real-time Simulation

Enhanced environment with real-time capabilities:

```python
from quadcopter.env import RealTimeQuadcopterEnv

# Run simulation at half real-time speed
env = RealTimeQuadcopterEnv(dt=0.02, real_time_factor=0.5)
```

### 4. Comprehensive Logging

Advanced logging with multiple export formats:

```python
from quadcopter.logging import simulate_with_logging

log = simulate_with_logging(duration=5.0, dt=0.02, controller=my_controller)
log.save_csv("simulation.csv")      # CSV for analysis
log.save_json("simulation.json")    # JSON for structured data
log.save_matlab("simulation.mat")   # MATLAB for advanced analysis
```

### 5. Academic Evaluation and Visualization

Comprehensive academic evaluation tools for research publications:

```python
from quadcopter.logging import simulate_with_academic_logging
from quadcopter.evaluation import AcademicEvaluator

# Run simulation with academic logging
log = simulate_with_academic_logging(
    duration=10.0, 
    dt=0.02, 
    controller=my_controller,
    ref_position=np.array([1.0, -1.0, 2.0]),
    controller_type="pid"
)

# Create academic evaluator
evaluator = AcademicEvaluator(log)

# Generate comprehensive analysis
metrics = evaluator.generate_comprehensive_analysis("results")

# Generate specific plots
evaluator.plot_3d_trajectory("trajectory.png")
evaluator.plot_state_tracking("tracking.png")
evaluator.plot_error_analysis("errors.png")
evaluator.plot_control_effort("control.png")
```

### 6. Enhanced Visualization

Comprehensive plotting capabilities:

```python
from quadcopter.plotting import (
    plot_trajectory, 
    plot_control_errors, 
    plot_3d_trajectory_comparison,
    plot_frequency_analysis
)

# Control error analysis
plot_control_errors(t, states, targets)

# Trajectory comparison
plot_3d_trajectory_comparison([
    (states1, "Controller A"),
    (states2, "Controller B")
])

# Frequency domain analysis
plot_frequency_analysis(t, signals, ["X Position", "Y Position", "Z Position"])
```

---

## API Reference

| Function / Class | Purpose | Key Arguments |
|------------------|---------|---------------|
| **Core Simulation** |
| `quadcopter.simulation.simulate` | One‑shot trajectory generator (adaptive RK45 or fixed‑step RK4) | `duration`, `dt`, `controller`, `method` |
| `quadcopter.dynamics.Params` | Physical constants (mass, arm length, thrust factor) | edit attributes to match your air‑frame |
| `quadcopter.dynamics.QuadState` | Minimal dataclass for the 13‑dim state | `.from_vector(vec)` / `.as_vector()` |
| **Environments** |
| `quadcopter.env.QuadcopterEnv` | Real‑time, fixed‑step RK4 environment | `dt`, `reset()`, `step()` |
| `quadcopter.env.RealTimeQuadcopterEnv` | Real-time environment with timing control | `dt`, `real_time_factor`, `reset()`, `step()` |
| `quadcopter.gym_env.QuadcopterGymEnv` | Gymnasium-compatible environment for RL training | `dt`, `max_steps` |
| **Controllers** |
| `quadcopter.controllers.PIDController` | PID controller with anti-windup and output limits | `kp`, `ki`, `kd`, `max_output` |
| `quadcopter.controllers.PositionController` | 3D position controller using PID for each axis | `x_pid`, `y_pid`, `z_pid`, `target_pos` |
| `quadcopter.controllers.LQRController` | Linear Quadratic Regulator controller | `A`, `B`, `Q`, `R` matrices |
| **Logging & Analysis** |
| `quadcopter.logging.SimulationLog` | Comprehensive logging with multiple export formats | `save_csv()`, `save_json()`, `save_matlab()` |
| `quadcopter.logging.AcademicLog` | Academic-grade logging for research publications | `add_entry()`, `save_csv()`, `save_json()`, `save_matlab()` |
| `quadcopter.logging.simulate_with_academic_logging` | Simulation with academic-grade logging | `duration`, `dt`, `controller`, `ref_position` |
| `quadcopter.evaluation.AcademicEvaluator` | Academic evaluation tools for performance analysis | `plot_3d_trajectory()`, `plot_state_tracking()`, `generate_performance_report()` |
| **Visualization** |
| `quadcopter.plotting.plot_trajectory` | Static 3‑D + time‑series figure | `t, states, controls` |
| `quadcopter.plotting.plot_control_errors` | Control error analysis over time | `t, states, targets` |
| `quadcopter.plotting.plot_3d_trajectory_comparison` | Compare multiple trajectories in 3D | `trajectories` |
| `quadcopter.plotting.plot_frequency_analysis` | Frequency domain analysis of signals | `t, signals, signal_names` |
| `quadcopter.plotting.animate_trajectory` | Matplotlib animation (MP4 / Jupyter) | `t, states`, `fps`, `save_path` |
| **Utilities** |
| `quadcopter.utils.create_pid_position_controller` | Create PID position controller with default gains | `target_pos`, `kp`, `ki`, `kd` |
| `quadcopter.utils.create_pid_attitude_controller` | Create PID attitude controller with default gains | `target_attitude`, `kp`, `ki`, `kd` |
| `quadcopter.utils.create_lqr_controller` | Create LQR controller with default matrices | `params`, `Q`, `R` |
| `quadcopter.utils.create_hover_controller` | Create simple hover controller | `params` |

---

## Usage Examples

### 1. Simplified PID Position Control

```python
from quadcopter import simulate, create_pid_position_controller
from quadcopter.dynamics import QuadState
from quadcopter.plotting import plot_trajectory
import numpy as np

# Create position controller using utility function
controller = create_pid_position_controller(
    target_pos=[1.0, -1.0, 2.0],
    kp=(2.0, 2.0, 4.0),
    ki=(0.1, 0.1, 0.2),
    kd=(0.5, 0.5, 1.0)
)

# Set up initial state
initial_state = QuadState(
    pos=np.array([0.0, 0.0, 0.0]),
    vel=np.array([0.0, 0.0, 0.0]),
    quat=np.array([1.0, 0.0, 0.0, 0.0]),
    ang_vel=np.array([0.0, 0.0, 0.0])
)

# Run simulation
t, states, controls = simulate(
    duration=10.0,
    dt=0.02,
    controller=controller,
    initial_state=initial_state,
    method="rk4"
)

# Plot results
plot_trajectory(t, states, controls)
```

### 2. Advanced PID Position Control

```python
from quadcopter import simulate, PositionController, PIDController
from quadcopter.dynamics import QuadState
from quadcopter.plotting import plot_trajectory
import numpy as np

# Create position controller
controller = PositionController(
    x_pid=PIDController(kp=2.0, ki=0.1, kd=0.5),
    y_pid=PIDController(kp=2.0, ki=0.1, kd=0.5),
    z_pid=PIDController(kp=4.0, ki=0.2, kd=1.0),
    target_pos=np.array([1.0, -1.0, 2.0])
)

# Set up initial state
initial_state = QuadState(
    pos=np.array([0.0, 0.0, 0.0]),
    vel=np.array([0.0, 0.0, 0.0]),
    quat=np.array([1.0, 0.0, 0.0, 0.0]),
    ang_vel=np.array([0.0, 0.0, 0.0])
)

# Run simulation
t, states, controls = simulate(
    duration=10.0,
    dt=0.02,
    controller=controller,
    initial_state=initial_state,
    method="rk4"
)

# Plot results
plot_trajectory(t, states, controls)
```

### 3. Reinforcement Learning

```python
from quadcopter.gym_env import QuadcopterGymEnv
import numpy as np

# Create RL environment
env = QuadcopterGymEnv()

# Simple policy
def simple_policy(observation):
    # Simple hover policy
    hover_speed = np.sqrt(0.65 * 9.81 / (4 * 3.25e-5))
    return np.full(4, hover_speed)

# Training loop
obs, info = env.reset()
for _ in range(1000):
    action = simple_policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### 4. Real-time Simulation

```python
from quadcopter.env import RealTimeQuadcopterEnv
import numpy as np

# Create real-time environment (half speed)
env = RealTimeQuadcopterEnv(dt=0.02, real_time_factor=0.5)
obs = env.reset()

# Simple hover controller
hover_speed = np.sqrt(0.65 * 9.81 / (4 * 3.25e-5))
motor_speeds = np.full(4, hover_speed)

# Run simulation
for _ in range(200):  # 4 seconds
    obs = env.step(motor_speeds)
    print(f"t={obs['t'][0]:.2f}s, pos={obs['pos']}")

print("Simulation completed!")
```

### 5. Comprehensive Logging

```python
from quadcopter.logging import simulate_with_logging
from quadcopter.controllers import PositionController, PIDController
import numpy as np

# Create controller
controller = PositionController(
    x_pid=PIDController(kp=2.0, ki=0.1, kd=0.5),
    y_pid=PIDController(kp=2.0, ki=0.1, kd=0.5),
    z_pid=PIDController(kp=4.0, ki=0.2, kd=1.0),
    target_pos=np.array([1.0, -1.0, 2.0])
)

# Run simulation with logging
log = simulate_with_logging(
    duration=5.0,
    dt=0.02,
    controller=controller,
    method="rk4"
)

# Export data in multiple formats
log.save_csv("trajectory_data.csv")
log.save_json("trajectory_data.json")
log.save_matlab("trajectory_data.mat")
```

### 6. Academic Evaluation

```python
from quadcopter.logging import simulate_with_academic_logging
from quadcopter.evaluation import AcademicEvaluator
from quadcopter.controllers import PositionController, PIDController
from quadcopter.dynamics import QuadState
import numpy as np

# Create position controller
controller = PositionController(
    x_pid=PIDController(kp=2.0, ki=0.1, kd=0.5),
    y_pid=PIDController(kp=2.0, ki=0.1, kd=0.5),
    z_pid=PIDController(kp=4.0, ki=0.2, kd=1.0),
    target_pos=np.array([1.0, -1.0, 2.0])
)

# Set up initial state
initial_state = QuadState(
    pos=np.array([0.0, 0.0, 0.0]),
    vel=np.array([0.0, 0.0, 0.0]),
    quat=np.array([1.0, 0.0, 0.0, 0.0]),
    ang_vel=np.array([0.0, 0.0, 0.0])
)

# Run simulation with academic logging
log = simulate_with_academic_logging(
    duration=10.0,
    dt=0.02,
    controller=controller,
    initial_state=initial_state,
    ref_position=np.array([1.0, -1.0, 2.0]),
    controller_type="pid"
)

# Create academic evaluator
evaluator = AcademicEvaluator(log)

# Generate comprehensive analysis with all plots and metrics
metrics = evaluator.generate_comprehensive_analysis("academic_results")

print("Academic evaluation completed! Results saved to 'academic_results' directory.")
```

---

## Examples & Notebooks

The library includes comprehensive examples demonstrating various features:

### Python Examples

Run any example with: `python examples/<example_name>.py`

- **`pid_control_example.py`** - PID position control demonstration
- **`lqr_control_example.py`** - LQR control implementation
- **`rl_training_example.py`** - Reinforcement learning training
- **`real_time_simulation.py`** - Real-time simulation with timing
- **`enhanced_plotting_example.py`** - Advanced visualization techniques
- **`enhanced_logging_example.py`** - Comprehensive data logging
- **`academic_evaluation_example.py`** - Academic evaluation and analysis

### Jupyter Notebooks

Run any notebook with: `jupyter notebook notebooks/<notebook_name>.ipynb`

- **`control_system_design.ipynb`** - Interactive PID/LQR tuning
- **`rl_training_tutorial.ipynb`** - RL experimentation and analysis
- **`data_analysis.ipynb`** - Log analysis and visualization
- **`performance_comparison.ipynb`** - Comparing different control methods

---

## Testing & Verification

```bash
pytest -q                        # Unit + performance tests (should be all dots)
mypy quadcopter                  # Static typing gate (should be 'Success')
python -m quadcopter --quiet     # CLI smoke test
```

All three commands should finish without errors. A 4s RK4 simulation typically takes ≈ 0.05–0.08s on a 2020‑era laptop.

---

## Academic Use

This library is designed for academic research and education. When using in research publications, please cite:

```bibtex
@software{quadcopter_dynamics_2025,
  author = {2black0},
  title = {Quadcopter-Sim: A Python Toolkit for 6-DoF Quadrotor Simulation},
  year = {2025},
  doi = {TBD},
  url = {https://github.com/2black0/quadcopter-sim-python}
}
```

---

## Roadmap

✅ **Completed Features:**
- Advanced control systems (PID, LQR, Fuzzy Logic)
- Gymnasium‑compatible wrapper for RL training
- Comprehensive logging for academic research
- Real-time simulation capabilities
- Enhanced visualization and analysis tools
- Academic evaluation and analysis tools for research publications
- Optional aerodynamic drag model
- Notebook benchmark for tuning PID / LQR / MPC / RL policies

🚧 **Future Development:**
- More controller types (MPC, Sliding Mode, etc.)
- Advanced disturbance models
- Multi-agent simulation capabilities
- Integration with hardware-in-the-loop testing

---

**Released under the MIT License.** Contributions and issues are very welcome!
