# 🛸 Quadcopter Dynamic Modeling in Python

This repository contains a simple physics-based simulation of a quadrotor (quadcopter) using Python. It includes the mathematical model, control inputs, simulation execution, and trajectory visualization.

> 💡 Suitable for educational and experimental purposes in control systems, dynamics, or robotics courses.

---

## 📁 Project Structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
└── Project
    ├── quadmodel.py     # Defines the dynamic equations of motion (EOM) of the quadrotor
    ├── quadplot.py      # Plots trajectory and orientation (Euler angles)
    ├── quadrun.py       # Main script that runs simulation
    └── quadvar.py       # Contains global variables and initial conditions
```

---

## 🚀 How to Run

### 1. Install Python dependencies

Make sure Python 3 and `pip` are installed. Then install required packages:

```bash
pip install -r requirements.txt
```

Alternatively, install manually:

```bash
pip install numpy matplotlib
```

### 2. Run the simulation

```bash
python3 Project/quadrun.py
```

---

## 📊 Features Simulated

* 3D translational motion (x, y, z)
* Rotational motion (roll `φ`, pitch `θ`, yaw `ψ`)
* Forces and torques from motor speeds (`w1`, `w2`, `w3`, `w4`)
* Uses Euler integration for numerical solution
* Realistic constants: mass, moments of inertia, thrust and drag coefficients

---

## 📈 Output

The simulation generates plots of:

* Position over time (x, y, z)
* Orientation (Euler angles φ, θ, ψ)

These are automatically shown after running the script.

---

## 🧠 Educational Value

This project is ideal for:

* Teaching basic flight dynamics
* Demonstrating Newton-Euler modeling
* Control algorithm prototyping (PID, LQR, etc.)

---

## 📄 License

This project is licensed under the terms of the **MIT License**. See the [LICENSE](./LICENSE) file for details.

---