from quadcopter.simulation import simulate
from quadcopter.controllers import AltitudePID
from quadcopter.plotting import plot_trajectory


def main() -> None:
    # PID tuned for a 1‑metre altitude hold
    controller = AltitudePID(setpoint=1.0, kp=3.0, ki=1.5, kd=1.0)

    # Run a 4‑second simulation at 20 ms steps
    t, states, controls = simulate(sim_time=4.0, dt=0.02, controller=controller)

    # Print step‑response metrics (rise‑time, overshoot, ISE, …)
    for name, value in controller.step_metrics(t, states[:, 2]).items():
        print(f"{name:20s}: {value:.3f}")

    # Quick visual check
    plot_trajectory(t, states, controls)


if __name__ == "__main__":
    main()
