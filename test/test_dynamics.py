import numpy as np
from quadcopter.dynamics import Params, QuadState, derivative

def test_derivative_shape_and_stability():
    """Hover test: all motors equal → vertical thrust only."""
    # ➊ nominal hover state (at origin, zero velocity & rates)
    state = QuadState(
        pos=np.zeros(3),
        vel=np.zeros(3),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),   # identity quaternion
        ang_vel=np.zeros(3),
    )

    # ➋ pack to a flat vector
    y0 = state.as_vector()

    # ➌ motor speeds chosen to balance weight:  w = √(m g / (4 b))
    p = Params()
    w_hover = np.sqrt(p.m * p.g / (4 * p.b))
    control = np.full(4, w_hover)

    # ➍ call the ODE right‑hand side
    dydt = derivative(t=0.0, y=y0, control=control, p=p)

    # ➎ basic sanity checks
    assert dydt.shape == (13,)
    # In perfect hover, vertical acceleration should be ≈ 0
    # (allow 1 e‑6 N rounding error)
    np.testing.assert_allclose(dydt[5], 0.0, atol=1e-6)   # z̈ component
    # No linear velocity → no horizontal accel
    np.testing.assert_allclose(dydt[3:5], 0.0, atol=1e-9)
    # No torques → zero angular acceleration
    np.testing.assert_allclose(dydt[10:13], 0.0, atol=1e-9)
