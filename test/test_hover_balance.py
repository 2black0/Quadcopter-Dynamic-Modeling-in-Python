# tests/test_hover_balance.py
import numpy as np
from quadcopter.dynamics import Params, derivative, QuadState

def test_hover_force_balance():
    p = Params()
    w_hover = np.sqrt(p.m * p.g / (4*p.b))
    y = QuadState(
        pos=np.zeros(3),
        vel=np.zeros(3),
        quat=np.array([1,0,0,0],dtype=float),
        ang_vel=np.zeros(3),
    ).as_vector()
    dy = derivative(0.0, y, np.full(4, w_hover), p)
    assert np.allclose(dy[3:6], 0.0, atol=1e-6)   # no vertical accel
