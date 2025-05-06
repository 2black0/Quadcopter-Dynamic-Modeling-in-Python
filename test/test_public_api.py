import numpy as np
from quadcopter import simulate, QuadcopterEnv
from quadcopter.dynamics import Params
from types import SimpleNamespace

def test_simulate_shapes():
    p = Params()
    w = np.sqrt(p.m * p.g / (4 * p.b))
    ctrl = SimpleNamespace(update=lambda *_: np.full(4, w))
    t, y, u = simulate(1.0, 0.02, ctrl, method="rk4")
    assert y.shape == (51, 13)
    assert u.shape == (51, 4)
    assert np.isfinite(y).all()

def test_env_step_integrity():
    env = QuadcopterEnv(dt=0.02)
    obs = env.reset()
    omega = np.full(4, 400.0)
    for _ in range(50):
        obs = env.step(omega)
    assert "pos" in obs and obs["pos"].shape == (3,)
    assert np.isfinite(obs["pos"]).all()
