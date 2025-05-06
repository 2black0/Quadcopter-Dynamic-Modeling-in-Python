import numpy as np
from quadcopter.simulation import simulate, HoverController

def test_hover_doesnt_drift():
    t, traj, _ = simulate(2.0, 0.02, HoverController())
    # final position should remain near origin (±1 cm)
    np.testing.assert_allclose(traj[-1, 0:3], 0.0, atol=1e-2)
