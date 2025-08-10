"""
utils.py - Utility functions for common quadcopter control tasks.

This module provides utility functions for creating commonly used controllers
and performing typical setup tasks, reducing code duplication in examples
and user applications.
"""

from typing import Optional
import numpy as np
from .controllers import PIDController, PositionController, LQRController, AttitudeController
from .dynamics import Params


def create_pid_position_controller(
    target_pos: np.ndarray,
    kp: tuple = (2.0, 2.0, 4.0),
    ki: tuple = (0.1, 0.1, 0.2),
    kd: tuple = (0.5, 0.5, 1.0)
) -> PositionController:
    """
    Create a PID position controller with default or custom gains.
    
    Parameters
    ----------
    target_pos : array-like
        Target position [x, y, z]
    kp : tuple, optional
        Kp gains for x, y, z axes (default: (2.0, 2.0, 4.0))
    ki : tuple, optional
        Ki gains for x, y, z axes (default: (0.1, 0.1, 0.2))
    kd : tuple, optional
        Kd gains for x, y, z axes (default: (0.5, 0.5, 1.0))
        
    Returns
    -------
    PositionController
        Configured position controller
    """
    return PositionController(
        x_pid=PIDController(kp=kp[0], ki=ki[0], kd=kd[0]),
        y_pid=PIDController(kp=kp[1], ki=ki[1], kd=kd[1]),
        z_pid=PIDController(kp=kp[2], ki=ki[2], kd=kd[2]),
        target_pos=np.array(target_pos)
    )


def create_pid_attitude_controller(
    target_attitude: np.ndarray = np.array([0.0, 0.0, 0.0]),
    kp: tuple = (1.0, 1.0, 1.0),
    ki: tuple = (0.01, 0.01, 0.01),
    kd: tuple = (0.1, 0.1, 0.1)
) -> AttitudeController:
    """
    Create a PID attitude controller with default or custom gains.
    
    Parameters
    ----------
    target_attitude : array-like, optional
        Target attitude [roll, pitch, yaw] (default: [0, 0, 0])
    kp : tuple, optional
        Kp gains for roll, pitch, yaw axes (default: (1.0, 1.0, 1.0))
    ki : tuple, optional
        Ki gains for roll, pitch, yaw axes (default: (0.01, 0.01, 0.01))
    kd : tuple, optional
        Kd gains for roll, pitch, yaw axes (default: (0.1, 0.1, 0.1))
        
    Returns
    -------
    AttitudeController
        Configured attitude controller
    """
    return AttitudeController(
        roll_pid=PIDController(kp=kp[0], ki=ki[0], kd=kd[0]),
        pitch_pid=PIDController(kp=kp[1], ki=ki[1], kd=kd[1]),
        yaw_pid=PIDController(kp=kp[2], ki=ki[2], kd=kd[2]),
        target_attitude=np.array(target_attitude)
    )


def create_lqr_controller(
    params: Optional[Params] = None,
    Q: Optional[np.ndarray] = None,
    R: Optional[np.ndarray] = None
) -> LQRController:
    """
    Create an LQR controller with default or custom parameters.
    
    Parameters
    ----------
    params : Params, optional
        Quadcopter parameters (default: Params())
    Q : ndarray, optional
        State cost matrix (default: identity matrix)
    R : ndarray, optional
        Input cost matrix (default: 0.1 * identity matrix)
        
    Returns
    -------
    LQRController
        Configured LQR controller
    """
    if params is None:
        params = Params()
        
    # Simple A and B matrices for demonstration
    # These would normally be computed by linearizing the dynamics
    A = np.zeros((12, 12))
    B = np.zeros((12, 4))
    
    # Fill with some basic values to make them non-singular
    A[:3, 3:6] = np.eye(3)  # Position dynamics
    A[3:6, 6:9] = np.eye(3)  # Velocity dynamics (simplified)
    A[6:9, 9:12] = np.eye(3)  # Orientation dynamics
    B[3:6, :] = np.ones((3, 4)) * 0.1  # Thrust input to acceleration
    B[9:12, :] = np.array([[1, -1, 0, 0], [0, 0, 1, -1], [1, 1, -1, -1]]) * 0.1  # Torque inputs
    
    if Q is None:
        Q = np.eye(12)
        
    if R is None:
        R = np.eye(4) * 0.1
        
    return LQRController(params=params, A=A, B=B, Q=Q, R=R)


def create_hover_controller(params: Optional[Params] = None) -> "HoverController":
    """
    Create a simple hover controller.
    
    Parameters
    ----------
    params : Params, optional
        Quadcopter parameters (default: Params())
        
    Returns
    -------
    HoverController
        Configured hover controller
    """
    from .simulation import HoverController
    if params is None:
        params = Params()
    return HoverController(params)


__all__ = [
    "create_pid_position_controller",
    "create_pid_attitude_controller",
    "create_lqr_controller",
    "create_hover_controller"
]