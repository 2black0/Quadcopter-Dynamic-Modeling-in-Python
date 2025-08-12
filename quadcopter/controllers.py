"""
controllers.py - Control systems for quadcopter dynamics.

This module implements various control systems that can be used with the quadcopter
dynamics model, including PID controllers, LQR controllers, and fuzzy logic controllers.

Classes:
    PIDController: Proportional-Integral-Derivative controller
    PositionController: PID-based position controller for x, y, z axes
    AttitudeController: PID-based attitude controller for roll, pitch, yaw
    LQRController: Linear Quadratic Regulator controller
    FuzzyLogicController: Fuzzy logic controller (template)
"""

from dataclasses import dataclass, field
from typing import Optional, cast
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are
from .dynamics import QuadState, Params
from .simulation import BaseController

@dataclass
class PIDController:
    """PID controller for quadcopter control systems.
    
    Implements a Proportional-Integral-Derivative controller with
    anti-windup mechanisms and output limits.
    """
    
    # PID gains
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    
    # Limits
    max_output: Optional[float] = None
    max_integral: Optional[float] = None
    
    # Internal state
    _integral: float = 0.0
    _prev_error: float = 0.0
    _prev_time: Optional[float] = None
    
    def update(self, error: float, dt: Optional[float] = None, time: Optional[float] = None) -> float:
        """Calculate PID output based on error.
        
        Parameters
        ----------
        error : float
            The error signal (setpoint - measured_value)
        dt : float, optional
            Time step. Required if time is not provided.
        time : float, optional
            Absolute time. If provided, dt is calculated from previous time.
            
        Returns
        -------
        float
            The controller output
        """
        # Determine time delta
        if time is not None and self._prev_time is not None:
            dt = time - self._prev_time
        elif dt is None:
            raise ValueError("Either dt or time must be provided")
            
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self._integral += error * dt
        if self.max_integral is not None:
            self._integral = np.clip(self._integral, -self.max_integral, self.max_integral)
        i_term = self.ki * self._integral
        
        # Derivative term
        if self._prev_time is not None and dt > 0:
            d_term = self.kd * (error - self._prev_error) / dt
        else:
            d_term = 0.0
            
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply limits
        if self.max_output is not None:
            output = np.clip(output, -self.max_output, self.max_output)
            
        # Update state
        self._prev_error = error
        self._prev_time = time if time is not None else (self._prev_time or 0) + dt
        
        return float(output)
    
    def reset(self) -> None:
        """Reset the controller's internal state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None


@dataclass
class PositionController(BaseController):
    """Position controller using PID for each axis."""
    
    # PID controllers for each axis
    x_pid: PIDController = field(default_factory=lambda: PIDController())
    y_pid: PIDController = field(default_factory=lambda: PIDController())
    z_pid: PIDController = field(default_factory=lambda: PIDController())
    
    # Target position
    target_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    
    def update(self, t: float, state: QuadState) -> NDArray[np.float64]:
        """Calculate motor speeds to reach target position."""
        # Calculate position errors
        pos_error = self.target_pos - state.pos
        
        # Get PID outputs for each axis (using dt=0.02 as a typical value)
        x_thrust = self.x_pid.update(pos_error[0], dt=0.02)
        y_thrust = self.y_pid.update(pos_error[1], dt=0.02)
        z_thrust = self.z_pid.update(pos_error[2], dt=0.02)
        
        # Convert to motor speeds (simplified control allocation)
        # This is a basic implementation - a full one would consider orientation and dynamics
        # Base hover thrust to counter gravity
        hover_thrust = 0.65 * 9.81  # mass * gravity
        
        # Total thrust needed (hover + z-position control)
        total_thrust = hover_thrust + z_thrust
        
        # Ensure thrust is positive
        total_thrust = max(total_thrust, 0.1)  # Minimum thrust to keep motors running
        
        # Convert thrust to motor speeds (assuming equal distribution)
        motor_speed_base = np.sqrt(total_thrust / (4 * 3.25e-5))
        
        # Add x,y control as differential thrust (simplified)
        # This is a very simplified model - real implementation would be more complex
        motor_speeds: NDArray[np.float64] = np.full(4, motor_speed_base)
        motor_speeds[0] += x_thrust * 0.1  # Front motors
        motor_speeds[1] -= x_thrust * 0.1  # Back motors
        motor_speeds[2] -= y_thrust * 0.1  # Left motors
        motor_speeds[3] += y_thrust * 0.1  # Right motors
        
        # Ensure all motor speeds are positive
        motor_speeds = np.maximum(motor_speeds, 10.0)  # Minimum motor speed
        
        return motor_speeds
    
    def reset(self) -> None:
        """Reset all PID controllers."""
        self.x_pid.reset()
        self.y_pid.reset()
        self.z_pid.reset()


@dataclass
class LQRController(BaseController):
    """Linear Quadratic Regulator controller for quadcopter."""
    
    params: Params = field(default_factory=Params)
    
    # State and input matrices (linearized around hover)
    A: np.ndarray = field(default_factory=lambda: np.zeros((12, 12)))  # State matrix
    B: np.ndarray = field(default_factory=lambda: np.zeros((12, 4)))   # Input matrix
    
    # Cost matrices
    Q: np.ndarray = field(default_factory=lambda: np.eye(12))  # State cost matrix
    R: np.ndarray = field(default_factory=lambda: np.eye(4))   # Input cost matrix
    
    # Feedback gain matrix
    K: np.ndarray = field(init=False)
    
    def __post_init__(self) -> None:
        """Compute the LQR gain matrix."""
        try:
            # Solve the algebraic Riccati equation
            P = solve_continuous_are(self.A, self.B, self.Q, self.R)
            # Compute the feedback gain matrix
            self.K = np.linalg.inv(self.R) @ self.B.T @ P
        except np.linalg.LinAlgError:
            # If the matrices are not suitable, use a simple proportional controller
            self.K = np.zeros((4, 12))
    
    def update(self, t: float, state: QuadState) -> NDArray[np.float64]:
        """Calculate control input using LQR."""
        # Linearize around hover point (simplified)
        # Extract relevant states (position, velocity, orientation, angular velocity)
        # For full 13-state vector, we exclude the quaternion constraint
        x = np.concatenate([
            state.pos,      # position (3)
            state.vel,      # velocity (3)
            state.quat[1:], # orientation (3) - using vector part of quaternion
            state.ang_vel   # angular velocity (3)
        ])
        
        # Reference state (hover at origin)
        x_ref = np.zeros(12)
        
        # Compute control input
        u: NDArray[np.float64]
        try:
            u = -self.K @ (x - x_ref)
        except Exception:
            # If LQR fails, use simple proportional control
            u = -0.1 * (x - x_ref)[:4]  # Only use position errors
        
        # Convert to motor speeds
        # This is a simplified mapping - a full implementation would be more complex
        motor_speed_base = np.sqrt(self.params.m * self.params.g / (4 * self.params.b))  # Hover speed
        motor_speeds = motor_speed_base + u.mean()  # Add control inputs
        
        # Ensure motor speeds are positive and reasonable
        motor_speeds = np.clip(motor_speeds, 100.0, 1000.0)
        
        return np.full(4, motor_speeds)


@dataclass
class AttitudeController(BaseController):
    """Attitude controller using PID for roll, pitch, yaw."""
    
    # PID controllers for angular rates
    roll_pid: PIDController = field(default_factory=lambda: PIDController())
    pitch_pid: PIDController = field(default_factory=lambda: PIDController())
    yaw_pid: PIDController = field(default_factory=lambda: PIDController())
    
    # Target attitude (roll, pitch, yaw)
    target_attitude: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    
    def update(self, t: float, state: QuadState) -> NDArray[np.float64]:
        """Calculate motor speeds to reach target attitude."""
        # Convert quaternion to Euler angles (simplified)
        # This is a simplified conversion - a full implementation would be more robust
        qw, qx, qy, qz = state.quat
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(2*(qw*qy - qz*qx))
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        
        # Calculate attitude errors
        attitude_error = self.target_attitude - np.array([roll, pitch, yaw])
        
        # Get PID outputs for each axis (using dt=0.02 as a typical value)
        roll_torque = self.roll_pid.update(attitude_error[0], dt=0.02)
        pitch_torque = self.pitch_pid.update(attitude_error[1], dt=0.02)
        yaw_torque = self.yaw_pid.update(attitude_error[2], dt=0.02)
        
        # Convert torques to motor speeds (simplified)
        # This is a placeholder - real implementation would properly allocate torques
        motor_speed_base = np.sqrt(0.65 * 9.81 / (4 * 3.25e-5))  # Hover speed
        motor_speeds = motor_speed_base + np.array([roll_torque, pitch_torque, yaw_torque, 0.0])
        
        # Ensure motor speeds are positive
        motor_speeds = np.maximum(motor_speeds, 0.0)
        
        return cast(NDArray[np.float64], motor_speeds)
    
    def reset(self) -> None:
        """Reset all PID controllers."""
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.yaw_pid.reset()


@dataclass
class FuzzyLogicController(BaseController):
    """Fuzzy logic controller template.
    
    This is a placeholder for a fuzzy logic controller implementation.
    A full implementation would require a fuzzy logic library like scikit-fuzzy.
    """
    
    def update(self, t: float, state: QuadState) -> NDArray[np.float64]:
        """Calculate motor speeds using fuzzy logic control."""
        # Placeholder implementation
        # A real implementation would:
        # 1. Define fuzzy sets for inputs (position error, velocity, etc.)
        # 2. Define fuzzy rules
        # 3. Perform fuzzification, inference, and defuzzification
        
        # For now, just return hover motor speeds
        hover_speed = np.sqrt(0.65 * 9.81 / (4 * 3.25e-5))
        return np.full(4, hover_speed)
    
    def reset(self) -> None:
        """Reset the controller's internal state."""
        pass

__all__ = [
    "PIDController",
    "PositionController",
    "AttitudeController",
    "LQRController",
    "FuzzyLogicController"
]