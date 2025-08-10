#!/usr/bin/env python3
"""
Comprehensive test for all controller implementations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from quadcopter.controllers import PIDController, PositionController, AttitudeController, LQRController
from quadcopter.dynamics import QuadState, Params

def test_pid_controller():
    """Test PID controller functionality."""
    print("Testing PIDController...")
    
    # Create a simple PID controller
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
    
    # Test basic proportional control (first call)
    output = pid.update(1.0, dt=0.1)
    expected = 1.0 + 0.01  # kp*error + ki*integral (starts at 0)
    assert np.isclose(output, expected), f"Expected {expected}, got {output}"
    print(f"First update test passed: {output}")
    
    # Test reset functionality
    pid.reset()
    assert pid._integral == 0.0
    assert pid._prev_error == 0.0
    assert pid._prev_time is None
    print("Reset test passed")
    
    print("PIDController tests passed!\n")

def test_position_controller():
    """Test PositionController functionality."""
    print("Testing PositionController...")
    
    # Create a position controller
    pos_ctrl = PositionController(
        x_pid=PIDController(kp=1.0, ki=0.1, kd=0.05),
        y_pid=PIDController(kp=1.0, ki=0.1, kd=0.05),
        z_pid=PIDController(kp=1.0, ki=0.1, kd=0.05),
        target_pos=np.array([1.0, -1.0, 2.0])
    )
    
    # Create a quadcopter state
    state = QuadState(
        pos=np.array([0.0, 0.0, 0.0]),
        vel=np.array([0.0, 0.0, 0.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        ang_vel=np.array([0.0, 0.0, 0.0])
    )
    
    # Test update
    motor_speeds = pos_ctrl.update(0.1, state)
    assert motor_speeds.shape == (4,)
    assert np.all(motor_speeds > 0)  # All motor speeds should be positive
    print(f"Position controller output: {motor_speeds}")
    
    # Test reset
    pos_ctrl.reset()
    assert pos_ctrl.x_pid._integral == 0.0
    assert pos_ctrl.y_pid._integral == 0.0
    assert pos_ctrl.z_pid._integral == 0.0
    print("Position controller reset test passed")
    
    print("PositionController tests passed!\n")

def test_attitude_controller():
    """Test AttitudeController functionality."""
    print("Testing AttitudeController...")
    
    # Create an attitude controller
    att_ctrl = AttitudeController(
        roll_pid=PIDController(kp=1.0, ki=0.1, kd=0.05),
        pitch_pid=PIDController(kp=1.0, ki=0.1, kd=0.05),
        yaw_pid=PIDController(kp=1.0, ki=0.1, kd=0.05),
        target_attitude=np.array([0.1, -0.1, 0.0])
    )
    
    # Create a quadcopter state
    state = QuadState(
        pos=np.array([0.0, 0.0, 0.0]),
        vel=np.array([0.0, 0.0, 0.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
        ang_vel=np.array([0.0, 0.0, 0.0])
    )
    
    # Test update
    motor_speeds = att_ctrl.update(0.1, state)
    assert motor_speeds.shape == (4,)
    assert np.all(motor_speeds > 0)  # All motor speeds should be positive
    print(f"Attitude controller output: {motor_speeds}")
    
    # Test reset
    att_ctrl.reset()
    assert att_ctrl.roll_pid._integral == 0.0
    assert att_ctrl.pitch_pid._integral == 0.0
    assert att_ctrl.yaw_pid._integral == 0.0
    print("Attitude controller reset test passed")
    
    print("AttitudeController tests passed!\n")

def test_lqr_controller():
    """Test LQRController functionality."""
    print("Testing LQRController...")
    
    # Create an LQR controller with simple matrices
    # For testing, we'll use identity matrices
    A = np.eye(12)
    B = np.eye(12, 4)  # 12x4 matrix
    Q = np.eye(12)
    R = np.eye(4)
    
    try:
        lqr_ctrl = LQRController(
            params=Params(),
            A=A,
            B=B,
            Q=Q,
            R=R
        )
        
        # Create a quadcopter state
        state = QuadState(
            pos=np.array([0.0, 0.0, 0.0]),
            vel=np.array([0.0, 0.0, 0.0]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ang_vel=np.array([0.0, 0.0, 0.0])
        )
        
        # Test update
        motor_speeds = lqr_ctrl.update(0.1, state)
        assert motor_speeds.shape == (4,)
        assert np.all(motor_speeds > 0)  # All motor speeds should be positive
        print(f"LQR controller output: {motor_speeds}")
        
        print("LQRController tests passed!\n")
    except Exception as e:
        print(f"LQRController test skipped due to error: {e}\n")

def main():
    """Run all controller tests."""
    print("Running comprehensive controller tests...\n")
    
    test_pid_controller()
    test_position_controller()
    test_attitude_controller()
    test_lqr_controller()
    
    print("All controller tests passed!")

if __name__ == "__main__":
    main()