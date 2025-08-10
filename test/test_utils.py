"""
Test utility functions for quadcopter package.
"""

import numpy as np
from quadcopter import (
    create_pid_position_controller, 
    create_pid_attitude_controller,
    create_lqr_controller,
    create_hover_controller
)
from quadcopter.dynamics import QuadState, Params


def test_create_pid_position_controller():
    """Test creating a PID position controller."""
    target_pos = [1.0, -1.0, 2.0]
    controller = create_pid_position_controller(target_pos)
    
    assert np.allclose(controller.target_pos, target_pos)
    # Check that default gains are set
    assert controller.x_pid.kp == 2.0
    assert controller.y_pid.ki == 0.1
    assert controller.z_pid.kd == 1.0
    print("✓ PID position controller creation test passed")


def test_create_pid_position_controller_custom_gains():
    """Test creating a PID position controller with custom gains."""
    target_pos = [1.0, -1.0, 2.0]
    kp = (3.0, 3.0, 5.0)
    ki = (0.2, 0.2, 0.3)
    kd = (0.6, 0.6, 1.2)
    
    controller = create_pid_position_controller(target_pos, kp, ki, kd)
    
    assert np.allclose(controller.target_pos, target_pos)
    assert controller.x_pid.kp == kp[0]
    assert controller.y_pid.ki == ki[1]
    assert controller.z_pid.kd == kd[2]
    print("✓ PID position controller with custom gains test passed")


def test_create_pid_attitude_controller():
    """Test creating a PID attitude controller."""
    target_attitude = [0.1, -0.1, 0.0]
    controller = create_pid_attitude_controller(target_attitude)
    
    assert np.allclose(controller.target_attitude, target_attitude)
    # Check that default gains are set
    assert controller.roll_pid.kp == 1.0
    assert controller.pitch_pid.ki == 0.01
    assert controller.yaw_pid.kd == 0.1
    print("✓ PID attitude controller creation test passed")


def test_create_lqr_controller():
    """Test creating an LQR controller."""
    controller = create_lqr_controller()
    
    # Check that matrices have the right shape
    assert controller.A.shape == (12, 12)
    assert controller.B.shape == (12, 4)
    assert controller.Q.shape == (12, 12)
    assert controller.R.shape == (4, 4)
    print("✓ LQR controller creation test passed")


def test_create_hover_controller():
    """Test creating a hover controller."""
    params = Params()
    controller = create_hover_controller(params)
    
    # Check that hover speed is calculated correctly
    expected_hover_speed = np.sqrt(params.m * params.g / (4.0 * params.b))
    assert controller.command[0] == expected_hover_speed
    print("✓ Hover controller creation test passed")


def test_controller_update_methods():
    """Test that all controllers have update methods."""
    # Create a state for testing
    state = QuadState(
        pos=np.array([0.0, 0.0, 0.0]),
        vel=np.array([0.0, 0.0, 0.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        ang_vel=np.array([0.0, 0.0, 0.0])
    )
    
    # Test PID position controller
    pid_controller = create_pid_position_controller([1.0, 1.0, 1.0])
    motor_speeds = pid_controller.update(0.0, state)
    assert len(motor_speeds) == 4
    
    # Test PID attitude controller
    attitude_controller = create_pid_attitude_controller([0.0, 0.0, 0.0])
    motor_speeds = attitude_controller.update(0.0, state)
    assert len(motor_speeds) == 4
    
    # Test LQR controller
    lqr_controller = create_lqr_controller()
    motor_speeds = lqr_controller.update(0.0, state)
    assert len(motor_speeds) == 4
    
    # Test hover controller
    hover_controller = create_hover_controller()
    motor_speeds = hover_controller.update(0.0, state)
    assert len(motor_speeds) == 4
    print("✓ All controller update methods test passed")


if __name__ == "__main__":
    test_create_pid_position_controller()
    test_create_pid_position_controller_custom_gains()
    test_create_pid_attitude_controller()
    test_create_lqr_controller()
    test_create_hover_controller()
    test_controller_update_methods()
    print("All tests passed! 🎉")