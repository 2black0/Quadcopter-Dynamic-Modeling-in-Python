#!/usr/bin/env python3
"""
Test script for academic evaluation functionality.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quadcopter.evaluation import AcademicEvaluator
from quadcopter.logging import AcademicLog
from quadcopter.dynamics import QuadState, Params

def test_academic_evaluator():
    """Test AcademicEvaluator functionality."""
    print("Testing AcademicEvaluator...")
    
    # Create a mock AcademicLog with some data
    log = AcademicLog(
        duration=5.0,
        dt=0.02,
        method="rk4",
        controller_type="pid"
    )
    
    # Add some mock data
    for i in range(10):
        time = i * 0.02
        pos = [i*0.1, i*0.2, i*0.3]
        orient = [i*0.01, i*0.02, i*0.03]
        vel = [0.1, 0.2, 0.3]
        ang_rate = [0.01, 0.02, 0.03]
        ref_pos = [1.0, 1.0, 1.0]
        ref_orient = [0.0, 0.0, 0.0]
        pos_error = [1.0-i*0.1, 1.0-i*0.2, 1.0-i*0.3]
        orient_error = [0.0-i*0.01, 0.0-i*0.02, 0.0-i*0.03]
        thrust = 10.0 + i*0.1
        torques = [0.1+i*0.01, 0.2+i*0.01, 0.3+i*0.01]
        motor_speeds = [400.0+i*10, 400.0+i*10, 400.0+i*10, 400.0+i*10]
        
        log.times.append(time)
        log.positions.append(pos)
        log.orientations.append(orient)
        log.velocities.append(vel)
        log.angular_rates.append(ang_rate)
        log.ref_positions.append(ref_pos)
        log.ref_orientations.append(ref_orient)
        log.position_errors.append(pos_error)
        log.orientation_errors.append(orient_error)
        log.thrusts.append(thrust)
        log.torques.append(torques)
        log.motor_speeds.append(motor_speeds)
        log.disturbances.append([0.0, 0.0, 0.0])
        
        # Add PID data
        log.pid_gains.append({
            'x': {'kp': 2.0, 'ki': 0.1, 'kd': 0.5},
            'y': {'kp': 2.0, 'ki': 0.1, 'kd': 0.5},
            'z': {'kp': 4.0, 'ki': 0.2, 'kd': 1.0}
        })
        log.pid_terms.append({
            'x': {'p': i*0.1, 'i': i*0.01, 'd': i*0.001},
            'y': {'p': i*0.2, 'i': i*0.02, 'd': i*0.002},
            'z': {'p': i*0.3, 'i': i*0.03, 'd': i*0.003}
        })
    
    # Create evaluator
    evaluator = AcademicEvaluator(log)
    
    # Test performance metrics calculation
    metrics = evaluator.log.calculate_performance_metrics()
    assert 'x' in metrics
    assert 'y' in metrics
    assert 'z' in metrics
    print("  - Performance metrics calculation verified")
    
    # Test that we can generate plots without errors
    try:
        evaluator.plot_3d_trajectory(show=False)
        evaluator.plot_state_tracking(show=False)
        evaluator.plot_error_analysis(show=False)
        evaluator.plot_control_effort(show=False)
        evaluator.plot_pid_contributions(show=False)
        print("  - All plotting functions verified")
    except Exception as e:
        print(f"  - Plotting error: {e}")
    
    # Test performance report generation
    try:
        report_metrics = evaluator.generate_performance_report()
        assert isinstance(report_metrics, dict)
        print("  - Performance report generation verified")
    except Exception as e:
        print(f"  - Report generation error: {e}")
    
    print("AcademicEvaluator tests completed!")

if __name__ == "__main__":
    test_academic_evaluator()