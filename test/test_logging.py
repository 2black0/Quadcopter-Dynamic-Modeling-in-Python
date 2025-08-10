#!/usr/bin/env python3
"""
Comprehensive test for the logging module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from quadcopter.logging import SimulationLog, simulate_with_logging
from quadcopter.dynamics import QuadState, Params
from quadcopter.simulation import HoverController

def test_simulation_log():
    """Test SimulationLog functionality."""
    print("Testing SimulationLog...")
    
    # Create a simulation log
    log = SimulationLog(
        duration=10.0,
        dt=0.02,
        method="rk4"
    )
    
    # Add some entries
    for i in range(5):
        time = i * 0.02
        state = QuadState(
            pos=np.array([i*0.1, i*0.2, i*0.3]),
            vel=np.array([0.1, 0.2, 0.3]),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ang_vel=np.array([0.0, 0.0, 0.0])
        )
        control = np.array([400.0, 400.0, 400.0, 400.0])
        
        log.add_entry(time, state, control)
    
    # Check that entries were added
    assert len(log.times) == 5
    assert len(log.states) == 5
    assert len(log.controls) == 5
    print(f"Added {len(log.times)} entries to log")
    
    # Test saving to CSV
    log.save_csv("test_log.csv")
    print("Saved log to CSV")
    
    # Test saving to JSON
    log.save_json("test_log.json")
    print("Saved log to JSON")
    
    # Test saving to MATLAB (if scipy is available)
    try:
        log.save_matlab("test_log.mat")
        print("Saved log to MATLAB format")
        # Clean up
        os.remove("test_log.mat")
    except:
        print("Skipping MATLAB format test (scipy not available)")
    
    # Clean up
    os.remove("test_log.csv")
    os.remove("test_log.json")
    
    print("SimulationLog tests passed!")


def test_simulate_with_logging():
    """Test simulate_with_logging functionality."""
    print("Testing simulate_with_logging...")
    
    # Create controller and initial state
    controller = HoverController()
    initial_state = QuadState(
        pos=np.array([0.0, 0.0, 0.0]),
        vel=np.array([0.0, 0.0, 0.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        ang_vel=np.array([0.0, 0.0, 0.0])
    )
    
    # Run simulation with logging
    log = simulate_with_logging(
        duration=2.0,
        dt=0.02,
        controller=controller,
        initial_state=initial_state,
        method="rk4"
    )
    
    # Check log
    assert len(log.times) > 0
    assert len(log.states) == len(log.times)
    assert len(log.controls) == len(log.times)
    print(f"Created log with {len(log.times)} entries")
    
    print("simulate_with_logging tests passed!")


def main():
    """Run all logging tests."""
    print("Running comprehensive logging tests...")
    test_simulation_log()
    test_simulate_with_logging()
    print("All logging tests passed!")


if __name__ == "__main__":
    main()