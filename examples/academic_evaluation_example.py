#!/usr/bin/env python3
"""
Academic evaluation example demonstrating comprehensive logging and analysis.

This example shows how to use the academic logging and evaluation capabilities
for generating publication-quality results for research papers.
"""

import numpy as np
from quadcopter.dynamics import QuadState
from quadcopter import create_pid_position_controller
from quadcopter.logging import simulate_with_academic_logging
from quadcopter.evaluation import AcademicEvaluator

def run_academic_evaluation_demo():
    """Run a comprehensive academic evaluation demonstration."""
    print("Running Academic Evaluation Demonstration...")
    print("=" * 50)
    
    # Create a position controller using utility function
    position_controller = create_pid_position_controller(
        target_pos=[1.0, -1.0, 2.0],  # Target position: (1, -1, 2)
        kp=(2.0, 2.0, 4.0),
        ki=(0.1, 0.1, 0.2),
        kd=(0.5, 0.5, 1.0)
    )
    
    # Set up initial state (quadcopter at origin, at rest)
    initial_state = QuadState(
        pos=np.array([0.0, 0.0, 0.0]),
        vel=np.array([0.0, 0.0, 0.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        ang_vel=np.array([0.0, 0.0, 0.0])
    )
    
    # Define reference trajectory
    ref_position = np.array([1.0, -1.0, 2.0])
    ref_orientation = np.array([0.0, 0.0, 0.0])
    
    # Run simulation with academic logging
    print("Running simulation with academic logging...")
    log = simulate_with_academic_logging(
        duration=10.0,
        dt=0.02,
        controller=position_controller,
        initial_state=initial_state,
        ref_position=ref_position,
        ref_orientation=ref_orientation,
        controller_type="pid",
        method="rk4"
    )
    
    print(f"Simulation completed with {len(log.times)} data points")
    
    # Create academic evaluator
    evaluator = AcademicEvaluator(log)
    
    # Generate comprehensive analysis
    print("\nGenerating comprehensive academic analysis...")
    metrics = evaluator.generate_comprehensive_analysis("academic_results")
    
    # Print summary
    print("\nSummary Results:")
    print("-" * 20)
    print(f"Final Position: [{log.positions[-1][0]:.3f}, {log.positions[-1][1]:.3f}, {log.positions[-1][2]:.3f}]")
    print(f"Target Position: [{ref_position[0]:.3f}, {ref_position[1]:.3f}, {ref_position[2]:.3f}]")
    print(f"Position Error: {np.linalg.norm(np.array(log.positions[-1]) - ref_position):.4f} meters")
    
    # Print key metrics
    if 'z' in metrics and metrics['z']:
        print(f"Z-axis Rise Time: {metrics['z'].get('rise_time', 'N/A'):.4f} s")
        print(f"Z-axis Settling Time: {metrics['z'].get('settling_time', 'N/A'):.4f} s")
        print(f"Z-axis Peak Overshoot: {metrics['z'].get('peak_overshoot', 'N/A'):.2f} %")
        print(f"Z-axis Steady-State Error: {metrics['z'].get('steady_state_error', 'N/A'):.6f} m")
    
    print("\nAcademic evaluation demonstration completed!")
    print("Results saved to 'academic_results' directory.")

if __name__ == "__main__":
    run_academic_evaluation_demo()