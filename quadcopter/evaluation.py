"""
evaluation.py - Academic evaluation and visualization tools for quadcopter simulations.

This module provides comprehensive evaluation capabilities for academic research,
including performance metrics calculation, visualization generation, and
statistical analysis tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from .logging import AcademicLog

class AcademicEvaluator:
    """Academic evaluator for quadcopter control system performance analysis."""
    
    def __init__(self, log: AcademicLog):
        """Initialize evaluator with simulation log data."""
        self.log = log
        self.times = np.array(log.times)
        
    def plot_3d_trajectory(self, save_path: Optional[str] = None, show: bool = True):
        """Plot 3D trajectory with reference path."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Actual trajectory
        positions = np.array(self.log.positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2, label='Actual Trajectory')
        
        # Reference trajectory
        ref_positions = np.array(self.log.ref_positions)
        ax.plot(ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2], 
                'r--', linewidth=2, label='Reference Trajectory')
        
        # Start and end points
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  c='green', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  c='red', s=100, label='End')
        
        ax.set_xlabel('X Position [m]')
        ax.set_ylabel('Y Position [m]')
        ax.set_zlabel('Z Position [m]')
        ax.set_title('3D Flight Trajectory')
        ax.legend()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_state_tracking(self, save_path: Optional[str] = None, show: bool = True):
        """Plot state variables against reference values over time."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('State Tracking Performance', fontsize=16)
        
        positions = np.array(self.log.positions)
        orientations = np.array(self.log.orientations)
        ref_positions = np.array(self.log.ref_positions)
        ref_orientations = np.array(self.log.ref_orientations)
        
        # Position tracking
        axes[0, 0].plot(self.times, positions[:, 0], 'b-', label='Actual X')
        axes[0, 0].plot(self.times, ref_positions[:, 0], 'r--', label='Reference X')
        axes[0, 0].set_ylabel('X Position [m]')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.times, positions[:, 1], 'b-', label='Actual Y')
        axes[0, 1].plot(self.times, ref_positions[:, 1], 'r--', label='Reference Y')
        axes[0, 1].set_ylabel('Y Position [m]')
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.times, positions[:, 2], 'b-', label='Actual Z')
        axes[1, 0].plot(self.times, ref_positions[:, 2], 'r--', label='Reference Z')
        axes[1, 0].set_ylabel('Z Position [m]')
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Orientation tracking
        axes[1, 1].plot(self.times, orientations[:, 0], 'b-', label='Actual Roll')
        axes[1, 1].plot(self.times, ref_orientations[:, 0], 'r--', label='Reference Roll')
        axes[1, 1].set_ylabel('Roll [rad]')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[2, 0].plot(self.times, orientations[:, 1], 'b-', label='Actual Pitch')
        axes[2, 0].plot(self.times, ref_orientations[:, 1], 'r--', label='Reference Pitch')
        axes[2, 0].set_ylabel('Pitch [rad]')
        axes[2, 0].set_xlabel('Time [s]')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(self.times, orientations[:, 2], 'b-', label='Actual Yaw')
        axes[2, 1].plot(self.times, ref_orientations[:, 2], 'r--', label='Reference Yaw')
        axes[2, 1].set_ylabel('Yaw [rad]')
        axes[2, 1].set_xlabel('Time [s]')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_error_analysis(self, save_path: Optional[str] = None, show: bool = True):
        """Plot error signals over time."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Error Analysis', fontsize=16)
        
        position_errors = np.array(self.log.position_errors)
        orientation_errors = np.array(self.log.orientation_errors)
        
        # Position errors
        axes[0, 0].plot(self.times, position_errors[:, 0], 'b-', linewidth=2)
        axes[0, 0].set_ylabel('X Position Error [m]')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        axes[0, 1].plot(self.times, position_errors[:, 1], 'b-', linewidth=2)
        axes[0, 1].set_ylabel('Y Position Error [m]')
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        axes[1, 0].plot(self.times, position_errors[:, 2], 'b-', linewidth=2)
        axes[1, 0].set_ylabel('Z Position Error [m]')
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        # Orientation errors
        axes[1, 1].plot(self.times, orientation_errors[:, 0], 'b-', linewidth=2)
        axes[1, 1].set_ylabel('Roll Error [rad]')
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        axes[2, 0].plot(self.times, orientation_errors[:, 1], 'b-', linewidth=2)
        axes[2, 0].set_ylabel('Pitch Error [rad]')
        axes[2, 0].set_xlabel('Time [s]')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        axes[2, 1].plot(self.times, orientation_errors[:, 2], 'b-', linewidth=2)
        axes[2, 1].set_ylabel('Yaw Error [rad]')
        axes[2, 1].set_xlabel('Time [s]')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_control_effort(self, save_path: Optional[str] = None, show: bool = True):
        """Plot control inputs over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Control Effort Analysis', fontsize=16)
        
        motor_speeds = np.array(self.log.motor_speeds)
        torques = np.array(self.log.torques)
        thrusts = np.array(self.log.thrusts)
        
        # Motor speeds
        axes[0, 0].plot(self.times, motor_speeds[:, 0], label='Motor 1')
        axes[0, 0].plot(self.times, motor_speeds[:, 1], label='Motor 2')
        axes[0, 0].plot(self.times, motor_speeds[:, 2], label='Motor 3')
        axes[0, 0].plot(self.times, motor_speeds[:, 3], label='Motor 4')
        axes[0, 0].set_ylabel('Motor Speed [rad/s]')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Torques
        axes[0, 1].plot(self.times, torques[:, 0], label='Roll Torque')
        axes[0, 1].plot(self.times, torques[:, 1], label='Pitch Torque')
        axes[0, 1].plot(self.times, torques[:, 2], label='Yaw Torque')
        axes[0, 1].set_ylabel('Torque [N·m]')
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total thrust
        axes[1, 0].plot(self.times, thrusts, 'b-', linewidth=2)
        axes[1, 0].set_ylabel('Total Thrust [N]')
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Control effort distribution
        axes[1, 1].bar(['Motor 1', 'Motor 2', 'Motor 3', 'Motor 4'], 
                       [np.mean(motor_speeds[:, i]) for i in range(4)])
        axes[1, 1].set_ylabel('Mean Motor Speed [rad/s]')
        axes[1, 1].set_title('Average Control Effort Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_pid_contributions(self, save_path: Optional[str] = None, show: bool = True):
        """Plot PID term contributions (for PID controllers)."""
        if not self.log.pid_terms or len(self.log.pid_terms) == 0:
            print("No PID data available for plotting")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('PID Controller Term Contributions', fontsize=16)
        
        # Extract PID terms for each axis
        pid_terms = self.log.pid_terms
        
        # X-axis PID terms
        p_terms_x = [term['x']['p'] for term in pid_terms]
        i_terms_x = [term['x']['i'] for term in pid_terms]
        d_terms_x = [term['x']['d'] for term in pid_terms]
        
        axes[0].plot(self.times, p_terms_x, label='Proportional')
        axes[0].plot(self.times, i_terms_x, label='Integral')
        axes[0].plot(self.times, d_terms_x, label='Derivative')
        axes[0].set_ylabel('X-Axis Control Terms')
        axes[0].set_xlabel('Time [s]')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('X-Axis PID Contributions')
        
        # Y-axis PID terms
        p_terms_y = [term['y']['p'] for term in pid_terms]
        i_terms_y = [term['y']['i'] for term in pid_terms]
        d_terms_y = [term['y']['d'] for term in pid_terms]
        
        axes[1].plot(self.times, p_terms_y, label='Proportional')
        axes[1].plot(self.times, i_terms_y, label='Integral')
        axes[1].plot(self.times, d_terms_y, label='Derivative')
        axes[1].set_ylabel('Y-Axis Control Terms')
        axes[1].set_xlabel('Time [s]')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Y-Axis PID Contributions')
        
        # Z-axis PID terms
        p_terms_z = [term['z']['p'] for term in pid_terms]
        i_terms_z = [term['z']['i'] for term in pid_terms]
        d_terms_z = [term['z']['d'] for term in pid_terms]
        
        axes[2].plot(self.times, p_terms_z, label='Proportional')
        axes[2].plot(self.times, i_terms_z, label='Integral')
        axes[2].plot(self.times, d_terms_z, label='Derivative')
        axes[2].set_ylabel('Z-Axis Control Terms')
        axes[2].set_xlabel('Time [s]')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Z-Axis PID Contributions')
        
        fig.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def plot_rl_learning_curve(self, save_path: Optional[str] = None, show: bool = True):
        """Plot RL learning curve (cumulative reward per episode)."""
        if not self.log.rl_rewards or len(self.log.rl_rewards) == 0:
            print("No RL data available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Assuming data is grouped by episodes
        # For simplicity, we'll plot reward vs step (in a real implementation,
        # this would be grouped by episodes)
        rewards = np.array(self.log.rl_rewards)
        cumulative_rewards = np.cumsum(rewards)
        
        ax.plot(self.times, cumulative_rewards, 'b-', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('RL Learning Curve')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def generate_performance_report(self, save_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Generate comprehensive performance metrics report."""
        metrics = self.log.calculate_performance_metrics()
        
        # Format report
        report = "Academic Performance Metrics Report\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Simulation Details:\n"
        report += f"  Duration: {self.log.duration} seconds\n"
        report += f"  Time Step: {self.log.dt} seconds\n"
        report += f"  Method: {self.log.method}\n"
        report += f"  Controller Type: {self.log.controller_type}\n\n"
        
        report += "Performance Metrics by Axis:\n"
        report += "-" * 30 + "\n\n"
        
        for axis, axis_metrics in metrics.items():
            report += f"{axis.upper()} Axis:\n"
            for metric_name, value in axis_metrics.items():
                if metric_name in ['rise_time', 'settling_time']:
                    report += f"  {metric_name.replace('_', ' ').title()}: {value:.4f} s\n"
                elif metric_name == 'peak_overshoot':
                    report += f"  {metric_name.replace('_', ' ').title()}: {value:.2f} %\n"
                else:
                    report += f"  {metric_name.replace('_', ' ').title()}: {value:.6f}\n"
            report += "\n"
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
        
        print(report)
        return metrics
    
    def generate_comprehensive_analysis(self, output_dir: str = "analysis_results"):
        """Generate comprehensive academic analysis with all plots and metrics."""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        print("Generating comprehensive analysis...")
        
        self.plot_3d_trajectory(
            save_path=f"{output_dir}/3d_trajectory.png", 
            show=False
        )
        print("  - 3D trajectory plot saved")
        
        self.plot_state_tracking(
            save_path=f"{output_dir}/state_tracking.png", 
            show=False
        )
        print("  - State tracking plot saved")
        
        self.plot_error_analysis(
            save_path=f"{output_dir}/error_analysis.png", 
            show=False
        )
        print("  - Error analysis plot saved")
        
        self.plot_control_effort(
            save_path=f"{output_dir}/control_effort.png", 
            show=False
        )
        print("  - Control effort plot saved")
        
        # Generate controller-specific plots
        if self.log.controller_type == "pid":
            self.plot_pid_contributions(
                save_path=f"{output_dir}/pid_contributions.png", 
                show=False
            )
            print("  - PID contributions plot saved")
        
        if self.log.controller_type == "rl":
            self.plot_rl_learning_curve(
                save_path=f"{output_dir}/rl_learning_curve.png", 
                show=False
            )
            print("  - RL learning curve plot saved")
        
        # Generate performance report
        metrics = self.generate_performance_report(
            save_path=f"{output_dir}/performance_metrics.txt"
        )
        print("  - Performance metrics report saved")
        
        print(f"\nAnalysis complete! Results saved to '{output_dir}' directory.")
        return metrics