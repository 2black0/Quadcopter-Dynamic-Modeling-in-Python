"""
logging.py - Enhanced logging capabilities for quadcopter simulations.

This module implements comprehensive logging functionality for quadcopter simulations,
enabling detailed data collection for analysis and academic reporting.

Classes:
    SimulationLog: Comprehensive log of simulation data (backward compatibility)
    AcademicLog: Enhanced logging for academic research with controller-specific data
"""

import json
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from datetime import datetime
from .dynamics import QuadState, Params
from .simulation import simulate, BaseController

@dataclass
class SimulationLog:
    """Comprehensive log of simulation data for analysis and reporting (backward compatibility)."""
    
    # Metadata
    timestamp: str = ""
    duration: float = 0.0
    dt: float = 0.02
    method: str = "rk4"
    
    # Simulation data
    times: List[float] = None
    states: List[Dict[str, Any]] = None
    controls: List[List[float]] = None
    rewards: Optional[List[float]] = None  # For RL applications
    
    def __post_init__(self):
        if self.times is None:
            self.times = []
        if self.states is None:
            self.states = []
        if self.controls is None:
            self.controls = []
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def add_entry(self, time: float, state: QuadState, control: np.ndarray, reward: Optional[float] = None):
        """Add a time step entry to the log."""
        self.times.append(time)
        self.states.append({
            "position": state.pos.tolist(),
            "velocity": state.vel.tolist(),
            "quaternion": state.quat.tolist(),
            "angular_velocity": state.ang_vel.tolist()
        })
        self.controls.append(control.tolist())
        if reward is not None and self.rewards is not None:
            self.rewards.append(reward)
    
    def save_json(self, filepath: str):
        """Save log to JSON format."""
        data = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_csv(self, filepath: str):
        """Save log to CSV format for analysis."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = [
                'time', 
                'pos_x', 'pos_y', 'pos_z',
                'vel_x', 'vel_y', 'vel_z',
                'quat_w', 'quat_x', 'quat_y', 'quat_z',
                'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
                'control_1', 'control_2', 'control_3', 'control_4'
            ]
            
            if self.rewards is not None:
                header.append('reward')
                
            writer.writerow(header)
            
            # Data rows
            for i in range(len(self.times)):
                row = [
                    self.times[i],
                    *self.states[i]['position'],
                    *self.states[i]['velocity'],
                    *self.states[i]['quaternion'],
                    *self.states[i]['angular_velocity'],
                    *self.controls[i]
                ]
                
                if self.rewards is not None and i < len(self.rewards):
                    row.append(self.rewards[i] if i < len(self.rewards) else 0.0)
                    
                writer.writerow(row)
    
    def save_matlab(self, filepath: str):
        """Save log to MATLAB .mat format."""
        try:
            from scipy.io import savemat
            
            # Convert to arrays
            data = {
                'time': np.array(self.times),
                'position': np.array([s['position'] for s in self.states]),
                'velocity': np.array([s['velocity'] for s in self.states]),
                'quaternion': np.array([s['quaternion'] for s in self.states]),
                'angular_velocity': np.array([s['angular_velocity'] for s in self.states]),
                'control': np.array(self.controls)
            }
            
            if self.rewards is not None:
                data['reward'] = np.array(self.rewards)
                
            savemat(filepath, data)
        except ImportError:
            print("scipy not available, cannot save MATLAB format")


def simulate_with_logging(
    duration: float,
    dt: float,
    controller: BaseController,
    initial_state: QuadState | None = None,
    params: Params = Params(),
    *,
    method: str = "rk4",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step: float | None = None,
) -> SimulationLog:
    """Wrapper around simulate that returns a SimulationLog."""
    
    # Run simulation
    t, states, controls = simulate(
        duration=duration,
        dt=dt,
        controller=controller,
        initial_state=initial_state,
        params=params,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step
    )
    
    # Create log
    log = SimulationLog(
        duration=duration,
        dt=dt,
        method=method
    )
    
    # Populate log
    for i in range(len(t)):
        state = QuadState.from_vector(states[i])
        log.add_entry(t[i], state, controls[i])
    
    return log


@dataclass
class AcademicLog:
    """Comprehensive academic-grade log for quadcopter simulations.
    
    This logger captures all data required for academic research and journal publications,
    including state variables, reference trajectories, error signals, control inputs,
    and controller-specific internal variables.
    """
    
    # Metadata
    timestamp: str = ""
    duration: float = 0.0
    dt: float = 0.02
    method: str = "rk4"
    controller_type: str = "unknown"
    
    # Core simulation data (applicable to all controllers)
    times: List[float] = None
    
    # State variables
    positions: List[List[float]] = None          # x, y, z
    orientations: List[List[float]] = None       # roll, pitch, yaw (Euler angles)
    velocities: List[List[float]] = None         # vx, vy, vz (body frame)
    angular_rates: List[List[float]] = None      # p, q, r (body frame)
    
    # Reference trajectories (setpoints)
    ref_positions: List[List[float]] = None      # x_ref, y_ref, z_ref
    ref_orientations: List[List[float]] = None   # phi_ref, theta_ref, psi_ref
    
    # Error signals
    position_errors: List[List[float]] = None    # e_x, e_y, e_z
    orientation_errors: List[List[float]] = None # e_phi, e_theta, e_psi
    
    # Control inputs
    thrusts: List[float] = None                  # Total thrust U1
    torques: List[List[float]] = None            # tau_phi, tau_theta, tau_psi
    motor_speeds: List[List[float]] = None       # omega_1, omega_2, omega_3, omega_4
    
    # External disturbances (if any)
    disturbances: List[List[float]] = None       # Force disturbances in x, y, z
    
    # Controller-specific data
    # PID Controller data
    pid_gains: List[Dict[str, Dict[str, float]]] = None  # Kp, Ki, Kd for each axis
    pid_terms: List[Dict[str, Dict[str, float]]] = None  # P, I, D term contributions
    
    # LQR Controller data
    lqr_cost_matrices: Optional[Dict[str, List[List[float]]]] = None  # Q, R matrices
    lqr_gain_matrix: Optional[List[List[float]]] = None               # K matrix
    lqr_instantaneous_costs: List[float] = None                       # x^T Q x + u^T R u
    
    # RL Controller data
    rl_rewards: List[float] = None                                    # Reward at each step
    rl_actions: List[List[float]] = None                              # Actions taken
    rl_states: List[List[float]] = None                               # States observed
    rl_next_states: List[List[float]] = None                          # Next states
    rl_policy_probabilities: List[List[float]] = None                 # Action probabilities
    rl_value_estimates: List[float] = None                            # Value function estimates
    rl_advantage_estimates: List[float] = None                        # Advantage estimates
    rl_training_losses: List[Dict[str, float]] = None                 # Policy, value, entropy losses
    
    def __post_init__(self):
        # Initialize all lists
        if self.times is None:
            self.times = []
        if self.positions is None:
            self.positions = []
        if self.orientations is None:
            self.orientations = []
        if self.velocities is None:
            self.velocities = []
        if self.angular_rates is None:
            self.angular_rates = []
        if self.ref_positions is None:
            self.ref_positions = []
        if self.ref_orientations is None:
            self.ref_orientations = []
        if self.position_errors is None:
            self.position_errors = []
        if self.orientation_errors is None:
            self.orientation_errors = []
        if self.thrusts is None:
            self.thrusts = []
        if self.torques is None:
            self.torques = []
        if self.motor_speeds is None:
            self.motor_speeds = []
        if self.disturbances is None:
            self.disturbances = []
        if self.pid_gains is None:
            self.pid_gains = []
        if self.pid_terms is None:
            self.pid_terms = []
        if self.lqr_instantaneous_costs is None:
            self.lqr_instantaneous_costs = []
        if self.rl_rewards is None:
            self.rl_rewards = []
        if self.rl_actions is None:
            self.rl_actions = []
        if self.rl_states is None:
            self.rl_states = []
        if self.rl_next_states is None:
            self.rl_next_states = []
        if self.rl_policy_probabilities is None:
            self.rl_policy_probabilities = []
        if self.rl_value_estimates is None:
            self.rl_value_estimates = []
        if self.rl_advantage_estimates is None:
            self.rl_advantage_estimates = []
        if self.rl_training_losses is None:
            self.rl_training_losses = []
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def add_entry(self, 
                  time: float,
                  state: QuadState,
                  params: Params,
                  ref_position: Optional[np.ndarray] = None,
                  ref_orientation: Optional[np.ndarray] = None,
                  thrust: Optional[float] = None,
                  torques: Optional[np.ndarray] = None,
                  motor_speeds: Optional[np.ndarray] = None,
                  reward: Optional[float] = None,
                  pid_data: Optional[Dict] = None,
                  lqr_data: Optional[Dict] = None,
                  rl_data: Optional[Dict] = None):
        """Add a comprehensive time step entry to the log."""
        self.times.append(time)
        
        # Convert quaternion to Euler angles
        qw, qx, qy, qz = state.quat
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(2*(qw*qy - qz*qx))
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        
        # State variables
        self.positions.append(state.pos.tolist())
        self.orientations.append([roll, pitch, yaw])
        self.velocities.append(state.vel.tolist())
        self.angular_rates.append(state.ang_vel.tolist())
        
        # Reference trajectories
        if ref_position is not None:
            self.ref_positions.append(ref_position.tolist())
        else:
            self.ref_positions.append([0.0, 0.0, 0.0])
            
        if ref_orientation is not None:
            self.ref_orientations.append(ref_orientation.tolist())
        else:
            self.ref_orientations.append([0.0, 0.0, 0.0])
        
        # Error signals
        pos_error = np.array(self.ref_positions[-1]) - state.pos
        orient_error = np.array(self.ref_orientations[-1]) - np.array([roll, pitch, yaw])
        self.position_errors.append(pos_error.tolist())
        self.orientation_errors.append(orient_error.tolist())
        
        # Control inputs
        if thrust is not None:
            self.thrusts.append(thrust)
        else:
            self.thrusts.append(0.0)
            
        if torques is not None:
            self.torques.append(torques.tolist())
        else:
            self.torques.append([0.0, 0.0, 0.0])
            
        if motor_speeds is not None:
            self.motor_speeds.append(motor_speeds.tolist())
        else:
            self.motor_speeds.append([0.0, 0.0, 0.0, 0.0])
        
        # Disturbances (placeholder)
        self.disturbances.append([0.0, 0.0, 0.0])
        
        # Controller-specific data
        if pid_data is not None:
            self.pid_gains.append(pid_data.get('gains', {'x': {'kp': 0, 'ki': 0, 'kd': 0}, 
                                                        'y': {'kp': 0, 'ki': 0, 'kd': 0}, 
                                                        'z': {'kp': 0, 'ki': 0, 'kd': 0}}))
            self.pid_terms.append(pid_data.get('terms', {'x': {'p': 0, 'i': 0, 'd': 0}, 
                                                        'y': {'p': 0, 'i': 0, 'd': 0}, 
                                                        'z': {'p': 0, 'i': 0, 'd': 0}}))
        else:
            self.pid_gains.append({'x': {'kp': 0, 'ki': 0, 'kd': 0}, 
                                  'y': {'kp': 0, 'ki': 0, 'kd': 0}, 
                                  'z': {'kp': 0, 'ki': 0, 'kd': 0}})
            self.pid_terms.append({'x': {'p': 0, 'i': 0, 'd': 0}, 
                                  'y': {'p': 0, 'i': 0, 'd': 0}, 
                                  'z': {'p': 0, 'i': 0, 'd': 0}})
        
        if lqr_data is not None:
            cost = lqr_data.get('instantaneous_cost', 0.0)
            self.lqr_instantaneous_costs.append(cost)
        else:
            self.lqr_instantaneous_costs.append(0.0)
        
        # RL data
        if reward is not None:
            self.rl_rewards.append(reward)
        else:
            self.rl_rewards.append(0.0)
            
        if rl_data is not None:
            self.rl_actions.append(rl_data.get('action', [0.0, 0.0, 0.0, 0.0]))
            self.rl_states.append(rl_data.get('state', [0.0] * 13))
            self.rl_next_states.append(rl_data.get('next_state', [0.0] * 13))
            self.rl_policy_probabilities.append(rl_data.get('policy_prob', [0.25, 0.25, 0.25, 0.25]))
            self.rl_value_estimates.append(rl_data.get('value_estimate', 0.0))
            self.rl_advantage_estimates.append(rl_data.get('advantage_estimate', 0.0))
            self.rl_training_losses.append(rl_data.get('losses', {'policy': 0.0, 'value': 0.0, 'entropy': 0.0}))
        else:
            self.rl_actions.append([0.0, 0.0, 0.0, 0.0])
            self.rl_states.append([0.0] * 13)
            self.rl_next_states.append([0.0] * 13)
            self.rl_policy_probabilities.append([0.25, 0.25, 0.25, 0.25])
            self.rl_value_estimates.append(0.0)
            self.rl_advantage_estimates.append(0.0)
            self.rl_training_losses.append({'policy': 0.0, 'value': 0.0, 'entropy': 0.0})
    
    def save_json(self, filepath: str):
        """Save log to JSON format."""
        data = asdict(self)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_csv(self, filepath: str):
        """Save log to CSV format for analysis."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header - comprehensive academic format
            header = [
                'time',
                # State variables
                'pos_x', 'pos_y', 'pos_z',
                'roll', 'pitch', 'yaw',
                'vel_x', 'vel_y', 'vel_z',
                'ang_rate_p', 'ang_rate_q', 'ang_rate_r',
                # Reference trajectories
                'ref_pos_x', 'ref_pos_y', 'ref_pos_z',
                'ref_roll', 'ref_pitch', 'ref_yaw',
                # Error signals
                'error_pos_x', 'error_pos_y', 'error_pos_z',
                'error_roll', 'error_pitch', 'error_yaw',
                # Control inputs
                'thrust', 'torque_roll', 'torque_pitch', 'torque_yaw',
                'motor_1', 'motor_2', 'motor_3', 'motor_4',
                # Disturbances
                'disturbance_x', 'disturbance_y', 'disturbance_z'
            ]
            
            # Add controller-specific headers if needed
            if self.controller_type == "pid":
                header.extend([
                    'pid_x_kp', 'pid_x_ki', 'pid_x_kd',
                    'pid_x_p_term', 'pid_x_i_term', 'pid_x_d_term',
                    'pid_y_kp', 'pid_y_ki', 'pid_y_kd',
                    'pid_y_p_term', 'pid_y_i_term', 'pid_y_d_term',
                    'pid_z_kp', 'pid_z_ki', 'pid_z_kd',
                    'pid_z_p_term', 'pid_z_i_term', 'pid_z_d_term'
                ])
            
            if self.controller_type == "lqr":
                header.append('lqr_instantaneous_cost')
            
            if self.controller_type == "rl":
                header.extend([
                    'rl_reward', 'rl_value_estimate', 'rl_advantage_estimate',
                    'rl_action_1', 'rl_action_2', 'rl_action_3', 'rl_action_4'
                ])
                
            writer.writerow(header)
            
            # Data rows
            for i in range(len(self.times)):
                row = [
                    self.times[i],
                    # State variables
                    *self.positions[i],
                    *self.orientations[i],
                    *self.velocities[i],
                    *self.angular_rates[i],
                    # Reference trajectories
                    *self.ref_positions[i],
                    *self.ref_orientations[i],
                    # Error signals
                    *self.position_errors[i],
                    *self.orientation_errors[i],
                    # Control inputs
                    self.thrusts[i],
                    *self.torques[i],
                    *self.motor_speeds[i],
                    # Disturbances
                    *self.disturbances[i]
                ]
                
                # Add controller-specific data if needed
                if self.controller_type == "pid" and i < len(self.pid_gains):
                    pid_gain = self.pid_gains[i]
                    pid_term = self.pid_terms[i]
                    row.extend([
                        pid_gain['x']['kp'], pid_gain['x']['ki'], pid_gain['x']['kd'],
                        pid_term['x']['p'], pid_term['x']['i'], pid_term['x']['d'],
                        pid_gain['y']['kp'], pid_gain['y']['ki'], pid_gain['y']['kd'],
                        pid_term['y']['p'], pid_term['y']['i'], pid_term['y']['d'],
                        pid_gain['z']['kp'], pid_gain['z']['ki'], pid_gain['z']['kd'],
                        pid_term['z']['p'], pid_term['z']['i'], pid_term['z']['d']
                    ])
                
                if self.controller_type == "lqr" and i < len(self.lqr_instantaneous_costs):
                    row.append(self.lqr_instantaneous_costs[i])
                
                if self.controller_type == "rl" and i < len(self.rl_rewards):
                    row.extend([
                        self.rl_rewards[i],
                        self.rl_value_estimates[i],
                        self.rl_advantage_estimates[i],
                        *self.rl_actions[i]
                    ])
                    
                writer.writerow(row)
    
    def save_matlab(self, filepath: str):
        """Save log to MATLAB .mat format."""
        try:
            from scipy.io import savemat
            
            # Convert to arrays for MATLAB
            data = {
                'time': np.array(self.times),
                'position': np.array(self.positions),
                'orientation': np.array(self.orientations),
                'velocity': np.array(self.velocities),
                'angular_rate': np.array(self.angular_rates),
                'ref_position': np.array(self.ref_positions),
                'ref_orientation': np.array(self.ref_orientations),
                'position_error': np.array(self.position_errors),
                'orientation_error': np.array(self.orientation_errors),
                'thrust': np.array(self.thrusts),
                'torque': np.array(self.torques),
                'motor_speed': np.array(self.motor_speeds),
                'disturbance': np.array(self.disturbances)
            }
            
            # Add controller-specific data
            if self.controller_type == "pid":
                # Extract PID data into arrays
                pid_kp_x = np.array([g['x']['kp'] for g in self.pid_gains])
                pid_ki_x = np.array([g['x']['ki'] for g in self.pid_gains])
                pid_kd_x = np.array([g['x']['kd'] for g in self.pid_gains])
                pid_p_x = np.array([t['x']['p'] for t in self.pid_terms])
                pid_i_x = np.array([t['x']['i'] for t in self.pid_terms])
                pid_d_x = np.array([t['x']['d'] for t in self.pid_terms])
                
                data.update({
                    'pid_kp_x': pid_kp_x,
                    'pid_ki_x': pid_ki_x,
                    'pid_kd_x': pid_kd_x,
                    'pid_p_x': pid_p_x,
                    'pid_i_x': pid_i_x,
                    'pid_d_x': pid_d_x
                })
            
            if self.controller_type == "lqr":
                data['lqr_instantaneous_cost'] = np.array(self.lqr_instantaneous_costs)
            
            if self.controller_type == "rl":
                data.update({
                    'rl_reward': np.array(self.rl_rewards),
                    'rl_action': np.array(self.rl_actions),
                    'rl_value_estimate': np.array(self.rl_value_estimates),
                    'rl_advantage_estimate': np.array(self.rl_advantage_estimates)
                })
                
            savemat(filepath, data)
        except ImportError:
            print("scipy not available, cannot save MATLAB format")
    
    def calculate_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for academic analysis.
        
        Returns performance metrics for each axis including rise time, settling time,
        peak overshoot, steady-state error, IAE, ISE, ITAE, and control effort.
        """
        if len(self.times) == 0:
            return {}
        
        metrics = {}
        dt = self.times[1] - self.times[0] if len(self.times) > 1 else self.dt
        
        # Calculate metrics for each axis
        axes = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        for i, axis in enumerate(axes):
            # Determine which data to use based on axis
            if axis in ['x', 'y', 'z']:
                actual = np.array(self.positions)[:, i]
                reference = np.array(self.ref_positions)[:, i]
                error = np.array(self.position_errors)[:, i]
            else:
                actual = np.array(self.orientations)[:, i-3]
                reference = np.array(self.ref_orientations)[:, i-3]
                error = np.array(self.orientation_errors)[:, i-3]
            
            # Find final reference value (assuming step input)
            final_ref = reference[-1] if len(reference) > 0 else 0.0
            
            # Calculate metrics
            axis_metrics = {}
            
            if len(actual) > 0 and final_ref != 0:
                # Rise time (10% to 90% of final value)
                if final_ref != 0:
                    ten_percent = 0.1 * final_ref
                    ninety_percent = 0.9 * final_ref
                    
                    # Find indices where response crosses these values
                    try:
                        rise_start_indices = np.where(actual >= ten_percent)[0]
                        rise_end_indices = np.where(actual >= ninety_percent)[0]
                        if len(rise_start_indices) > 0 and len(rise_end_indices) > 0:
                            rise_start_idx = rise_start_indices[0]
                            rise_end_idx = rise_end_indices[0]
                            if rise_end_idx > rise_start_idx:
                                rise_time = self.times[rise_end_idx] - self.times[rise_start_idx]
                                axis_metrics['rise_time'] = rise_time
                            else:
                                axis_metrics['rise_time'] = float('inf')
                        else:
                            axis_metrics['rise_time'] = float('inf')
                    except:
                        axis_metrics['rise_time'] = float('inf')
                else:
                    axis_metrics['rise_time'] = float('inf')
                
                # Settling time (within 2% of final value)
                settling_band = 0.02 * abs(final_ref) if final_ref != 0 else 0.01
                settled = np.abs(actual - final_ref) <= settling_band
                if np.any(settled):
                    # Find last time when it was outside settling band
                    last_unsettled = np.where(~settled)[0]
                    if len(last_unsettled) > 0:
                        settling_idx = last_unsettled[-1] + 1
                        if settling_idx < len(self.times):
                            settling_time = self.times[settling_idx]
                            axis_metrics['settling_time'] = settling_time
                        else:
                            axis_metrics['settling_time'] = self.times[-1]
                    else:
                        axis_metrics['settling_time'] = 0.0
                else:
                    axis_metrics['settling_time'] = float('inf')
                
                # Peak overshoot
                if final_ref > 0:
                    peak = np.max(actual)
                    overshoot = ((peak - final_ref) / final_ref) * 100
                else:
                    peak = np.min(actual)
                    overshoot = ((final_ref - peak) / abs(final_ref)) * 100
                axis_metrics['peak_overshoot'] = overshoot
                
                # Steady-state error
                steady_state_error = abs(actual[-1] - final_ref)
                axis_metrics['steady_state_error'] = steady_state_error
                
                # IAE (Integral of Absolute Error)
                iae = np.sum(np.abs(error)) * dt
                axis_metrics['iae'] = iae
                
                # ISE (Integral of Squared Error)
                ise = np.sum(error**2) * dt
                axis_metrics['ise'] = ise
                
                # ITAE (Integral of Time-weighted Absolute Error)
                itae = np.sum(np.arange(len(error)) * dt * np.abs(error)) * dt
                axis_metrics['itae'] = itae
                
                # Control effort (for position control, use motor speeds)
                if len(self.motor_speeds) > 0:
                    control_effort = np.sum(np.array(self.motor_speeds)**2) * dt
                    axis_metrics['control_effort'] = control_effort
            
            metrics[axis] = axis_metrics
        
        return metrics


def simulate_with_academic_logging(
    duration: float,
    dt: float,
    controller: BaseController,
    initial_state: QuadState | None = None,
    params: Params = Params(),
    ref_position: Optional[np.ndarray] = None,
    ref_orientation: Optional[np.ndarray] = None,
    *,
    method: str = "rk4",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step: float | None = None,
    controller_type: str = "unknown"
) -> AcademicLog:
    """Wrapper around simulate that returns an AcademicLog."""
    
    # Run simulation
    t, states, controls = simulate(
        duration=duration,
        dt=dt,
        controller=controller,
        initial_state=initial_state,
        params=params,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step
    )
    
    # Create academic log
    log = AcademicLog(
        duration=duration,
        dt=dt,
        method=method,
        controller_type=controller_type
    )
    
    # Populate log
    for i in range(len(t)):
        state = QuadState.from_vector(states[i])
        # For academic logging, we need more detailed information
        # This is a simplified version - in practice, controllers would provide more data
        log.add_entry(
            time=t[i],
            state=state,
            params=params,
            ref_position=ref_position,
            ref_orientation=ref_orientation,
            motor_speeds=controls[i],
            thrust=np.sum(controls[i])  # Simplified thrust calculation
        )
    
    return log


# TODO: Implement LoggingQuadcopterEnv class

__all__ = ["SimulationLog", "simulate_with_logging", "AcademicLog", "simulate_with_academic_logging", "LoggingQuadcopterEnv"]