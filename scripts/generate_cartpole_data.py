"""
Cart-Pole Data Generation Module with Structured Excitation

Generates training data for pHNN by collecting trajectories from Gymnasium CartPole-v1
with three types of structured control inputs to properly decouple the Port-Hamiltonian
structure:

A. Zero-Input Set (20%): u=0 to learn Hamiltonian H and damping R
B. Chirp Set (40%): Sinusoidal with increasing frequency to learn coupling J
C. Step/Random Set (40%): Random step inputs to learn inertia/mass matrix

States are rearranged to canonical form [x, theta, x_dot, theta_dot].

Usage:
    python scripts/generate_cartpole_data.py
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
import yaml
from pathlib import Path


def load_config(config_path="cartpole_mpc_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class NoTerminationCartPole(Wrapper):
    """
    Wrapper for CartPole that never terminates episodes.

    This allows collecting long trajectories with sufficient excitation
    to properly learn the Port-Hamiltonian structure.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Override termination - never end the episode
        return obs, reward, False, False, info


def rearrange_state(gym_state):
    """
    Rearrange Gymnasium state to pHNN canonical form.

    Gymnasium CartPole-v1 provides: [x, x_dot, theta, theta_dot]
    Rearrange to: [x, theta, x_dot, theta_dot] (positions, then velocities)

    Args:
        gym_state: State from Gymnasium environment (numpy array or list)

    Returns:
        Rearranged state as numpy array
    """
    x, x_dot, theta, theta_dot = gym_state
    return np.array([x, theta, x_dot, theta_dot], dtype=np.float32)


def generate_zero_input_trajectory(env, max_steps, dt, seed):
    """
    Generate Zero-Input trajectory: u = 0 for entire episode.

    Purpose: Isolates Hamiltonian (energy conservation) and damping R matrix.
    Learns how friction slows the system without external forcing.

    Args:
        env: Gymnasium environment
        max_steps: Maximum steps per trajectory
        dt: Time step
        seed: Random seed for initial condition

    Returns:
        Tuple of (states, controls, derivatives) lists
    """
    gym_state, _ = env.reset(seed=seed)
    state = rearrange_state(gym_state)

    traj_states = []
    traj_controls = []
    traj_derivatives = []

    for step in range(max_steps):
        # Zero control
        force = 0.0
        action = 1  # Action doesn't matter with zero force, but needed for gym

        traj_states.append(state.copy())
        traj_controls.append([force])

        # Step with no applied force (alternating actions to average out discrete effects)
        next_gym_state, _, _, _, _ = env.step(step % 2)
        next_state = rearrange_state(next_gym_state)

        # Compute derivative
        derivative = (next_state - state) / dt
        traj_derivatives.append(derivative)

        state = next_state

    return traj_states, traj_controls, traj_derivatives


def generate_chirp_trajectory(env, max_steps, dt, seed, amplitude=10.0, f_start=0.5, f_end=5.0):
    """
    Generate Chirp trajectory: u(t) = A * sin(ω(t) * t) with increasing frequency.

    Purpose: Excites system at different frequencies to learn coupling J matrix.
    Shows how energy transfers between cart and pole.

    Args:
        env: Gymnasium environment
        max_steps: Maximum steps per trajectory
        dt: Time step
        seed: Random seed for initial condition
        amplitude: Force amplitude
        f_start: Starting frequency (Hz)
        f_end: Ending frequency (Hz)

    Returns:
        Tuple of (states, controls, derivatives) lists
    """
    gym_state, _ = env.reset(seed=seed)
    state = rearrange_state(gym_state)

    traj_states = []
    traj_controls = []
    traj_derivatives = []

    # Chirp parameters: linearly increasing frequency
    T = max_steps * dt

    for step in range(max_steps):
        t = step * dt

        # Linear chirp: frequency increases from f_start to f_end
        omega = 2 * np.pi * (f_start + (f_end - f_start) * t / T)
        force = amplitude * np.sin(omega * t)

        traj_states.append(state.copy())
        traj_controls.append([force])

        # Map continuous force to discrete action
        action = 1 if force > 0 else 0

        next_gym_state, _, _, _, _ = env.step(action)
        next_state = rearrange_state(next_gym_state)

        derivative = (next_state - state) / dt
        traj_derivatives.append(derivative)

        state = next_state

    return traj_states, traj_controls, traj_derivatives


def generate_step_random_trajectory(env, max_steps, dt, seed, force_max=10.0, hold_steps=10):
    """
    Generate Step/Random trajectory: Random force held for k steps, then switch.

    Purpose: High-impact transients to learn inertia (mass matrix).
    F=ma is most visible when F changes abruptly.

    Args:
        env: Gymnasium environment
        max_steps: Maximum steps per trajectory
        dt: Time step
        seed: Random seed
        force_max: Maximum force magnitude
        hold_steps: Number of steps to hold each random force

    Returns:
        Tuple of (states, controls, derivatives) lists
    """
    np.random.seed(seed)
    gym_state, _ = env.reset(seed=seed)
    state = rearrange_state(gym_state)

    traj_states = []
    traj_controls = []
    traj_derivatives = []

    # Generate random force sequence
    num_switches = (max_steps // hold_steps) + 1
    force_sequence = np.random.uniform(-force_max, force_max, num_switches)

    for step in range(max_steps):
        # Get current force from sequence
        force_idx = step // hold_steps
        force = force_sequence[min(force_idx, len(force_sequence) - 1)]

        traj_states.append(state.copy())
        traj_controls.append([force])

        # Map to discrete action
        action = 1 if force > 0 else 0

        next_gym_state, _, _, _, _ = env.step(action)
        next_state = rearrange_state(next_gym_state)

        derivative = (next_state - state) / dt
        traj_derivatives.append(derivative)

        state = next_state

    return traj_states, traj_controls, traj_derivatives


def generate_cartpole_data(config):
    """
    Generate training data with structured excitation.

    Data composition:
    - 20% Zero-Input (learn H and R)
    - 40% Chirp (learn J)
    - 40% Step/Random (learn inertia)

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (states, controls, derivatives) tensors
    """
    # Extract parameters
    dt = config['cartpole']['dt']
    num_traj = config['cartpole']['num_traj']
    max_steps = config['cartpole']['max_steps']
    random_seed = config['data']['random_seed']

    # Determine number of trajectories for each type
    num_zero = int(0.2 * num_traj)
    num_chirp = int(0.4 * num_traj)
    num_step = num_traj - num_zero - num_chirp  # Remaining for step/random

    print(f"Generating {num_traj} trajectories with structured excitation:")
    print(f"  - Zero-Input: {num_zero} trajectories (20%)")
    print(f"  - Chirp: {num_chirp} trajectories (40%)")
    print(f"  - Step/Random: {num_step} trajectories (40%)")
    print(f"  - Steps per trajectory: {max_steps}")
    print(f"  - Time step: {dt}s")
    print()

    # Create environment with no termination
    base_env = gym.make('CartPole-v1')
    env = NoTerminationCartPole(base_env)

    all_states = []
    all_controls = []
    all_derivatives = []

    # Generate Zero-Input trajectories
    print("Generating Zero-Input trajectories...")
    for i in range(num_zero):
        states, controls, derivatives = generate_zero_input_trajectory(
            env, max_steps, dt, random_seed + i
        )
        all_states.append(states)
        all_controls.append(controls)
        all_derivatives.append(derivatives)

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{num_zero} complete")

    # Generate Chirp trajectories
    print("\nGenerating Chirp trajectories...")
    for i in range(num_chirp):
        # Vary chirp parameters for diversity
        f_start = np.random.uniform(0.3, 1.0)
        f_end = np.random.uniform(3.0, 7.0)
        amplitude = np.random.uniform(5.0, 15.0)

        states, controls, derivatives = generate_chirp_trajectory(
            env, max_steps, dt, random_seed + num_zero + i,
            amplitude=amplitude, f_start=f_start, f_end=f_end
        )
        all_states.append(states)
        all_controls.append(controls)
        all_derivatives.append(derivatives)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{num_chirp} complete")

    # Generate Step/Random trajectories
    print("\nGenerating Step/Random trajectories...")
    for i in range(num_step):
        # Vary hold time for diversity
        hold_steps = np.random.randint(5, 20)
        force_max = np.random.uniform(8.0, 20.0)

        states, controls, derivatives = generate_step_random_trajectory(
            env, max_steps, dt, random_seed + num_zero + num_chirp + i,
            force_max=force_max, hold_steps=hold_steps
        )
        all_states.append(states)
        all_controls.append(controls)
        all_derivatives.append(derivatives)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{num_step} complete")

    env.close()

    # All trajectories have same length (no padding needed)
    print(f"\nAll trajectories have length: {max_steps} steps")

    # Convert to tensors
    states_tensor = torch.tensor(np.array(all_states), dtype=torch.float32)
    controls_tensor = torch.tensor(np.array(all_controls), dtype=torch.float32)
    derivatives_tensor = torch.tensor(np.array(all_derivatives), dtype=torch.float32)

    print(f"\nDataset shapes:")
    print(f"  States: {states_tensor.shape}")  # (num_traj, max_steps, 4)
    print(f"  Controls: {controls_tensor.shape}")  # (num_traj, max_steps, 1)
    print(f"  Derivatives: {derivatives_tensor.shape}")  # (num_traj, max_steps, 4)

    # Data statistics
    print(f"\nData statistics:")
    print(f"  State ranges:")
    print(f"    x: [{states_tensor[:, :, 0].min():.3f}, {states_tensor[:, :, 0].max():.3f}]")
    print(f"    theta: [{states_tensor[:, :, 1].min():.3f}, {states_tensor[:, :, 1].max():.3f}]")
    print(f"    x_dot: [{states_tensor[:, :, 2].min():.3f}, {states_tensor[:, :, 2].max():.3f}]")
    print(f"    theta_dot: [{states_tensor[:, :, 3].min():.3f}, {states_tensor[:, :, 3].max():.3f}]")
    print(f"  Control range: [{controls_tensor.min():.3f}, {controls_tensor.max():.3f}]")

    return states_tensor, controls_tensor, derivatives_tensor


def save_data(states, controls, derivatives, save_path):
    """Save generated data to disk."""
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Save as dictionary
    data_dict = {
        'states': states,
        'controls': controls,
        'derivatives': derivatives
    }

    torch.save(data_dict, save_path)
    print(f"\nData saved to {save_path}")


def validate_saved_data(save_path):
    """Load and validate saved data."""
    print(f"\nValidating saved data from {save_path}...")

    data = torch.load(save_path)
    states = data['states']
    controls = data['controls']
    derivatives = data['derivatives']

    print(f"Loaded data shapes:")
    print(f"  States: {states.shape}")
    print(f"  Controls: {controls.shape}")
    print(f"  Derivatives: {derivatives.shape}")

    print(f"\nSample trajectories:")
    print(f"  First state of trajectory 0: {states[0, 0]}")
    print(f"  First control of trajectory 0: {controls[0, 0]}")

    # Check for diversity
    print(f"\nControl diversity check:")
    print(f"  Trajectory 0 control std: {controls[0].std():.3f} (should be 0 for zero-input)")
    print(f"  Trajectory {int(0.2*len(states))} control std: {controls[int(0.2*len(states))].std():.3f} (chirp - should be high)")
    print(f"  Trajectory {int(0.6*len(states))} control std: {controls[int(0.6*len(states))].std():.3f} (step - should be high)")

    print("\nValidation complete!")


def main():
    """Main function to generate and save cart-pole training data."""
    print("=" * 80)
    print("Cart-Pole Data Generation with Structured Excitation")
    print("=" * 80)
    print()
    print("This script generates three types of trajectories to properly decouple")
    print("the Port-Hamiltonian structure (J, R, H, G):")
    print()
    print("  A. Zero-Input (20%): Learns Hamiltonian H and damping R")
    print("  B. Chirp (40%): Learns coupling J matrix")
    print("  C. Step/Random (40%): Learns inertia/mass matrix")
    print()
    print("=" * 80)
    print()

    # Load configuration
    config = load_config()

    # Generate data
    states, controls, derivatives = generate_cartpole_data(config)

    # Save data
    save_path = config['data']['save_path']
    save_data(states, controls, derivatives, save_path)

    # Validate saved data
    validate_saved_data(save_path)

    print("\n" + "=" * 80)
    print("Data generation complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Train pHNN: python scripts/train_cartpole_phnn.py")
    print("  2. Run MPC: ./RUN_MPC.sh")


if __name__ == "__main__":
    main()
