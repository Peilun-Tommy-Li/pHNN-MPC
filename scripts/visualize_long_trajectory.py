"""
Long Trajectory Visualization (No Termination)

Same as visualize_phnn_accuracy.py but uses a CartPole wrapper that doesn't
terminate early, allowing us to see longer trajectories and accumulated errors.

Usage:
    python scripts/visualize_long_trajectory.py
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')
from pHNN import pHNN


class NoTerminationCartPole(Wrapper):
    """CartPole wrapper that never terminates episodes."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Never terminate - always return False
        return obs, reward, False, False, info


def load_config(config_path="cartpole_mpc_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_trained_model(config_path, weights_path):
    """Load trained pHNN model."""
    print(f"Loading trained pHNN model from {weights_path}...")
    model = pHNN(config_path)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    print("Model loaded successfully!")
    return model


def rearrange_state(gym_state):
    """Rearrange Gymnasium state to pHNN canonical form."""
    x, x_dot, theta, theta_dot = gym_state
    return np.array([x, theta, x_dot, theta_dot], dtype=np.float32)


def simulate_both(model, num_steps=50, force_magnitude=10.0, seed=42, dt=0.02):
    """
    Simulate both true and predicted dynamics with same control sequence.

    Returns:
        true_states, pred_states, controls, actions
    """
    np.random.seed(seed)

    # Create environment without termination
    env = NoTerminationCartPole(gym.make('CartPole-v1'))

    # Generate random control sequence
    actions = np.random.randint(0, 2, size=num_steps)
    controls = np.where(actions == 1, force_magnitude, -force_magnitude)

    # Simulate true dynamics
    gym_state, _ = env.reset(seed=seed)
    initial_state = rearrange_state(gym_state)

    true_states = [initial_state.copy()]

    for action in actions:
        next_gym_state, _, _, _, _ = env.step(action)
        next_state = rearrange_state(next_gym_state)
        true_states.append(next_state.copy())

    true_states = np.array(true_states)
    env.close()

    # Simulate pHNN dynamics
    state = torch.tensor(initial_state, dtype=torch.float32)
    pred_states = [state.clone()]

    for control in controls:
        x_current = state.unsqueeze(0).requires_grad_(True)
        u_current = torch.tensor([[control]], dtype=torch.float32)

        dx, _ = model(x_current, u_current)
        state = state.detach() + dt * dx.squeeze(0).detach()
        pred_states.append(state.clone())

    pred_states = torch.stack(pred_states).detach().numpy()

    return true_states, pred_states, controls, actions


def plot_detailed_comparison(true_states, pred_states, controls, dt, save_path):
    """Create detailed comparison plot."""
    time = np.arange(len(true_states)) * dt
    errors = np.abs(true_states - pred_states)

    fig = plt.figure(figsize=(18, 12))

    state_names = ['Cart Position x', 'Pole Angle θ', 'Cart Velocity ẋ', 'Pole Angular Velocity θ̇']
    units = ['(m)', '(rad)', '(m/s)', '(rad/s)']

    # Row 1: State comparisons
    for i in range(4):
        ax = plt.subplot(4, 3, i*3 + 1)
        ax.plot(time, true_states[:, i], 'b-', linewidth=2.5, label='True (Gym)', alpha=0.8)
        ax.plot(time, pred_states[:, i], 'r--', linewidth=2, label='Predicted (pHNN)', alpha=0.8)
        ax.set_ylabel(f'{state_names[i]} {units[i]}', fontsize=11, fontweight='bold')
        if i == 0:
            ax.set_title('State Trajectories', fontsize=13, fontweight='bold')
        if i == 3:
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    # Row 2: Absolute errors
    for i in range(4):
        ax = plt.subplot(4, 3, i*3 + 2)
        ax.plot(time, errors[:, i], 'g-', linewidth=2)
        ax.fill_between(time, 0, errors[:, i], alpha=0.3, color='green')
        ax.set_ylabel(f'Error {units[i]}', fontsize=10, fontweight='bold')
        if i == 0:
            ax.set_title('Absolute Prediction Errors', fontsize=13, fontweight='bold')
        if i == 3:
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add mean error annotation
        mean_err = np.mean(errors[:, i])
        ax.axhline(mean_err, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax.text(0.02, 0.95, f'Mean: {mean_err:.4f}', transform=ax.transAxes,
                fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

    # Row 3: Percentage errors
    for i in range(4):
        ax = plt.subplot(4, 3, i*3 + 3)
        # Avoid division by zero
        true_range = np.max(np.abs(true_states[:, i])) + 1e-6
        percent_errors = 100 * errors[:, i] / true_range
        ax.plot(time, percent_errors, 'purple', linewidth=2)
        ax.fill_between(time, 0, percent_errors, alpha=0.3, color='purple')
        ax.set_ylabel('Error (%)', fontsize=10, fontweight='bold')
        if i == 0:
            ax.set_title('Relative Errors', fontsize=13, fontweight='bold')
        if i == 3:
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add mean percentage error
        mean_pct = np.mean(percent_errors)
        ax.axhline(mean_pct, color='red', linestyle='--', linewidth=1, alpha=0.6)
        ax.text(0.02, 0.95, f'Mean: {mean_pct:.2f}%', transform=ax.transAxes,
                fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

    plt.suptitle('pHNN vs True CartPole Dynamics - Long Trajectory Analysis',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Detailed comparison saved to {save_path}")
    plt.close()


def plot_cumulative_error(true_states, pred_states, dt, save_path):
    """Plot cumulative error growth over time."""
    errors = np.abs(true_states - pred_states)
    cumulative_errors = np.cumsum(errors, axis=0)
    time = np.arange(len(true_states)) * dt

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    state_names = ['Cart Position x', 'Pole Angle θ', 'Cart Velocity ẋ', 'Pole Angular Velocity θ̇']

    for i in range(4):
        ax = axs[i // 2, i % 2]

        # Plot instantaneous and cumulative error
        ax2 = ax.twinx()

        l1 = ax.plot(time, errors[:, i], 'b-', linewidth=2, label='Instantaneous Error', alpha=0.7)
        l2 = ax2.plot(time, cumulative_errors[:, i], 'r-', linewidth=2, label='Cumulative Error', alpha=0.7)

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Instantaneous Error', fontsize=10, color='b')
        ax2.set_ylabel('Cumulative Error', fontsize=10, color='r')
        ax.set_title(f'{state_names[i]}', fontsize=12, fontweight='bold')

        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')

        # Combined legend
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=9)

        ax.grid(True, alpha=0.3)

    plt.suptitle('Error Accumulation Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Cumulative error plot saved to {save_path}")
    plt.close()


def print_statistics(true_states, pred_states):
    """Print comprehensive statistics."""
    errors = true_states - pred_states
    abs_errors = np.abs(errors)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE PREDICTION STATISTICS")
    print("=" * 80)

    print(f"\nTrajectory Length: {len(true_states)} time steps")

    state_names = ['Cart Position x', 'Pole Angle θ', 'Cart Velocity ẋ', 'Pole Angular Velocity θ̇']

    print(f"\n{'State':<25} {'Mean Abs Err':<15} {'Max Abs Err':<15} {'Final Err':<15}")
    print("-" * 80)

    for i, name in enumerate(state_names):
        mean_err = np.mean(abs_errors[:, i])
        max_err = np.max(abs_errors[:, i])
        final_err = abs_errors[-1, i]
        print(f"{name:<25} {mean_err:<15.6f} {max_err:<15.6f} {final_err:<15.6f}")

    # Overall metrics
    print(f"\n{'Overall Metrics':<25}")
    print("-" * 80)
    print(f"{'MSE (all states)':<25} {np.mean(errors**2):.6f}")
    print(f"{'MAE (all states)':<25} {np.mean(abs_errors):.6f}")
    print(f"{'RMSE (all states)':<25} {np.sqrt(np.mean(errors**2)):.6f}")

    # Error growth rate
    print(f"\n{'Error Growth Analysis':<25}")
    print("-" * 80)
    initial_error = np.mean(abs_errors[0])
    final_error = np.mean(abs_errors[-1])
    growth_rate = (final_error - initial_error) / len(true_states)
    print(f"{'Initial avg error':<25} {initial_error:.6f}")
    print(f"{'Final avg error':<25} {final_error:.6f}")
    print(f"{'Error growth rate':<25} {growth_rate:.6f} per step")

    print("=" * 80)


def main():
    """Main function."""
    print("=" * 80)
    print("Long Trajectory Visualization (No Early Termination)")
    print("=" * 80)

    # Load config and model
    config = load_config()
    config_path = "cartpole_mpc_config.yaml"
    weights_path = config['training']['model_save_path']
    model = load_trained_model(config_path, weights_path)

    dt = config['cartpole']['dt']
    force_magnitude = config['cartpole']['force_magnitude']

    # Simulate for different lengths
    trajectory_lengths = [30, 50, 100]

    for num_steps in trajectory_lengths:
        print(f"\n{'='*80}")
        print(f"Simulating {num_steps}-step trajectory...")
        print(f"{'='*80}")

        true_states, pred_states, controls, actions = simulate_both(
            model, num_steps=num_steps, force_magnitude=force_magnitude, dt=dt
        )

        # Print statistics
        print_statistics(true_states, pred_states)

        # Create visualizations
        plot_detailed_comparison(
            true_states, pred_states, controls, dt,
            f'results/long_trajectory_{num_steps}steps.png'
        )

        plot_cumulative_error(
            true_states, pred_states, dt,
            f'results/cumulative_error_{num_steps}steps.png'
        )

    print("\n" + "=" * 80)
    print("Long trajectory visualization complete!")
    print("=" * 80)
    print("\nGenerated files:")
    for num_steps in trajectory_lengths:
        print(f"  - results/long_trajectory_{num_steps}steps.png")
        print(f"  - results/cumulative_error_{num_steps}steps.png")


if __name__ == "__main__":
    main()
