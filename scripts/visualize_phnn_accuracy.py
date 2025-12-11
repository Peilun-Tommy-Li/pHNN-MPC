"""
Visualize pHNN Prediction Accuracy

Applies the same control sequence to both:
1. Actual Gymnasium CartPole simulation
2. Trained pHNN model prediction

Then visualizes the differences to assess model accuracy.

Usage:
    python scripts/visualize_phnn_accuracy.py
"""

import torch
import numpy as np
import gymnasium as gym
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')
from pHNN import pHNN


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


def generate_control_sequence(num_steps, force_magnitude=10.0, seed=42):
    """
    Generate a random control sequence.

    Args:
        num_steps: Number of control steps
        force_magnitude: Magnitude of control force
        seed: Random seed for reproducibility

    Returns:
        Array of control forces (num_steps,)
    """
    np.random.seed(seed)
    # Random actions (0 or 1)
    actions = np.random.randint(0, 2, size=num_steps)
    # Convert to forces
    controls = np.where(actions == 1, force_magnitude, -force_magnitude)
    return controls, actions


def simulate_true_dynamics(env, controls, actions, dt):
    """
    Simulate true CartPole dynamics with given control sequence.

    Args:
        env: Gymnasium environment
        controls: Control forces (for recording)
        actions: Discrete actions (for env.step)
        dt: Time step

    Returns:
        Array of states (num_steps+1, 4) in pHNN format [x, theta, x_dot, theta_dot]
    """
    gym_state, _ = env.reset(seed=42)
    state = rearrange_state(gym_state)

    true_states = [state.copy()]

    for action in actions:
        next_gym_state, _, terminated, truncated, _ = env.step(action)
        next_state = rearrange_state(next_gym_state)
        true_states.append(next_state.copy())

        if terminated or truncated:
            print(f"  Episode terminated at step {len(true_states)-1}")
            break

    return np.array(true_states)


def simulate_phnn_dynamics(model, initial_state, controls, dt):
    """
    Simulate pHNN model predictions with given control sequence.

    Args:
        model: Trained pHNN model
        initial_state: Initial state (4,)
        controls: Control forces (num_steps,)
        dt: Time step

    Returns:
        Array of predicted states (num_steps+1, 4)
    """
    state = torch.tensor(initial_state, dtype=torch.float32)
    pred_states = [state.clone()]

    for control in controls:
        # Prepare inputs - ensure grad is enabled for x
        x_current = state.unsqueeze(0).requires_grad_(True)
        u_current = torch.tensor([[control]], dtype=torch.float32)

        # Predict derivative (don't use no_grad, model needs gradients for Hamiltonian)
        dx, _ = model(x_current, u_current)

        # Integrate forward and detach for next iteration
        state = state.detach() + dt * dx.squeeze(0).detach()
        pred_states.append(state.clone())

    # Convert to numpy
    pred_states = torch.stack(pred_states).detach().numpy()
    return pred_states


def compute_errors(true_states, pred_states):
    """Compute error statistics between true and predicted states."""
    # Ensure same length
    min_len = min(len(true_states), len(pred_states))
    true_states = true_states[:min_len]
    pred_states = pred_states[:min_len]

    # Compute errors
    errors = true_states - pred_states
    abs_errors = np.abs(errors)

    # Statistics
    mse = np.mean(errors**2)
    mae = np.mean(abs_errors)
    max_error = np.max(abs_errors, axis=0)

    stats = {
        'mse': mse,
        'mae': mae,
        'max_errors': max_error,
        'errors': errors
    }

    return stats


def plot_comparison(true_states, pred_states, controls, dt, save_path='results/phnn_vs_true.png'):
    """
    Plot comparison between true and predicted trajectories.

    Args:
        true_states: True states from simulation (T, 4)
        pred_states: Predicted states from pHNN (T, 4)
        controls: Control forces (T-1,)
        dt: Time step
        save_path: Path to save plot
    """
    # Ensure same length
    min_len = min(len(true_states), len(pred_states))
    true_states = true_states[:min_len]
    pred_states = pred_states[:min_len]

    time = np.arange(min_len) * dt
    state_names = ['x (cart position)', 'θ (pole angle)', 'ẋ (cart velocity)', 'θ̇ (pole angular velocity)']

    fig = plt.figure(figsize=(16, 10))

    # State comparisons (4 subplots)
    for i in range(4):
        ax = plt.subplot(3, 2, i+1)
        ax.plot(time, true_states[:, i], 'b-', linewidth=2, label='True (Gym)')
        ax.plot(time, pred_states[:, i], 'r--', linewidth=2, label='Predicted (pHNN)')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(state_names[i], fontsize=10)
        ax.set_title(f'{state_names[i]}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Error plots
    errors = true_states - pred_states

    # Subplot 5: Absolute errors per state
    ax = plt.subplot(3, 2, 5)
    for i in range(4):
        ax.plot(time, np.abs(errors[:, i]), linewidth=2, label=f'{state_names[i]}')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Absolute Error', fontsize=10)
    ax.set_title('Prediction Errors Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Subplot 6: Control sequence
    ax = plt.subplot(3, 2, 6)
    ax.plot(time[:-1], controls[:min_len-1], 'g-', linewidth=2, label='Control Force')
    ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Control Force (N)', fontsize=10)
    ax.set_title('Applied Control Sequence', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nComparison plot saved to {save_path}")
    plt.close()


def plot_phase_space(true_states, pred_states, save_path='results/phase_space_comparison.png'):
    """
    Plot phase space trajectories (position vs velocity for cart and pole).

    Args:
        true_states: True states (T, 4)
        pred_states: Predicted states (T, 4)
        save_path: Path to save plot
    """
    min_len = min(len(true_states), len(pred_states))
    true_states = true_states[:min_len]
    pred_states = pred_states[:min_len]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Cart phase space: x vs x_dot
    ax = axs[0]
    ax.plot(true_states[:, 0], true_states[:, 2], 'b-', linewidth=2, label='True', alpha=0.7)
    ax.plot(pred_states[:, 0], pred_states[:, 2], 'r--', linewidth=2, label='Predicted', alpha=0.7)
    ax.scatter(true_states[0, 0], true_states[0, 2], c='green', s=100, marker='o',
               label='Start', zorder=5, edgecolors='black')
    ax.scatter(true_states[-1, 0], true_states[-1, 2], c='red', s=100, marker='X',
               label='End (True)', zorder=5, edgecolors='black')
    ax.scatter(pred_states[-1, 0], pred_states[-1, 2], c='orange', s=100, marker='X',
               label='End (Pred)', zorder=5, edgecolors='black')
    ax.set_xlabel('Cart Position x (m)', fontsize=11)
    ax.set_ylabel('Cart Velocity ẋ (m/s)', fontsize=11)
    ax.set_title('Cart Phase Space', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Pole phase space: theta vs theta_dot
    ax = axs[1]
    ax.plot(true_states[:, 1], true_states[:, 3], 'b-', linewidth=2, label='True', alpha=0.7)
    ax.plot(pred_states[:, 1], pred_states[:, 3], 'r--', linewidth=2, label='Predicted', alpha=0.7)
    ax.scatter(true_states[0, 1], true_states[0, 3], c='green', s=100, marker='o',
               label='Start', zorder=5, edgecolors='black')
    ax.scatter(true_states[-1, 1], true_states[-1, 3], c='red', s=100, marker='X',
               label='End (True)', zorder=5, edgecolors='black')
    ax.scatter(pred_states[-1, 1], pred_states[-1, 3], c='orange', s=100, marker='X',
               label='End (Pred)', zorder=5, edgecolors='black')
    ax.set_xlabel('Pole Angle θ (rad)', fontsize=11)
    ax.set_ylabel('Pole Angular Velocity θ̇ (rad/s)', fontsize=11)
    ax.set_title('Pole Phase Space', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Phase space plot saved to {save_path}")
    plt.close()


def print_error_statistics(stats, true_states, pred_states):
    """Print detailed error statistics."""
    print("\n" + "=" * 70)
    print("PREDICTION ERROR STATISTICS")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  Mean Squared Error (MSE): {stats['mse']:.6f}")
    print(f"  Mean Absolute Error (MAE): {stats['mae']:.6f}")

    print(f"\nMaximum Absolute Errors:")
    state_names = ['x (cart pos)', 'θ (pole angle)', 'ẋ (cart vel)', 'θ̇ (pole ang vel)']
    for i, name in enumerate(state_names):
        print(f"  {name:20s}: {stats['max_errors'][i]:.6f}")

    print(f"\nTrajectory Length:")
    min_len = min(len(true_states), len(pred_states))
    print(f"  True simulation:  {len(true_states)} steps")
    print(f"  pHNN prediction:  {len(pred_states)} steps")
    print(f"  Compared steps:   {min_len} steps")

    # Final state comparison
    if len(true_states) == len(pred_states):
        print(f"\nFinal State Comparison:")
        print(f"  {'State':20s} {'True':>12s} {'Predicted':>12s} {'Error':>12s}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        final_idx = min_len - 1
        for i, name in enumerate(state_names):
            true_val = true_states[final_idx, i]
            pred_val = pred_states[final_idx, i]
            error = true_val - pred_val
            print(f"  {name:20s} {true_val:12.6f} {pred_val:12.6f} {error:12.6f}")

    print("=" * 70)


def main():
    """Main visualization function."""
    print("=" * 70)
    print("pHNN Prediction Accuracy Visualization")
    print("=" * 70)

    # Load configuration
    config = load_config()

    # Load trained model
    config_path = "cartpole_mpc_config.yaml"
    weights_path = config['training']['model_save_path']
    model = load_trained_model(config_path, weights_path)

    # Parameters
    dt = config['cartpole']['dt']
    num_steps = 100  # Number of control steps
    force_magnitude = config['cartpole']['force_magnitude']

    # Generate control sequence
    print(f"\nGenerating random control sequence ({num_steps} steps)...")
    controls, actions = generate_control_sequence(num_steps, force_magnitude)
    print(f"Control range: [{controls.min():.1f}, {controls.max():.1f}] N")

    # Simulate true dynamics
    print(f"\nSimulating true CartPole dynamics...")
    env = gym.make('CartPole-v1')
    true_states = simulate_true_dynamics(env, controls, actions, dt)
    env.close()
    print(f"True simulation: {len(true_states)} states")

    # Simulate pHNN dynamics
    print(f"\nSimulating pHNN predictions...")
    initial_state = true_states[0]
    pred_states = simulate_phnn_dynamics(model, initial_state, controls, dt)
    print(f"pHNN prediction: {len(pred_states)} states")

    # Compute errors
    stats = compute_errors(true_states, pred_states)

    # Print statistics
    print_error_statistics(stats, true_states, pred_states)

    # Create visualizations
    print(f"\nGenerating visualizations...")

    # Main comparison plot
    plot_comparison(true_states, pred_states, controls, dt)

    # Phase space plot
    plot_phase_space(true_states, pred_states)

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - results/phnn_vs_true.png (trajectory comparison)")
    print("  - results/phase_space_comparison.png (phase space)")


if __name__ == "__main__":
    main()
