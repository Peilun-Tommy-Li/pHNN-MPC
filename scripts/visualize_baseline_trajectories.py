"""
Visualize baseline model (MLP, Neural ODE) long trajectory predictions.

Similar to visualize_phnn_long_trajectories.py but for baseline models.
Shows pole angle, angular velocity, control input, and energy evolution.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')
from baseline_mlp import VanillaMLP
from baseline_node import NeuralODE


def load_config(config_path="cartpole_mpc_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_baseline_model(model_type, checkpoint_path, config):
    """
    Load trained baseline model.

    Args:
        model_type: 'mlp' or 'node'
        checkpoint_path: Path to checkpoint
        config: Configuration dict

    Returns:
        model: Loaded model
    """
    print(f"Loading {model_type.upper()} model from {checkpoint_path}...")

    if model_type == 'mlp':
        model = VanillaMLP(
            state_dim=4,
            action_dim=1,
            hidden_sizes=[256, 256, 256, 256],
            activation='relu',
            use_residual=True,
            dropout=0.1,
            layer_norm=False
        )
    elif model_type == 'node':
        model = NeuralODE(
            state_dim=4,
            action_dim=1,
            hidden_sizes=[128, 128, 128],
            activation='tanh',
            layer_norm=False,
            solver='dopri5',
            rtol=1e-3,
            atol=1e-4
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded (epoch {checkpoint['epoch']})")
    return model


def wrap_angle(angle):
    """Wrap angle to [-pi, pi] range."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def rollout_trajectory(model, x0, controls, dt, model_type):
    """
    Rollout trajectory using baseline model.

    Args:
        model: Baseline model (MLP or Neural ODE)
        x0: Initial state (4,) - tensor
        controls: Control sequence (seq_len, 1) - tensor
        dt: Time step
        model_type: 'mlp' or 'node'

    Returns:
        states: Predicted states (seq_len+1, 4) - tensor
    """
    states = [x0.unsqueeze(0)]

    with torch.no_grad():
        for t in range(len(controls)):
            x_current = states[-1]
            u_current = controls[t].unsqueeze(0)

            # Predict next state
            if model_type == 'mlp':
                x_next = model(x_current, u_current)
            elif model_type == 'node':
                x_next = model(x_current, u_current, dt)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            states.append(x_next)

    return torch.cat(states, dim=0)  # (seq_len+1, 4)


def compute_errors(x_pred, x_true):
    """Compute prediction errors."""
    errors = torch.abs(x_pred - x_true)
    return {
        'x': errors[:, 0].numpy(),
        'theta': errors[:, 1].numpy(),
        'x_dot': errors[:, 2].numpy(),
        'theta_dot': errors[:, 3].numpy(),
    }


def plot_trajectory(x_pred, x_true, controls, dt, model_type, sample_idx, save_path):
    """
    Plot trajectory prediction for a single sample.

    Args:
        x_pred: Predicted states (seq_len+1, 4)
        x_true: True states (seq_len, 4)
        controls: Control inputs (seq_len-1, 1)
        dt: Time step
        model_type: 'mlp' or 'node'
        sample_idx: Sample index for title
        save_path: Path to save figure
    """
    time = np.arange(len(x_true)) * dt

    # Compute errors
    errors = compute_errors(x_pred, x_true)

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Pole Angle θ (top-left)
    ax = axes[0, 0]
    pred_wrapped = wrap_angle(x_pred[:, 1].numpy())
    true_wrapped = wrap_angle(x_true[:, 1].numpy())
    ax.plot(time, true_wrapped, 'b-', linewidth=2, label='True', alpha=0.7)
    ax.plot(time, pred_wrapped, 'r--', linewidth=2, label='Predicted', alpha=0.7)
    ax.axhline(np.pi, color='gray', linestyle=':', alpha=0.5, label='±π')
    ax.axhline(-np.pi, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Pole Angle θ (rad)', fontsize=11)
    ax.set_title('Pole Angle θ (rad)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show error stats
    mean_theta_err = np.mean(errors['theta'])
    max_theta_err = np.max(errors['theta'])
    ax.text(0.02, 0.98, f'Mean Error: {mean_theta_err:.4f} rad\nMax Error: {max_theta_err:.4f} rad',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 2. Pole Angular Velocity θ̇ (top-right)
    ax = axes[0, 1]
    ax.plot(time, x_true[:, 3].numpy(), 'b-', linewidth=2, label='True', alpha=0.7)
    ax.plot(time, x_pred[:, 3].numpy(), 'r--', linewidth=2, label='Predicted', alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Pole Angular Velocity θ̇ (rad/s)', fontsize=11)
    ax.set_title('Pole Angular Velocity θ̇ (rad/s)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show error stats
    mean_vel_theta_err = np.mean(errors['theta_dot'])
    max_vel_theta_err = np.max(errors['theta_dot'])
    ax.text(0.02, 0.98, f'Mean Error: {mean_vel_theta_err:.4f} rad/s\nMax Error: {max_vel_theta_err:.4f} rad/s',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 3. Control Input (bottom-left)
    ax = axes[1, 0]
    time_control = np.arange(len(controls)) * dt
    ax.plot(time_control, controls.numpy(), 'k-', linewidth=2, label='Control Force')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Control Force (N)', fontsize=11)
    ax.set_title('Control Input', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show control stats
    mean_control = np.mean(np.abs(controls.numpy()))
    max_control = np.max(np.abs(controls.numpy()))
    ax.text(0.02, 0.98, f'Mean |u|: {mean_control:.4f} N\nMax |u|: {max_control:.4f} N',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 4. Energy Evolution (bottom-right)
    ax = axes[1, 1]
    # Approximate kinetic + potential energy
    # KE = 0.5 * m * v^2, PE ∝ (1 - cos(theta))
    ke_true = 0.5 * (x_true[:, 2]**2 + x_true[:, 3]**2)
    pe_true = 1 - torch.cos(x_true[:, 1])
    energy_true = ke_true + pe_true

    ke_pred = 0.5 * (x_pred[:, 2]**2 + x_pred[:, 3]**2)
    pe_pred = 1 - torch.cos(x_pred[:, 1])
    energy_pred = ke_pred + pe_pred

    ax.plot(time, energy_true.numpy(), 'b-', linewidth=2, label='True', alpha=0.7)
    ax.plot(time, energy_pred.numpy(), 'r--', linewidth=2, label='Predicted', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Approximate Energy', fontsize=11)
    ax.set_title('Energy Evolution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show energy error stats
    energy_error = np.abs(energy_pred.numpy() - energy_true.numpy())
    mean_energy_err = np.mean(energy_error)
    max_energy_err = np.max(energy_error)
    ax.text(0.02, 0.98, f'Mean Error: {mean_energy_err:.4f}\nMax Error: {max_energy_err:.4f}',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.suptitle(f'{model_type.upper()} Long Trajectory Prediction - Sample {sample_idx + 1}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def print_statistics(all_errors):
    """Print error statistics across all samples."""
    print("\n" + "=" * 80)
    print("Prediction Error Statistics (across all samples)")
    print("=" * 80)

    # Aggregate errors
    theta_errors = []
    theta_dot_errors = []
    x_errors = []
    x_dot_errors = []

    for errors in all_errors:
        theta_errors.extend(errors['theta'])
        theta_dot_errors.extend(errors['theta_dot'])
        x_errors.extend(errors['x'])
        x_dot_errors.extend(errors['x_dot'])

    # Compute statistics
    print(f"\nPole Angle (θ):")
    print(f"  Mean error: {np.mean(theta_errors):.6f} rad")
    print(f"  Std error:  {np.std(theta_errors):.6f} rad")
    print(f"  Max error:  {np.max(theta_errors):.6f} rad")

    print(f"\nPole Angular Velocity (θ̇):")
    print(f"  Mean error: {np.mean(theta_dot_errors):.6f} rad/s")
    print(f"  Std error:  {np.std(theta_dot_errors):.6f} rad/s")
    print(f"  Max error:  {np.max(theta_dot_errors):.6f} rad/s")

    print(f"\nCart Position (x):")
    print(f"  Mean error: {np.mean(x_errors):.6f} m")
    print(f"  Std error:  {np.std(x_errors):.6f} m")
    print(f"  Max error:  {np.max(x_errors):.6f} m")

    print(f"\nCart Velocity (ẋ):")
    print(f"  Mean error: {np.mean(x_dot_errors):.6f} m/s")
    print(f"  Std error:  {np.std(x_dot_errors):.6f} m/s")
    print(f"  Max error:  {np.max(x_dot_errors):.6f} m/s")

    print("=" * 80 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize baseline model trajectory predictions')
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'node'],
                        help='Model type: mlp or node')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: baseline/{model}/best_model.pth)')
    parser.add_argument('--config', type=str, default='cartpole_mpc_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of trajectories to visualize')
    parser.add_argument('--output_dir', type=str, default='results/baseline_trajectories',
                        help='Output directory for plots')

    args = parser.parse_args()

    print("=" * 80)
    print(f"{args.model.upper()} Trajectory Visualization")
    print("=" * 80)

    # Load configuration
    config = load_config(args.config)
    dt = config['cartpole']['dt']

    # Determine checkpoint path
    if args.checkpoint is None:
        checkpoint_path = f"baseline/{args.model}/best_model.pth"
    else:
        checkpoint_path = args.checkpoint

    # Load model
    model = load_baseline_model(args.model, checkpoint_path, config)

    # Load data
    data_path = config['data']['save_path']
    print(f"\nLoading data from {data_path}...")
    data = torch.load(data_path, weights_only=True)

    states = data['states']  # (num_traj, seq_len, 4)
    controls = data['controls']  # (num_traj, seq_len, 1)

    print(f"Data loaded: {states.shape[0]} trajectories, {states.shape[1]} steps each")

    # Select sample trajectories
    num_samples = min(args.num_samples, states.shape[0])
    num_traj = states.shape[0]
    sample_indices = np.linspace(0, num_traj - 1, num_samples, dtype=int)

    print(f"\nVisualizing {num_samples} sample trajectories:")
    print(f"Sample indices: {sample_indices}")

    # Create output directory
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize each sample
    all_errors = []

    for i, idx in enumerate(sample_indices):
        print(f"\nProcessing sample {i + 1}/{num_samples} (trajectory {idx})...")

        # Get ground truth trajectory (first half)
        x_true = states[idx]
        half_len = len(x_true) // 2
        x_true = x_true[:half_len, :]

        u_true = controls[idx, :-1]
        u_true = u_true[:half_len-1]

        # Rollout prediction
        x0 = x_true[0]
        x_pred = rollout_trajectory(model, x0, u_true, dt, args.model)

        # Compute errors
        errors = compute_errors(x_pred, x_true)
        all_errors.append(errors)

        # Plot trajectory
        save_path = output_dir / f"trajectory_sample_{i+1}_idx_{idx}.png"
        plot_trajectory(x_pred, x_true, u_true, dt, args.model, i, save_path)

    # Print statistics
    print_statistics(all_errors)

    print("=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print(f"\nGenerated {num_samples} trajectory plots in: {output_dir}")
    print(f"Files:")
    for i in range(num_samples):
        print(f"  - trajectory_sample_{i+1}_idx_{sample_indices[i]}.png")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
