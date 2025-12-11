"""
Visualize pHNN Long Trajectory Predictions with Angle Wrapping

Compares pHNN predictions vs ground truth for 100-step trajectories.
Properly handles 2π periodicity of angles.

Usage:
    python scripts/visualize_phnn_long_trajectories.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')
from pHNN import pHNN


def load_config(config_path="cartpole_mpc_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(config_path, weights_path):
    """Load trained pHNN model."""
    print(f"Loading model from {weights_path}...")
    model = pHNN(config_path)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    print("Model loaded successfully!")
    return model


def wrap_angle(angle):
    """
    Wrap angle to [-π, π] range.

    Handles 2π periodicity: 0 and 2π are the same angle.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_difference(angle1, angle2):
    """
    Compute shortest angular difference between two angles.

    Returns difference in [-π, π] range.
    """
    diff = angle1 - angle2
    return wrap_angle(diff)


def rollout_trajectory(model, x0, controls, dt):
    """
    Rollout pHNN predictions for given controls.

    Args:
        model: Trained pHNN model
        x0: Initial state (4,)
        controls: Control sequence (T, 1)
        dt: Time step

    Returns:
        Predicted states (T+1, 4)
    """
    x_pred = [x0.unsqueeze(0).detach()]

    for t in range(len(controls)):
        # Create new tensor with requires_grad for pHNN Hamiltonian gradient
        x_current = x_pred[-1].detach().requires_grad_(True)

        with torch.no_grad():
            # Compute derivative (but pHNN internally needs grad for Hamiltonian)
            pass

        # Call model (allows internal gradient computation for Hamiltonian)
        dx, _ = model(x_current, controls[t].unsqueeze(0))

        # Integrate (detach to avoid building computation graph)
        x_next = x_pred[-1].detach() + dt * dx.detach()
        x_pred.append(x_next)

    x_pred = torch.cat(x_pred, dim=0)  # (T+1, 4)
    return x_pred


def compute_errors(x_pred, x_true):
    """
    Compute prediction errors with proper angle wrapping.

    State: [x, theta, x_dot, theta_dot]

    Returns:
        Dictionary of errors
    """
    # Position error (standard)
    pos_error = torch.abs(x_pred[:, 0] - x_true[:, 0])

    # Angle error (wrapped to [-π, π])
    theta_pred = x_pred[:, 1].numpy()
    theta_true = x_true[:, 1].numpy()
    theta_error = np.abs(angle_difference(theta_pred, theta_true))

    # Velocity errors (standard)
    vel_x_error = torch.abs(x_pred[:, 2] - x_true[:, 2])
    vel_theta_error = torch.abs(x_pred[:, 3] - x_true[:, 3])

    return {
        'pos': pos_error.numpy(),
        'theta': theta_error,
        'vel_x': vel_x_error.numpy(),
        'vel_theta': vel_theta_error.numpy()
    }


def plot_trajectory_comparison(x_pred, x_true, controls, dt, sample_idx, save_path):
    """
    Plot comparison of predicted vs true trajectory with proper angle handling.

    Args:
        x_pred: Predicted states (T, 4)
        x_true: True states (T, 4)
        controls: Control inputs (T-1, 1)
        dt: Time step
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
    mean_vel_theta_err = np.mean(errors['vel_theta'])
    max_vel_theta_err = np.max(errors['vel_theta'])
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

    plt.suptitle(f'pHNN Long Trajectory Prediction - Sample {sample_idx + 1}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    """Main function to visualize long trajectories."""
    print("=" * 80)
    print("pHNN Long Trajectory Visualization (100 steps, 5 samples)")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Load trained model
    weights_path = config['training']['model_save_path']
    model = load_model("cartpole_mpc_config.yaml", weights_path)

    # Load test data
    data_path = config['data']['save_path']
    print(f"\nLoading test data from {data_path}...")
    data = torch.load(data_path, weights_only=True)

    states = data['states']  # (num_traj, seq_len, 4)
    controls = data['controls']  # (num_traj, seq_len, 1)

    print(f"Data loaded: {states.shape[0]} trajectories, {states.shape[1]} steps each")

    dt = config['cartpole']['dt']

    # Select 5 diverse samples
    num_samples = 10
    num_traj = states.shape[0]

    # Sample evenly spaced trajectories for diversity
    sample_indices = np.linspace(0, num_traj - 1, num_samples, dtype=int)

    print(f"\nVisualizing {num_samples} sample trajectories (100 steps):")
    print(f"Sample indices: {sample_indices}")

    # Create output directory
    output_dir = Path("results/trajectory_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize each sample
    all_errors = []

    for i, idx in enumerate(sample_indices):
        print(f"\nProcessing sample {i + 1}/{num_samples} (trajectory {idx})...")

        # Get ground truth trajectory (first half of original length)
        x_true = states[idx]  # (seq_len, 4)
        half_len = len(x_true) // 2  # Integer division to get half length
        x_true = x_true[:half_len, :]  # Take first half

        u_true = controls[idx, :-1]  # (seq_len-1, 1)
        u_true = u_true[:half_len-1]  # Take first half (minus 1 since controls are one shorter)

        # Rollout pHNN prediction
        x0 = x_true[0]
        x_pred = rollout_trajectory(model, x0, u_true, dt)

        # Compute errors
        errors = compute_errors(x_pred, x_true)
        all_errors.append(errors)

        # Print summary
        print(f"  Mean angle error: {np.mean(errors['theta']):.4f} rad ({np.degrees(np.mean(errors['theta'])):.2f}°)")
        print(f"  Max angle error:  {np.max(errors['theta']):.4f} rad ({np.degrees(np.max(errors['theta'])):.2f}°)")
        print(f"  Final angle error: {errors['theta'][-1]:.4f} rad ({np.degrees(errors['theta'][-1]):.2f}°)")

        # Plot comparison
        save_path = output_dir / f"trajectory_sample_{i + 1}_idx_{idx}.png"
        plot_trajectory_comparison(x_pred, x_true, u_true, dt, i, save_path)

    # Aggregate statistics
    print("\n" + "=" * 80)
    print("Aggregate Statistics Across All Samples")
    print("=" * 80)

    # Combine errors
    all_pos_err = np.concatenate([e['pos'] for e in all_errors])
    all_theta_err = np.concatenate([e['theta'] for e in all_errors])
    all_vel_x_err = np.concatenate([e['vel_x'] for e in all_errors])
    all_vel_theta_err = np.concatenate([e['vel_theta'] for e in all_errors])

    print(f"\nPosition Error:")
    print(f"  Mean: {np.mean(all_pos_err):.4f} m")
    print(f"  Std:  {np.std(all_pos_err):.4f} m")
    print(f"  Max:  {np.max(all_pos_err):.4f} m")

    print(f"\nAngle Error (with wrapping):")
    print(f"  Mean: {np.mean(all_theta_err):.4f} rad ({np.degrees(np.mean(all_theta_err)):.2f}°)")
    print(f"  Std:  {np.std(all_theta_err):.4f} rad ({np.degrees(np.std(all_theta_err)):.2f}°)")
    print(f"  Max:  {np.max(all_theta_err):.4f} rad ({np.degrees(np.max(all_theta_err)):.2f}°)")

    print(f"\nCart Velocity Error:")
    print(f"  Mean: {np.mean(all_vel_x_err):.4f} m/s")
    print(f"  Std:  {np.std(all_vel_x_err):.4f} m/s")
    print(f"  Max:  {np.max(all_vel_x_err):.4f} m/s")

    print(f"\nAngular Velocity Error:")
    print(f"  Mean: {np.mean(all_vel_theta_err):.4f} rad/s")
    print(f"  Std:  {np.std(all_vel_theta_err):.4f} rad/s")
    print(f"  Max:  {np.max(all_vel_theta_err):.4f} rad/s")

    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)
    print(f"\nGenerated files in: {output_dir}/")
    print("  - trajectory_sample_1_idx_X.png")
    print("  - trajectory_sample_2_idx_X.png")
    print("  - trajectory_sample_3_idx_X.png")
    print("  - trajectory_sample_4_idx_X.png")
    print("  - trajectory_sample_5_idx_X.png")


if __name__ == "__main__":
    main()
