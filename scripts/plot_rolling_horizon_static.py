"""
Static plot visualization of rolling horizon predictions.

Shows multiple prediction tasks (every N steps) on the same figure.
Each prediction task shows:
- Initial condition point (current state)
- Predicted trajectory (dashed line)
- Ground truth trajectory (solid line)
- Corresponding control inputs
- Vertical dotted lines connecting predictions/ground truth to control inputs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
import argparse
from pathlib import Path
import sys

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
    checkpoint = torch.load(weights_path, map_location='cpu')

    # Handle both dict and direct state_dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✓ Model loaded successfully!")
    return model


def wrap_angle(angle):
    """Wrap angle to [-pi, pi] range."""
    if isinstance(angle, torch.Tensor):
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    else:
        return np.arctan2(np.sin(angle), np.cos(angle))


def rollout_prediction(model, x0, controls, dt):
    """
    Rollout pHNN predictions for given controls starting from x0.

    Args:
        model: Trained pHNN model
        x0: Initial state (4,) - tensor
        controls: Control sequence (H, 1) - tensor
        dt: Time step

    Returns:
        Predicted states (H+1, 4) - tensor (includes initial state)
    """
    x_pred = [x0.unsqueeze(0).detach()]  # (1, 4)

    for t in range(len(controls)):
        # Get current state (requires grad for Hamiltonian computation in pHNN)
        x_current = x_pred[-1].clone().requires_grad_(True)
        u_current = controls[t].unsqueeze(0).clone()

        # Compute derivative (pHNN needs gradients internally)
        dx, _ = model(x_current, u_current)

        # Integrate forward (detach to avoid building computation graph)
        x_next = x_pred[-1] + dt * dx.detach()
        x_pred.append(x_next.detach())

    return torch.cat(x_pred, dim=0)  # (H+1, 4)


def plot_rolling_horizon_static(model, states, controls, dt, horizon=10,
                                prediction_interval=20, save_path='results/rolling_horizon_static.png'):
    """
    Create static plot showing multiple rolling horizon predictions.

    Args:
        model: Trained pHNN model
        states: Ground truth states (T, 4) - tensor
        controls: Control inputs (T-1, 1) - tensor
        dt: Time step
        horizon: Prediction horizon
        prediction_interval: Steps between prediction tasks
        save_path: Path to save figure
    """
    print(f"\nCreating rolling horizon static plot...")
    print(f"  Horizon: {horizon} steps")
    print(f"  Prediction interval: {prediction_interval} steps")

    # Determine prediction starting points
    T_total = len(states) - 1
    prediction_starts = list(range(0, T_total - horizon, prediction_interval))
    num_predictions = len(prediction_starts)

    print(f"  Number of predictions: {num_predictions}")
    print(f"  Prediction starts: {prediction_starts}")

    # Create figure with 3 stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Overall time axis
    total_time = (len(states) - 1) * dt
    time_full = np.arange(len(states)) * dt

    # Plot full ground truth trajectory (faded background)
    axes[0].plot(time_full, wrap_angle(states[:, 1].numpy()),
                color='lightgray', linewidth=1, alpha=0.5, label='Full Ground Truth')
    axes[1].plot(time_full, states[:, 3].numpy(),
                color='lightgray', linewidth=1, alpha=0.5, label='Full Ground Truth')

    time_control_full = np.arange(len(controls)) * dt
    axes[2].plot(time_control_full, controls[:, 0].numpy(),
                color='lightgray', linewidth=1, alpha=0.5, label='Full Control')

    # Color palette for different predictions
    colors = plt.cm.viridis(np.linspace(0, 0.9, num_predictions))

    # Process each prediction task
    for i, t_start in enumerate(prediction_starts):
        print(f"\n  Processing prediction {i+1}/{num_predictions} (t={t_start})...")

        # Extract data for this prediction
        t_end = min(t_start + horizon, T_total)
        actual_horizon = t_end - t_start

        # Initial condition
        x0 = states[t_start]

        # Control sequence and ground truth
        u_seq = controls[t_start:t_end]
        x_true_seq = states[t_start:t_end+1]  # Includes initial state

        # Run prediction
        x_pred_seq = rollout_prediction(model, x0, u_seq, dt)

        # Time arrays for this prediction
        time_pred = np.arange(t_start, t_end + 1) * dt
        time_control = np.arange(t_start, t_end) * dt

        color = colors[i]
        label_true = f'Ground Truth (t={t_start})' if i < 3 else None
        label_pred = f'Prediction (t={t_start})' if i < 3 else None

        # 1. Pole Angle subplot
        ax = axes[0]

        # Plot initial condition (large marker)
        ax.plot(time_pred[0], wrap_angle(x_true_seq[0, 1].item()),
               'o', color=color, markersize=10, zorder=10,
               label=f'IC (t={t_start})' if i < 3 else None)

        # Plot ground truth
        ax.plot(time_pred, wrap_angle(x_true_seq[:, 1].numpy()),
               '-', color=color, linewidth=2, alpha=0.8, label=label_true)

        # Plot prediction
        ax.plot(time_pred, wrap_angle(x_pred_seq[:, 1].numpy()),
               '--', color=color, linewidth=2, alpha=0.8, label=label_pred)

        # 2. Pole Angular Velocity subplot
        ax = axes[1]

        # Plot initial condition
        ax.plot(time_pred[0], x_true_seq[0, 3].item(),
               'o', color=color, markersize=10, zorder=10)

        # Plot ground truth
        ax.plot(time_pred, x_true_seq[:, 3].numpy(),
               '-', color=color, linewidth=2, alpha=0.8)

        # Plot prediction
        ax.plot(time_pred, x_pred_seq[:, 3].numpy(),
               '--', color=color, linewidth=2, alpha=0.8)

        # 3. Control Input subplot (plot the corresponding controls)
        ax = axes[2]
        ax.plot(time_control, u_seq[:, 0].numpy(),
               '-', color=color, linewidth=2.5, alpha=0.9)

        # Add vertical dotted lines connecting state predictions to control inputs
        # Draw at the start and end of each prediction window
        for ax in axes[:2]:  # Only for state plots
            ax.axvline(time_pred[0], color=color, linestyle=':', alpha=0.3, linewidth=1)
            ax.axvline(time_pred[-1], color=color, linestyle=':', alpha=0.3, linewidth=1)

        # Vertical lines for control plot
        axes[2].axvline(time_pred[0], color=color, linestyle=':', alpha=0.3, linewidth=1)
        axes[2].axvline(time_pred[-1], color=color, linestyle=':', alpha=0.3, linewidth=1)

    # Configure axes
    axes[0].set_ylabel(r'Pole Angle $\theta$ (rad)', fontsize=13)
    axes[0].set_title(r'Rolling Horizon Predictions: Pole Angle $\theta$',
                     fontsize=14, fontweight='bold')
    axes[0].axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axes[0].axhline(np.pi, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    axes[0].axhline(-np.pi, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=9, ncol=2)

    axes[1].set_ylabel(r'Pole Ang. Vel. $\dot{\theta}$ (rad/s)', fontsize=13)
    axes[1].set_title(r'Rolling Horizon Predictions: Pole Angular Velocity $\dot{\theta}$',
                     fontsize=14, fontweight='bold')
    axes[1].axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel(r'Control Input $u$ (N)', fontsize=13)
    axes[2].set_xlabel('Time (s)', fontsize=13)
    axes[2].set_title(r'Control Input $u$ (Corresponding to Each Prediction Window)',
                     fontsize=14, fontweight='bold')
    axes[2].axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axes[2].grid(True, alpha=0.3)

    # Set x-axis limits
    for ax in axes:
        ax.set_xlim(0, total_time)

    # Add text box with information
    info_text = (f'Horizon: {horizon} steps ({horizon*dt:.2f}s)\n'
                f'Prediction interval: {prediction_interval} steps\n'
                f'Total predictions: {num_predictions}\n'
                f'Solid lines: Ground truth\n'
                f'Dashed lines: pHNN predictions\n'
                f'Dots: Initial conditions\n'
                f'Vertical dotted lines: Prediction windows')

    axes[0].text(0.02, 0.02, info_text, transform=axes[0].transAxes,
                verticalalignment='bottom', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.suptitle(f'Rolling Horizon pHNN Predictions (Multiple Prediction Tasks)',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Figure saved to {save_path}")
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create static rolling horizon plot')
    parser.add_argument('--config', type=str, default='cartpole_mpc_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--horizon', type=int, default=10,
                       help='Prediction horizon (number of steps)')
    parser.add_argument('--interval', type=int, default=20,
                       help='Steps between prediction tasks')
    parser.add_argument('--sample', type=int, default=0,
                       help='Sample trajectory index to visualize')
    parser.add_argument('--output', type=str, default='results/rolling_horizon_static.png',
                       help='Output figure path')

    args = parser.parse_args()

    print("=" * 80)
    print("Rolling Horizon pHNN Prediction - Static Plot")
    print("=" * 80)

    # Load configuration
    config = load_config(args.config)
    dt = config['cartpole']['dt']
    model_path = config.get('model', {}).get('checkpoint', 'models/checkpoint_epoch_860.pth')

    # Load model
    model = load_model(args.config, model_path)

    # Load data
    data_path = config['data']['save_path']
    print(f"\nLoading test data from {data_path}...")
    data = torch.load(data_path, weights_only=True)

    states = data['states']  # (num_traj, seq_len, 4)
    controls = data['controls']  # (num_traj, seq_len, 1)

    print(f"✓ Data loaded: {states.shape[0]} trajectories, {states.shape[1]} steps each")

    # Select sample trajectory
    sample_idx = args.sample
    print(f"\nUsing trajectory: {sample_idx}")

    # Get trajectory (first half)
    x_traj = states[sample_idx]
    half_len = len(x_traj) // 2
    x_traj = x_traj[:half_len, :]

    u_traj = controls[sample_idx, :-1]
    u_traj = u_traj[:half_len-1]

    traj_length = len(x_traj)
    print(f"Trajectory length: {traj_length} steps ({(traj_length-1)*dt:.2f}s)")

    # Visualization settings
    print(f"\nVisualization settings:")
    print(f"  Prediction horizon: {args.horizon} steps ({args.horizon*dt:.2f}s)")
    print(f"  Prediction interval: {args.interval} steps ({args.interval*dt:.2f}s)")

    # Create static plot
    plot_rolling_horizon_static(
        model=model,
        states=x_traj,
        controls=u_traj,
        dt=dt,
        horizon=args.horizon,
        prediction_interval=args.interval,
        save_path=args.output
    )

    print("\n" + "=" * 80)
    print("Static Plot Complete!")
    print("=" * 80)
    print(f"\nGenerated file: {args.output}")
    print("\nTo view with different settings:")
    print(f"  python {__file__} --horizon 15 --interval 10 --sample 5")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
