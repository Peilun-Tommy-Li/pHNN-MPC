"""
Rolling Horizon pHNN Prediction Animation

Creates an animation showing how the pHNN model predicts future trajectories
at each time step using a rolling horizon approach.

Usage:
    python scripts/visualize_rolling_horizon_prediction.py
    python scripts/visualize_rolling_horizon_prediction.py --horizon 15 --sample 0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
import sys
import argparse
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
    """Wrap angle to [-pi, pi] range.

    Args:
        angle: Angle in radians (can be tensor or numpy array)

    Returns:
        Wrapped angle in [-pi, pi]
    """
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


def create_rolling_horizon_animation(model, states, controls, dt, horizon=10,
                                     save_path='results/rolling_horizon_animation.mp4', fps=10):
    """
    Create rolling horizon prediction animation.

    Args:
        model: Trained pHNN model
        states: Ground truth states (T, 4) - tensor
        controls: Control inputs (T-1, 1) - tensor
        dt: Time step
        horizon: Prediction horizon
        save_path: Path to save animation
        fps: Frames per second
    """
    print(f"\nCreating rolling horizon animation (H={horizon})...")

    T_total = len(states) - 1  # Total timesteps (excluding last state)
    num_frames = T_total - horizon  # Number of animation frames

    # Create figure with 3 stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Subplot titles and labels
    axes[0].set_ylabel(r'Pole Angle $\theta$ (rad)', fontsize=12)
    axes[0].set_title(r'Rolling Horizon Prediction: Pole Angle $\theta$', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    axes[0].axhline(np.pi, color='gray', linestyle=':', linewidth=1, alpha=0.4, label='±π')
    axes[0].axhline(-np.pi, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    axes[1].set_ylabel(r'Pole Ang. Vel. $\dot{\theta}$ (rad/s)', fontsize=12)
    axes[1].set_title(r'Rolling Horizon Prediction: Pole Angular Velocity $\dot{\theta}$', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    axes[2].set_ylabel(r'Control Input $u$ (N)', fontsize=12)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_title(r'Control Input $u$', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    # Initialize line objects
    # History lines (solid)
    theta_hist_true, = axes[0].plot([], [], 'b-', linewidth=2, label='True (history)', alpha=0.9)
    theta_hist_pred, = axes[0].plot([], [], 'r--', linewidth=2, label='Predicted (history)', alpha=0.9)

    thetadot_hist_true, = axes[1].plot([], [], 'b-', linewidth=2, label='True (history)', alpha=0.9)
    thetadot_hist_pred, = axes[1].plot([], [], 'r--', linewidth=2, label='Predicted (history)', alpha=0.9)

    control_hist, = axes[2].plot([], [], 'k-', linewidth=2, label='Control (history)', alpha=0.9)

    # Current state markers (solid dots)
    theta_current, = axes[0].plot([], [], 'bo', markersize=10, label='Current state', zorder=5)
    thetadot_current, = axes[1].plot([], [], 'bo', markersize=10, label='Current state', zorder=5)
    control_current, = axes[2].plot([], [], 'ko', markersize=10, label='Current input', zorder=5)

    # Future predictions (faded/transparent)
    theta_future_true, = axes[0].plot([], [], 'b-', linewidth=1.5, alpha=0.4, label='True (future)')
    theta_future_pred, = axes[0].plot([], [], 'r--', linewidth=1.5, alpha=0.4, label='Predicted (future)')

    thetadot_future_true, = axes[1].plot([], [], 'b-', linewidth=1.5, alpha=0.4, label='True (future)')
    thetadot_future_pred, = axes[1].plot([], [], 'r--', linewidth=1.5, alpha=0.4, label='Predicted (future)')

    # Legends
    axes[0].legend(loc='upper right', fontsize=9)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[2].legend(loc='upper right', fontsize=9)

    # Time text
    time_text = fig.text(0.5, 0.96, '', ha='center', fontsize=14, fontweight='bold')

    # Store predictions for history
    all_predictions = []

    def init():
        """Initialize animation."""
        theta_hist_true.set_data([], [])
        theta_hist_pred.set_data([], [])
        thetadot_hist_true.set_data([], [])
        thetadot_hist_pred.set_data([], [])
        control_hist.set_data([], [])

        theta_current.set_data([], [])
        thetadot_current.set_data([], [])
        control_current.set_data([], [])

        theta_future_true.set_data([], [])
        theta_future_pred.set_data([], [])
        thetadot_future_true.set_data([], [])
        thetadot_future_pred.set_data([], [])

        time_text.set_text('')

        return (theta_hist_true, theta_hist_pred, thetadot_hist_true, thetadot_hist_pred,
                control_hist, theta_current, thetadot_current, control_current,
                theta_future_true, theta_future_pred, thetadot_future_true,
                thetadot_future_pred, time_text)

    def animate(frame):
        """Update animation for each frame."""
        t = frame  # Current time index

        # ===== Data Preparation =====
        # Current true state at time t
        x_true_current = states[t]  # (4,)

        # Future control sequence (next H steps)
        u_future = controls[t:t+horizon]  # (H, 1)

        # Future ground truth states
        x_true_future = states[t+1:t+horizon+1]  # (H, 4)

        # ===== Model Prediction =====
        # CRUCIAL: Reset model state to exact current true state
        x0 = x_true_current.clone()

        # Run open-loop rollout for H steps
        x_pred_future = rollout_prediction(model, x0, u_future, dt)  # (H+1, 4)

        # Store prediction for history (only the prediction part, not x0)
        all_predictions.append(x_pred_future[1:])  # (H, 4)

        # ===== Time arrays =====
        time_hist = np.arange(t+1) * dt  # History: [0, t]
        time_current = t * dt  # Current time
        time_future = np.arange(t+1, t+horizon+1) * dt  # Future: [t+1, t+H]

        # ===== Plot History (up to time t) =====
        # True history (wrap angles to [-pi, pi])
        theta_hist_true.set_data(time_hist, wrap_angle(states[:t+1, 1].numpy()))
        thetadot_hist_true.set_data(time_hist, states[:t+1, 3].numpy())

        # Predicted history (concatenate all previous predictions' first steps)
        if t > 0:
            # Take the first prediction from each previous timestep
            theta_pred_hist = np.array([all_predictions[i][0, 1].item() for i in range(t)])
            thetadot_pred_hist = np.array([all_predictions[i][0, 3].item() for i in range(t)])

            # Prepend initial state
            theta_pred_full = np.concatenate([[states[0, 1].item()], theta_pred_hist])
            thetadot_pred_full = np.concatenate([[states[0, 3].item()], thetadot_pred_hist])

            # Wrap angles to [-pi, pi]
            theta_hist_pred.set_data(time_hist, wrap_angle(theta_pred_full))
            thetadot_hist_pred.set_data(time_hist, thetadot_pred_full)
        else:
            theta_hist_pred.set_data([time_hist[0]], [wrap_angle(states[0, 1].item())])
            thetadot_hist_pred.set_data([time_hist[0]], [states[0, 3].item()])

        # Control history
        if t > 0:
            control_hist.set_data(time_hist[:-1], controls[:t, 0].numpy())

        # ===== Plot Current State (time t) =====
        theta_current.set_data([time_current], [wrap_angle(x_true_current[1].item())])
        thetadot_current.set_data([time_current], [x_true_current[3].item()])

        # Current control input
        if t < len(controls):
            control_current.set_data([time_current], [controls[t, 0].item()])

        # ===== Plot Future (faded/transparent) =====
        # Future ground truth (wrap angles to [-pi, pi])
        theta_future_true.set_data(time_future, wrap_angle(x_true_future[:, 1].numpy()))
        thetadot_future_true.set_data(time_future, x_true_future[:, 3].numpy())

        # Future predictions (wrap angles to [-pi, pi])
        theta_future_pred.set_data(time_future, wrap_angle(x_pred_future[1:, 1].numpy()))
        thetadot_future_pred.set_data(time_future, x_pred_future[1:, 3].numpy())

        # ===== Update axis limits (FIXED time axis) =====
        # Set fixed x-axis based on total simulation time
        total_time = (len(states) - 1) * dt
        x_min = 0
        x_max = total_time

        for ax in axes:
            ax.set_xlim(x_min, x_max)

        # Auto-scale y-axis based on ALL data (computed once)
        if t == 0:
            # Compute ranges for entire trajectory
            # Wrap theta to [-pi, pi] for proper range calculation
            theta_range = wrap_angle(states[:, 1].numpy())
            thetadot_range = states[:, 3].numpy()
            control_range = controls[:, 0].numpy()

            theta_margin = (theta_range.max() - theta_range.min()) * 0.2 + 0.1
            thetadot_margin = (thetadot_range.max() - thetadot_range.min()) * 0.2 + 0.1
            control_margin = (control_range.max() - control_range.min()) * 0.2 + 0.1

            axes[0].set_ylim(theta_range.min() - theta_margin, theta_range.max() + theta_margin)
            axes[1].set_ylim(thetadot_range.min() - thetadot_margin, thetadot_range.max() + thetadot_margin)
            axes[2].set_ylim(control_range.min() - control_margin, control_range.max() + control_margin)

        # Update time text
        time_text.set_text(f'Time: {time_current:.2f}s | Frame: {frame+1}/{num_frames} | Horizon: {horizon} steps')

        return (theta_hist_true, theta_hist_pred, thetadot_hist_true, thetadot_hist_pred,
                control_hist, theta_current, thetadot_current, control_current,
                theta_future_true, theta_future_pred, thetadot_future_true,
                thetadot_future_pred, time_text)

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=num_frames, interval=1000/fps, blit=True
    )

    # Save animation
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(save_path, writer=writer)
        print(f"✓ Animation saved to {save_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("  Make sure ffmpeg is installed")

    plt.close()
    return anim


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Rolling Horizon pHNN Prediction Animation')
    parser.add_argument('--config', type=str, default='cartpole_mpc_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--horizon', type=int, default=10,
                        help='Prediction horizon (default: 10 steps)')
    parser.add_argument('--sample', type=int, default=0,
                        help='Which trajectory sample to use (default: 0)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Animation frames per second (default: 10)')
    parser.add_argument('--output', type=str, default='results/rolling_horizon_animation.mp4',
                        help='Output animation path')
    args = parser.parse_args()

    print("=" * 80)
    print("Rolling Horizon pHNN Prediction Animation")
    print("=" * 80)

    # Load configuration
    config = load_config(args.config)

    # Load trained model
    weights_path = config['training']['model_save_path']
    model = load_model(args.config, weights_path)

    # Load test data
    data_path = config['data']['save_path']
    print(f"\nLoading test data from {data_path}...")
    data = torch.load(data_path, weights_only=True)

    states = data['states']  # (num_traj, seq_len, 4)
    controls = data['controls']  # (num_traj, seq_len, 1)

    print(f"✓ Data loaded: {states.shape[0]} trajectories, {states.shape[1]} steps each")

    # Select sample trajectory
    sample_idx = args.sample
    if sample_idx >= states.shape[0]:
        print(f"Warning: sample {sample_idx} out of range, using sample 0")
        sample_idx = 0

    # Get trajectory (first half as requested in previous modification)
    x_true = states[sample_idx]  # (seq_len, 4)
    half_len = len(x_true) // 2
    x_true = x_true[:half_len, :]  # (half_len, 4)

    u_true = controls[sample_idx, :-1]  # (seq_len-1, 1)
    u_true = u_true[:half_len-1]  # (half_len-1, 1)

    dt = config['cartpole']['dt']

    print(f"\nVisualization settings:")
    print(f"  Sample trajectory: {sample_idx}")
    print(f"  Trajectory length: {len(x_true)} steps ({len(x_true)*dt:.2f}s)")
    print(f"  Prediction horizon: {args.horizon} steps ({args.horizon*dt:.2f}s)")
    print(f"  Animation FPS: {args.fps}")

    # Create animation
    create_rolling_horizon_animation(
        model, x_true, u_true, dt,
        horizon=args.horizon,
        save_path=args.output,
        fps=args.fps
    )

    print("\n" + "=" * 80)
    print("Animation complete!")
    print("=" * 80)
    print(f"\nGenerated file: {args.output}")
    print("\nTo view with different settings:")
    print(f"  python {__file__} --horizon 15 --sample 5 --fps 15")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
