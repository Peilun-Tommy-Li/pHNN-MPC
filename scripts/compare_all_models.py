"""
Compare all three models (MLP, Neural ODE, pHNN) side by side.

Creates comprehensive comparison plots showing trajectory predictions
for all models on the same test trajectories.
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
from pHNN import pHNN


def load_config(config_path="cartpole_mpc_config.yaml"):
    """Load configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def wrap_angle(angle):
    """Wrap angle to [-pi, pi] range."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def load_mlp(checkpoint_path):
    """Load MLP model."""
    print(f"Loading MLP from {checkpoint_path}...")
    model = VanillaMLP(
        state_dim=4, action_dim=1,
        hidden_sizes=[256, 256, 256, 256],
        activation='relu', use_residual=True,
        dropout=0.1, layer_norm=False
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ Loaded (epoch {checkpoint['epoch']})")
    return model


def load_node(checkpoint_path):
    """Load Neural ODE model."""
    print(f"Loading Neural ODE from {checkpoint_path}...")
    model = NeuralODE(
        state_dim=4, action_dim=1,
        hidden_sizes=[128, 128, 128],
        activation='tanh', layer_norm=False,
        solver='dopri5', rtol=1e-3, atol=1e-4
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ Loaded (epoch {checkpoint['epoch']})")
    return model


def load_phnn(config_path, checkpoint_path):
    """Load pHNN model."""
    print(f"Loading pHNN from {checkpoint_path}...")
    model = pHNN(config_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"  ✓ Loaded")
    return model


def rollout_mlp(model, x0, controls):
    """Rollout MLP trajectory."""
    states = [x0.unsqueeze(0)]
    with torch.no_grad():
        for u in controls:
            x_next = model(states[-1], u.unsqueeze(0))
            states.append(x_next)
    return torch.cat(states, dim=0)


def rollout_node(model, x0, controls, dt):
    """Rollout Neural ODE trajectory."""
    states = [x0.unsqueeze(0)]
    with torch.no_grad():
        for u in controls:
            x_next = model(states[-1], u.unsqueeze(0), dt)
            states.append(x_next)
    return torch.cat(states, dim=0)


def rollout_phnn(model, x0, controls, dt):
    """Rollout pHNN trajectory."""
    states = [x0.unsqueeze(0).detach()]
    for u in controls:
        x_current = states[-1].clone().requires_grad_(True)
        u_current = u.unsqueeze(0).clone()
        dx, _ = model(x_current, u_current)
        x_next = states[-1] + dt * dx.detach()
        states.append(x_next.detach())
    return torch.cat(states, dim=0)


def plot_comparison(x_true, x_mlp, x_node, x_phnn, controls, dt, sample_idx, save_path):
    """
    Create comparison plot for all models.

    Args:
        x_true: Ground truth states (seq_len, 4)
        x_mlp: MLP predictions (seq_len+1, 4)
        x_node: Neural ODE predictions (seq_len+1, 4)
        x_phnn: pHNN predictions (seq_len+1, 4)
        controls: Control inputs (seq_len-1, 1)
        dt: Time step
        sample_idx: Sample index
        save_path: Save path
    """
    time = np.arange(len(x_true)) * dt

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Color scheme
    colors = {'true': 'black', 'mlp': 'orange', 'node': 'green', 'phnn': 'blue'}

    # 1. Pole Angle θ (top-left)
    ax = axes[0, 0]
    ax.plot(time, wrap_angle(x_true[:, 1].numpy()),
            color=colors['true'], linewidth=2.5, label='Ground Truth', alpha=0.9)
    ax.plot(time, wrap_angle(x_mlp[:, 1].numpy()),
            color=colors['mlp'], linewidth=2, linestyle='--', label='MLP', alpha=0.8)
    ax.plot(time, wrap_angle(x_node[:, 1].numpy()),
            color=colors['node'], linewidth=2, linestyle='--', label='Neural ODE', alpha=0.8)
    ax.plot(time, wrap_angle(x_phnn[:, 1].numpy()),
            color=colors['phnn'], linewidth=2, linestyle='--', label='pHNN', alpha=0.8)

    ax.axhline(np.pi, color='gray', linestyle=':', alpha=0.4)
    ax.axhline(-np.pi, color='gray', linestyle=':', alpha=0.4)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Pole Angle θ (rad)', fontsize=11)
    ax.set_title('Pole Angle θ', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Compute errors
    mlp_err = np.mean(np.abs(wrap_angle(x_mlp[:, 1].numpy()) - wrap_angle(x_true[:, 1].numpy())))
    node_err = np.mean(np.abs(wrap_angle(x_node[:, 1].numpy()) - wrap_angle(x_true[:, 1].numpy())))
    phnn_err = np.mean(np.abs(wrap_angle(x_phnn[:, 1].numpy()) - wrap_angle(x_true[:, 1].numpy())))

    ax.text(0.02, 0.02,
            f'Mean Errors:\n  MLP: {mlp_err:.4f}\n  NODE: {node_err:.4f}\n  pHNN: {phnn_err:.4f}',
            transform=ax.transAxes, verticalalignment='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2. Pole Angular Velocity θ̇ (top-right)
    ax = axes[0, 1]
    ax.plot(time, x_true[:, 3].numpy(),
            color=colors['true'], linewidth=2.5, label='Ground Truth', alpha=0.9)
    ax.plot(time, x_mlp[:, 3].numpy(),
            color=colors['mlp'], linewidth=2, linestyle='--', label='MLP', alpha=0.8)
    ax.plot(time, x_node[:, 3].numpy(),
            color=colors['green'], linewidth=2, linestyle='--', label='Neural ODE', alpha=0.8)
    ax.plot(time, x_phnn[:, 3].numpy(),
            color=colors['phnn'], linewidth=2, linestyle='--', label='pHNN', alpha=0.8)

    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Pole Angular Velocity θ̇ (rad/s)', fontsize=11)
    ax.set_title('Pole Angular Velocity θ̇', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Compute errors
    mlp_err = np.mean(np.abs(x_mlp[:, 3].numpy() - x_true[:, 3].numpy()))
    node_err = np.mean(np.abs(x_node[:, 3].numpy() - x_true[:, 3].numpy()))
    phnn_err = np.mean(np.abs(x_phnn[:, 3].numpy() - x_true[:, 3].numpy()))

    ax.text(0.02, 0.02,
            f'Mean Errors:\n  MLP: {mlp_err:.4f}\n  NODE: {node_err:.4f}\n  pHNN: {phnn_err:.4f}',
            transform=ax.transAxes, verticalalignment='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. Control Input (bottom-left)
    ax = axes[1, 0]
    time_control = np.arange(len(controls)) * dt
    ax.plot(time_control, controls.numpy(),
            color='black', linewidth=2, label='Control Force', alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Control Force (N)', fontsize=11)
    ax.set_title('Control Input', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Energy Evolution (bottom-right)
    ax = axes[1, 1]

    # Compute energies
    ke_true = 0.5 * (x_true[:, 2]**2 + x_true[:, 3]**2)
    pe_true = 1 - torch.cos(x_true[:, 1])
    energy_true = ke_true + pe_true

    ke_mlp = 0.5 * (x_mlp[:, 2]**2 + x_mlp[:, 3]**2)
    pe_mlp = 1 - torch.cos(x_mlp[:, 1])
    energy_mlp = ke_mlp + pe_mlp

    ke_node = 0.5 * (x_node[:, 2]**2 + x_node[:, 3]**2)
    pe_node = 1 - torch.cos(x_node[:, 1])
    energy_node = ke_node + pe_node

    ke_phnn = 0.5 * (x_phnn[:, 2]**2 + x_phnn[:, 3]**2)
    pe_phnn = 1 - torch.cos(x_phnn[:, 1])
    energy_phnn = ke_phnn + pe_phnn

    ax.plot(time, energy_true.numpy(),
            color=colors['true'], linewidth=2.5, label='Ground Truth', alpha=0.9)
    ax.plot(time, energy_mlp.numpy(),
            color=colors['mlp'], linewidth=2, linestyle='--', label='MLP', alpha=0.8)
    ax.plot(time, energy_node.numpy(),
            color=colors['green'], linewidth=2, linestyle='--', label='Neural ODE', alpha=0.8)
    ax.plot(time, energy_phnn.numpy(),
            color=colors['phnn'], linewidth=2, linestyle='--', label='pHNN', alpha=0.8)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Approximate Energy', fontsize=11)
    ax.set_title('Energy Evolution', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Compute energy errors
    mlp_err = np.mean(np.abs(energy_mlp.numpy() - energy_true.numpy()))
    node_err = np.mean(np.abs(energy_node.numpy() - energy_true.numpy()))
    phnn_err = np.mean(np.abs(energy_phnn.numpy() - energy_true.numpy()))

    ax.text(0.02, 0.98,
            f'Mean Errors:\n  MLP: {mlp_err:.4f}\n  NODE: {node_err:.4f}\n  pHNN: {phnn_err:.4f}',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f'Model Comparison - Sample {sample_idx + 1}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare all models')
    parser.add_argument('--mlp_checkpoint', type=str, default='baseline/mlp/best_model.pth')
    parser.add_argument('--node_checkpoint', type=str, default='baseline/node/best_model.pth')
    parser.add_argument('--phnn_checkpoint', type=str, default='models/checkpoint_epoch_860.pth')
    parser.add_argument('--config', type=str, default='cartpole_mpc_config.yaml')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='results/model_comparison')
    args = parser.parse_args()

    print("=" * 80)
    print("Comparing MLP, Neural ODE, and pHNN")
    print("=" * 80)

    # Load configuration
    config = load_config(args.config)
    dt = config['cartpole']['dt']

    # Load models
    mlp = load_mlp(args.mlp_checkpoint)
    node = load_node(args.node_checkpoint)
    phnn = load_phnn(args.config, args.phnn_checkpoint)

    # Load data
    data_path = config['data']['save_path']
    print(f"\nLoading data from {data_path}...")
    data = torch.load(data_path, weights_only=True)
    states = data['states']
    controls = data['controls']
    print(f"✓ Data loaded: {states.shape[0]} trajectories")

    # Select samples
    num_samples = min(args.num_samples, states.shape[0])
    sample_indices = np.linspace(0, states.shape[0] - 1, num_samples, dtype=int)
    print(f"\nComparing {num_samples} trajectories: {sample_indices}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each sample
    for i, idx in enumerate(sample_indices):
        print(f"\nProcessing sample {i + 1}/{num_samples} (trajectory {idx})...")

        # Get trajectory (first half)
        x_true = states[idx]
        half_len = len(x_true) // 2
        x_true = x_true[:half_len, :]

        u_true = controls[idx, :-1]
        u_true = u_true[:half_len-1]

        x0 = x_true[0]

        # Rollout all models
        x_mlp = rollout_mlp(mlp, x0, u_true)
        x_node = rollout_node(node, x0, u_true, dt)
        x_phnn = rollout_phnn(phnn, x0, u_true, dt)

        # Plot comparison
        save_path = output_dir / f"comparison_sample_{i+1}_idx_{idx}.png"
        plot_comparison(x_true, x_mlp, x_node, x_phnn, u_true, dt, i, save_path)

    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print("=" * 80)
    print(f"\nGenerated {num_samples} comparison plots in: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
