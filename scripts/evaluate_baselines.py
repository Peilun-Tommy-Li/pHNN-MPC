"""
Evaluate and compare baseline models (MLP, Neural ODE) with pHNN.

Compares models on:
1. One-step prediction accuracy
2. Multi-step trajectory rollout
3. Long-horizon prediction error
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


def load_phnn_model(config_path, weights_path):
    """Load pHNN model."""
    print(f"Loading pHNN model from {weights_path}...")
    model = pHNN(config_path)
    checkpoint = torch.load(weights_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✓ pHNN model loaded")
    return model


def rollout_mlp(model, x0, controls):
    """Rollout MLP trajectory."""
    states = [x0.unsqueeze(0)]
    for u in controls:
        x_next = model(states[-1], u.unsqueeze(0))
        states.append(x_next)
    return torch.cat(states, dim=0)


def rollout_node(model, x0, controls, dt):
    """Rollout Neural ODE trajectory."""
    states = [x0.unsqueeze(0)]
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


def compute_trajectory_errors(pred_states, true_states):
    """Compute per-dimension trajectory errors."""
    errors = torch.abs(pred_states - true_states)
    return {
        'x': errors[:, 0].numpy(),
        'theta': errors[:, 1].numpy(),
        'x_dot': errors[:, 2].numpy(),
        'theta_dot': errors[:, 3].numpy(),
        'total': errors.mean(dim=1).numpy()
    }


def evaluate_models(models, data, dt, num_samples=10):
    """
    Evaluate all models on trajectory rollout.

    Args:
        models: Dict of {model_name: model}
        data: Data dict with states and controls
        dt: Time step
        num_samples: Number of trajectories to evaluate

    Returns:
        results: Dict of evaluation results
    """
    states = data['states']
    controls = data['controls']

    num_traj = states.shape[0]
    sample_indices = np.linspace(0, num_traj - 1, num_samples, dtype=int)

    results = {name: [] for name in models.keys()}
    results['ground_truth'] = []

    print(f"\nEvaluating models on {num_samples} trajectories...")

    for i, idx in enumerate(sample_indices):
        # Get trajectory (first half)
        x_true = states[idx]
        half_len = len(x_true) // 2
        x_true = x_true[:half_len, :]

        u_true = controls[idx, :-1]
        u_true = u_true[:half_len-1]

        x0 = x_true[0]

        # Store ground truth
        results['ground_truth'].append(x_true.numpy())

        # Rollout each model
        for name, model in models.items():
            with torch.no_grad():
                if 'mlp' in name.lower():
                    pred_states = rollout_mlp(model, x0, u_true)
                elif 'node' in name.lower():
                    pred_states = rollout_node(model, x0, u_true, dt)
                elif 'phnn' in name.lower():
                    pred_states = rollout_phnn(model, x0, u_true, dt)
                else:
                    raise ValueError(f"Unknown model type: {name}")

            # Compute errors
            errors = compute_trajectory_errors(pred_states, x_true)
            results[name].append(errors)

        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{num_samples} trajectories")

    return results


def plot_comparison(results, dt, save_path='results/baseline_comparison.png'):
    """Plot model comparison."""
    num_samples = len(results['ground_truth'])
    model_names = [k for k in results.keys() if k != 'ground_truth']

    # Average errors over all samples
    avg_errors = {}
    for name in model_names:
        # Stack all trajectories
        all_errors = {
            'x': [],
            'theta': [],
            'x_dot': [],
            'theta_dot': [],
            'total': []
        }
        for traj_errors in results[name]:
            for key in all_errors:
                all_errors[key].append(traj_errors[key])

        # Average across trajectories (padding to same length)
        max_len = max(len(e) for e in all_errors['total'])
        avg_errors[name] = {}
        for key in all_errors:
            padded = []
            for err in all_errors[key]:
                if len(err) < max_len:
                    err = np.concatenate([err, np.full(max_len - len(err), np.nan)])
                padded.append(err)
            avg_errors[name][key] = np.nanmean(np.array(padded), axis=0)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison: Trajectory Rollout Errors', fontsize=16, fontweight='bold')

    variables = ['x', 'theta', 'x_dot', 'theta_dot']
    titles = ['Cart Position x', 'Pole Angle θ', 'Cart Velocity ẋ', 'Pole Angular Velocity θ̇']
    colors = {'mlp': 'orange', 'node': 'green', 'phnn': 'blue'}

    for ax, var, title in zip(axes.flat, variables, titles):
        for name in model_names:
            errors = avg_errors[name][var]
            time = np.arange(len(errors)) * dt
            color = colors.get(name.lower(), 'gray')
            ax.plot(time, errors, label=name.upper(), linewidth=2, color=color, alpha=0.8)

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Absolute Error', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {save_path}")
    plt.close()


def print_summary(results):
    """Print summary statistics."""
    model_names = [k for k in results.keys() if k != 'ground_truth']

    print("\n" + "=" * 80)
    print("Model Performance Summary (Average across all test trajectories)")
    print("=" * 80)

    # Compute mean errors
    for name in model_names:
        print(f"\n{name.upper()}:")

        all_final_errors = []
        for traj_errors in results[name]:
            # Final time step error
            final_error = traj_errors['total'][-1]
            all_final_errors.append(final_error)

        mean_final_error = np.mean(all_final_errors)
        std_final_error = np.std(all_final_errors)

        print(f"  Final timestep error: {mean_final_error:.6f} ± {std_final_error:.6f}")

        # Average over entire trajectory
        all_mean_errors = []
        for traj_errors in results[name]:
            all_mean_errors.append(np.mean(traj_errors['total']))

        mean_traj_error = np.mean(all_mean_errors)
        std_traj_error = np.std(all_mean_errors)

        print(f"  Mean trajectory error: {mean_traj_error:.6f} ± {std_traj_error:.6f}")

    print("=" * 80 + "\n")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate baseline models')
    parser.add_argument('--mlp_checkpoint', type=str, default='baseline/mlp/best_model.pth',
                        help='Path to MLP checkpoint')
    parser.add_argument('--node_checkpoint', type=str, default='baseline/node/best_model.pth',
                        help='Path to Neural ODE checkpoint')
    parser.add_argument('--phnn_checkpoint', type=str, default='models/checkpoint_epoch_860.pth',
                        help='Path to pHNN checkpoint')
    parser.add_argument('--config', type=str, default='cartpole_mpc_config.yaml',
                        help='Configuration file')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of test trajectories')
    parser.add_argument('--output', type=str, default='results/baseline_comparison.png',
                        help='Output plot path')

    args = parser.parse_args()

    print("=" * 80)
    print("Baseline Model Evaluation")
    print("=" * 80)

    # Load configuration
    config = load_config(args.config)
    dt = config['cartpole']['dt']

    # Load data
    data_path = config['data']['save_path']
    print(f"\nLoading data from {data_path}...")
    data = torch.load(data_path, weights_only=True)
    print(f"✓ Data loaded: {data['states'].shape[0]} trajectories")

    # Load models
    models = {}

    if Path(args.mlp_checkpoint).exists():
        models['MLP'] = load_baseline_model('mlp', args.mlp_checkpoint, config)
    else:
        print(f"Warning: MLP checkpoint not found at {args.mlp_checkpoint}")

    if Path(args.node_checkpoint).exists():
        models['NODE'] = load_baseline_model('node', args.node_checkpoint, config)
    else:
        print(f"Warning: Neural ODE checkpoint not found at {args.node_checkpoint}")

    if Path(args.phnn_checkpoint).exists():
        models['pHNN'] = load_phnn_model(args.config, args.phnn_checkpoint)
    else:
        print(f"Warning: pHNN checkpoint not found at {args.phnn_checkpoint}")

    if len(models) == 0:
        print("Error: No models found!")
        return

    # Evaluate models
    results = evaluate_models(models, data, dt, num_samples=args.num_samples)

    # Print summary
    print_summary(results)

    # Plot comparison
    plot_comparison(results, dt, save_path=args.output)

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
