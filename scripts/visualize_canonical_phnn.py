"""
Visualization tools for canonical pHNN model.

Provides:
1. Mass matrix parameter monitoring (a, b, c)
2. Trajectory prediction vs ground truth (q and p)
3. Velocity reconstruction comparison (reconstructed vs actual)
4. Energy plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.append('src')

from pHNN_canonical import pHNN_Canonical
from TrajectoryStepDataset import TrajectoryStepDataset
from integrators import rollout_trajectory_differentiable


def print_mass_matrix_params(model):
    """
    Print current learned mass matrix parameters.

    For CartPoleMassMatrix: M(θ) = [[a, b*cos(θ)], [b*cos(θ), c]]
    """
    print("\n" + "="*70)
    print("Mass Matrix Parameters")
    print("="*70)

    M_net = model.M_net

    if hasattr(M_net, 'log_a'):
        # CartPoleMassMatrix
        with torch.no_grad():
            a = torch.exp(M_net.log_a).item() + 1e-3
            b = M_net.b.item()
            c = torch.exp(M_net.log_c).item() + 1e-3

        print(f"\nStructure: M(θ) = [[a, b*cos(θ)], [b*cos(θ), c]]")
        print(f"\nLearned parameters:")
        print(f"  a = {a:.6f}  (cart mass term)")
        print(f"  b = {b:.6f}  (coupling term)")
        print(f"  c = {c:.6f}  (pole inertia term)")

        # Check positive definiteness condition
        det = a * c - b**2
        print(f"\nPositive definiteness:")
        print(f"  det(M) = a*c - b² = {det:.6f}")
        if det > 0:
            print(f"  ✓ Matrix is positive definite (det > 0)")
        else:
            print(f"  ✗ WARNING: Matrix is NOT positive definite!")

        # Physical interpretation
        print(f"\nPhysical interpretation:")
        print(f"  Expected a ≈ 1.0 (cart mass)")
        print(f"  Expected b ≈ 0.5 (pole mass × pole length / 2)")
        print(f"  Expected c ≈ 0.33 (pole inertia)")
    else:
        print("Generic mass matrix (not CartPoleMassMatrix)")

    print("="*70 + "\n")


def print_dissipation_params(model):
    """Print diagonal dissipation matrix parameters."""
    print("\n" + "="*70)
    print("Dissipation Matrix Parameters")
    print("="*70)

    with torch.no_grad():
        R_diag = torch.nn.functional.softplus(model.R_diag_raw) + 1e-4

    print(f"\nStructure: R = diag(r1, r2, r3, r4)")
    print(f"\nLearned diagonal values:")
    for i, val in enumerate(R_diag):
        coord_name = ['x', 'θ', 'p_x', 'p_θ'][i] if i < 4 else f'{i}'
        print(f"  r_{i+1} ({coord_name:3s}) = {val.item():.6f}")

    print("="*70 + "\n")


def visualize_trajectory_predictions(
    model,
    dataset,
    num_samples=3,
    num_steps=50,
    dt=0.02,
    save_path=None
):
    """
    Visualize predicted trajectories vs ground truth.

    Shows:
    - Position coordinates (x, θ)
    - Momentum coordinates (p_x, p_θ)
    - Velocity reconstruction comparison

    Args:
        model: Trained pHNN_Canonical model
        dataset: TrajectoryStepDataset
        num_samples: Number of trajectories to plot
        num_steps: Number of steps to predict
        dt: Time step
        save_path: Optional path to save figure
    """
    model.eval()

    # Get random trajectory indices (not flat sample indices)
    traj_indices = np.random.choice(dataset.num_traj, num_samples, replace=False)

    fig, axes = plt.subplots(4, num_samples, figsize=(5*num_samples, 12))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    for idx, traj_idx in enumerate(traj_indices):
        # Get full trajectory
        traj = dataset.states[traj_idx]  # (timesteps, state_dim)
        controls = dataset.inputs[traj_idx]  # (timesteps, input_dim)

        # Truncate to num_steps
        max_steps = min(num_steps, len(traj) - 1)
        traj = traj[:max_steps+1]
        controls = controls[:max_steps]

        # Initial state
        y0 = traj[0:1]  # (1, state_dim)

        # Rollout model prediction
        # Note: Cannot use no_grad() because model needs gradients for Hamiltonian
        y_pred = rollout_trajectory_differentiable(
            model=model,
            y0=y0.requires_grad_(True),
            controls=controls.unsqueeze(0),  # (1, max_steps, input_dim)
            dt=dt,
            integrator='rk4',
            return_energies=False
        )

        # Convert to numpy
        y_pred = y_pred[0].detach().numpy()  # (num_steps+1, state_dim)
        y_true = traj.numpy()  # (num_steps+1, state_dim)

        # Split into position and velocity
        q_true = y_true[:, :2]  # (num_steps+1, 2)
        q_dot_true = y_true[:, 2:]  # (num_steps+1, 2)

        q_pred = y_pred[:, :2]
        q_dot_pred = y_pred[:, 2:]

        # Convert predicted velocities to momentum for comparison
        # We need to compute p = M(q) @ q_dot for predicted trajectory
        p_pred_list = []
        p_true_list = []
        q_dot_recon_list = []

        for t in range(len(y_true)):
            y_t = torch.tensor(y_true[t:t+1], dtype=torch.float32)

            # Get momentum from model's coordinate transform
            with torch.no_grad():
                from coordinate_transforms import kinematic_to_canonical, canonical_to_kinematic, split_state

                # True trajectory in canonical coords
                z_true_t = kinematic_to_canonical(y_t, model.M_net)
                _, p_true_t = split_state(z_true_t)
                p_true_list.append(p_true_t.numpy()[0])

                # Predicted trajectory in canonical coords
                y_pred_t = torch.tensor(y_pred[t:t+1], dtype=torch.float32)
                z_pred_t = kinematic_to_canonical(y_pred_t, model.M_net)
                _, p_pred_t = split_state(z_pred_t)
                p_pred_list.append(p_pred_t.numpy()[0])

                # Reconstructed velocity from predicted momentum
                y_recon_t = canonical_to_kinematic(z_pred_t, model.M_net)
                _, q_dot_recon_t = split_state(y_recon_t)
                q_dot_recon_list.append(q_dot_recon_t.numpy()[0])

        p_true = np.array(p_true_list)  # (num_steps+1, 2)
        p_pred = np.array(p_pred_list)
        q_dot_recon = np.array(q_dot_recon_list)

        time = np.arange(len(y_true)) * dt

        # Plot 1: Position x
        ax = axes[0, idx]
        ax.plot(time, q_true[:, 0], 'b-', label='True', linewidth=2)
        ax.plot(time, q_pred[:, 0], 'r--', label='Predicted', linewidth=2)
        ax.set_ylabel('x (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        if idx == 0:
            ax.set_title(f'Sample {idx+1}\nPosition x', fontsize=12)
        else:
            ax.set_title(f'Sample {idx+1}', fontsize=12)

        # Plot 2: Angle θ
        ax = axes[1, idx]
        ax.plot(time, q_true[:, 1], 'b-', label='True', linewidth=2)
        ax.plot(time, q_pred[:, 1], 'r--', label='Predicted', linewidth=2)
        ax.set_ylabel('θ (rad)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Plot 3: Momentum p_x
        ax = axes[2, idx]
        ax.plot(time, p_true[:, 0], 'b-', label='True', linewidth=2)
        ax.plot(time, p_pred[:, 0], 'r--', label='Predicted', linewidth=2)
        ax.set_ylabel('p_x', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Plot 4: Velocity reconstruction comparison
        ax = axes[3, idx]
        ax.plot(time, q_dot_true[:, 0], 'b-', label='True ẋ', linewidth=2)
        ax.plot(time, q_dot_recon[:, 0], 'r--', label='Reconstructed ẋ', linewidth=2)
        ax.set_ylabel('ẋ (m/s)', fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.suptitle('Canonical pHNN: Trajectory Predictions vs Ground Truth',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory visualization to {save_path}")

    plt.show()


def visualize_velocity_reconstruction(
    model,
    dataset,
    num_samples=100,
    save_path=None
):
    """
    Scatter plot comparing reconstructed velocities vs true velocities.

    Tests the coordinate transform: q_dot_reconstructed = M^{-1}(q) @ p

    Args:
        model: Trained pHNN_Canonical model
        dataset: TrajectoryStepDataset
        num_samples: Number of random states to sample
        save_path: Optional path to save figure
    """
    model.eval()

    # Collect samples
    q_dot_true_list = []
    q_dot_recon_list = []

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        x, u, _ = dataset[idx]  # x is (seq_len, state_dim)

        with torch.no_grad():
            # Split state
            from coordinate_transforms import split_state, kinematic_to_canonical, canonical_to_kinematic

            # Use first timestep from sequence
            y = x[0:1]  # (1, state_dim)
            q, q_dot_true = split_state(y)

            # Convert to canonical (momentum)
            z = kinematic_to_canonical(y, model.M_net)

            # Reconstruct kinematic (velocity)
            y_recon = canonical_to_kinematic(z, model.M_net)
            _, q_dot_recon = split_state(y_recon)

            q_dot_true_list.append(q_dot_true.numpy()[0])
            q_dot_recon_list.append(q_dot_recon.numpy()[0])

    q_dot_true_arr = np.array(q_dot_true_list)  # (num_samples, 2)
    q_dot_recon_arr = np.array(q_dot_recon_list)  # (num_samples, 2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ẋ comparison
    ax = axes[0]
    ax.scatter(q_dot_true_arr[:, 0], q_dot_recon_arr[:, 0], alpha=0.5, s=20)

    # Perfect reconstruction line
    lim = [q_dot_true_arr[:, 0].min(), q_dot_true_arr[:, 0].max()]
    ax.plot(lim, lim, 'r--', linewidth=2, label='Perfect reconstruction')

    ax.set_xlabel('True ẋ (m/s)', fontsize=12)
    ax.set_ylabel('Reconstructed ẋ (m/s)', fontsize=12)
    ax.set_title('Cart Velocity Reconstruction', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_aspect('equal', adjustable='box')

    # Compute error
    mse_x = np.mean((q_dot_true_arr[:, 0] - q_dot_recon_arr[:, 0])**2)
    ax.text(0.05, 0.95, f'MSE: {mse_x:.6f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # θ̇ comparison
    ax = axes[1]
    ax.scatter(q_dot_true_arr[:, 1], q_dot_recon_arr[:, 1], alpha=0.5, s=20)

    # Perfect reconstruction line
    lim = [q_dot_true_arr[:, 1].min(), q_dot_true_arr[:, 1].max()]
    ax.plot(lim, lim, 'r--', linewidth=2, label='Perfect reconstruction')

    ax.set_xlabel('True θ̇ (rad/s)', fontsize=12)
    ax.set_ylabel('Reconstructed θ̇ (rad/s)', fontsize=12)
    ax.set_title('Pole Angular Velocity Reconstruction', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_aspect('equal', adjustable='box')

    # Compute error
    mse_theta = np.mean((q_dot_true_arr[:, 1] - q_dot_recon_arr[:, 1])**2)
    ax.text(0.05, 0.95, f'MSE: {mse_theta:.6f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Velocity Reconstruction: q̇ = M⁻¹(q) @ p',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved velocity reconstruction plot to {save_path}")

    plt.show()


def visualize_energy_conservation(
    model,
    dataset,
    num_samples=3,
    num_steps=100,
    dt=0.02,
    save_path=None
):
    """
    Visualize energy evolution over trajectories.

    For uncontrolled trajectories, energy should be approximately conserved
    (with small drift due to dissipation R).

    Args:
        model: Trained pHNN_Canonical model
        dataset: TrajectoryStepDataset
        num_samples: Number of trajectories to plot
        num_steps: Number of steps
        dt: Time step
        save_path: Optional path to save figure
    """
    model.eval()

    # Get random trajectory indices
    traj_indices = np.random.choice(dataset.num_traj, num_samples, replace=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for idx, traj_idx in enumerate(traj_indices):
        # Get full trajectory
        traj = dataset.states[traj_idx]  # (timesteps, state_dim)
        controls = dataset.inputs[traj_idx]  # (timesteps, input_dim)

        # Truncate
        max_steps = min(num_steps, len(traj) - 1)
        traj = traj[:max_steps+1]
        controls = controls[:max_steps]

        y0 = traj[0:1]

        # Rollout with energy tracking
        # Note: Cannot use no_grad() because model needs gradients for Hamiltonian
        y_pred, energies = rollout_trajectory_differentiable(
            model=model,
            y0=y0.requires_grad_(True),
            controls=controls.unsqueeze(0),
            dt=dt,
            integrator='rk4',
            return_energies=True
        )

        energies = energies[0].detach().numpy()  # (num_steps+1,)
        time = np.arange(len(energies)) * dt

        # Plot
        ax.plot(time, energies, linewidth=2, label=f'Trajectory {idx+1}')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Hamiltonian Energy H(q,p)', fontsize=12)
    ax.set_title('Energy Evolution (Port-Hamiltonian System)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved energy plot to {save_path}")

    plt.show()


def main():
    """Run all visualizations."""
    print("\n" + "="*70)
    print("Canonical pHNN Visualization")
    print("="*70)

    # Load configuration
    config_path = "cartpole_mpc_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    print("\nLoading trained model...")
    model_path = config['training']['model_save_path']

    try:
        model = pHNN_Canonical(config_path)
        checkpoint = torch.load(model_path, map_location='cpu')

        # Handle both formats: direct state_dict or dict with 'model_state_dict' key
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
        else:
            model.load_state_dict(checkpoint)
            epoch = 'unknown'

        model.eval()
        print(f"✓ Loaded model from {model_path}")
        print(f"  Training epoch: {epoch}")
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        print("  Please train the model first using: python scripts/train_cartpole_phnn_canonical.py")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("  Please check the model file and try retraining if necessary.")
        return

    # Print learned parameters
    print_mass_matrix_params(model)
    print_dissipation_params(model)

    # Load dataset
    print("\nLoading dataset...")
    data_path = config['data'].get('train_data_path') or config['data'].get('save_path')
    if not data_path:
        print("✗ Could not find data path in config (tried 'train_data_path' and 'save_path')")
        return

    try:
        # Load data file
        data = torch.load(data_path)
        states = data['states']
        controls = data['controls']
        derivatives = data['derivatives']

        # Create dataset
        seq_len = config['training'].get('seq_len', 16)
        dataset = TrajectoryStepDataset(states, controls, derivatives, seq_len)
        print(f"✓ Loaded {len(dataset)} trajectory steps (seq_len={seq_len})")
    except FileNotFoundError:
        print(f"✗ Dataset not found: {data_path}")
        print("  Please generate data first using: python scripts/generate_cartpole_data.py")
        return
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    dt = config['cartpole']['dt']

    # Create visualizations
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    print("\n1. Trajectory predictions vs ground truth...")
    visualize_trajectory_predictions(
        model, dataset,
        num_samples=3,
        num_steps=50,
        dt=dt,
        save_path='results/canonical_phnn_trajectories.png'
    )

    print("\n2. Velocity reconstruction accuracy...")
    visualize_velocity_reconstruction(
        model, dataset,
        num_samples=200,
        save_path='results/canonical_phnn_velocity_reconstruction.png'
    )

    print("\n3. Energy conservation...")
    visualize_energy_conservation(
        model, dataset,
        num_samples=3,
        num_steps=100,
        dt=dt,
        save_path='results/canonical_phnn_energy.png'
    )

    print("\n" + "="*70)
    print("Visualization Complete!")
    print("="*70)
    print("\nGenerated plots:")
    print("  - results/canonical_phnn_trajectories.png")
    print("  - results/canonical_phnn_velocity_reconstruction.png")
    print("  - results/canonical_phnn_energy.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
