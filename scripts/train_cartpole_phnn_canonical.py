"""
Cart-Pole pHNN Canonical Training Module

Trains a Port-Hamiltonian Neural Network using canonical momentum coordinates
with RK4-integrated physics loss.

Key differences from standard training:
- Uses pHNN_Canonical with [q, p] coordinates
- RK4 integration for trajectory prediction
- Combined position and velocity reconstruction loss

Usage:
    python scripts/train_cartpole_phnn_canonical.py
    python scripts/train_cartpole_phnn_canonical.py --resume models/checkpoint_epoch_200.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

# Add src to path
sys.path.append('src')
from pHNN_canonical import pHNN_Canonical
from TrajectoryStepDataset import TrajectoryStepDataset
from integrators import rollout_trajectory_differentiable
from coordinate_transforms import split_state, compute_velocity_reconstruction_error


def load_config(config_path="cartpole_mpc_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_training_data(data_path):
    """Load preprocessed training data."""
    print(f"Loading training data from {data_path}...")
    data = torch.load(data_path)

    states = data['states']
    controls = data['controls']
    derivatives = data['derivatives']

    print(f"Loaded data shapes:")
    print(f"  States: {states.shape}")
    print(f"  Controls: {controls.shape}")
    print(f"  Derivatives: {derivatives.shape}")

    return states, controls, derivatives


def create_dataloader(states, controls, derivatives, config):
    """Create PyTorch DataLoader for training."""
    use_traj_dataset = config['training']['preserve_traj']

    if use_traj_dataset:
        seq_len = config['training']['seq_len']
        dataset = TrajectoryStepDataset(states, controls, derivatives, seq_len)
        print(f"\nUsing TrajectoryStepDataset with seq_len={seq_len}")
    else:
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(states, controls, derivatives)
        print(f"\nUsing TensorDataset")

    batch_size = config['training']['batch_size']
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    print(f"DataLoader created with batch_size={batch_size}")
    print(f"Number of batches: {len(dataloader)}")

    return dataloader


def compute_integrated_loss(
    model,
    x_batch,
    u_batch,
    dt,
    loss_weights,
    device,
    integrator='rk4'
):
    """
    Compute physics-integrated loss with selectable integrator.

    Combines:
    1. Position tracking loss
    2. Velocity reconstruction loss

    Args:
        model: pHNN_Canonical model
        x_batch: (batch, seq_len, state_dim) ground truth states [q, q̇]
        u_batch: (batch, seq_len, input_dim) control inputs
        dt: Time step
        loss_weights: Dict with 'position' and 'velocity' weights
        device: torch.device
        integrator: 'rk4' or 'euler' (default: 'rk4')

    Returns:
        total_loss: Combined loss
        loss_dict: Dict with individual loss components
    """
    batch_size, seq_len, state_dim = x_batch.shape
    q_dim = state_dim // 2

    # Get initial state
    y0 = x_batch[:, 0, :].requires_grad_(True)

    # Rollout trajectory
    # Note: We need to roll out for seq_len-1 steps
    u_rollout = u_batch[:, :-1, :]  # (batch, seq_len-1, input_dim)

    # Select integrator
    if integrator == 'rk4':
        from integrators import rk4_step
        step_fn = rk4_step
    elif integrator == 'euler':
        from integrators import euler_step
        step_fn = euler_step
    else:
        raise ValueError(f"Unknown integrator: {integrator}")

    # OPTIMIZED: Manual rollout with cached intermediate values
    y_pred_list = [y0]
    vel_recon_errors = []

    for t in range(seq_len - 1):
        y_current = y_pred_list[-1]
        u_t = u_rollout[:, t, :]

        # Get dynamics with intermediate values (avoids recomputation)
        dy_dt, _, intermediate = model(y_current, u_t, return_intermediate=True)

        # Integration step (manual Euler for now - cleaner)
        y_next = y_current + dt * dy_dt
        y_pred_list.append(y_next)

        # OPTIMIZATION: Reuse cached velocity reconstruction from forward pass
        # This avoids a second coordinate transform!
        q_dot_recon = intermediate['q_dot_reconstructed']

        # Ground truth velocity
        y_true = x_batch[:, t, :]
        q_true, q_dot_true = split_state(y_true)

        # Reconstruction error
        vel_error = torch.sum((q_dot_recon - q_dot_true) ** 2, dim=1)  # (batch,)
        vel_recon_errors.append(vel_error.mean())

    # Stack predicted trajectory
    y_pred = torch.stack(y_pred_list, dim=1)  # (batch, seq_len, state_dim)

    # === 1. Position Loss ===
    # Extract positions from predicted and true trajectories
    q_pred = y_pred[:, :, :q_dim]  # (batch, seq_len, q_dim)
    q_true = x_batch[:, :, :q_dim]

    # Cart position loss (x)
    x_pred = q_pred[:, :, 0]
    x_true = q_true[:, :, 0]
    l_position_x = torch.mean((x_pred - x_true) ** 2)

    # Pole angle loss (theta) - use cosine distance for periodicity
    theta_pred = q_pred[:, :, 1]
    theta_true = q_true[:, :, 1]
    l_position_theta = torch.mean(1 - torch.cos(theta_pred - theta_true))

    l_position = l_position_x + l_position_theta

    # === 2. Velocity Reconstruction Loss ===
    l_velocity = torch.mean(torch.stack(vel_recon_errors))

    # === 3. Combined Loss ===
    w_pos = loss_weights.get('position', 1.0)
    w_vel = loss_weights.get('velocity', 1.0)

    total_loss = w_pos * l_position + w_vel * l_velocity

    loss_dict = {
        'position': l_position.item(),
        'position_x': l_position_x.item(),
        'position_theta': l_position_theta.item(),
        'velocity_reconstruction': l_velocity.item(),
        'total': total_loss.item()
    }

    return total_loss, loss_dict


def train_phnn_canonical(model, dataloader, config, device, start_epoch=0):
    """
    Train pHNN_Canonical model with RK4-integrated loss.

    Args:
        model: pHNN_Canonical model
        dataloader: Training data loader
        config: Configuration dictionary
        device: torch.device for GPU/CPU
        start_epoch: Epoch to start from (for resuming training)
    """
    lr = config['training']['lr']
    epochs = config['training']['epochs']
    dt = config['cartpole']['dt']

    # Loss weights
    loss_weights = {
        'position': config['training'].get('loss_weight_position', 1.0),
        'velocity': config['training'].get('loss_weight_velocity', 1.0)
    }

    # Integration method
    integrator = config['training'].get('integrator', 'rk4')  # 'rk4' or 'euler'

    # Move model to device
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if start_epoch > 0:
        print(f"\nContinuing training from epoch {start_epoch + 1} to {epochs}")
    else:
        print(f"\nTraining pHNN_Canonical for {epochs} epochs with lr={lr}")

    print(f"Loss weights: position={loss_weights['position']}, velocity={loss_weights['velocity']}")
    print(f"Integrator: {integrator}")
    print(f"Time step dt={dt}")
    print(f"Device: {device}")

    # Track losses
    epoch_losses = {
        'total': [],
        'position': [],
        'position_x': [],
        'position_theta': [],
        'velocity_reconstruction': []
    }

    for epoch in range(start_epoch, epochs):
        epoch_loss_accum = {
            'total': 0.0,
            'position': 0.0,
            'position_x': 0.0,
            'position_theta': 0.0,
            'velocity_reconstruction': 0.0
        }

        for x_batch, u_batch, _ in dataloader:
            # Move batch to device
            x_batch = x_batch.to(device)
            u_batch = u_batch.to(device)

            optimizer.zero_grad()

            # Compute physics-integrated loss
            loss, loss_dict = compute_integrated_loss(
                model, x_batch, u_batch, dt, loss_weights, device, integrator
            )

            # Gradient clipping for stability
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses
            for key in epoch_loss_accum:
                epoch_loss_accum[key] += loss_dict.get(key, 0.0)

        # Average losses
        num_batches = len(dataloader)
        for key in epoch_loss_accum:
            avg_loss = epoch_loss_accum[key] / num_batches
            epoch_losses[key].append(avg_loss)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Total: {epoch_losses['total'][-1]:.6f}, "
                  f"Pos: {epoch_losses['position'][-1]:.6f}, "
                  f"Vel: {epoch_losses['velocity_reconstruction'][-1]:.6f}")

            # Print mass matrix parameters
            if hasattr(model.M_net, 'log_a'):
                with torch.no_grad():
                    a = torch.exp(model.M_net.log_a).item() + 1e-3
                    b = model.M_net.b.item()
                    c = torch.exp(model.M_net.log_c).item() + 1e-3
                print(f"  Mass matrix: a={a:.4f}, b={b:.4f}, c={c:.4f}")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = f"models/canonical_checkpoint_epoch_{epoch + 1}.pth"
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")

    print("\nTraining complete!")

    # Plot loss curves
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(epoch_losses['total'])
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Total Loss')
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].grid(True)
    axs[0, 0].set_yscale('log')

    axs[0, 1].plot(epoch_losses['position'], label='Total Position')
    axs[0, 1].plot(epoch_losses['position_x'], '--', alpha=0.7, label='Cart x')
    axs[0, 1].plot(epoch_losses['position_theta'], '--', alpha=0.7, label='Pole θ')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Position Loss')
    axs[0, 1].set_title('Position Tracking Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].set_yscale('log')

    axs[1, 0].plot(epoch_losses['velocity_reconstruction'])
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Velocity Reconstruction Loss')
    axs[1, 0].set_title('Velocity Reconstruction Loss')
    axs[1, 0].grid(True)
    axs[1, 0].set_yscale('log')

    # Plot ratio of velocity to position loss
    vel_pos_ratio = [v / (p + 1e-8) for v, p in zip(
        epoch_losses['velocity_reconstruction'],
        epoch_losses['position']
    )]
    axs[1, 1].plot(vel_pos_ratio)
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Vel Loss / Pos Loss')
    axs[1, 1].set_title('Loss Component Ratio')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('results/training_loss_canonical.png')
    print("Loss curves saved to results/training_loss_canonical.png")
    plt.close()

    return model


def save_model(model, save_path):
    """Save trained model weights."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights saved to {save_path}")


def evaluate_model(model, states, controls, derivatives, config, device):
    """Evaluate trained canonical pHNN model on a sample trajectory."""
    print("\nEvaluating canonical pHNN model on sample trajectory...")

    dt = config['cartpole']['dt']
    model.eval()

    # Take first trajectory and move to device
    x_true = states[0].to(device)  # (seq_len, 4)
    u_true = controls[0].to(device)  # (seq_len, 1)

    # Rollout pHNN predictions using RK4
    from integrators import rk4_step

    x_pred = [x_true[0].unsqueeze(0).requires_grad_(True)]

    for t in range(len(x_true) - 1):
        x_current = x_pred[-1]
        u_t = u_true[t].unsqueeze(0)

        x_next = rk4_step(model, x_current, u_t, dt)
        x_pred.append(x_next.detach().requires_grad_(True))

    x_pred = torch.cat(x_pred, dim=0)

    # Compute MSE
    mse = nn.MSELoss()(x_pred, x_true).item()
    print(f"Trajectory MSE: {mse:.6f}")

    # Compute velocity reconstruction error
    q_true, q_dot_true = split_state(x_true.unsqueeze(0))
    q_dot_recon = model.get_velocity_reconstruction(x_true.unsqueeze(0))
    vel_recon_mse = torch.mean((q_dot_recon - q_dot_true.squeeze(0)) ** 2).item()
    print(f"Velocity Reconstruction MSE: {vel_recon_mse:.6f}")

    # Plot comparison
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    state_names = ['x (cart position)', 'theta (pole angle)',
                   'x_dot (cart velocity)', 'theta_dot (pole angular velocity)']

    for i in range(4):
        ax = axs[i // 2, i % 2]
        ax.plot(x_true[:, i].cpu().numpy(), label='True', linewidth=2)
        ax.plot(x_pred[:, i].detach().cpu().numpy(), '--', label='Predicted', linewidth=2)
        ax.set_xlabel('Time step')
        ax.set_ylabel(state_names[i])
        ax.set_title(f'State {i}: {state_names[i]}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('results/model_evaluation_canonical.png')
    print("Evaluation plot saved to results/model_evaluation_canonical.png")
    plt.close()

    # Print learned mass matrix at equilibrium
    print("\nLearned mass matrix M(q) at equilibrium [0, 0]:")
    with torch.no_grad():
        q_eq = torch.zeros(1, 2, device=device)  # [x=0, theta=0]
        M_eq = model.M_net(q_eq)[0].cpu().numpy()
        print(M_eq)
        print(f"Eigenvalues: {torch.linalg.eigvalsh(model.M_net(q_eq))[0].cpu().numpy()}")

    # Print canonical J matrix
    print("\nFixed canonical J matrix:")
    print(model.J.cpu().numpy())

    # Print learned R matrix
    print("\nLearned constant R matrix:")
    R = model.get_R_matrix(1)[0].cpu().numpy()
    print(R)
    print(f"Eigenvalues: {torch.linalg.eigvalsh(model.get_R_matrix(1).to(device))[0].cpu().numpy()}")

    model.train()


def main(resume_from=None):
    """
    Main training function for canonical pHNN.

    Args:
        resume_from: Optional path to checkpoint to resume training from
    """
    print("=" * 60)
    print("Cart-Pole pHNN Canonical Training (Momentum Coordinates)")
    print("=" * 60)

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("\nNo GPU detected, using CPU")

    # Load configuration
    config = load_config()

    # Load training data
    data_path = config['data']['save_path']
    states, controls, derivatives = load_training_data(data_path)

    # Create dataloader
    dataloader = create_dataloader(states, controls, derivatives, config)

    # Create pHNN_Canonical model
    print("\nCreating pHNN_Canonical model with momentum coordinates...")
    model = pHNN_Canonical("cartpole_mpc_config.yaml")
    print(f"Model created with {sum(p.numel() for p in model.parameters())} trainable parameters")

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_from is not None:
        print(f"\n{'='*60}")
        print(f"Resuming training from checkpoint: {resume_from}")
        print(f"{'='*60}")

        if Path(resume_from).exists():
            checkpoint = torch.load(resume_from, weights_only=True)
            model.load_state_dict(checkpoint)

            # Extract epoch number
            import re
            match = re.search(r'epoch_(\d+)', resume_from)
            if match:
                start_epoch = int(match.group(1))
                print(f"Resuming from epoch {start_epoch}")
            else:
                print(f"Checkpoint loaded, starting from epoch 0")

            print(f"Weights loaded successfully!")
        else:
            print(f"WARNING: Checkpoint file not found: {resume_from}")
            print(f"Starting training from scratch...")
    else:
        print("\nStarting training from scratch...")

    # Train model
    trained_model = train_phnn_canonical(model, dataloader, config, device, start_epoch)

    # Move model back to CPU for saving
    trained_model = trained_model.cpu()

    # Save model
    save_path = config['training']['model_save_path'].replace('.pth', '_canonical.pth')
    save_model(trained_model, save_path)

    # Evaluate model
    trained_model = trained_model.to(device)
    evaluate_model(trained_model, states, controls, derivatives, config, device)
    trained_model = trained_model.cpu()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train canonical pHNN model with momentum coordinates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python scripts/train_cartpole_phnn_canonical.py

  # Resume from checkpoint at epoch 200
  python scripts/train_cartpole_phnn_canonical.py --resume models/canonical_checkpoint_epoch_200.pth
        """
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )

    args = parser.parse_args()
    main(resume_from=args.resume)
