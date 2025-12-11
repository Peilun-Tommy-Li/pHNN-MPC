"""
Cart-Pole pHNN Training Module

Trains a Port-Hamiltonian Neural Network on cart-pole dynamics data with fixed G matrix.
State representation: [x, theta, x_dot, theta_dot]

Usage:
    python scripts/train_cartpole_phnn.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')
from pHNN import pHNN
from TrajectoryStepDataset import TrajectoryStepDataset


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


def train_phnn(model, dataloader, config, device, start_epoch=0):
    """
    Train pHNN model on cart-pole data with GPU acceleration.

    Loss combines trajectory prediction MSE and derivative MSE.

    Args:
        model: pHNN model
        dataloader: Training data loader
        config: Configuration dictionary
        device: torch.device for GPU/CPU
        start_epoch: Epoch to start from (for resuming training)
    """
    lr = config['training']['lr']
    epochs = config['training']['epochs']
    dt = config['cartpole']['dt']

    # Move model to device
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if start_epoch > 0:
        print(f"\nContinuing training from epoch {start_epoch + 1} to {epochs}")
    else:
        print(f"\nTraining pHNN for {epochs} epochs with lr={lr}")
    print(f"Time step dt={dt}")
    print(f"Device: {device}")

    # Track losses
    epoch_losses = []
    epoch_dx_losses = []

    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        total_loss_dx = 0.0

        for x_batch, u_batch, dx_batch in dataloader:
            # Move batch to device
            x_batch = x_batch.to(device)
            u_batch = u_batch.to(device)
            dx_batch = dx_batch.to(device)

            optimizer.zero_grad()

            # Get initial states for the batch
            # x_batch shape: (batch_size, seq_len, state_dim)
            x0_batch = x_batch[:, 0, :].requires_grad_(True)

            # Predict trajectory using pHNN forward dynamics
            X_pred = [x0_batch]
            dX_pred = []

            for t in range(x_batch.shape[1] - 1):
                # Predict derivative at current state
                dx, _ = model(X_pred[-1], u_batch[:, t, :])
                dX_pred.append(dx)

                # Integrate forward: x_{t+1} = x_t + dt * dx_t
                X_pred.append(X_pred[-1] + dt * dx)

            # Stack predictions
            X_pred = torch.stack(X_pred, dim=1)  # (batch_size, seq_len, state_dim)
            dX_pred = torch.stack(dX_pred, dim=1)  # (batch_size, seq_len-1, state_dim)

            # ===== IMPROVED LOSS FUNCTION =====
            # State ordering: [x, theta, x_dot, theta_dot]

            # Extract components from predictions and ground truth
            pos_pred = X_pred[:, :, 0]      # Cart position
            theta_pred = X_pred[:, :, 1]    # Pole angle
            vel_pred = X_pred[:, :, 2:]     # Velocities [x_dot, theta_dot]

            pos_true = x_batch[:, :, 0]
            theta_true = x_batch[:, :, 1]
            vel_true = x_batch[:, :, 2:]

            # 1. Cart Position Loss (Standard MSE)
            l_pos = loss_fn(pos_pred, pos_true)

            # 2. Angle Loss (Cosine Distance for circular topology)
            # 1 - cos(error) is robust for large angle errors
            # Approximates error^2/2 for small errors but handles wrap-around
            l_theta = torch.mean(1 - torch.cos(theta_pred - theta_true))

            # 3. Velocity Loss (Standard MSE)
            # CRITICAL: Weight equally to capture dynamics
            l_vel = loss_fn(vel_pred, vel_true)

            # 4. Energy Anchor (Force energy at equilibrium [0,0,0,0] to be 0)
            # This ensures the Hamiltonian has correct reference point
            zero_state = torch.zeros(1, 4, device=device, requires_grad=True)
            zero_control = torch.zeros(1, 1, device=device)
            _, H_zero = model(zero_state, zero_control)
            l_energy_anchor = torch.mean(H_zero ** 2)

            # Combined weighted loss
            combined_loss = (1.0 * l_pos +
                           1.0 * l_theta +
                           1.0 * l_vel +
                           0.01 * l_energy_anchor)

            combined_loss.backward()
            optimizer.step()

            # Track individual losses for monitoring
            total_loss += (l_pos.item() + l_theta.item() + l_vel.item())
            total_loss_dx += l_energy_anchor.item()

        avg_loss = total_loss / len(dataloader)
        avg_energy_loss = total_loss_dx / len(dataloader)

        epoch_losses.append(avg_loss)
        epoch_dx_losses.append(avg_energy_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - State Loss: {avg_loss:.6f}, Energy Anchor: {avg_energy_loss:.6f}")

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = f"models/checkpoint_epoch_{epoch + 1}.pth"
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")

    print("\nTraining complete!")

    # Plot loss curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('State Loss (Pos + Angle + Vel)')
    plt.title('Training Loss (State Components)')
    plt.grid(True)
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    plt.plot(epoch_dx_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Energy Anchor Loss')
    plt.title('Training Loss (Energy Equilibrium)')
    plt.grid(True)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('results/training_loss.png')
    print("Loss curves saved to results/training_loss.png")
    plt.close()

    return model


def save_model(model, save_path):
    """Save trained model weights."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights saved to {save_path}")


def evaluate_model(model, states, controls, derivatives, config, device):
    """Evaluate trained model on a sample trajectory."""
    print("\nEvaluating model on sample trajectory...")

    dt = config['cartpole']['dt']
    model.eval()

    # Take first trajectory and move to device
    x_true = states[0].to(device)  # (seq_len, 4)
    u_true = controls[0].to(device)  # (seq_len, 1)

    # Rollout pHNN predictions
    x_pred = [x_true[0].unsqueeze(0)]

    for t in range(len(x_true) - 1):
        x_current = x_pred[-1].requires_grad_(True)
        dx, _ = model(x_current, u_true[t].unsqueeze(0))
        x_next = x_pred[-1].detach() + dt * dx.detach()
        x_pred.append(x_next)

    x_pred = torch.cat(x_pred, dim=0)

    # Compute MSE
    mse = nn.MSELoss()(x_pred, x_true).item()
    print(f"Trajectory MSE: {mse:.6f}")

    # Plot comparison (move to CPU for plotting)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    state_names = ['x (cart position)', 'theta (pole angle)', 'x_dot (cart velocity)', 'theta_dot (pole angular velocity)']

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
    plt.savefig('results/model_evaluation.png')
    print("Evaluation plot saved to results/model_evaluation.png")
    plt.close()

    # Print learned J matrix
    print("\nLearned skew-symmetric J matrix:")
    J_learned = (model.J.data - model.J.data.T) / 2
    print(J_learned.cpu().numpy())

    # Print fixed G matrix
    if hasattr(model, 'G_fixed'):
        print("\nFixed G matrix:")
        print(model.G_fixed.cpu().numpy())

    model.train()


def main(resume_from=None):
    """
    Main training function with GPU acceleration.

    Args:
        resume_from: Optional path to checkpoint to resume training from
    """
    print("=" * 60)
    print("Cart-Pole pHNN Training with GPU Acceleration")
    print("=" * 60)

    # Detect device (GPU if available, else CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # Use first GPU
        print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
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

    # Create pHNN model
    print("\nCreating pHNN model with fixed G matrix...")
    model = pHNN("cartpole_mpc_config.yaml")
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

            # Try to extract epoch number from filename
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

    # Verify fixed G configuration
    if hasattr(model, 'G_fixed'):
        print(f"Fixed G matrix: {model.G_fixed.squeeze().numpy()}")
    else:
        print("WARNING: G matrix is learned, not fixed!")

    # Train model with GPU
    trained_model = train_phnn(model, dataloader, config, device, start_epoch)

    # Move model back to CPU for saving
    trained_model = trained_model.cpu()

    # Save model
    save_path = config['training']['model_save_path']
    save_model(trained_model, save_path)

    # Evaluate model (on device for speed)
    trained_model = trained_model.to(device)
    evaluate_model(trained_model, states, controls, derivatives, config, device)
    trained_model = trained_model.cpu()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Train pHNN model with optional checkpoint resumption',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from scratch
  python scripts/train_cartpole_phnn.py

  # Resume from checkpoint at epoch 200
  python scripts/train_cartpole_phnn.py --resume models/checkpoint_epoch_200.pth

  # Resume from latest checkpoint
  python scripts/train_cartpole_phnn.py --resume models/checkpoint_epoch_500.pth
        """
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from (e.g., models/checkpoint_epoch_200.pth)'
    )

    args = parser.parse_args()
    main(resume_from=args.resume)
