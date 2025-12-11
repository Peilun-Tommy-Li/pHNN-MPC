"""
Training script for baseline models (Vanilla MLP and Neural ODE).

Trains models to predict next state given current state and action.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import argparse
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append('src')
from baseline_mlp import VanillaMLP
from baseline_node import NeuralODE
from BaselineDataset import BaselineDataset


def load_config(config_path="cartpole_mpc_config.yaml"):
    """Load configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(model_type, config):
    """
    Create baseline model.

    Args:
        model_type: 'mlp' or 'node'
        config: Configuration dict

    Returns:
        model: Baseline model
    """
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

    return model


def compute_loss(model, states, actions, next_states, dt, model_type):
    """
    Compute prediction loss.

    Args:
        model: Baseline model
        states: (batch_size, state_dim) current states
        actions: (batch_size, action_dim) actions
        next_states: (batch_size, state_dim) ground truth next states
        dt: Time step
        model_type: 'mlp' or 'node'

    Returns:
        loss: Scalar loss
        metrics: Dict of metrics
    """
    # Predict next state
    if model_type == 'mlp':
        pred_next_states = model(states, actions)
    elif model_type == 'node':
        pred_next_states = model(states, actions, dt)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # MSE loss on next state
    loss = nn.functional.mse_loss(pred_next_states, next_states)

    # Compute per-dimension errors
    with torch.no_grad():
        errors = torch.abs(pred_next_states - next_states)
        mean_errors = errors.mean(dim=0)

    metrics = {
        'loss': loss.item(),
        'x_error': mean_errors[0].item(),
        'theta_error': mean_errors[1].item(),
        'x_dot_error': mean_errors[2].item(),
        'theta_dot_error': mean_errors[3].item(),
    }

    return loss, metrics


def train_epoch(model, dataloader, optimizer, dt, model_type, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        states = batch['state'].to(device)
        actions = batch['control'].to(device)
        next_states = batch['next_state'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Compute loss
        loss, metrics = compute_loss(model, states, actions, next_states, dt, model_type)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, dt, model_type, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_metrics = {
        'x_error': 0,
        'theta_error': 0,
        'x_dot_error': 0,
        'theta_dot_error': 0,
    }
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            states = batch['state'].to(device)
            actions = batch['control'].to(device)
            next_states = batch['next_state'].to(device)

            loss, metrics = compute_loss(model, states, actions, next_states, dt, model_type)

            total_loss += loss.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metrics = {key: val / num_batches for key, val in total_metrics.items()}
    avg_metrics['loss'] = avg_loss

    return avg_metrics


def train_baseline(model_type, config, epochs=500, batch_size=32, lr=1e-3,
                   save_interval=50, device='cpu'):
    """
    Train baseline model.

    Args:
        model_type: 'mlp' or 'node'
        config: Configuration dict
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        save_interval: Save checkpoint every N epochs
        device: Device to train on
    """
    print("=" * 80)
    print(f"Training {model_type.upper()} Baseline Model")
    print("=" * 80)

    # Load data
    data_path = config['data']['save_path']
    print(f"\nLoading data from {data_path}...")
    data = torch.load(data_path, weights_only=True)

    states = data['states']  # (num_traj, seq_len, 4)
    controls = data['controls']  # (num_traj, seq_len, 1)
    dt = config['cartpole']['dt']

    print(f"Data loaded: {states.shape[0]} trajectories, {states.shape[1]} steps each")
    print(f"Time step dt: {dt}s")

    # Create datasets
    train_dataset = BaselineDataset(states, controls)

    # Split into train/val (80/20)
    dataset_size = len(train_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    print(f"\nCreating {model_type.upper()} model...")
    model = create_model(model_type, config).to(device)

    model_info = model.get_model_info()
    print(f"Model info:")
    for key, val in model_info.items():
        print(f"  {key}: {val}")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    # Create save directory
    save_dir = Path(f"baseline/{model_type}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("=" * 80)

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, dt, model_type, device)

        # Validate
        val_metrics = validate(model, val_loader, dt, model_type, device)
        val_loss = val_metrics['loss']

        # Update learning rate
        scheduler.step(val_loss)

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Time: {elapsed:.1f}s")
            print(f"  Val Errors: "
                  f"x={val_metrics['x_error']:.6f}, "
                  f"θ={val_metrics['theta_error']:.6f}, "
                  f"ẋ={val_metrics['x_dot_error']:.6f}, "
                  f"θ̇={val_metrics['theta_dot_error']:.6f}")

        # Save checkpoint
        if epoch % save_interval == 0 or epoch == epochs:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'model_info': model_info,
            }, checkpoint_path)

            if epoch % 100 == 0:
                print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = save_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'model_info': model_info,
            }, best_path)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {save_dir}")
    print("=" * 80 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--model', type=str, required=True, choices=['mlp', 'node'],
                        help='Model type: mlp or node')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--config', type=str, default='cartpole_mpc_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Train model
    train_baseline(
        model_type=args.model,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_interval=args.save_interval,
        device=device
    )


if __name__ == "__main__":
    main()
