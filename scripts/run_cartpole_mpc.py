"""
Cart-Pole MPC Control & Visualization Module

Uses trained pHNN model with MPC controller to perform real-time control of
Gymnasium CartPole environment and generates GIF animations.

Usage:
    python scripts/run_cartpole_mpc.py
"""

import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append('src')
from pHNN import pHNN
from mpc_controller import MPCController
from cartpole_simulator import CartPoleSimulator


def load_config(config_path="cartpole_mpc_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_trained_model(config_path, weights_path):
    """Load trained pHNN model."""
    print(f"Loading trained pHNN model from {weights_path}...")

    model = pHNN(config_path)
    checkpoint = torch.load(weights_path, map_location='cpu')

    # Handle both dict and direct state_dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("✓ Model loaded successfully!")

    # Verify fixed G
    if hasattr(model, 'G_fixed'):
        print(f"  Fixed G matrix: {model.G_fixed.squeeze().numpy()}")

    return model


def create_mpc_from_config(phnn_model, config):
    """
    Create MPC controller from configuration.

    Args:
        phnn_model: Trained pHNN model
        config: Configuration dictionary

    Returns:
        MPCController instance
    """
    mpc_config = config['mpc']

    # Handle duplicate mpc sections - use the second one with more complete config
    Q_diag = mpc_config.get('Q_diag', [10.0, 100.0, 1.0, 10.0])
    R_diag = mpc_config.get('R_diag', [0.01])

    controller = MPCController(
        phnn_model=phnn_model,
        horizon=mpc_config.get('horizon', 20),
        dt=config['cartpole']['dt'],
        Q=Q_diag,
        R=R_diag[0],  # R is scalar for this controller
        target_state=mpc_config.get('x_target', [0.0, 0.0, 0.0, 0.0]),
        u_min=mpc_config.get('u_min', -10.0),
        u_max=mpc_config.get('u_max', 10.0),
        optimizer_type='Adam',
        lr=mpc_config.get('learning_rate', 0.1),
        max_iterations=mpc_config.get('optimizer_steps', 50)
    )

    return controller


def run_mpc_control(simulator, mpc_controller, initial_state, num_steps, config, verbose=True):
    """
    Run closed-loop MPC control on CartPole simulator.

    Args:
        simulator: CartPoleSimulator instance
        mpc_controller: MPC controller instance
        initial_state: Initial state [x, theta, x_dot, theta_dot]
        num_steps: Number of simulation steps
        config: Configuration dictionary
        verbose: Print progress

    Returns:
        Tuple of (states, controls, hamiltonians, stability_achieved, stable_duration)
    """
    print(f"\nRunning MPC control for {num_steps} steps...")

    # Reset simulator
    simulator.reset(initial_state)
    state = initial_state.copy()

    # Storage
    states = [state.copy()]
    controls = []
    hamiltonians = []

    # Stability tracking
    target_state = np.array(config['mpc']['x_target'])
    tolerance = np.array(config['stability']['tolerance'])
    min_stable_duration = config['stability']['min_duration']
    dt = config['cartpole']['dt']

    stable_start_step = None
    stability_achieved = False
    stable_duration = 0.0

    for step in range(num_steps):
        # Compute MPC control
        control = mpc_controller.compute_control(state)
        controls.append(control.item() if isinstance(control, np.ndarray) else control)

        # Log Hamiltonian (needs gradients for pHNN forward pass)
        state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        control_tensor = torch.tensor([[control]], dtype=torch.float32)
        _, H = mpc_controller.model(state_tensor, control_tensor)
        hamiltonians.append(H.detach().item())

        # Check stability
        state_error = state - target_state
        within_tolerance = np.all(np.abs(state_error) <= tolerance)

        if within_tolerance:
            if stable_start_step is None:
                stable_start_step = step
                if verbose:
                    print(f"  Entered stable region at step {step} ({step * dt:.2f}s)")

            stable_duration = (step - stable_start_step + 1) * dt

            if stable_duration >= min_stable_duration and not stability_achieved:
                stability_achieved = True
                if verbose:
                    print(f"  ✓ Stability achieved! Stayed within tolerance for {stable_duration:.2f}s")
                    print(f"    State: x={state[0]:.4f}, θ={state[1]:.4f}, ẋ={state[2]:.4f}, θ̇={state[3]:.4f}")
        else:
            if stable_start_step is not None and verbose:
                print(f"  Left stable region at step {step} (was stable for {stable_duration:.2f}s)")
            stable_start_step = None
            stable_duration = 0.0

        # Step simulator
        next_state, done = simulator.step(control)

        # Store
        states.append(next_state.copy())

        # Update state
        state = next_state

        # Check termination
        if done:
            if verbose:
                print(f"Episode terminated at step {step + 1}")
            break

        if verbose and (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps}: x={state[0]:.3f}, θ={state[1]:.3f}")

    if verbose:
        print(f"Control episode completed with {len(states)} states")

    return np.array(states), np.array(controls), np.array(hamiltonians), stability_achieved, stable_duration


def compute_metrics(states, controls, target_state):
    """
    Compute control performance metrics.

    Args:
        states: State trajectory (T, state_dim)
        controls: Control inputs (T-1,)
        target_state: Target state (state_dim,)

    Returns:
        Dictionary of metrics
    """
    # State tracking MSE
    state_errors = states - target_state
    mse = np.mean(np.sum(state_errors**2, axis=1))

    # Control effort
    control_effort = np.sum(controls**2)

    # Average state error per dimension
    avg_errors = np.mean(np.abs(state_errors), axis=0)

    metrics = {
        'mse': mse,
        'control_effort': control_effort,
        'avg_x_error': avg_errors[0],
        'avg_theta_error': avg_errors[1],
        'avg_x_dot_error': avg_errors[2],
        'avg_theta_dot_error': avg_errors[3]
    }

    return metrics


def print_metrics(metrics):
    """Print control performance metrics."""
    print("\n" + "=" * 60)
    print("Control Performance Metrics")
    print("=" * 60)
    print(f"State Tracking MSE: {metrics['mse']:.6f}")
    print(f"Total Control Effort: {metrics['control_effort']:.6f}")
    print(f"\nAverage Absolute Errors:")
    print(f"  Cart Position (x): {metrics['avg_x_error']:.6f}")
    print(f"  Pole Angle (theta): {metrics['avg_theta_error']:.6f}")
    print(f"  Cart Velocity (x_dot): {metrics['avg_x_dot_error']:.6f}")
    print(f"  Pole Angular Velocity (theta_dot): {metrics['avg_theta_dot_error']:.6f}")
    print("=" * 60)


def create_animation(states, dt, save_path='results/mpc_control_animation.mp4', fps=30):
    """
    Create animation of cart-pole motion.

    Args:
        states: (num_steps, 4) state trajectory
        dt: Time step
        save_path: Path to save animation
        fps: Frames per second
    """
    print(f"\nGenerating animation with {len(states)} frames...")

    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Cart-pole parameters
    cart_width = 0.3
    cart_height = 0.2
    pole_length = 1.0

    # Axis limits
    x_min = states[:, 0].min() - 1.0
    x_max = states[:, 0].max() + 1.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, pole_length + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Initialize elements
    cart = plt.Rectangle((0, 0), cart_width, cart_height, fc='blue', ec='black')
    pole_line, = ax.plot([], [], 'r-', linewidth=3)
    pole_mass = plt.Circle((0, 0), 0.1, fc='red', ec='black')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.add_patch(cart)
    ax.add_patch(pole_mass)
    ax.axhline(0, color='black', linewidth=1)

    def init():
        cart.set_xy((0, 0))
        pole_line.set_data([], [])
        pole_mass.center = (0, 0)
        time_text.set_text('')
        return cart, pole_line, pole_mass, time_text

    def animate(i):
        x, theta = states[i, 0], states[i, 1]

        # Cart position
        cart.set_xy((x - cart_width/2, 0))

        # Pole position
        pole_x = x + pole_length * np.sin(theta)
        pole_y = pole_length * np.cos(theta)
        pole_line.set_data([x, pole_x], [cart_height, pole_y + cart_height])
        pole_mass.center = (pole_x, pole_y + cart_height)

        # Time
        time_text.set_text(f'Time: {i * dt:.2f} s')

        return cart, pole_line, pole_mass, time_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(states), interval=dt*1000, blit=True
    )

    # Save as MP4 if ffmpeg available
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"✓ Animation saved to {save_path}")
    except Exception as e:
        print(f"Warning: Could not save animation: {e}")
        print("  Install ffmpeg to enable video generation")

    plt.close()
    return anim


def plot_trajectories(states, controls, hamiltonians, config):
    """Plot state trajectories, control inputs, and Hamiltonian."""
    target_state = np.array(config['mpc'].get('x_target', config['mpc'].get('target_state', [0.0, 0.0, 0.0, 0.0])))
    dt = config['cartpole']['dt']
    time = np.arange(len(states)) * dt

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))

    # State plots
    state_names = ['x (cart position)', 'theta (pole angle)', 'x_dot (cart velocity)', 'theta_dot (pole angular velocity)']

    for i in range(4):
        ax = axs[i // 2, i % 2]
        ax.plot(time, states[:, i], label='Actual', linewidth=2)
        ax.axhline(target_state[i], color='r', linestyle='--', label='Target', linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(state_names[i])
        ax.set_title(f'{state_names[i]}')
        ax.legend()
        ax.grid(True)

    # Control input
    ax = axs[2, 0]
    ax.plot(time[:-1], controls, label='Control Force', linewidth=2, color='green')
    if 'u_min' in config['mpc'] and 'u_max' in config['mpc']:
        ax.axhline(config['mpc']['u_min'], color='r', linestyle='--', linewidth=1, label='Constraints')
        ax.axhline(config['mpc']['u_max'], color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Force')
    ax.set_title('MPC Control Input')
    ax.legend()
    ax.grid(True)

    # Hamiltonian
    ax = axs[2, 1]
    ax.plot(time[:-1], hamiltonians, label='Hamiltonian (Energy)', linewidth=2, color='purple')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Hamiltonian')
    ax.set_title('System Energy (Hamiltonian)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('results/control_trajectories.png')
    print("Trajectory plots saved to results/control_trajectories.png")
    plt.close()


def main():
    """Main function for MPC control and visualization."""
    print("=" * 70)
    print("Cart-Pole MPC Control with Original pHNN")
    print("=" * 70)

    # Load configuration
    config = load_config()

    # Load trained pHNN model
    config_path = "cartpole_mpc_config.yaml"
    weights_path = config['training']['model_save_path']
    model = load_trained_model(config_path, weights_path)

    # Create MPC controller
    print("\nCreating MPC controller...")
    mpc_controller = create_mpc_from_config(model, config)
    print(f"✓ MPC controller created")
    print(f"  Horizon: {mpc_controller.horizon} steps")
    print(f"  Control bounds: [{mpc_controller.u_min:.1f}, {mpc_controller.u_max:.1f}] N")
    print(f"  Optimization iterations: {mpc_controller.max_iterations}")

    # Create simulator
    dt = config['cartpole']['dt']
    simulator = CartPoleSimulator(dt=dt)
    print(f"✓ Created cart-pole simulator (dt={dt}s)")

    # Initial state (slightly off from upright)
    initial_state = np.array([0.0, 0.1, 0.0, 0.0])  # [x, theta, x_dot, theta_dot]

    # Run MPC control
    num_steps = config.get('mpc', {}).get('simulation_steps', 300)
    states, controls, hamiltonians, stability_achieved, stable_duration = run_mpc_control(
        simulator, mpc_controller, initial_state, num_steps, config, verbose=True
    )

    # Create output directory
    os.makedirs('results', exist_ok=True)

    # Compute and print metrics
    target_state = np.array(config['mpc']['x_target'])
    metrics = compute_metrics(states, controls, target_state)
    print_metrics(metrics)

    # Print stability results
    print("\nStability Metrics:")
    tolerance = np.array(config['stability']['tolerance'])
    min_duration = config['stability']['min_duration']
    print(f"  Tolerance: x=±{tolerance[0]}, θ=±{tolerance[1]}, ẋ=±{tolerance[2]}, θ̇=±{tolerance[3]}")
    print(f"  Required duration: {min_duration}s")
    if stability_achieved:
        print(f"  ✓ STABLE - Maintained target for {stable_duration:.2f}s")
    else:
        print(f"  ✗ NOT STABLE - Best duration: {stable_duration:.2f}s")

    # Plot trajectories
    print("\nGenerating plots...")
    plot_trajectories(states, controls, hamiltonians, config)

    # Create animation
    create_animation(states, dt, save_path='results/mpc_control_animation.mp4', fps=int(1.0/dt))

    # Save trajectory data
    print("\nSaving trajectory data...")
    np.savez(
        'results/mpc_control_trajectory.npz',
        states=states,
        controls=controls,
        hamiltonians=hamiltonians,
        dt=dt
    )
    print("✓ Saved trajectory data to results/mpc_control_trajectory.npz")

    print("\n" + "=" * 70)
    print("MPC Control Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - results/control_trajectories.png")
    print("  - results/mpc_control_animation.mp4")
    print("  - results/mpc_control_trajectory.npz")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
