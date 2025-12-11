"""
Pole Stabilization MPC Control

Uses MPC to stabilize only the pole angle and angular velocity,
ignoring cart position. This is useful for testing the pole balancing
capability without worrying about cart position constraints.

Usage:
    python scripts/run_pole_stabilization_mpc.py
    python scripts/run_pole_stabilization_mpc.py --config pole_stabilization_config.yaml
"""

import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import sys
import os
import argparse

sys.path.append('src')
from pHNN import pHNN
from mpc_controller import MPCController
from cartpole_simulator import CartPoleSimulator


def load_config(config_path="pole_stabilization_config.yaml"):
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

    if hasattr(model, 'G_fixed'):
        print(f"  Fixed G matrix: {model.G_fixed.squeeze().numpy()}")

    return model


def create_mpc_from_config(phnn_model, config):
    """Create MPC controller from configuration."""
    mpc_config = config['mpc']

    Q_diag = mpc_config.get('Q_diag', [0.0, 500.0, 0.0, 50.0])
    R_diag = mpc_config.get('R_diag', [0.001])

    controller = MPCController(
        phnn_model=phnn_model,
        horizon=mpc_config.get('horizon', 25),
        dt=config['cartpole']['dt'],
        Q=Q_diag,
        R=R_diag[0],
        target_state=mpc_config.get('x_target', [0.0, 0.0, 0.0, 0.0]),
        u_min=mpc_config.get('u_min', -20.0),
        u_max=mpc_config.get('u_max', 20.0),
        optimizer_type='Adam',
        lr=mpc_config.get('learning_rate', 0.2),
        max_iterations=mpc_config.get('optimizer_steps', 40)
    )

    return controller


def run_pole_stabilization_mpc(simulator, mpc_controller, initial_state, num_steps, config, verbose=True):
    """
    Run MPC control focusing only on pole stabilization.

    Args:
        simulator: CartPoleSimulator instance
        mpc_controller: MPC controller instance
        initial_state: Initial state [x, theta, x_dot, theta_dot]
        num_steps: Number of simulation steps
        config: Configuration dictionary
        verbose: Print progress

    Returns:
        Dictionary with states, controls, metrics, etc.
    """
    if verbose:
        print("\n" + "="*70)
        print("Pole Stabilization MPC Control")
        print("="*70)
        print(f"Goal: Stabilize pole angle and angular velocity to zero")
        print(f"Cart position is NOT penalized (free to move)")
        print(f"Simulation: {num_steps} steps ({num_steps * config['cartpole']['dt']:.1f}s)")
        print("="*70 + "\n")

    # Reset simulator
    simulator.reset(initial_state)
    state = initial_state.copy()

    # Storage
    states = [state.copy()]
    controls = []
    hamiltonians = []

    # Stability tracking (only for pole)
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
        control_value = control.item() if isinstance(control, np.ndarray) else control
        controls.append(control_value)

        # Log Hamiltonian (needs gradients for pHNN forward pass)
        state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        control_tensor = torch.tensor([[control_value]], dtype=torch.float32)
        _, H = mpc_controller.model(state_tensor, control_tensor)
        hamiltonians.append(H.detach().item())

        # Check pole stability (ignore cart position)
        state_error = state - target_state
        # Only check theta and theta_dot (indices 1 and 3)
        pole_within_tolerance = (np.abs(state_error[1]) <= tolerance[1] and
                                 np.abs(state_error[3]) <= tolerance[3])

        # Update stability tracking
        if pole_within_tolerance:
            if stable_start_step is None:
                stable_start_step = step
                if verbose:
                    print(f"  Entered stable region at step {step} ({step * dt:.2f}s)")
                    print(f"    θ={state[1]:.4f} rad, θ̇={state[3]:.4f} rad/s")

            stable_duration = (step - stable_start_step + 1) * dt

            if stable_duration >= min_stable_duration and not stability_achieved:
                stability_achieved = True
                if verbose:
                    print(f"\n  ✓ POLE STABILIZED! Maintained for {stable_duration:.2f}s")
                    print(f"    Final state: x={state[0]:.3f}m, θ={state[1]:.4f}rad, ẋ={state[2]:.3f}m/s, θ̇={state[3]:.4f}rad/s")
        else:
            if stable_start_step is not None and verbose:
                print(f"  Left stable region at step {step} (was stable for {stable_duration:.2f}s)")
            stable_start_step = None
            stable_duration = 0.0

        # Step simulator
        next_state, done = simulator.step(control_value)

        # Store
        states.append(next_state.copy())

        # Update state
        state = next_state

        # Check termination (pole fell over)
        if done:
            if verbose:
                print(f"\n  ✗ Pole fell over at step {step + 1} ({(step+1)*dt:.2f}s)")
                print(f"    Final θ={state[1]:.4f} rad ({np.degrees(state[1]):.1f}°)")
            break

        if verbose and (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{num_steps}: x={state[0]:.2f}m, θ={state[1]:.4f}rad ({np.degrees(state[1]):.1f}°)")

    if verbose:
        print(f"\nControl episode completed with {len(states)} states")

    return {
        'states': np.array(states),
        'controls': np.array(controls),
        'hamiltonians': np.array(hamiltonians),
        'stability_achieved': stability_achieved,
        'stable_duration': stable_duration,
        'dt': dt
    }


def plot_pole_stabilization_results(data, config, save_path='results/pole_stabilization_results.png'):
    """Plot pole stabilization results."""
    states = data['states']
    controls = data['controls']
    hamiltonians = data['hamiltonians']
    dt = data['dt']

    time = np.arange(len(states)) * dt
    time_control = np.arange(len(controls)) * dt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Pole Angle (top-left)
    ax = axes[0, 0]
    ax.plot(time, states[:, 1], 'b-', linewidth=2, label='Pole Angle θ')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    tolerance = config['stability']['tolerance'][1]
    ax.axhline(tolerance, color='red', linestyle=':', linewidth=1, alpha=0.5, label=f'±{tolerance} rad tolerance')
    ax.axhline(-tolerance, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Pole Angle θ (rad)', fontsize=11)
    ax.set_title('Pole Angle θ (Primary Objective)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Statistics
    mean_theta = np.mean(np.abs(states[:, 1]))
    max_theta = np.max(np.abs(states[:, 1]))
    final_theta = states[-1, 1]
    ax.text(0.02, 0.98, f'Mean |θ|: {mean_theta:.4f} rad\nMax |θ|: {max_theta:.4f} rad\nFinal θ: {final_theta:.4f} rad',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 2. Pole Angular Velocity (top-right)
    ax = axes[0, 1]
    ax.plot(time, states[:, 3], 'r-', linewidth=2, label='Pole Angular Velocity θ̇')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    tolerance_vel = config['stability']['tolerance'][3]
    ax.axhline(tolerance_vel, color='red', linestyle=':', linewidth=1, alpha=0.5, label=f'±{tolerance_vel} rad/s tolerance')
    ax.axhline(-tolerance_vel, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Pole Angular Velocity θ̇ (rad/s)', fontsize=11)
    ax.set_title('Pole Angular Velocity θ̇ (Primary Objective)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Statistics
    mean_theta_dot = np.mean(np.abs(states[:, 3]))
    max_theta_dot = np.max(np.abs(states[:, 3]))
    final_theta_dot = states[-1, 3]
    ax.text(0.02, 0.98, f'Mean |θ̇|: {mean_theta_dot:.4f} rad/s\nMax |θ̇|: {max_theta_dot:.4f} rad/s\nFinal θ̇: {final_theta_dot:.4f} rad/s',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 3. Cart Position (bottom-left) - not penalized but shown for reference
    ax = axes[1, 0]
    ax.plot(time, states[:, 0], 'g-', linewidth=2, label='Cart Position x')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Cart Position x (m)', fontsize=11)
    ax.set_title('Cart Position x (NOT penalized)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Statistics
    cart_drift = np.abs(states[-1, 0] - states[0, 0])
    max_cart_pos = np.max(np.abs(states[:, 0]))
    ax.text(0.02, 0.98, f'Cart drift: {cart_drift:.3f} m\nMax |x|: {max_cart_pos:.3f} m\nFinal x: {states[-1, 0]:.3f} m',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # 4. Control Input (bottom-right)
    ax = axes[1, 1]
    ax.plot(time_control, controls, 'purple', linewidth=2, label='Control Force')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    u_min = config['mpc']['u_min']
    u_max = config['mpc']['u_max']
    ax.axhline(u_min, color='red', linestyle=':', linewidth=1, alpha=0.5, label=f'Control limits')
    ax.axhline(u_max, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Control Force (N)', fontsize=11)
    ax.set_title('Control Input', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Statistics
    mean_control = np.mean(np.abs(controls))
    max_control = np.max(np.abs(controls))
    control_effort = np.sum(controls**2)
    ax.text(0.02, 0.98, f'Mean |u|: {mean_control:.3f} N\nMax |u|: {max_control:.3f} N\nEffort: {control_effort:.1f}',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Overall title
    stability_text = "STABILIZED ✓" if data['stability_achieved'] else "NOT STABLE ✗"
    plt.suptitle(f'Pole Stabilization MPC Results - {stability_text}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved results to {save_path}")
    plt.close()


def create_animation(states, dt, save_path='results/pole_stabilization.mp4', fps=30):
    """Create animation of pole stabilization."""
    print(f"\nGenerating animation with {len(states)} frames...")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Cart-pole parameters
    cart_width = 0.3
    cart_height = 0.2
    pole_length = 1.0

    # Axis limits (follow cart)
    x_range = 3.0
    ax.set_xlim(-x_range, x_range)
    ax.set_ylim(-0.5, pole_length + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Initialize elements
    cart = plt.Rectangle((0, 0), cart_width, cart_height, fc='blue', ec='black')
    pole_line, = ax.plot([], [], 'r-', linewidth=3)
    pole_mass = plt.Circle((0, 0), 0.1, fc='red', ec='black')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    info_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax.add_patch(cart)
    ax.add_patch(pole_mass)
    ax.axhline(0, color='black', linewidth=1)

    def init():
        cart.set_xy((0, 0))
        pole_line.set_data([], [])
        pole_mass.center = (0, 0)
        time_text.set_text('')
        info_text.set_text('')
        return cart, pole_line, pole_mass, time_text, info_text

    def animate(i):
        x, theta = states[i, 0], states[i, 1]

        # Update axis to follow cart
        ax.set_xlim(x - x_range, x + x_range)

        # Cart position
        cart.set_xy((x - cart_width/2, 0))

        # Pole position
        pole_x = x + pole_length * np.sin(theta)
        pole_y = pole_length * np.cos(theta)
        pole_line.set_data([x, pole_x], [cart_height, pole_y + cart_height])
        pole_mass.center = (pole_x, pole_y + cart_height)

        # Time
        time_text.set_text(f'Time: {i * dt:.2f} s')

        # Info
        info_text.set_text(f'θ: {theta:.4f} rad ({np.degrees(theta):.1f}°)\nx: {x:.2f} m')

        return cart, pole_line, pole_mass, time_text, info_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(states), interval=dt*1000, blit=True
    )

    # Save as MP4
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"✓ Animation saved to {save_path}")
    except Exception as e:
        print(f"Warning: Could not save animation: {e}")
        print("  Install ffmpeg to enable video generation")

    plt.close()


def main():
    """Main function for pole stabilization MPC."""
    parser = argparse.ArgumentParser(description='Pole Stabilization MPC Control')
    parser.add_argument('--config', type=str, default='pole_stabilization_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--no-animation', action='store_true',
                        help='Skip animation generation')
    args = parser.parse_args()

    print("=" * 80)
    print("Pole Stabilization MPC - Original pHNN Model")
    print("=" * 80)

    # Load configuration
    config = load_config(args.config)
    print(f"\n✓ Loaded config from {args.config}")

    # Load trained pHNN model
    model_config_path = config['model']['config_path']
    weights_path = config['model']['weights_path']
    model = load_trained_model(model_config_path, weights_path)

    # Create MPC controller
    print("\nCreating MPC controller...")
    mpc_controller = create_mpc_from_config(model, config)
    print(f"✓ MPC controller created")
    print(f"  Horizon: {mpc_controller.horizon} steps")
    print(f"  Q weights: {config['mpc']['Q_diag']} (only pole penalized)")
    print(f"  Control bounds: [{mpc_controller.u_min:.1f}, {mpc_controller.u_max:.1f}] N")
    print(f"  Optimization iterations: {mpc_controller.max_iterations}")

    # Create simulator
    dt = config['cartpole']['dt']
    simulator = CartPoleSimulator(dt=dt)
    print(f"✓ Created cart-pole simulator (dt={dt}s)")

    # Get initial state
    initial_state = np.array(config['mpc']['initial_state'], dtype=np.float32)
    print(f"\nInitial state: x={initial_state[0]:.2f}m, θ={initial_state[1]:.3f}rad ({np.degrees(initial_state[1]):.1f}°)")

    # Run MPC control
    num_steps = config['mpc']['simulation_steps']
    data = run_pole_stabilization_mpc(
        simulator, mpc_controller, initial_state, num_steps, config, verbose=True
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print(f"Simulation time: {len(data['states']) * dt:.2f}s ({len(data['states'])} steps)")
    print(f"Pole stability achieved: {'YES ✓' if data['stability_achieved'] else 'NO ✗'}")
    if data['stability_achieved']:
        print(f"  Stable for: {data['stable_duration']:.2f}s")
    print(f"\nFinal state:")
    print(f"  Cart position: {data['states'][-1, 0]:.3f} m (drift: {abs(data['states'][-1, 0] - data['states'][0, 0]):.3f} m)")
    print(f"  Pole angle: {data['states'][-1, 1]:.4f} rad ({np.degrees(data['states'][-1, 1]):.2f}°)")
    print(f"  Cart velocity: {data['states'][-1, 2]:.3f} m/s")
    print(f"  Pole ang. vel: {data['states'][-1, 3]:.4f} rad/s")
    print(f"\nControl effort: {np.sum(data['controls']**2):.2f}")
    print(f"Mean |control|: {np.mean(np.abs(data['controls'])):.3f} N")
    print("=" * 80)

    # Plot results
    plot_pole_stabilization_results(data, config)

    # Create animation
    if not args.no_animation:
        create_animation(data['states'], dt, save_path='results/pole_stabilization.mp4')

    # Save trajectory data
    os.makedirs('results', exist_ok=True)
    np.savez(
        'results/pole_stabilization_trajectory.npz',
        states=data['states'],
        controls=data['controls'],
        hamiltonians=data['hamiltonians'],
        dt=dt
    )
    print("\n✓ Saved trajectory data to results/pole_stabilization_trajectory.npz")

    print("\n" + "=" * 80)
    print("Generated files:")
    print("  - results/pole_stabilization_results.png")
    print("  - results/pole_stabilization.mp4")
    print("  - results/pole_stabilization_trajectory.npz")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
