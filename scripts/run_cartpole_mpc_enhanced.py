"""
Enhanced Cart-Pole MPC Control with Information Overlay

Generates a higher-quality video with state information, control signals,
and performance metrics overlaid on the animation.

Usage:
    python scripts/run_cartpole_mpc_enhanced.py
"""

import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from pathlib import Path
import sys
import os

sys.path.append('src')
from pHNN import pHNN
from mpc_controller import MPCController
from cartpole_simulator import CartPoleSimulator


def create_mpc_from_config(phnn_model, config):
    """Create MPC controller from configuration."""
    mpc_config = config['mpc']

    Q_diag = mpc_config.get('Q_diag', [10.0, 100.0, 1.0, 10.0])
    R_diag = mpc_config.get('R_diag', [0.01])

    controller = MPCController(
        phnn_model=phnn_model,
        horizon=mpc_config.get('horizon', 20),
        dt=config['cartpole']['dt'],
        Q=Q_diag,
        R=R_diag[0],
        target_state=mpc_config.get('x_target', [0.0, 0.0, 0.0, 0.0]),
        u_min=mpc_config.get('u_min', -10.0),
        u_max=mpc_config.get('u_max', 10.0),
        optimizer_type='Adam',
        lr=mpc_config.get('learning_rate', 0.1),
        max_iterations=mpc_config.get('optimizer_steps', 50)
    )

    return controller


def load_config(config_path="cartpole_mpc_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_trained_model(config_path, weights_path):
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


def render_cartpole_frame(state, width=400, height=300):
    """Render a cart-pole frame from state for animation."""
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)

    # Cart-pole parameters
    cart_width = 0.3
    cart_height = 0.2
    pole_length = 1.0

    x, theta = state[0], state[1]

    # Set axis limits
    ax.set_xlim(x - 2.5, x + 2.5)
    ax.set_ylim(-0.5, pole_length + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=1)

    # Draw cart
    cart = Rectangle((x - cart_width/2, 0), cart_width, cart_height,
                     fc='blue', ec='black', linewidth=2)
    ax.add_patch(cart)

    # Draw pole
    pole_x = x + pole_length * np.sin(theta)
    pole_y = pole_length * np.cos(theta)
    ax.plot([x, pole_x], [cart_height, pole_y + cart_height],
            'r-', linewidth=3)
    ax.plot(pole_x, pole_y + cart_height, 'ro', markersize=10)

    # Convert to RGB array
    fig.canvas.draw()
    # Use buffer_rgba() and convert to RGB
    buf = np.array(fig.canvas.buffer_rgba())
    frame = buf[:, :, :3]  # Drop alpha channel to get RGB
    plt.close(fig)

    return frame


def run_mpc_control_enhanced(simulator, mpc_controller, config, random_init=True):
    """Run MPC control and collect detailed data for visualization.

    Args:
        simulator: CartPoleSimulator instance
        mpc_controller: MPC controller instance
        config: Configuration dictionary
        random_init: If True, start from random initial condition
    """
    max_steps = config['visualization']['max_control_steps']

    print(f"\nRunning MPC control for up to {max_steps} steps...")

    # Set initial state
    if random_init:
        # Sample random initial state
        # x: cart position in [-1, 1] meters
        # theta: pole angle in [-0.3, 0.3] radians (~17 degrees)
        # x_dot: cart velocity in [-0.5, 0.5] m/s
        # theta_dot: angular velocity in [-0.5, 0.5] rad/s
        x_init = np.random.uniform(-1.0, 1.0)
        theta_init = np.random.uniform(-0.3, 0.3)
        x_dot_init = np.random.uniform(-0.5, 0.5)
        theta_dot_init = np.random.uniform(-0.5, 0.5)

        initial_state = np.array([x_init, theta_init, x_dot_init, theta_dot_init], dtype=np.float32)

        print(f"Random initial condition:")
        print(f"  Cart position: {x_init:.3f} m")
        print(f"  Pole angle: {theta_init:.3f} rad ({np.degrees(theta_init):.1f}°)")
        print(f"  Cart velocity: {x_dot_init:.3f} m/s")
        print(f"  Angular velocity: {theta_dot_init:.3f} rad/s")
    else:
        # Default initial state (upright with small perturbation)
        initial_state = np.array([0.0, 0.05, 0.0, 0.0], dtype=np.float32)

    # Reset simulator
    simulator.reset(initial_state)
    state = initial_state.copy()

    # Storage
    states = [state.copy()]
    controls = []
    hamiltonians = []
    frames = []
    mpc_costs = []

    # Stability tracking
    target_state = np.array(config['mpc']['x_target'])
    tolerance = np.array(config['stability']['tolerance'])
    min_stable_duration = config['stability']['min_duration']
    dt = config['cartpole']['dt']

    stable_start_step = None
    stability_achieved = False
    stable_duration = 0.0

    # Capture initial frame
    frames.append(render_cartpole_frame(state))

    for step in range(max_steps):
        # Compute MPC control
        control = mpc_controller.compute_control(state)
        control_value = control.item() if isinstance(control, np.ndarray) else control
        controls.append(control_value)

        # Log Hamiltonian (needs gradients for pHNN forward pass)
        state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=True).unsqueeze(0)
        control_tensor = torch.tensor([[control_value]], dtype=torch.float32)
        _, H = mpc_controller.model(state_tensor, control_tensor)
        hamiltonians.append(H.detach().item())

        # Compute instantaneous cost
        state_error = state - target_state
        Q_diag = np.array(config['mpc']['Q_diag'])
        R_val = config['mpc']['R_diag'][0]
        instant_cost = np.sum(Q_diag * state_error**2) + R_val * control_value**2
        mpc_costs.append(instant_cost)

        # Check if within tolerance
        within_tolerance = np.all(np.abs(state_error) <= tolerance)

        # Update stability tracking
        if within_tolerance:
            if stable_start_step is None:
                stable_start_step = step
                print(f"  Entered stable region at step {step} ({step * dt:.2f}s)")

            stable_duration = (step - stable_start_step + 1) * dt

            if stable_duration >= min_stable_duration and not stability_achieved:
                stability_achieved = True
                print(f"  ✓ Stability achieved! Stayed within tolerance for {stable_duration:.2f}s")
                print(f"    State: x={state[0]:.4f}, θ={state[1]:.4f}, ẋ={state[2]:.4f}, θ̇={state[3]:.4f}")
        else:
            if stable_start_step is not None:
                print(f"  Left stable region at step {step} (was stable for {stable_duration:.2f}s)")
            stable_start_step = None
            stable_duration = 0.0

        # Step simulator
        next_state, done = simulator.step(control_value)

        # Store
        states.append(next_state.copy())

        # Capture frame
        frames.append(render_cartpole_frame(next_state))

        # Update state
        state = next_state

        # Check termination
        if done:
            print(f"Episode terminated at step {step + 1}")
            break

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{max_steps}")

    print(f"Control episode completed with {len(states)} states")

    return {
        'states': np.array(states),
        'controls': np.array(controls),
        'hamiltonians': np.array(hamiltonians),
        'costs': np.array(mpc_costs),
        'frames': frames,
        'stability_achieved': stability_achieved,
        'stable_duration': stable_duration
    }


def create_enhanced_video(data, config, save_path='results/cartpole_mpc_enhanced.mp4'):
    """
    Create enhanced video with information overlay using matplotlib animation.
    """
    print(f"\nCreating enhanced video with information overlay...")

    frames = data['frames']
    states = data['states']
    controls = data['controls']
    hamiltonians = data['hamiltonians']
    costs = data['costs']

    dt = config['cartpole']['dt']
    time = np.arange(len(states)) * dt

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 9))

    # Main animation area (left side)
    ax_anim = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    ax_anim.axis('off')

    # State plots (right side, top)
    ax_states = plt.subplot2grid((3, 2), (0, 1))
    ax_control = plt.subplot2grid((3, 2), (1, 1))
    ax_energy = plt.subplot2grid((3, 2), (2, 1))

    # Initialize animation image
    im = ax_anim.imshow(frames[0])
    title = ax_anim.set_title('', fontsize=14, fontweight='bold', loc='left')

    # State plots
    state_names = ['x', 'θ', 'ẋ', 'θ̇']
    colors = ['blue', 'red', 'green', 'purple']
    lines_states = []
    for i in range(4):
        line, = ax_states.plot([], [], color=colors[i], linewidth=2, label=state_names[i])
        lines_states.append(line)

    ax_states.set_xlim(0, time[-1])
    y_min = np.min(states) - 0.1
    y_max = np.max(states) + 0.1
    ax_states.set_ylim(y_min, y_max)
    ax_states.set_ylabel('State Values', fontsize=10, fontweight='bold')
    ax_states.legend(loc='upper right', fontsize=8)
    ax_states.grid(True, alpha=0.3)
    ax_states.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)

    # Control plot
    line_control, = ax_control.plot([], [], color='darkgreen', linewidth=2, label='Control Force')
    ax_control.set_xlim(0, time[-1])
    ax_control.set_ylim(np.min(controls) - 1, np.max(controls) + 1)
    ax_control.set_ylabel('Control (N)', fontsize=10, fontweight='bold')
    ax_control.legend(loc='upper right', fontsize=8)
    ax_control.grid(True, alpha=0.3)
    ax_control.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)

    # Energy and cost plot
    line_hamiltonian, = ax_energy.plot([], [], color='orange', linewidth=2, label='Hamiltonian')
    ax_energy_twin = ax_energy.twinx()
    line_cost, = ax_energy_twin.plot([], [], color='brown', linewidth=2, label='Cost', linestyle='--')
    ax_energy.set_xlim(0, time[-1])
    ax_energy.set_ylim(np.min(hamiltonians) - 0.5, np.max(hamiltonians) + 0.5)
    ax_energy_twin.set_ylim(np.min(costs) - 1, np.max(costs) + 1)
    ax_energy.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
    ax_energy.set_ylabel('Hamiltonian', fontsize=10, fontweight='bold', color='orange')
    ax_energy_twin.set_ylabel('Cost', fontsize=10, fontweight='bold', color='brown')
    ax_energy.tick_params(axis='y', labelcolor='orange')
    ax_energy_twin.tick_params(axis='y', labelcolor='brown')
    lines = [line_hamiltonian, line_cost]
    labels = [l.get_label() for l in lines]
    ax_energy.legend(lines, labels, loc='upper right', fontsize=8)
    ax_energy.grid(True, alpha=0.3)

    # Current time marker (vertical line)
    vline_states = ax_states.axvline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    vline_control = ax_control.axvline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    vline_energy = ax_energy.axvline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)

    def update(frame_idx):
        # Update animation image
        im.set_array(frames[frame_idx])

        # Update title with current state info
        t = frame_idx * dt
        if frame_idx < len(states):
            state = states[frame_idx]
            title_text = (f"Time: {t:.2f}s  |  "
                         f"x: {state[0]:+.3f}m  |  "
                         f"θ: {state[1]:+.3f}rad  |  "
                         f"ẋ: {state[2]:+.3f}m/s  |  "
                         f"θ̇: {state[3]:+.3f}rad/s")
            title.set_text(title_text)

        # Update state plots
        for i in range(4):
            lines_states[i].set_data(time[:frame_idx+1], states[:frame_idx+1, i])

        # Update control plot
        if frame_idx > 0:
            line_control.set_data(time[:frame_idx], controls[:frame_idx])

        # Update energy and cost plots
        if frame_idx > 0:
            line_hamiltonian.set_data(time[:frame_idx], hamiltonians[:frame_idx])
            line_cost.set_data(time[:frame_idx], costs[:frame_idx])

        # Update time markers
        vline_states.set_xdata([t, t])
        vline_control.set_xdata([t, t])
        vline_energy.set_xdata([t, t])

        return [im, title] + lines_states + [line_control, line_hamiltonian, line_cost, vline_states, vline_control, vline_energy]

    # Create animation
    fps = config['visualization']['gif_fps']
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000/fps, blit=True
    )

    # Save as MP4 (requires ffmpeg) or GIF
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    if save_path.endswith('.mp4'):
        # Save as MP4 (better quality, smaller size)
        try:
            anim.save(save_path, writer='ffmpeg', fps=fps, dpi=150,
                     extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            print(f"Enhanced video saved to {save_path}")
        except Exception as e:
            print(f"Could not save MP4 (ffmpeg may not be installed): {e}")
            print("Falling back to GIF...")
            gif_path = save_path.replace('.mp4', '.gif')
            anim.save(gif_path, writer='pillow', fps=fps)
            print(f"Enhanced GIF saved to {gif_path}")
    else:
        # Save as GIF
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Enhanced GIF saved to {save_path}")

    plt.close()


def main(random_init=True):
    """Main function for enhanced MPC visualization.

    Args:
        random_init: If True, use random initial condition. If False, use default initial state.
    """
    print("=" * 80)
    print("Cart-Pole MPC Control - Enhanced Visualization with Original pHNN")
    print("=" * 80)

    if random_init:
        print("\nMode: RANDOM initial condition")
    else:
        print("\nMode: DEFAULT initial condition (0.1 rad perturbation)")

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

    # Run MPC control with random or default initialization
    data = run_mpc_control_enhanced(simulator, mpc_controller, config, random_init=random_init)

    # Compute metrics
    target_state = np.array(config['mpc']['x_target'])
    state_errors = data['states'] - target_state
    mse = np.mean(np.sum(state_errors**2, axis=1))
    control_effort = np.sum(data['controls']**2)

    print("\n" + "=" * 80)
    print("Control Performance Metrics")
    print("=" * 80)
    print(f"State Tracking MSE: {mse:.6f}")
    print(f"Total Control Effort: {control_effort:.6f}")
    print(f"Episode Length: {len(data['states'])} steps ({len(data['states']) * config['cartpole']['dt']:.2f}s)")
    print(f"Average Cost: {np.mean(data['costs']):.6f}")
    print()
    print("Stability Metrics:")
    tolerance = np.array(config['stability']['tolerance'])
    min_duration = config['stability']['min_duration']
    print(f"  Tolerance: x=±{tolerance[0]}, θ=±{tolerance[1]}, ẋ=±{tolerance[2]}, θ̇=±{tolerance[3]}")
    print(f"  Required duration: {min_duration}s")
    if data['stability_achieved']:
        print(f"  ✓ STABLE - Maintained target for {data['stable_duration']:.2f}s")
    else:
        print(f"  ✗ NOT STABLE - Best duration: {data['stable_duration']:.2f}s")
    print("=" * 80)

    # Create enhanced video
    create_enhanced_video(data, config, 'results/cartpole_mpc_enhanced.mp4')

    # Also create simple GIF (from original script)
    print("\nCreating simple GIF animation...")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    im = ax.imshow(data['frames'][0])

    def update_simple(frame_idx):
        im.set_array(data['frames'][frame_idx])
        return [im]

    anim = animation.FuncAnimation(
        fig, update_simple, frames=len(data['frames']),
        interval=1000/config['visualization']['gif_fps'], blit=True
    )

    anim.save('results/cartpole_mpc_simple.gif', writer='pillow',
              fps=config['visualization']['gif_fps'])
    plt.close()
    print("Simple GIF saved to results/cartpole_mpc_simple.gif")

    print("\n" + "=" * 80)
    print("Enhanced MPC visualization complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - results/cartpole_mpc_enhanced.mp4 (or .gif if ffmpeg unavailable)")
    print("  - results/cartpole_mpc_simple.gif")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run MPC control with pHNN dynamics model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with random initial condition (default)
  python scripts/run_cartpole_mpc_enhanced.py

  # Run from default Gymnasium initial state
  python scripts/run_cartpole_mpc_enhanced.py --no-random

  # Run multiple times with different random seeds
  python scripts/run_cartpole_mpc_enhanced.py --seed 42
        """
    )

    parser.add_argument(
        '--no-random',
        action='store_true',
        help='Use default Gymnasium initial state instead of random'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible initial conditions'
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    main(random_init=not args.no_random)
