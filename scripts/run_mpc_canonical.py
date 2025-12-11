"""
Run MPC controller with canonical pHNN model and generate visualizations.

This script:
1. Loads trained canonical pHNN model
2. Creates MPC controller
3. Simulates cart-pole control
4. Saves results and animations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import yaml
import sys
import os
sys.path.append('src')

from pHNN_canonical import pHNN_Canonical
from mpc_controller_canonical import create_mpc_controller
from cartpole_simulator import CartPoleSimulator



def simulate_mpc_control(
    simulator,
    controller,
    initial_state,
    num_steps=500,
    verbose=True
):
    """
    Simulate MPC control of cart-pole.

    Args:
        simulator: CartPoleSimulator instance
        controller: MPCControllerCanonical
        initial_state: Initial state [x, theta, x_dot, theta_dot]
        num_steps: Number of simulation steps
        verbose: Print progress

    Returns:
        states: (num_steps+1, state_dim) state trajectory
        controls: (num_steps, input_dim) control inputs
        costs: (num_steps,) MPC costs
        solve_times: (num_steps,) MPC solve times
    """
    states = [initial_state]
    controls = []
    costs = []
    solve_times = []

    # Set initial state
    simulator.reset(initial_state)

    x_current = initial_state
    u_prev = None

    if verbose:
        print(f"\nRunning MPC simulation for {num_steps} steps...")
        print(f"Initial state: x={x_current[0]:.3f}, θ={x_current[1]:.3f}, "
              f"ẋ={x_current[2]:.3f}, θ̇={x_current[3]:.3f}")

    for step in range(num_steps):
        # Compute MPC control
        u, info = controller.control(x_current, u_prev)

        # Apply control
        x_next, done = simulator.step(u)

        # Store
        controls.append(u)
        states.append(x_next)
        costs.append(info['optimization']['final_cost'])
        solve_times.append(info['solve_time'])

        # Update for next iteration
        x_current = x_next
        u_prev = info['u_sequence']

        # Print progress
        if verbose and (step % 50 == 0 or step == num_steps - 1):
            print(f"Step {step:3d}: x={x_current[0]:.3f}, θ={x_current[1]:.3f}, "
                  f"u={u[0]:.3f}, cost={costs[-1]:.2f}, "
                  f"solve_time={solve_times[-1]*1000:.1f}ms")

        # Check if terminated
        if done:
            if verbose:
                print(f"Episode terminated at step {step}")
            break

    states = np.array(states)
    controls = np.array(controls)
    costs = np.array(costs)
    solve_times = np.array(solve_times)

    if verbose:
        print(f"\nSimulation complete!")
        print(f"  Average solve time: {np.mean(solve_times)*1000:.1f} ms")
        print(f"  Final state: x={states[-1, 0]:.3f}, θ={states[-1, 1]:.3f}")

    return states, controls, costs, solve_times


def plot_results(states, controls, costs, solve_times, dt, save_path=None):
    """Plot MPC control results."""
    time = np.arange(len(states)) * dt

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Position
    ax = axes[0, 0]
    ax.plot(time, states[:, 0], 'b-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5, label='Target')
    ax.set_ylabel('Cart Position x (m)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Cart Position', fontweight='bold')

    # Angle
    ax = axes[0, 1]
    ax.plot(time, states[:, 1], 'b-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5, label='Target (upright)')
    ax.set_ylabel('Pole Angle θ (rad)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Pole Angle', fontweight='bold')

    # Cart velocity
    ax = axes[1, 0]
    ax.plot(time, states[:, 2], 'g-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('Cart Velocity ẋ (m/s)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('Cart Velocity', fontweight='bold')

    # Pole angular velocity
    ax = axes[1, 1]
    ax.plot(time, states[:, 3], 'g-', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('Pole Velocity θ̇ (rad/s)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('Pole Angular Velocity', fontweight='bold')

    # Control input
    ax = axes[2, 0]
    control_time = time[:-1]
    ax.plot(control_time, controls[:, 0], 'm-', linewidth=2)
    ax.set_ylabel('Control Force u (N)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('MPC Control Input', fontweight='bold')

    # MPC cost and solve time
    ax = axes[2, 1]
    ax2 = ax.twinx()

    line1 = ax.plot(control_time, costs, 'r-', linewidth=2, label='MPC Cost')
    ax.set_ylabel('MPC Cost', fontsize=11, color='r')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.tick_params(axis='y', labelcolor='r')

    line2 = ax2.plot(control_time, solve_times * 1000, 'b-', linewidth=2, label='Solve Time')
    ax2.set_ylabel('Solve Time (ms)', fontsize=11, color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('MPC Performance', fontweight='bold')

    plt.suptitle('Canonical pHNN-MPC Control Results',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved results plot to {save_path}")

    plt.close()


def create_animation(states, dt, save_path=None, fps=30):
    """
    Create animation of cart-pole motion.

    Args:
        states: (num_steps, 4) state trajectory
        dt: Time step
        save_path: Path to save animation
        fps: Frames per second
    """
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

    if save_path:
        print(f"Saving animation to {save_path} (this may take a while)...")
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"✓ Saved animation to {save_path}")

    plt.close()
    return anim


def main():
    """Main execution function."""
    print("="*70)
    print("Canonical pHNN-MPC Controller")
    print("="*70)

    # Load configuration
    config_path = "cartpole_mpc_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load trained model
    print("\nLoading trained canonical pHNN model...")
    model_path = config['training']['model_save_path']

    model = pHNN_Canonical(config_path)
    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"✓ Loaded model from {model_path}")

    # Create MPC controller
    print("\nCreating MPC controller...")
    controller = create_mpc_controller(model, config)
    print(f"✓ MPC controller created")
    print(f"  Horizon: {controller.horizon} steps")
    print(f"  Control bounds: [{controller.u_min:.1f}, {controller.u_max:.1f}] N")
    print(f"  Optimization steps: {controller.optimizer_steps}")

    # Create simulator
    dt = config['cartpole']['dt']
    simulator = CartPoleSimulator(dt=dt)
    print(f"✓ Created cart-pole simulator (dt={dt}s)")

    # Initial state (slightly off from upright)
    initial_state = np.array([0.0, 0.1, 0.0, 0.0])  # [x, theta, x_dot, theta_dot]

    # Run simulation
    num_steps = config.get('mpc', {}).get('simulation_steps', 300)

    states, controls, costs, solve_times = simulate_mpc_control(
        simulator, controller, initial_state, num_steps=num_steps, verbose=True
    )

    # Create output directory
    os.makedirs('results', exist_ok=True)

    # Plot results
    print("\nGenerating plots...")
    plot_results(
        states, controls, costs, solve_times, dt,
        save_path='results/mpc_canonical_results.png'
    )

    # Create animation
    print("\nGenerating animation...")
    create_animation(
        states, dt,
        save_path='results/mpc_canonical_animation.mp4',
        fps=int(1.0 / dt)
    )

    # Save trajectory data
    print("\nSaving trajectory data...")
    np.savez(
        'results/mpc_canonical_trajectory.npz',
        states=states,
        controls=controls,
        costs=costs,
        solve_times=solve_times,
        dt=dt
    )
    print("✓ Saved trajectory data to results/mpc_canonical_trajectory.npz")

    print("\n" + "="*70)
    print("MPC Simulation Complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/mpc_canonical_results.png")
    print("  - results/mpc_canonical_animation.mp4")
    print("  - results/mpc_canonical_trajectory.npz")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
