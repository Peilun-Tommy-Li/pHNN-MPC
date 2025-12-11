"""
Numerical Integration Methods for Port-Hamiltonian Neural Networks

Implements differentiable integrators for training and simulation.
All integrators preserve gradient flow for backpropagation.
"""

import torch
import torch.nn as nn
from typing import Callable, Tuple


def euler_step(
    model: nn.Module,
    y: torch.Tensor,
    u: torch.Tensor,
    dt: float
) -> torch.Tensor:
    """
    Single Euler integration step (first-order).

    y_{t+1} = y_t + dt * f(y_t, u_t)

    Args:
        model: pHNN model with forward(y, u) -> (dy_dt, H, ...) or (dy_dt, H)
        y: (batch, state_dim) current state
        u: (batch, input_dim) control input
        dt: Time step size

    Returns:
        y_next: (batch, state_dim) next state
    """
    result = model(y, u)
    dy_dt = result[0]  # First output is always dy_dt
    y_next = y + dt * dy_dt
    return y_next


def rk4_step(
    model: nn.Module,
    y: torch.Tensor,
    u: torch.Tensor,
    dt: float
) -> torch.Tensor:
    """
    Single Runge-Kutta 4th order (RK4) integration step.

    More accurate than Euler, preserves gradients for backpropagation.

    Algorithm:
        k1 = f(y_t, u_t)
        k2 = f(y_t + dt/2 * k1, u_t)
        k3 = f(y_t + dt/2 * k2, u_t)
        k4 = f(y_t + dt * k3, u_t)
        y_{t+1} = y_t + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    Args:
        model: pHNN model with forward(y, u) -> (dy_dt, H)
        y: (batch, state_dim) current state
        u: (batch, input_dim) control input
        dt: Time step size

    Returns:
        y_next: (batch, state_dim) next state
    """
    # k1 = f(y_t, u_t)
    k1 = model(y, u)[0]

    # k2 = f(y_t + dt/2 * k1, u_t)
    y_temp = y + (dt / 2) * k1
    k2 = model(y_temp, u)[0]

    # k3 = f(y_t + dt/2 * k2, u_t)
    y_temp = y + (dt / 2) * k2
    k3 = model(y_temp, u)[0]

    # k4 = f(y_t + dt * k3, u_t)
    y_temp = y + dt * k3
    k4 = model(y_temp, u)[0]

    # Combine: y_{t+1} = y_t + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return y_next


def rk4_step_with_energy(
    model: nn.Module,
    y: torch.Tensor,
    u: torch.Tensor,
    dt: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RK4 integration step that also returns Hamiltonian energy.

    Args:
        model: pHNN model with forward(y, u) -> (dy_dt, H, ...)
        y: (batch, state_dim) current state
        u: (batch, input_dim) control input
        dt: Time step size

    Returns:
        y_next: (batch, state_dim) next state
        H: (batch,) Hamiltonian energy at current state
    """
    # k1 = f(y_t, u_t)
    result = model(y, u)
    k1, H = result[0], result[1]

    # k2 = f(y_t + dt/2 * k1, u_t)
    y_temp = y + (dt / 2) * k1
    k2 = model(y_temp, u)[0]

    # k3 = f(y_t + dt/2 * k2, u_t)
    y_temp = y + (dt / 2) * k2
    k3 = model(y_temp, u)[0]

    # k4 = f(y_t + dt * k3, u_t)
    y_temp = y + dt * k3
    k4 = model(y_temp, u)[0]

    # Combine
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return y_next, H


def rollout_trajectory(
    model: nn.Module,
    y0: torch.Tensor,
    controls: torch.Tensor,
    dt: float,
    integrator: str = 'rk4'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rollout trajectory using specified integrator.

    Args:
        model: pHNN model
        y0: (batch, state_dim) initial state
        controls: (batch, num_steps, input_dim) control inputs
        dt: Time step size
        integrator: 'euler' or 'rk4'

    Returns:
        trajectory: (batch, num_steps + 1, state_dim) state trajectory
        energies: (batch, num_steps + 1) Hamiltonian energies
    """
    batch_size, num_steps, input_dim = controls.shape
    state_dim = y0.shape[1]

    # Storage
    trajectory = torch.zeros(batch_size, num_steps + 1, state_dim,
                            dtype=y0.dtype, device=y0.device)
    energies = torch.zeros(batch_size, num_steps + 1,
                          dtype=y0.dtype, device=y0.device)

    # Initial state
    trajectory[:, 0] = y0

    # Initial energy
    result = model(y0, controls[:, 0])
    H0 = result[1]
    energies[:, 0] = H0

    # Select integrator
    if integrator == 'rk4':
        step_fn = rk4_step
    elif integrator == 'euler':
        step_fn = euler_step
    else:
        raise ValueError(f"Unknown integrator: {integrator}")

    # Rollout
    y_current = y0
    for t in range(num_steps):
        u_t = controls[:, t]  # (batch, input_dim)
        y_next = step_fn(model, y_current, u_t, dt)

        # Store
        trajectory[:, t + 1] = y_next

        # Compute energy at next state
        H = model(y_next, u_t)[1]
        energies[:, t + 1] = H

        y_current = y_next

    return trajectory, energies


def rollout_trajectory_differentiable(
    model: nn.Module,
    y0: torch.Tensor,
    controls: torch.Tensor,
    dt: float,
    integrator: str = 'rk4',
    return_energies: bool = False
) -> torch.Tensor:
    """
    Differentiable trajectory rollout for training.

    This version maintains gradients through the entire rollout for backpropagation.

    Args:
        model: pHNN model
        y0: (batch, state_dim) initial state
        controls: (batch, num_steps, input_dim) control inputs
        dt: Time step size
        integrator: 'euler' or 'rk4'
        return_energies: If True, also return energies

    Returns:
        trajectory: (batch, num_steps + 1, state_dim) state trajectory
        energies: (batch, num_steps + 1) [optional] Hamiltonian energies
    """
    batch_size, num_steps, input_dim = controls.shape
    state_dim = y0.shape[1]

    # Select integrator
    if integrator == 'rk4':
        step_fn = rk4_step_with_energy if return_energies else rk4_step
    elif integrator == 'euler':
        step_fn = lambda m, y, u, dt: (euler_step(m, y, u, dt), m(y, u)[1]) if return_energies else euler_step(m, y, u, dt)
    else:
        raise ValueError(f"Unknown integrator: {integrator}")

    # Initialize trajectory list (will stack at end)
    traj_list = [y0]
    energy_list = [] if return_energies else None

    # Initial energy
    if return_energies:
        H0 = model(y0, controls[:, 0])[1]
        energy_list.append(H0)

    # Rollout
    y_current = y0
    for t in range(num_steps):
        u_t = controls[:, t]

        if return_energies:
            y_next, H = step_fn(model, y_current, u_t, dt)
            energy_list.append(H)
        else:
            y_next = step_fn(model, y_current, u_t, dt)

        traj_list.append(y_next)
        y_current = y_next

    # Stack trajectory
    trajectory = torch.stack(traj_list, dim=1)  # (batch, num_steps+1, state_dim)

    if return_energies:
        energies = torch.stack(energy_list, dim=1)  # (batch, num_steps+1)
        return trajectory, energies
    else:
        return trajectory


def compare_integrators(
    model: nn.Module,
    y0: torch.Tensor,
    controls: torch.Tensor,
    dt: float
) -> dict:
    """
    Compare Euler vs RK4 integration accuracy.

    Args:
        model: pHNN model
        y0: (batch, state_dim) initial state
        controls: (batch, num_steps, input_dim) control inputs
        dt: Time step size

    Returns:
        comparison: Dict with 'euler_traj', 'rk4_traj', 'error', etc.
    """
    with torch.no_grad():
        # Euler
        euler_traj, euler_energies = rollout_trajectory(
            model, y0, controls, dt, integrator='euler'
        )

        # RK4
        rk4_traj, rk4_energies = rollout_trajectory(
            model, y0, controls, dt, integrator='rk4'
        )

        # Compare (if ground truth available, use controls as proxy)
        # For now, just compute difference
        traj_error = torch.norm(euler_traj - rk4_traj, dim=-1)  # (batch, num_steps+1)

        # Energy conservation
        euler_energy_drift = torch.abs(euler_energies[:, -1] - euler_energies[:, 0])
        rk4_energy_drift = torch.abs(rk4_energies[:, -1] - rk4_energies[:, 0])

    return {
        'euler_trajectory': euler_traj,
        'rk4_trajectory': rk4_traj,
        'trajectory_difference': traj_error,
        'euler_energies': euler_energies,
        'rk4_energies': rk4_energies,
        'euler_energy_drift': euler_energy_drift,
        'rk4_energy_drift': rk4_energy_drift
    }
