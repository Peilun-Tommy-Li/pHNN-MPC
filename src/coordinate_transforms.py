"""
Coordinate Transformations for Port-Hamiltonian Neural Networks

Utilities for converting between:
- Kinematic/observation space: y = [q, q̇] (position, velocity)
- Canonical/momentum space: z = [q, p] (position, momentum)

The transformation is mediated by the learned mass matrix M(q): p = M(q) · q̇
"""

import torch
from typing import Tuple

try:
    from .mass_matrix import MassMatrixNetwork
except ImportError:
    from mass_matrix import MassMatrixNetwork


def velocity_to_momentum(
    q: torch.Tensor,
    q_dot: torch.Tensor,
    M_net: MassMatrixNetwork
) -> torch.Tensor:
    """
    Transform velocity to momentum: p = M(q) · q̇

    Args:
        q: (batch, q_dim) position coordinates
        q_dot: (batch, q_dim) velocity coordinates
        M_net: Mass matrix network

    Returns:
        p: (batch, q_dim) momentum coordinates
    """
    M = M_net(q)  # (batch, q_dim, q_dim)
    p = torch.bmm(M, q_dot.unsqueeze(-1)).squeeze(-1)  # (batch, q_dim)
    return p


def momentum_to_velocity(
    q: torch.Tensor,
    p: torch.Tensor,
    M_net: MassMatrixNetwork
) -> torch.Tensor:
    """
    Transform momentum to velocity: q̇ = M^{-1}(q) · p

    Args:
        q: (batch, q_dim) position coordinates
        p: (batch, q_dim) momentum coordinates
        M_net: Mass matrix network

    Returns:
        q_dot: (batch, q_dim) velocity coordinates
    """
    M_inv = M_net.inverse(q)  # (batch, q_dim, q_dim)
    q_dot = torch.bmm(M_inv, p.unsqueeze(-1)).squeeze(-1)  # (batch, q_dim)
    return q_dot


def kinematic_to_canonical(
    y: torch.Tensor,
    M_net: MassMatrixNetwork
) -> torch.Tensor:
    """
    Transform from kinematic to canonical coordinates.

    y = [q, q̇] → z = [q, p]

    Args:
        y: (batch, 2*q_dim) kinematic state [q, q̇]
        M_net: Mass matrix network

    Returns:
        z: (batch, 2*q_dim) canonical state [q, p]
    """
    q_dim = y.shape[1] // 2
    q = y[:, :q_dim]  # Position
    q_dot = y[:, q_dim:]  # Velocity

    p = velocity_to_momentum(q, q_dot, M_net)

    z = torch.cat([q, p], dim=1)  # [q, p]
    return z


def canonical_to_kinematic(
    z: torch.Tensor,
    M_net: MassMatrixNetwork
) -> torch.Tensor:
    """
    Transform from canonical to kinematic coordinates.

    z = [q, p] → y = [q, q̇]

    Args:
        z: (batch, 2*q_dim) canonical state [q, p]
        M_net: Mass matrix network

    Returns:
        y: (batch, 2*q_dim) kinematic state [q, q̇]
    """
    q_dim = z.shape[1] // 2
    q = z[:, :q_dim]  # Position
    p = z[:, q_dim:]  # Momentum

    q_dot = momentum_to_velocity(q, p, M_net)

    y = torch.cat([q, q_dot], dim=1)  # [q, q̇]
    return y


def split_state(state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split state into position and velocity/momentum.

    Works for both kinematic [q, q̇] and canonical [q, p] states.

    Args:
        state: (batch, 2*q_dim) full state

    Returns:
        q: (batch, q_dim) position coordinates
        v: (batch, q_dim) velocity or momentum coordinates
    """
    q_dim = state.shape[1] // 2
    q = state[:, :q_dim]
    v = state[:, q_dim:]
    return q, v


def combine_state(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Combine position and velocity/momentum into full state.

    Args:
        q: (batch, q_dim) position coordinates
        v: (batch, q_dim) velocity or momentum coordinates

    Returns:
        state: (batch, 2*q_dim) full state [q, v]
    """
    return torch.cat([q, v], dim=1)


def batch_matrix_vector_product(
    A: torch.Tensor,
    v: torch.Tensor
) -> torch.Tensor:
    """
    Batched matrix-vector product: A @ v

    Args:
        A: (batch, n, n) batch of matrices
        v: (batch, n) batch of vectors

    Returns:
        result: (batch, n) batch of vectors A @ v
    """
    return torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)


def compute_kinetic_energy(
    q: torch.Tensor,
    p: torch.Tensor,
    M_net: MassMatrixNetwork
) -> torch.Tensor:
    """
    Compute kinetic energy: T = (1/2) p^T M^{-1}(q) p

    Args:
        q: (batch, q_dim) position coordinates
        p: (batch, q_dim) momentum coordinates
        M_net: Mass matrix network

    Returns:
        T: (batch,) kinetic energy
    """
    q_dot = momentum_to_velocity(q, p, M_net)  # M^{-1} p
    # T = (1/2) p^T q̇ = (1/2) p^T M^{-1} p
    T = 0.5 * torch.sum(p * q_dot, dim=1)  # (batch,)
    return T


def verify_coordinate_transform(
    y: torch.Tensor,
    M_net: MassMatrixNetwork,
    tol: float = 1e-5
) -> Tuple[bool, float]:
    """
    Verify round-trip coordinate transformation: y → z → y'

    Checks if y ≈ y' (should be exact up to numerical precision).

    Args:
        y: (batch, 2*q_dim) original kinematic state
        M_net: Mass matrix network
        tol: Tolerance for round-trip error

    Returns:
        is_valid: True if round-trip error < tol
        max_error: Maximum absolute error in round-trip
    """
    z = kinematic_to_canonical(y, M_net)
    y_reconstructed = canonical_to_kinematic(z, M_net)

    error = torch.abs(y - y_reconstructed)
    max_error = torch.max(error).item()
    is_valid = max_error < tol

    return is_valid, max_error


def compute_velocity_reconstruction_error(
    q: torch.Tensor,
    q_dot_true: torch.Tensor,
    p: torch.Tensor,
    M_net: MassMatrixNetwork
) -> torch.Tensor:
    """
    Compute velocity reconstruction error: ||q̇_true - M^{-1}(q) p||²

    This is a key component of the training loss.

    Args:
        q: (batch, q_dim) position coordinates
        q_dot_true: (batch, q_dim) true velocity from data
        p: (batch, q_dim) predicted momentum from model
        M_net: Mass matrix network

    Returns:
        error: (batch,) MSE for each sample in batch
    """
    q_dot_recon = momentum_to_velocity(q, p, M_net)
    error = torch.sum((q_dot_recon - q_dot_true) ** 2, dim=1)  # (batch,)
    return error
