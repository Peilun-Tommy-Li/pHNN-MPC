"""
Canonical Port-Hamiltonian Neural Network

This implementation uses proper canonical coordinates [q, p] (position, momentum)
instead of kinematic coordinates [q, q̇] (position, velocity).

Key differences from standard pHNN:
1. Fixed canonical J matrix (not learnable)
2. Constant learnable PSD R matrix (not state-dependent)
3. Hamiltonian defined on [q, p] coordinates
4. Coordinate transforms via learned mass matrix M(q)
"""

import torch
import torch.nn as nn
import yaml
from typing import Tuple, Optional

# Import utilities
try:
    from src.NN import MLP
    from src.mass_matrix import MassMatrixNetwork, CartPoleMassMatrix
    from src.coordinate_transforms import (
        kinematic_to_canonical,
        canonical_to_kinematic,
        momentum_to_velocity,
        split_state
    )
except ImportError:
    from NN import MLP
    from mass_matrix import MassMatrixNetwork, CartPoleMassMatrix
    from coordinate_transforms import (
        kinematic_to_canonical,
        canonical_to_kinematic,
        momentum_to_velocity,
        split_state
    )


class pHNN_Canonical(nn.Module):
    """
    Port-Hamiltonian Neural Network with Canonical Momentum Coordinates.

    Architecture:
    - Input: y = [q, q̇] (kinematic/observation space)
    - Internal: z = [q, p] (canonical coordinates)
    - Output: dy/dt in observation space

    Components:
    - M_net: Learnable mass matrix M(q) for coordinate transforms
    - H_net: Hamiltonian network H(q, p)
    - J: Fixed canonical skew-symmetric matrix
    - R: Constant learnable PSD dissipation matrix
    - G: Control input matrix (fixed)
    """

    def __init__(self, config_path: str):
        super().__init__()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.state_dim = config['model']['state_dim']
        self.input_dim = config['model']['input_dim']
        self.q_dim = self.state_dim // 2  # Position dimension

        # 1. Mass Matrix - Cart-Pole Specific Structure
        mass_config = config['model'].get('mass_matrix', {})
        mass_type = mass_config.get('type', 'cartpole')

        if mass_type == 'cartpole':
            # Use structured cart-pole mass matrix: M = [[a, b*cos(θ)], [b*cos(θ), c]]
            self.M_net = CartPoleMassMatrix(
                init_a=mass_config.get('init_a', 1.0),
                init_b=mass_config.get('init_b', 0.1),
                init_c=mass_config.get('init_c', 1.0)
            )
        else:
            # Fallback to general mass matrix network
            self.M_net = MassMatrixNetwork(
                q_dim=self.q_dim,
                mass_type=mass_type,
                hidden_sizes=mass_config.get('hidden_sizes', [64, 64]),
                activation=self._get_activation(mass_config.get('activation', 'nn.Tanh')),
                init_scale=mass_config.get('init_scale', 1.0)
            )

        # 2. Hamiltonian Network H(q, p)
        self.H_net = self._create_mlp(
            config['model']['H_mlp'],
            input_dim=self.state_dim,  # Takes [q, p]
            output_dim=1
        )

        # 3. Fixed Canonical J Matrix (skew-symmetric)
        # For 2-DOF system: J = [[0, I], [-I, 0]]
        J = self._create_canonical_J(self.q_dim)
        self.register_buffer('J', J)  # Not trainable

        # 4. Diagonal Learnable R Matrix (PSD)
        # Only learn diagonal elements (enforce non-negative via softplus)
        R_diag_init = torch.ones(self.state_dim) * 0.1
        self.R_diag_raw = nn.Parameter(R_diag_init)

        # 5. Fixed G Matrix for Control Input
        if config['model'].get('fixed_G', False):
            fixed_G_value = torch.tensor(config['model']['G_value'], dtype=torch.float32)
            self.register_buffer('G', fixed_G_value)
        else:
            raise ValueError("pHNN_Canonical requires fixed_G=True")

    def _create_canonical_J(self, q_dim: int) -> torch.Tensor:
        """
        Create canonical Port-Hamiltonian J matrix.

        For q_dim=2 (cart-pole):
        J = [ 0  0  1  0 ]
            [ 0  0  0  1 ]
            [-1  0  0  0 ]
            [ 0 -1  0  0 ]
        """
        n = 2 * q_dim
        J = torch.zeros(n, n)

        # Upper-right block: I
        J[:q_dim, q_dim:] = torch.eye(q_dim)

        # Lower-left block: -I
        J[q_dim:, :q_dim] = -torch.eye(q_dim)

        return J

    def _get_activation(self, activation_str: str) -> nn.Module:
        """Parse activation function from config string."""
        activation_class = getattr(nn, activation_str.split('.')[-1])
        return activation_class()

    def _create_mlp(self, params, input_dim, output_dim):
        """Create MLP from config parameters."""
        activation_class = getattr(nn, params['activation'].split('.')[-1])
        return MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=tuple(params['hidden_sizes']),
            activation=activation_class,
            dropout=params['dropout'],
            layer_norm=params['layer_norm'],
            bias=params['bias']
        )

    def get_R_matrix(self, batch_size: int) -> torch.Tensor:
        """
        Compute diagonal R matrix enforced to be PSD.

        Args:
            batch_size: Number of samples in batch

        Returns:
            R: (batch, state_dim, state_dim) diagonal PSD dissipation matrices
        """
        # Ensure diagonal elements are non-negative via softplus
        R_diag = torch.nn.functional.softplus(self.R_diag_raw) + 1e-4

        # Create diagonal matrix
        R = torch.diag(R_diag)

        # Expand to batch
        R = R.unsqueeze(0).expand(batch_size, -1, -1)

        return R

    def forward(
        self,
        y: torch.Tensor,
        u: torch.Tensor,
        return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Forward pass in canonical Port-Hamiltonian form.

        Args:
            y: (batch, state_dim) kinematic state [q, q̇]
            u: (batch, input_dim) control input
            return_intermediate: If True, return intermediate values

        Returns:
            dy_dt: (batch, state_dim) time derivative in kinematic space
            H: (batch,) Hamiltonian energy
            intermediate: Optional dict with z, p, q_dot_reconstructed, etc.
        """
        # Handle dimension expansion
        if y.ndim == 1:
            y = y.unsqueeze(0)
        elif y.ndim > 2:
            y = y.view(-1, y.shape[-1])

        if u.ndim == 1:
            u = u.unsqueeze(0)
        elif u.ndim > 2:
            u = u.view(-1, u.shape[-1])

        batch_size = y.shape[0]

        # 1. Transform to canonical coordinates: y = [q, q̇] → z = [q, p]
        z = kinematic_to_canonical(y, self.M_net)  # (batch, state_dim)

        # 2. Compute Hamiltonian H(q, p)
        H = self.H_net(z)  # (batch, 1)

        # 3. Compute gradient ∂H/∂z
        dH_dz = torch.autograd.grad(
            H.sum(),
            z,
            create_graph=True
        )[0]  # (batch, state_dim)

        # 4. Get R matrix (constant, PSD)
        R = self.get_R_matrix(batch_size)  # (batch, state_dim, state_dim)

        # 5. Compute canonical dynamics: dz/dt = (J - R) ∂H/∂z + G u
        J_batch = self.J.unsqueeze(0).expand(batch_size, -1, -1)
        G_batch = self.G.unsqueeze(0).expand(batch_size, -1, -1)

        dH_dz_col = dH_dz.unsqueeze(2)  # (batch, state_dim, 1)
        u_col = u.unsqueeze(2)  # (batch, input_dim, 1)

        dz_dt = torch.bmm(J_batch - R, dH_dz_col) + torch.bmm(G_batch, u_col)
        dz_dt = dz_dt.squeeze(2)  # (batch, state_dim)

        # 6. Transform back to kinematic space: dz/dt → dy/dt
        # This requires computing d/dt[q, M(q)q̇] and converting to d/dt[q, q̇]
        #
        # For now, use simplified approach:
        # dq/dt = q̇ (from canonical coords)
        # dp/dt = (from canonical dynamics)
        # Then: dy/dt = d/dt[q, q̇] = [q̇, q̈]
        #
        # More precisely:
        # q̇ = M^{-1}(q) p
        # dq/dt = q̇
        # q̈ = d/dt[M^{-1}(q) p] = M^{-1} dp/dt + (dM^{-1}/dq)(dq/dt) p
        #
        # For simplicity (avoiding dM/dq computation), approximate:
        # q̈ ≈ M^{-1} dp/dt

        q, p = split_state(z)
        dq_dt, dp_dt = split_state(dz_dt)

        # Reconstruct velocity
        q_dot = momentum_to_velocity(q, p, self.M_net)

        # Approximate acceleration (neglecting dM/dq term for now)
        q_ddot = momentum_to_velocity(q, dp_dt, self.M_net)

        # Combine: dy/dt = [dq/dt, d(q̇)/dt] = [q̇, q̈]
        dy_dt = torch.cat([q_dot, q_ddot], dim=1)

        # Prepare output
        H_scalar = H.squeeze(1)  # (batch,)

        if return_intermediate:
            intermediate = {
                'z': z,
                'q': q,
                'p': p,
                'q_dot_reconstructed': q_dot,
                'dH_dz': dH_dz,
                'dz_dt': dz_dt,
                'R': R
            }
            return dy_dt, H_scalar, intermediate
        else:
            return dy_dt, H_scalar, None

    def get_velocity_reconstruction(self, y: torch.Tensor) -> torch.Tensor:
        """
        Get reconstructed velocity from kinematic state.

        Useful for computing velocity reconstruction loss.

        Args:
            y: (batch, state_dim) kinematic state [q, q̇]

        Returns:
            q_dot_reconstructed: (batch, q_dim) reconstructed velocity
        """
        z = kinematic_to_canonical(y, self.M_net)
        q, p = split_state(z)
        q_dot = momentum_to_velocity(q, p, self.M_net)
        return q_dot
