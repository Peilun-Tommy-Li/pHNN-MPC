"""
Mass Matrix Network for Port-Hamiltonian Neural Networks

This module implements learnable mass matrix M(q) for transforming between
kinematic coordinates (position, velocity) and canonical coordinates (position, momentum).

The mass matrix is enforced to be positive-definite via Cholesky parameterization.
"""

import torch
import torch.nn as nn
from typing import Literal


class MassMatrixNetwork(nn.Module):
    """
    Learns configuration-dependent mass/inertia matrix M(q).

    Enforces positive-definiteness via Cholesky factorization: M = L @ L.T
    where L is lower-triangular.

    Supports three parameterization types:
    - 'constant': Single learnable matrix (independent of q)
    - 'diagonal': Configuration-dependent diagonal matrix
    - 'full': Configuration-dependent symmetric positive-definite matrix
    """

    def __init__(
        self,
        q_dim: int,
        mass_type: Literal['constant', 'diagonal', 'full'] = 'diagonal',
        hidden_sizes: list[int] = [64, 64],
        activation: nn.Module = nn.Tanh(),
        init_scale: float = 1.0
    ):
        """
        Args:
            q_dim: Dimension of position coordinates (e.g., 2 for cart-pole)
            mass_type: Type of mass matrix parameterization
            hidden_sizes: MLP hidden layer sizes (for diagonal/full types)
            activation: Activation function for MLP
            init_scale: Initial scale for mass matrix (centered around identity)
        """
        super().__init__()

        self.q_dim = q_dim
        self.mass_type = mass_type
        self.init_scale = init_scale

        if mass_type == 'constant':
            # Single learnable Cholesky factor
            # Initialize near identity: M ≈ I at start
            L_init = torch.eye(q_dim) * init_scale
            self.L_tril = nn.Parameter(L_init)
            self.mlp = None

        elif mass_type == 'diagonal':
            # Output q_dim positive values for diagonal of M
            # Use log parameterization: M_ii = exp(mlp_i(q))
            layers = []
            prev_size = q_dim
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(activation)
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, q_dim))
            self.mlp = nn.Sequential(*layers)

            # Initialize to output near zero (so M ≈ I initially)
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.constant_(self.mlp[-1].bias, 0.0)

        elif mass_type == 'full':
            # Output lower-triangular Cholesky factor
            # For q_dim=2: need 3 elements [L_00, L_10, L_11]
            # For q_dim=n: need n(n+1)/2 elements
            num_tril_elements = q_dim * (q_dim + 1) // 2

            layers = []
            prev_size = q_dim
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(activation)
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, num_tril_elements))
            self.mlp = nn.Sequential(*layers)

            # Initialize to output near identity
            nn.init.zeros_(self.mlp[-1].weight)
            # Bias for diagonal elements initialized to log(init_scale)
            # so exp(...) ≈ init_scale
            bias_init = torch.zeros(num_tril_elements)
            diag_indices = self._get_diagonal_indices(q_dim)
            bias_init[diag_indices] = torch.log(torch.tensor(init_scale))
            nn.init.constant_(self.mlp[-1].bias, 0.0)
            self.mlp[-1].bias.data = bias_init

        else:
            raise ValueError(f"Unknown mass_type: {mass_type}")

    def _get_diagonal_indices(self, n: int) -> list[int]:
        """Get indices of diagonal elements in lower-triangular vectorization."""
        indices = []
        idx = 0
        for i in range(n):
            indices.append(idx)
            idx += (i + 2)  # Next diagonal element
        return indices

    def _vector_to_lower_triangular(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Convert vector to lower-triangular matrix.

        Args:
            vec: (batch, n(n+1)/2) vector of lower-triangular elements

        Returns:
            L: (batch, n, n) lower-triangular matrices
        """
        batch_size = vec.shape[0]
        L = torch.zeros(batch_size, self.q_dim, self.q_dim,
                       dtype=vec.dtype, device=vec.device)

        # Fill lower triangle
        tril_indices = torch.tril_indices(self.q_dim, self.q_dim, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = vec

        return L

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute mass matrix M(q).

        Args:
            q: (batch, q_dim) position coordinates

        Returns:
            M: (batch, q_dim, q_dim) positive-definite mass matrices
        """
        batch_size = q.shape[0]

        if self.mass_type == 'constant':
            # Same M for all configurations
            # Enforce positive-definite via Cholesky
            L = torch.tril(self.L_tril)  # Ensure lower-triangular
            # Make diagonal positive via softplus
            L = L.clone()
            diag_indices = torch.arange(self.q_dim)
            L[diag_indices, diag_indices] = torch.nn.functional.softplus(
                L[diag_indices, diag_indices]
            ) + 1e-3  # Ensure strictly positive

            M = L @ L.T
            # Expand to batch
            M = M.unsqueeze(0).expand(batch_size, -1, -1)

        elif self.mass_type == 'diagonal':
            # Diagonal M(q)
            log_diag = self.mlp(q)  # (batch, q_dim)
            diag = torch.exp(log_diag) + 1e-3  # Ensure positive
            M = torch.diag_embed(diag)  # (batch, q_dim, q_dim)

        elif self.mass_type == 'full':
            # Full symmetric M(q)
            L_vec = self.mlp(q)  # (batch, n(n+1)/2)
            L = self._vector_to_lower_triangular(L_vec)

            # Make diagonal positive via softplus
            diag_indices = torch.arange(self.q_dim)
            L[:, diag_indices, diag_indices] = torch.nn.functional.softplus(
                L[:, diag_indices, diag_indices]
            ) + 1e-3

            M = torch.bmm(L, L.transpose(1, 2))  # (batch, n, n)

        return M

    def inverse(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse mass matrix M^{-1}(q).

        More efficient than computing M then inverting.

        Args:
            q: (batch, q_dim) position coordinates

        Returns:
            M_inv: (batch, q_dim, q_dim) inverse mass matrices
        """
        batch_size = q.shape[0]

        if self.mass_type == 'constant':
            L = torch.tril(self.L_tril)
            diag_indices = torch.arange(self.q_dim)
            L = L.clone()
            L[diag_indices, diag_indices] = torch.nn.functional.softplus(
                L[diag_indices, diag_indices]
            ) + 1e-3

            # M = L @ L.T, so M^{-1} = L^{-T} @ L^{-1}
            L_inv = torch.inverse(L)
            M_inv = L_inv.T @ L_inv
            M_inv = M_inv.unsqueeze(0).expand(batch_size, -1, -1)

        elif self.mass_type == 'diagonal':
            log_diag = self.mlp(q)
            diag = torch.exp(log_diag) + 1e-3
            diag_inv = 1.0 / diag
            M_inv = torch.diag_embed(diag_inv)

        elif self.mass_type == 'full':
            # Use torch.linalg.inv for full matrices
            M = self.forward(q)
            M_inv = torch.linalg.inv(M)

        return M_inv


class IdentityMassMatrix(nn.Module):
    """
    Trivial mass matrix M(q) = I for baseline comparisons.

    Useful for ablation studies to measure benefit of learning M(q).
    """

    def __init__(self, q_dim: int):
        super().__init__()
        self.q_dim = q_dim
        self.register_buffer('I', torch.eye(q_dim))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        batch_size = q.shape[0]
        return self.I.unsqueeze(0).expand(batch_size, -1, -1)

    def inverse(self, q: torch.Tensor) -> torch.Tensor:
        return self.forward(q)  # I^{-1} = I


class CartPoleMassMatrix(nn.Module):
    """
    Cart-pole specific mass matrix with known structure.

    For cart-pole system, the mass matrix has the form:
        M(θ) = [ a          b*cos(θ) ]
               [ b*cos(θ)   c        ]

    where:
    - a: effective mass for cart motion (> 0)
    - b: coupling term between cart and pole (can be positive or negative)
    - c: effective inertia for pole rotation (> 0)

    Only 3 learnable parameters instead of full 2×2 matrix.
    """

    def __init__(self, init_a: float = 1.0, init_b: float = 0.1, init_c: float = 1.0):
        """
        Args:
            init_a: Initial value for cart mass parameter (> 0)
            init_b: Initial value for coupling parameter
            init_c: Initial value for pole inertia parameter (> 0)
        """
        super().__init__()

        # Learnable parameters
        # Use log parameterization for a and c to ensure positivity
        self.log_a = nn.Parameter(torch.log(torch.tensor(init_a)))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.log_c = nn.Parameter(torch.log(torch.tensor(init_c)))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute mass matrix M(θ).

        Args:
            q: (batch, 2) position coordinates [x, θ]

        Returns:
            M: (batch, 2, 2) mass matrices
        """
        batch_size = q.shape[0]

        # Extract angle θ
        theta = q[:, 1]  # (batch,)

        # Compute parameters (ensure a, c > 0)
        a = torch.exp(self.log_a) + 1e-3  # Ensure strictly positive
        b = self.b
        c = torch.exp(self.log_c) + 1e-3

        # Compute cos(θ)
        cos_theta = torch.cos(theta)  # (batch,)

        # Build mass matrix
        # M = [ a        b*cos(θ) ]
        #     [ b*cos(θ)  c       ]
        M = torch.zeros(batch_size, 2, 2, dtype=q.dtype, device=q.device)

        # Expand scalars to batch dimension if needed
        a_expanded = torch.full((batch_size,), a.item() if a.dim() == 0 else a, dtype=q.dtype, device=q.device)
        b_expanded = torch.full((batch_size,), b.item() if b.dim() == 0 else b, dtype=q.dtype, device=q.device)
        c_expanded = torch.full((batch_size,), c.item() if c.dim() == 0 else c, dtype=q.dtype, device=q.device)

        M[:, 0, 0] = a_expanded
        M[:, 0, 1] = b_expanded * cos_theta
        M[:, 1, 0] = b_expanded * cos_theta
        M[:, 1, 1] = c_expanded

        return M

    def inverse(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse mass matrix M^{-1}(θ).

        For 2×2 symmetric matrix:
        M^{-1} = (1/det) * [ c        -b*cos(θ) ]
                           [ -b*cos(θ)  a        ]

        where det = a*c - b²*cos²(θ)

        Args:
            q: (batch, 2) position coordinates [x, θ]

        Returns:
            M_inv: (batch, 2, 2) inverse mass matrices
        """
        batch_size = q.shape[0]

        # Extract angle θ
        theta = q[:, 1]

        # Compute parameters
        a = torch.exp(self.log_a) + 1e-3
        b = self.b
        c = torch.exp(self.log_c) + 1e-3

        # Extract scalar values and expand to batch dimension
        a_val = a.item() if a.dim() == 0 else a
        b_val = b.item() if b.dim() == 0 else b
        c_val = c.item() if c.dim() == 0 else c

        a_expanded = torch.full((batch_size,), a_val, dtype=q.dtype, device=q.device)
        b_expanded = torch.full((batch_size,), b_val, dtype=q.dtype, device=q.device)
        c_expanded = torch.full((batch_size,), c_val, dtype=q.dtype, device=q.device)

        # Compute cos(θ)
        cos_theta = torch.cos(theta)
        b_cos_theta = b_expanded * cos_theta

        # Compute determinant
        det = a_expanded * c_expanded - b_cos_theta ** 2  # (batch,)

        # Ensure det > 0 (should be guaranteed if a, c > 0 and |b| small enough)
        det = det + 1e-6  # Add small epsilon for numerical stability

        # Build inverse matrix
        M_inv = torch.zeros(batch_size, 2, 2, dtype=q.dtype, device=q.device)
        M_inv[:, 0, 0] = c_expanded / det
        M_inv[:, 0, 1] = -b_cos_theta / det
        M_inv[:, 1, 0] = -b_cos_theta / det
        M_inv[:, 1, 1] = a_expanded / det

        return M_inv

    def get_parameters_dict(self) -> dict:
        """Get current parameter values for logging/debugging."""
        return {
            'a': (torch.exp(self.log_a) + 1e-3).item(),
            'b': self.b.item(),
            'c': (torch.exp(self.log_c) + 1e-3).item()
        }
