"""
MPC Controller using Canonical pHNN Model

This controller uses the learned canonical Port-Hamiltonian Neural Network
to predict future trajectories and optimize control inputs.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import time


class MPCControllerCanonical:
    """
    Model Predictive Control using canonical pHNN for predictions.

    Uses the learned dynamics model to predict future states and optimizes
    control inputs to minimize a cost function over a prediction horizon.
    """

    def __init__(
        self,
        model: nn.Module,
        horizon: int = 20,
        dt: float = 0.02,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        x_target: Optional[np.ndarray] = None,
        u_min: float = -10.0,
        u_max: float = 10.0,
        optimizer_steps: int = 50,
        learning_rate: float = 0.1,
        verbose: bool = False
    ):
        """
        Initialize MPC controller.

        Args:
            model: Trained pHNN_Canonical model
            horizon: Prediction horizon (number of steps)
            dt: Time step size
            Q: State cost matrix (state_dim x state_dim)
            R: Control cost matrix (input_dim x input_dim)
            x_target: Target state (state_dim,)
            u_min: Minimum control input
            u_max: Maximum control input
            optimizer_steps: Number of optimization steps per MPC solve
            learning_rate: Learning rate for control optimization
            verbose: Print optimization progress
        """
        self.model = model
        self.model.eval()

        self.horizon = horizon
        self.dt = dt
        self.optimizer_steps = optimizer_steps
        self.learning_rate = learning_rate
        self.verbose = verbose

        # State and input dimensions
        self.state_dim = model.state_dim
        self.input_dim = model.input_dim

        # Cost matrices
        if Q is None:
            # Default: penalize position and velocity errors
            Q = np.eye(self.state_dim)
            Q[0, 0] = 10.0  # x position
            Q[1, 1] = 100.0  # theta (angle) - most important
            Q[2, 2] = 1.0   # x velocity
            Q[3, 3] = 10.0  # theta velocity
        self.Q = torch.tensor(Q, dtype=torch.float32)

        if R is None:
            # Default: small control penalty
            R = 0.01 * np.eye(self.input_dim)
        self.R = torch.tensor(R, dtype=torch.float32)

        # Target state
        if x_target is None:
            # Default: upright equilibrium at origin
            x_target = np.array([0.0, 0.0, 0.0, 0.0])  # [x, theta, x_dot, theta_dot]
        self.x_target = torch.tensor(x_target, dtype=torch.float32)

        # Control constraints
        self.u_min = u_min
        self.u_max = u_max

    def compute_cost(
        self,
        x_pred: torch.Tensor,
        u_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MPC cost function.

        Cost = sum_t [(x_t - x_target)^T Q (x_t - x_target) + u_t^T R u_t]

        Args:
            x_pred: (horizon+1, state_dim) predicted state trajectory
            u_seq: (horizon, input_dim) control sequence

        Returns:
            cost: Scalar cost
        """
        state_cost = 0.0
        control_cost = 0.0

        # State cost over horizon
        for t in range(self.horizon + 1):
            x_error = x_pred[t] - self.x_target
            state_cost += x_error @ self.Q @ x_error

        # Control cost over horizon
        for t in range(self.horizon):
            control_cost += u_seq[t] @ self.R @ u_seq[t]

        return state_cost + control_cost

    def rollout(
        self,
        x0: torch.Tensor,
        u_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Rollout trajectory using learned pHNN model.

        Args:
            x0: (state_dim,) initial state
            u_seq: (horizon, input_dim) control sequence

        Returns:
            x_traj: (horizon+1, state_dim) predicted state trajectory
        """
        from integrators import euler_step

        # Ensure x0 requires grad for backprop
        if not x0.requires_grad:
            x0 = x0.detach().requires_grad_(True)

        x_traj = [x0]
        x_current = x0

        for t in range(self.horizon):
            u_t = u_seq[t:t+1]  # (1, input_dim)

            # Single step prediction
            x_next = euler_step(
                self.model,
                x_current.unsqueeze(0),  # (1, state_dim)
                u_t.unsqueeze(0),  # (1, input_dim)
                self.dt
            )
            x_next = x_next.squeeze(0)  # (state_dim,)

            x_traj.append(x_next)
            x_current = x_next

        return torch.stack(x_traj)  # (horizon+1, state_dim)

    def optimize_control(
        self,
        x0: torch.Tensor,
        u_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Optimize control sequence using gradient descent.

        Args:
            x0: (state_dim,) current state
            u_init: (horizon, input_dim) initial control guess

        Returns:
            u_opt: (horizon, input_dim) optimized control sequence
            info: Dict with optimization info
        """
        # Initialize control sequence
        if u_init is None:
            u_seq = torch.zeros(self.horizon, self.input_dim, requires_grad=True)
        else:
            u_seq = u_init.clone().detach().requires_grad_(True)

        # Optimizer
        optimizer = torch.optim.Adam([u_seq], lr=self.learning_rate)

        costs = []
        best_cost = float('inf')
        best_u = None

        for step in range(self.optimizer_steps):
            optimizer.zero_grad()

            # Clamp controls to constraints (soft constraint during optimization)
            u_clamped = torch.clamp(u_seq, self.u_min, self.u_max)

            # Rollout trajectory
            x_pred = self.rollout(x0, u_clamped)

            # Compute cost
            cost = self.compute_cost(x_pred, u_clamped)

            # Backpropagation
            cost.backward()
            optimizer.step()

            # Track best solution
            cost_val = cost.item()
            costs.append(cost_val)

            if cost_val < best_cost:
                best_cost = cost_val
                best_u = u_clamped.detach().clone()

            if self.verbose and (step % 10 == 0 or step == self.optimizer_steps - 1):
                print(f"  Step {step:3d}: cost = {cost_val:.4f}")

        # Final clamping
        u_opt = torch.clamp(best_u, self.u_min, self.u_max)

        info = {
            'costs': costs,
            'final_cost': best_cost,
            'num_steps': self.optimizer_steps
        }

        return u_opt, info

    def control(
        self,
        x_current: np.ndarray,
        u_prev: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute MPC control action for current state.

        Args:
            x_current: (state_dim,) current state
            u_prev: (horizon, input_dim) previous control sequence (for warm start)

        Returns:
            u: (input_dim,) control action to apply
            info: Dict with MPC info
        """
        start_time = time.time()

        # Convert to torch
        x0 = torch.tensor(x_current, dtype=torch.float32)

        # Warm start from previous solution (shifted)
        if u_prev is not None:
            u_init = torch.tensor(u_prev, dtype=torch.float32)
            # Shift previous solution: [u1, u2, ..., uN] -> [u2, u3, ..., uN, 0]
            u_init = torch.cat([u_init[1:], torch.zeros(1, self.input_dim)], dim=0)
        else:
            u_init = None

        # Optimize control sequence
        u_seq_opt, opt_info = self.optimize_control(x0, u_init)

        # Return first control action (receding horizon)
        u = u_seq_opt[0].detach().numpy()

        solve_time = time.time() - start_time

        info = {
            'u_sequence': u_seq_opt.detach().numpy(),
            'solve_time': solve_time,
            'optimization': opt_info
        }

        return u, info


def create_mpc_controller(
    model: nn.Module,
    config: dict
) -> MPCControllerCanonical:
    """
    Create MPC controller from config.

    Args:
        model: Trained pHNN_Canonical model
        config: Configuration dict

    Returns:
        controller: MPCControllerCanonical instance
    """
    mpc_config = config.get('mpc', {})

    # Cost matrices
    Q_diag = mpc_config.get('Q_diag', [10.0, 100.0, 1.0, 10.0])
    R_diag = mpc_config.get('R_diag', [0.01])

    Q = np.diag(Q_diag)
    R = np.diag(R_diag)

    # Target state
    x_target = np.array(mpc_config.get('x_target', [0.0, 0.0, 0.0, 0.0]))

    controller = MPCControllerCanonical(
        model=model,
        horizon=mpc_config.get('horizon', 20),
        dt=config['cartpole']['dt'],
        Q=Q,
        R=R,
        x_target=x_target,
        u_min=mpc_config.get('u_min', -10.0),
        u_max=mpc_config.get('u_max', 10.0),
        optimizer_steps=mpc_config.get('optimizer_steps', 50),
        learning_rate=mpc_config.get('learning_rate', 0.1),
        verbose=mpc_config.get('verbose', False)
    )

    return controller
