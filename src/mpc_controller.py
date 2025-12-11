"""
Model Predictive Control (MPC) Controller using pHNN dynamics model.

The MPC optimizes a sequence of control inputs over a finite horizon to minimize
a quadratic cost function while respecting the learned pHNN dynamics.

Cost function: sum_t [||x_t - x_target||^2_Q + ||u_t||^2_R]
Dynamics: x_{t+1} = x_t + dt * pHNN(x_t, u_t)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MPCController:
    """
    Model Predictive Control controller using pHNN as the predictive model.
    """

    def __init__(self, phnn_model, horizon, dt, Q, R, target_state=None,
                 u_min=None, u_max=None, x_min=None, x_max=None,
                 optimizer_type="Adam", lr=0.1, max_iterations=50):
        """
        Initialize MPC controller.

        Args:
            phnn_model: Trained pHNN model for dynamics prediction
            horizon: Prediction horizon (number of time steps)
            dt: Time step size
            Q: State cost weights (list or array of length state_dim)
            R: Control effort cost (scalar)
            target_state: Target state to track (default: zeros)
            u_min: Minimum control value (for constraints)
            u_max: Maximum control value (for constraints)
            x_min: Minimum state values (list/array, None for unbounded)
            x_max: Maximum state values (list/array, None for unbounded)
            optimizer_type: Optimizer type ("Adam" or "LBFGS")
            lr: Learning rate for optimizer
            max_iterations: Maximum optimization iterations
        """
        self.model = phnn_model
        self.model.eval()  # Set to evaluation mode

        self.horizon = horizon
        self.dt = dt

        # Cost weights
        self.Q = torch.diag(torch.tensor(Q, dtype=torch.float32)) if isinstance(Q, list) else torch.diag(Q)
        self.R = R

        # Target state
        if target_state is None:
            self.target_state = torch.zeros(phnn_model.J.shape[0])
        else:
            self.target_state = torch.tensor(target_state, dtype=torch.float32)

        # Control constraints
        self.u_min = u_min
        self.u_max = u_max

        # State constraints
        self.x_min = torch.tensor(x_min, dtype=torch.float32) if x_min is not None else None
        self.x_max = torch.tensor(x_max, dtype=torch.float32) if x_max is not None else None

        # Optimization settings
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.max_iterations = max_iterations

        # State dimension
        self.state_dim = phnn_model.J.shape[0]

    def compute_cost(self, states, controls):
        """
        Compute quadratic cost over trajectory with state constraints.

        Cost = sum_t [||x_t - x_target||^2_Q + ||u_t||^2_R] + barrier(state_constraints)

        Args:
            states: Predicted state sequence (horizon+1, state_dim)
            controls: Control sequence (horizon, input_dim)

        Returns:
            Total cost (scalar tensor)
        """
        cost = 0.0

        # State tracking cost
        for t in range(self.horizon + 1):
            state_error = states[t] - self.target_state
            cost += state_error @ self.Q @ state_error

            # State constraint penalties (soft barrier)
            if self.x_min is not None or self.x_max is not None:
                barrier_weight = 1000.0  # Large penalty for constraint violation

                if self.x_min is not None:
                    # Penalty for violating lower bound: exp(-k*(x - x_min)) when x < x_min
                    violation = self.x_min - states[t]
                    cost += barrier_weight * torch.sum(torch.relu(violation) ** 2)

                if self.x_max is not None:
                    # Penalty for violating upper bound: exp(k*(x - x_max)) when x > x_max
                    violation = states[t] - self.x_max
                    cost += barrier_weight * torch.sum(torch.relu(violation) ** 2)

        # Control effort cost
        for t in range(self.horizon):
            control_magnitude = controls[t] @ controls[t]  # ||u||^2
            cost += self.R * control_magnitude

        return cost

    def rollout_dynamics(self, x0, controls):
        """
        Rollout pHNN dynamics given initial state and control sequence.

        Args:
            x0: Initial state (state_dim,)
            controls: Control sequence (horizon, input_dim)

        Returns:
            State trajectory (horizon+1, state_dim)
        """
        states = [x0]

        for t in range(self.horizon):
            # Get current state and control - ensure requires_grad for pHNN
            x_t = states[-1].unsqueeze(0).requires_grad_(True)  # (1, state_dim)
            u_t = controls[t].unsqueeze(0)  # (1, input_dim)

            # Predict derivative using pHNN
            dx, _ = self.model(x_t, u_t)

            # Integrate forward: x_{t+1} = x_t + dt * dx
            x_next = states[-1] + self.dt * dx.squeeze(0)
            states.append(x_next)

        return torch.stack(states)

    def compute_control(self, current_state):
        """
        Compute optimal control input using MPC.

        Solves the optimization problem:
            min_{u_0, ..., u_{H-1}} sum_t [||x_t - x_target||^2_Q + ||u_t||^2_R]
            s.t. x_{t+1} = x_t + dt * pHNN(x_t, u_t)

        Returns only the first control input (receding horizon).

        Args:
            current_state: Current state (numpy array or torch tensor)

        Returns:
            Optimal control input (scalar or 1D array)
        """
        # Convert to tensor if needed
        if isinstance(current_state, np.ndarray):
            current_state = torch.tensor(current_state, dtype=torch.float32)

        # Initialize control sequence (start with zeros or use warm-start)
        control_sequence = torch.zeros(self.horizon, 1, requires_grad=True)

        # Create optimizer
        if self.optimizer_type == "Adam":
            optimizer = optim.Adam([control_sequence], lr=self.lr)
        elif self.optimizer_type == "LBFGS":
            optimizer = optim.LBFGS([control_sequence], lr=self.lr, max_iter=20)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # Optimization loop
        for iteration in range(self.max_iterations):
            def closure():
                optimizer.zero_grad()

                # Apply control constraints if specified
                if self.u_min is not None and self.u_max is not None:
                    controls = torch.clamp(control_sequence, self.u_min, self.u_max)
                else:
                    controls = control_sequence

                # Rollout dynamics
                states = self.rollout_dynamics(current_state, controls)

                # Compute cost
                cost = self.compute_cost(states, controls)

                # Backpropagate
                cost.backward()

                return cost

            if self.optimizer_type == "LBFGS":
                optimizer.step(closure)
            else:
                cost = closure()
                optimizer.step()

        # Extract first control from optimized sequence
        with torch.no_grad():
            if self.u_min is not None and self.u_max is not None:
                optimal_control = torch.clamp(control_sequence[0], self.u_min, self.u_max)
            else:
                optimal_control = control_sequence[0]

        return optimal_control.detach().numpy()


def create_mpc_from_config(phnn_model, config):
    """
    Create MPC controller from configuration dictionary.

    Args:
        phnn_model: Trained pHNN model
        config: Configuration dictionary with 'mpc' section

    Returns:
        MPCController instance
    """
    mpc_config = config['mpc']

    controller = MPCController(
        phnn_model=phnn_model,
        horizon=mpc_config['horizon'],
        dt=mpc_config['dt'],
        Q=mpc_config['Q'],
        R=mpc_config['R'],
        target_state=mpc_config.get('target_state', None),
        u_min=mpc_config.get('u_min', None),
        u_max=mpc_config.get('u_max', None),
        x_min=mpc_config.get('x_min', None),
        x_max=mpc_config.get('x_max', None),
        optimizer_type=mpc_config.get('optimizer', 'Adam'),
        lr=mpc_config.get('lr', 0.1),
        max_iterations=mpc_config.get('max_iterations', 50)
    )

    return controller
