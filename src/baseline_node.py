"""
Neural ODE Baseline for Cart-Pole Dynamics Learning

Uses torchdiffeq to learn continuous-time dynamics: dx/dt = f(x, u, t)
where f is parameterized by a neural network.
"""

import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. Install with: pip install torchdiffeq")


class ODEFunc(nn.Module):
    """
    Neural network that defines the ODE: dx/dt = f(x, u).

    The control input u is assumed constant over each integration step.
    """

    def __init__(self,
                 state_dim=4,
                 action_dim=1,
                 hidden_sizes=[128, 128, 128],
                 activation='tanh',
                 layer_norm=False):
        """
        Initialize ODE function network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: List of hidden layer sizes
            activation: Activation function
            layer_norm: Whether to use layer normalization
        """
        super(ODEFunc, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.current_action = None  # Will be set during integration

        # Choose activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network: f(x, u) -> dx/dt
        layers = []
        input_dim = state_dim + action_dim
        prev_size = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(act_fn())
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, state_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t, state):
        """
        Compute dx/dt at time t.

        Args:
            t: Time (scalar or tensor) - not used, but required by odeint
            state: (batch_size, state_dim) current state

        Returns:
            dx_dt: (batch_size, state_dim) state derivative
        """
        # Ensure current_action is set
        if self.current_action is None:
            raise RuntimeError("current_action must be set before calling forward")

        # Concatenate state and action
        batch_size = state.shape[0]
        action = self.current_action

        # Expand action to match batch size if needed
        if action.shape[0] == 1 and batch_size > 1:
            action = action.expand(batch_size, -1)

        x = torch.cat([state, action], dim=-1)

        # Compute derivative
        dx_dt = self.network(x)

        return dx_dt


class NeuralODE(nn.Module):
    """
    Neural ODE model for dynamics learning.

    Integrates the learned ODE to predict next state.
    """

    def __init__(self,
                 state_dim=4,
                 action_dim=1,
                 hidden_sizes=[128, 128, 128],
                 activation='tanh',
                 layer_norm=False,
                 solver='dopri5',
                 rtol=1e-3,
                 atol=1e-4):
        """
        Initialize Neural ODE.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: Hidden layer sizes for ODE function
            activation: Activation function
            layer_norm: Whether to use layer normalization
            solver: ODE solver ('dopri5', 'rk4', 'euler', 'adaptive_heun')
            rtol: Relative tolerance for adaptive solvers
            atol: Absolute tolerance for adaptive solvers
        """
        super(NeuralODE, self).__init__()

        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError("torchdiffeq is required. Install with: pip install torchdiffeq")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

        # ODE function
        self.ode_func = ODEFunc(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            layer_norm=layer_norm
        )

    def forward(self, state, action, dt):
        """
        Predict next state by integrating ODE.

        Args:
            state: (batch_size, state_dim) current state
            action: (batch_size, action_dim) control input (assumed constant)
            dt: Time step (scalar or tensor)

        Returns:
            next_state: (batch_size, state_dim) predicted next state
        """
        # Set the current action in the ODE function
        self.ode_func.current_action = action

        # Integration time points: [0, dt]
        # Handle dt as scalar or tensor, ensure on correct device
        if isinstance(dt, (int, float)):
            t = torch.tensor([0.0, dt], dtype=state.dtype, device=state.device)
        else:
            t = torch.cat([torch.zeros(1, dtype=state.dtype, device=state.device),
                          dt.to(state.device).unsqueeze(0) if dt.dim() == 0 else dt.to(state.device)])

        # Integrate ODE
        trajectory = odeint(
            self.ode_func,
            state,
            t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )

        # Return state at t=dt
        next_state = trajectory[-1]

        return next_state

    def predict_trajectory(self, initial_state, controls, dt):
        """
        Rollout trajectory given initial state and control sequence.

        Args:
            initial_state: (batch_size, state_dim) or (state_dim,)
            controls: (seq_len, action_dim) or (batch_size, seq_len, action_dim)
            dt: Time step (scalar)

        Returns:
            states: (seq_len+1, state_dim) or (batch_size, seq_len+1, state_dim)
        """
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        if controls.dim() == 2:
            controls = controls.unsqueeze(0)

        batch_size = initial_state.shape[0]
        seq_len = controls.shape[1]

        states = [initial_state]
        current_state = initial_state

        for t in range(seq_len):
            action = controls[:, t, :]
            next_state = self.forward(current_state, action, dt)
            states.append(next_state)
            current_state = next_state

        states = torch.stack(states, dim=1)  # (batch_size, seq_len+1, state_dim)

        if squeeze_output:
            states = states.squeeze(0)

        return states

    def get_model_info(self):
        """Return model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_type': 'NeuralODE',
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'solver': self.solver,
            'rtol': self.rtol,
            'atol': self.atol,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }
