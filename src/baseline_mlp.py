"""
Vanilla MLP Baseline for Cart-Pole Dynamics Learning

A deep feedforward neural network that learns to predict next state
given current state and action: f(x_t, u_t) -> x_{t+1}
"""

import torch
import torch.nn as nn


class VanillaMLP(nn.Module):
    """
    Vanilla MLP for dynamics learning.

    Architecture:
        Input: [state (4), action (1)] -> 5D
        Hidden layers: Configurable depth and width
        Output: next_state (4)
    """

    def __init__(self,
                 state_dim=4,
                 action_dim=1,
                 hidden_sizes=[256, 256, 256, 256],
                 activation='relu',
                 use_residual=True,
                 dropout=0.0,
                 layer_norm=False):
        """
        Initialize Vanilla MLP.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu', 'gelu')
            use_residual: If True, predict state delta instead of next state
            dropout: Dropout probability
            layer_norm: Whether to use layer normalization
        """
        super(VanillaMLP, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_residual = use_residual

        # Input dimension: state + action
        input_dim = state_dim + action_dim
        output_dim = state_dim

        # Choose activation function
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

        # Build MLP layers
        layers = []
        prev_size = input_dim

        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))

            # Layer normalization (optional)
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))

            # Activation
            layers.append(act_fn())

            # Dropout (optional)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_size = hidden_size

        # Output layer (no activation)
        layers.append(nn.Linear(prev_size, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state, action):
        """
        Forward pass.

        Args:
            state: (batch_size, state_dim) current state
            action: (batch_size, action_dim) control input

        Returns:
            next_state: (batch_size, state_dim) predicted next state
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)

        # Pass through network
        output = self.network(x)

        # Residual connection: predict state delta
        if self.use_residual:
            next_state = state + output
        else:
            next_state = output

        return next_state

    def predict_trajectory(self, initial_state, controls, dt=None):
        """
        Rollout trajectory given initial state and control sequence.

        Args:
            initial_state: (batch_size, state_dim) or (state_dim,)
            controls: (seq_len, action_dim) or (batch_size, seq_len, action_dim)
            dt: Time step (not used by MLP, kept for API compatibility)

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
            next_state = self.forward(current_state, action)
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
            'model_type': 'VanillaMLP',
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'use_residual': self.use_residual,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }
