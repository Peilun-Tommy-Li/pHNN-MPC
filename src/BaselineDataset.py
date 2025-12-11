"""
Simple dataset for baseline models (MLP, Neural ODE).

Returns (state, action, next_state) pairs for one-step prediction.
"""

import torch
from torch.utils.data import Dataset


class BaselineDataset(Dataset):
    """
    Dataset for baseline models that predict next state from current state and action.

    Each sample is a tuple: (state_t, action_t, state_{t+1})
    """

    def __init__(self, states, controls):
        """
        Args:
            states: (num_traj, seq_len, state_dim) - state trajectories
            controls: (num_traj, seq_len, action_dim) - control inputs

        Note: controls should have same seq_len as states, but we only use
              controls[:-1] since the last control doesn't have a next state.
        """
        self.states = states
        self.controls = controls
        self.num_traj, self.seq_len, self.state_dim = states.shape
        self.action_dim = controls.shape[-1]

        # Create all (state, action, next_state) pairs
        self.data = []

        for traj_idx in range(self.num_traj):
            for t in range(self.seq_len - 1):  # -1 because we need next state
                state = states[traj_idx, t]          # (state_dim,)
                action = controls[traj_idx, t]       # (action_dim,)
                next_state = states[traj_idx, t + 1] # (state_dim,)

                self.data.append({
                    'state': state,
                    'control': action,
                    'next_state': next_state
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
