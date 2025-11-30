from torch.utils.data import Dataset

# -------------------------------
# This class preserve the entire trajectory in each batch, so we you sample, you always get data from 1 traj
# It AUTOMATICALLY AVOID batch from different trajectory
# -------------------------------


class TrajectoryStepDataset(Dataset):

    def __init__(self, states, inputs, derivatives, seq_len):
        """
        Args:
            states (torch.Tensor): (num_traj, timesteps, state_dim)
            inputs (torch.Tensor): (num_traj, timesteps, input_dim)
            derivatives (torch.Tensor): (num_traj, timesteps, state_dim)
            seq_len (int): The length of the continuous sequence to sample.
        """
        self.states = states
        self.inputs = inputs
        self.derivatives = derivatives
        self.num_traj, self.timesteps, _ = states.shape
        self.seq_len = seq_len

    def __len__(self):
        # Total number of possible sequences to sample
        return self.num_traj * (self.timesteps - self.seq_len + 1)

    def __getitem__(self, idx):
        # Map a flat index to a specific sequence within a specific trajectory
        traj_idx = idx // (self.timesteps - self.seq_len + 1)
        start_idx = idx % (self.timesteps - self.seq_len + 1)

        states_seq = self.states[traj_idx, start_idx: start_idx + self.seq_len]
        inputs_seq = self.inputs[traj_idx, start_idx: start_idx + self.seq_len]
        derivatives_seq = self.derivatives[traj_idx, start_idx: start_idx + self.seq_len]

        return states_seq, inputs_seq, derivatives_seq