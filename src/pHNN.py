import torch
import torch.nn as nn
import yaml

# Import MLP - handle both direct and module import
try:
    from src.NN import MLP
except ImportError:
    from NN import MLP


class pHNN(nn.Module):
    def __init__(self, config_path: str):
        super(pHNN, self).__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        state_dim = config['model']['state_dim']
        input_dim = config['model']['input_dim']

        # Constant learnable J
        self.J = nn.Parameter(torch.randn(state_dim, state_dim))

        # Make J skew-symmetric: J = (J - J^T)/2
        # This will be enforced in forward

        self.R_net = self._create_mlp(config['model']['R_mlp'], state_dim, state_dim * state_dim)
        self.H_net = self._create_mlp(config['model']['H_mlp'], state_dim, 1)

        # G matrix: either fixed or learned
        if config['model'].get('fixed_G', False):
            # Fixed G matrix (non-trainable)
            fixed_G_value = torch.tensor(config['model']['G_value'], dtype=torch.float32)
            self.register_buffer('G_fixed', fixed_G_value)
            self.G_net = None
        else:
            # Learned G matrix (state-dependent)
            self.G_net = self._create_mlp(config['model']['G_mlp'], state_dim, input_dim * state_dim)

    def _create_mlp(self, params, input_dim, output_dim):
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

    def forward(self, x, u):
        """
        x: (B, n)
        u: (B, m)
        returns: dx/dt
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim > 2:
            x = x.view(-1, x.shape[-1])

        if u.ndim == 1:
            u = u.unsqueeze(0)
        elif u.ndim > 2:
            u = u.view(-1, u.shape[-1])

        B, n = x.shape
        _, m = u.shape

        # Hamiltonian
        H = self.H_net(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]  # (B, n)

        # PSD R
        # 1. Output the n*n elements from the MLP
        R_raw = self.R_net(x).view(B, n, n)
        # 2. Make it symmetric and then use a method to enforce PSD, like this simple one:
        R_sym = (R_raw + R_raw.transpose(1, 2)) / 2
        R = torch.bmm(R_sym, R_sym.transpose(1, 2))

        # constant J
        J_batch = (self.J - self.J.T).unsqueeze(0).expand(B, n, n)

        # control - either fixed or learned G
        if self.G_net is None:
            # Use fixed G matrix
            G = self.G_fixed.unsqueeze(0).expand(B, -1, -1)  # (B, n, m)
        else:
            # Use learned G matrix
            G = self.G_net(x)  # (B, n*m)
            G = G.view(B, n, m)  # (B, n, m)

        u = u.unsqueeze(2)  # (B, m, 1)
        dH = dH.unsqueeze(2)  # (B, n, 1)

        dx = torch.bmm(J_batch - R, dH) + torch.bmm(G, u)  # (B, n, 1)
        dx = dx.squeeze(2)  # (B, n)

        return dx, H.squeeze(1)

