import torch.nn as nn
import math


# The MLP class remains unchanged.
class MLP(nn.Module):
    """
    Flexible MLP:
      - input_dim -> hidden_sizes (list) -> output_dim
      - activation can be a torch.nn class (e.g. nn.ReLU, nn.SiLU)
      - optional dropout and layernorm
    """
    def __init__(self, input_dim, output_dim, hidden_sizes=(128,128),
                 activation=nn.SiLU, dropout=0.0, layer_norm=False, bias=True):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h, bias=bias))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, output_dim, bias=bias))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        return self.net(x)