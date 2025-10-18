# estlib/il/model.py
import torch
import torch.nn as nn

class PolicyMLP(nn.Module):
    """
    Simple MLP policy.
    Input:  x_t ∈ R^{in_dim}
    Output: a_t ∈ R^{act_dim}
    """
    def __init__(self, in_dim: int, act_dim: int, hidden: int = 512, layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * layers + [act_dim]
        layers_ = []
        for i in range(len(dims) - 2):
            layers_ += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers_ += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
