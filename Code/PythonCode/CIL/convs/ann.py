from torch import nn
from torch import Tensor


class ANN(nn.Module):
    def __init__(self, input_dim: int):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x: Tensor):
        return self.layers(x)


def get_ann(input_dim: int = 121):
    """Basic fully connected net with input -> Linear -> Relu -> Linear -> Relu"""
    return ANN(input_dim=input_dim)
