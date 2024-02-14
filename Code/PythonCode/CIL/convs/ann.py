from torch import nn
from torch import Tensor


class ANN(nn.Module):
    def __init__(self, input_dim: int):
        super(ANN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.out_dim = 64

    def forward(self, x: Tensor):
        return {"features": self.layers(x)}


def get_ann(input_dim: int = 121):
    """Basic fully connected net with input -> Linear -> Relu -> Linear -> Relu"""
    return ANN(input_dim=input_dim)


class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
        )
        self.out_dim = 64

    def forward(self, x: Tensor):
        return {"features": self.layers(x)}


def get_dnn(input_dim: int = 121):
    """Basic fully connected net with input -> Linear -> Relu -> Linear -> Relu"""
    return DNN(input_dim=input_dim)
