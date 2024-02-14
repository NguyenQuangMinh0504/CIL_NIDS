from torch import Tensor
from torch import nn


class GeneralizedFC(nn.Module):
    def __init__(self, input_dim: int):
        super(GeneralizedFC, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )

    def forward(self, x: Tensor):
        return self.layers(x)


class SpecializedFC(nn.Module):
    feature_dim: int
    """Output dimension of net"""
    def __init__(self):
        self.feature_dim = 64
        super(SpecializedFC, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x: Tensor):
        return self.layers(x)


class DeepGeneralizedFC(nn.Module):
    def __init__(self, input_dim: int):
        super(DeepGeneralizedFC, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

    def forward(self, x: Tensor):
        return self.layers(x)


class DeepSpecializedFC(nn.Module):
    feature_dim: int
    """Output dimension of net"""
    def __init__(self):
        self.feature_dim = 64
        super(DeepSpecializedFC, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, x: Tensor):
        return self.layers(x)


def get_kdd_fc() -> (nn.Module, nn.Module):
    return GeneralizedFC(input_dim=121), SpecializedFC()


def get_memo_ann(input_dim: int) -> (nn.Module, nn.Module):
    return GeneralizedFC(input_dim=input_dim), SpecializedFC()


def get_memo_dnn(input_dim: int) -> (nn.Module, nn.Module):
    return DeepGeneralizedFC(input_dim=input_dim), DeepSpecializedFC()
