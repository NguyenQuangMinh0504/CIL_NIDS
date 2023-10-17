import json
from torch import nn


class ConfigEncoder(json.JSONEncoder):
    pass


def count_parameters(model: nn.Module, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
