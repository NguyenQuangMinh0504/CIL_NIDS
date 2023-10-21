import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task: int = -1
        """Current trained task"""
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return self._data_memory, self._targets_memory

    def _compute_accuracy(self, model: nn.Module, loader: DataLoader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == targets).sum()
                total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
