import numpy as np


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task: int = -1
        """Current trained task"""
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return self._data_memory, self._targets_memory
