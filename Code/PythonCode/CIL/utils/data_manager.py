import numpy as np
from utils.data import iData, iCIFAR10, iCIFAR100


class DataManager(object):
    dataset_name: str
    _increments: list
    """_increments[i]: Number of classes at training iterator i"""

    def __init__(self, dataset_name: str, shuffle: bool, seed: int, init_cls: int, increment: int):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "Not enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        """Return number of total classes at training iterator task"""
        return self._increments[task]

    def _setup_data(self, dataset_name, shuffle: bool, seed):
        idata: iData = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed=seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order


def _get_idata(dataset_name: str):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))
