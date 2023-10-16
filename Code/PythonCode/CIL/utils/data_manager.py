from utils.data import iCIFAR10


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()


def _get_idata(dataset_name: str):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))
