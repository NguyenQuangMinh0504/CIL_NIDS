from torchvision.datasets.cifar import CIFAR10, CIFAR100
import numpy as np
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize


class iData(object):
    train_trsf: dict
    test_trsf: dict
    common_trsf: dict
    class_order: dict
    train_data: np.ndarray
    train_targets: np.ndarray
    test_data: np.ndarray
    test_targets: np.ndarray
    use_path: bool


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        RandomCrop(size=32, padding=4),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=63/255),
    ]
    test_trsf = []
    common_trsf = [
        ToTensor(),
        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ]
    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset: CIFAR10 = CIFAR10("./data", train=True, download=True)
        test_dataset: CIFAR10 = CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        RandomCrop(size=32, padding=4),
        RandomHorizontalFlip(),
        ColorJitter(brightness=63/255),
    ]
    test_trsf = []
    common_trsf = [
        ToTensor(),
        Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ]
    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset: CIFAR100 = CIFAR100("./data", train=True, download=True)
        test_dataset: CIFAR100 = CIFAR100("./data", train=True, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)
