import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from utils.data import iData
from sklearn.datasets import make_classification


class Random_Data(iData):
    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = [ToTensor()]

    def __init__(self, **kwargs):
        self.pre_processing = kwargs["pre_processing"]

    def download_data(self):
        X, y = make_classification(random_state=42,
                                   n_classes=10,
                                   n_samples=10000,
                                   n_informative=15,
                                   n_features=15,
                                   n_clusters_per_class=1,
                                   class_sep=6,
                                   n_redundant=0,
                                   flip_y=0.0)

        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            X, y, test_size=0.2, random_state=42)

        self.train_data = self.train_data.astype(np.float32)
        self.test_data = self.test_data.astype(np.float32)
