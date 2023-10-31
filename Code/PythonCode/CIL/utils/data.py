from torchvision.datasets.cifar import CIFAR10, CIFAR100
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from utils.helper import encode_numeric_zscore, encode_text_dummy, encode_text_index
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


class KDD99(iData):
    use_path = False
    train_trsf = []
    test_trsf = []

    def download_data(self):
        # try:
        #     path = get_file('kddcup.data_10_percent.gz', origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')
        # except Exception:
        #     print("Error downloading")
        #     raise
        path = "/Users/nguyenquangminh/Desktop/Thesis/Dataset/kddcup.data_10_percent"

        # Pre processing
        df = pd.read_csv(path, header=None)
        df.dropna(inplace=True, axis=1)
        df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                      'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                      'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
                      'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                      'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                      'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                      'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                      'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                      'dst_host_srv_rerror_rate', 'outcome']

        encode_numeric_zscore(df, 'duration')
        encode_text_dummy(df, 'protocol_type')
        encode_text_dummy(df, 'service')
        encode_text_dummy(df, 'flag')
        encode_numeric_zscore(df, 'src_bytes')
        encode_numeric_zscore(df, 'dst_bytes')
        encode_text_dummy(df, 'land')
        encode_numeric_zscore(df, 'wrong_fragment')
        encode_numeric_zscore(df, 'urgent')
        encode_numeric_zscore(df, 'hot')
        encode_numeric_zscore(df, 'num_failed_logins')
        encode_text_dummy(df, 'logged_in')
        encode_numeric_zscore(df, 'num_compromised')
        encode_numeric_zscore(df, 'root_shell')
        encode_numeric_zscore(df, 'su_attempted')
        encode_numeric_zscore(df, 'num_root')
        encode_numeric_zscore(df, 'num_file_creations')
        encode_numeric_zscore(df, 'num_shells')
        encode_numeric_zscore(df, 'num_access_files')
        encode_numeric_zscore(df, 'num_outbound_cmds')
        encode_text_dummy(df, 'is_host_login')
        encode_text_dummy(df, 'is_guest_login')
        encode_numeric_zscore(df, 'count')
        encode_numeric_zscore(df, 'srv_count')
        encode_numeric_zscore(df, 'serror_rate')
        encode_numeric_zscore(df, 'srv_serror_rate')
        encode_numeric_zscore(df, 'rerror_rate')
        encode_numeric_zscore(df, 'srv_rerror_rate')
        encode_numeric_zscore(df, 'same_srv_rate')
        encode_numeric_zscore(df, 'diff_srv_rate')
        encode_numeric_zscore(df, 'srv_diff_host_rate')
        encode_numeric_zscore(df, 'dst_host_count')
        encode_numeric_zscore(df, 'dst_host_srv_count')
        encode_numeric_zscore(df, 'dst_host_same_srv_rate')
        encode_numeric_zscore(df, 'dst_host_diff_srv_rate')
        encode_numeric_zscore(df, 'dst_host_same_src_port_rate')
        encode_numeric_zscore(df, 'dst_host_srv_diff_host_rate')
        encode_numeric_zscore(df, 'dst_host_serror_rate')
        encode_numeric_zscore(df, 'dst_host_srv_serror_rate')
        encode_numeric_zscore(df, 'dst_host_rerror_rate')
        encode_numeric_zscore(df, 'dst_host_srv_rerror_rate')
        outcomes = encode_text_index(df, 'outcome')
        df.dropna()
        logging.info(outcomes)

        y = df["outcome"].to_numpy()
        df.drop(labels="outcome", axis=1)

        self.train_data, self.train_targets, self.test_data, self.test_targets = train_test_split(
            df.to_numpy(), y, test_size=0.2, random_state=42)

        del df
