from torchvision.datasets.cifar import CIFAR10, CIFAR100
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from utils.helper import encode_numeric_zscore, encode_text_dummy, encode_text_index
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize


class iData(object):
    train_trsf: list
    test_trsf: list
    common_trsf: list
    class_order: dict
    train_data: np.ndarray
    train_targets: np.ndarray
    test_data: np.ndarray
    test_targets: np.ndarray
    use_path: bool
    """True: load from array, False: load from real image path"""

    def download_data(self):
        """Downloading data"""
        logging.info("Downloading data........")


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
    common_trsf = [ToTensor()]

    def download_data(self):
        super().download_data()
        # try:
        #     path = get_file('kddcup.data_10_percent.gz', origin='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz')
        # except Exception:
        #     print("Error downloading")
        #     raise
        path = "../../../Dataset/kddcup.data_10_percent"

        # Pre processing
        df = pd.read_csv(path, header=None)

        df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                      'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                      'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
                      'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                      'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                      'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                      'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                      'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                      'dst_host_srv_rerror_rate', 'outcome']

        # logging.info(df["protocol_type"].value_counts(normalize=False, sort=True))
        # smurf_df = df[df["outcome"] == "smurf."]
        # normal_df = df[df["outcome"] == "normal."]
        # neptune_df = df[df["outcome"] == "neptune."]
        # back_df = df[df["outcome"] == "back."]
        # pod_df = df[df["outcome"] == "pod."]

        # logging.info("Protocol type ....")
        # logging.info(df["protocol_type"].value_counts(normalize=False, sort=True))
        # logging.info("smurf protocol type ....")
        # logging.info(smurf_df["protocol_type"].value_counts(normalize=False, sort=True))
        # logging.info("normal protocol type ....")
        # logging.info(normal_df["protocol_type"].value_counts(normalize=False, sort=True))

        # info of service attribute
        # logging.info(df["service"].unique())
        # logging.info("service value counts...")
        # logging.info(df["service"].value_counts(normalize=False, sort=True))
        # logging.info("service smurf ...")
        # logging.info(smurf_df["service"].value_counts(normalize=False, sort=True))
        # logging.info("service normal ...")
        # logging.info(normal_df["service"].value_counts(normalize=False, sort=True))

        # logging.info("outcome....")
        # logging.info(df["outcome"].value_counts(normalize=False, sort=True))
        # logging.info("src bytes...")
        # logging.info(df["src_bytes"].value_counts(normalize=False, sort=True))
        # logging.info("src bytes statistic ... ")
        # logging.info(df["src_bytes"].describe(percentiles=[0, 0.5, 0.99]))
        # logging.info("smurf src bytes ...")
        # logging.info(smurf_df["src_bytes"].value_counts(normalize=False, sort=True))
        # plt.scatter(df["src_bytes"], df["src_bytes"])

        # info of flag attribute
        # logging.info("Flag..................")
        # logging.info(df["flag"].unique())
        # logging.info(df["flag"].value_counts(normalize=False, sort=True))
        # logging.info(smurf_df["flag"].value_counts(normalize=False, sort=True))
        # logging.info(normal_df["flag"].value_counts(normalize=False, sort=True))

        # # info of source bytes attribute
        # logging.info("Src byte.............")
        # logging.info(df["src_bytes"].unique())
        # logging.info(df["src_bytes"].value_counts(normalize=False, sort=True))
        # logging.info(smurf_df["src_bytes"].value_counts(normalize=False, sort=True))
        # logging.info(normal_df["src_bytes"].value_counts(normalize=False, sort=True))

        # # info of neptune source bytes
        # logging.info("Source bytes of neptune dataset......")
        # logging.info(f"Len of neptune df is: {len(neptune_df)}")
        # logging.info(neptune_df["src_bytes"].value_counts(normalize=False, sort=True))
        # logging.info(neptune_df["dst_bytes"].value_counts(normalize=False, sort=True))

        # info of back attack
        # logging.info(back_df["src_bytes"].value_counts(normalize=False, sort=True))

        # logging.info("Wrong fragment...")
        # logging.info(df["wrong_fragment"].value_counts(normalize=False, sort=True))
        # logging.info("Pod wrong fragment")
        # logging.info(pod_df["wrong_fragment"].value_counts(normalize=False, sort=True))

        # ---------------- Droping all value that has less than 200 records.
        y_drop = ["spy.", "perl.", "phf.", "multihop.",
                  "ftp_write.", "loadmodule.", "rootkit.", "imap.",
                  "warezmaster.", "land.", "buffer_overflow.", "guess_passwd."]
        df.drop(df[df["outcome"].isin(y_drop)].index, inplace=True)
        # ----------------

        logging.info(df["outcome"].value_counts())

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

        # logging.info(df["outcome"].value_counts(normalize=False, sort=True))
        # logging.info((df[df["outcome"] == "smurf."]))
        # corr = df.corr()
        # logging.info(corr['outcome'].sort_values(ascending=False))
        # logging.info(corr)
        # outcomes = encode_text_index(df, 'outcome')
        self.label_dict = encode_text_index(df, "outcome")

        df.dropna(inplace=True, axis=1)

        # corr = df.corr()
        # print(corr)
        # print(corr['outcome'].sort_values(ascending=False))
        # logging.info(df['outcome'])
        # logging.info(outcomes)

        y = df["outcome"].to_numpy()

        df.drop(labels="outcome", axis=1)

        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            df.to_numpy(), y, test_size=0.2, random_state=42)

        self.train_data = self.train_data.astype(np.float32)
        self.test_data = self.test_data.astype(np.float32)
        # self.train_targets = self.train_targets.astype(np.float32)
        # self.test_targets = self.test_targets.astype(np.float32)

        del df


class TON_IoT_Network(iData):
    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = [ToTensor()]

    def download_data(self):
        path = "../../../Dataset/TON_IOT/Train_Test_datasets/Train_Test_Network_dataset/Train_Test_Network.csv"
        dataset = pd.read_csv(path)
        dataset.drop(columns=["ts", "src_ip", "dst_ip"], inplace=True)
        dataset.drop(columns=["dns_query"], inplace=True)
        dataset.drop(columns=["label"], inplace=True)

        for column in dataset.columns:
            if column != "type":
                if dataset[column].dtype == "object":
                    encode_text_dummy(dataset, column)
                else:
                    encode_numeric_zscore(dataset, column)
            else:
                self.label_dict = encode_text_index(dataset, column)

        dataset.dropna(axis=1, inplace=True)
        y = dataset["type"].to_numpy()
        dataset.drop(labels="type", axis=1)

        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            dataset.to_numpy(), y, test_size=0.2, random_state=42)

        self.train_data = self.train_data.astype(np.float32)
        self.test_data = self.test_data.astype(np.float32)

        del dataset
