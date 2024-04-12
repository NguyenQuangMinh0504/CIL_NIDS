import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from utils.data import iData
from utils.helper import encode_text_index, encode_text_dummy, encode_numeric_zscore, encode_numeric_min_max_scale, check_invalid_data


class TON_IoT_Network(iData):
    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = [ToTensor()]

    def __init__(self, **kwargs):
        self.pre_processing = kwargs["pre_processing"]

    def download_data(self):
        path = "../../../Dataset/TON_IOT/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv"
        dataset = pd.read_csv(path)

        logging.info(dataset["type"].value_counts())

        dataset.drop(columns=["src_ip", "dst_ip"], inplace=True)
        dataset.drop(columns=["dns_query"], inplace=True)
        dataset.drop(columns=["label"], inplace=True)
        check_invalid_data(df=dataset)
        logging.info("Dropping duplicates data ...")
        dataset.drop_duplicates(inplace=True)
        for column in dataset.columns:
            if column != "type":
                if dataset[column].dtype == "object":
                    encode_text_dummy(dataset, column)
                else:
                    if self.pre_processing == "z_score":
                        encode_numeric_zscore(dataset, name=column)
                    elif self.pre_processing == "min_max":
                        encode_numeric_min_max_scale(dataset, name=column)
            else:
                self.label_dict = encode_text_index(dataset, column)

        dataset.dropna(axis=1, inplace=True)
        y = dataset["type"].to_numpy()
        dataset.drop(labels="type", axis=1)

        logging.info("After dropping")
        check_invalid_data(df=dataset)
        logging.info(dataset["type"].value_counts())

        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            dataset.to_numpy(), y, test_size=0.2, random_state=42)

        self.train_data = self.train_data.astype(np.float32)
        self.test_data = self.test_data.astype(np.float32)

        del dataset
