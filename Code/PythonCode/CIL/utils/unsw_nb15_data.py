import pandas as pd
from torchvision.transforms import ToTensor
from utils.data import iData
from utils.helper import encode_text_dummy, encode_numeric_zscore, encode_text_index, encode_numeric_min_max_scale, check_invalid_data
from sklearn.model_selection import train_test_split
import numpy as np
import logging

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class UNSW_NB15(iData):

    def __init__(self, **kwargs):
        self.pre_processing = kwargs["pre_processing"]

    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = [ToTensor()]

    def download_data(self):
        path = "../../../Dataset/UNSW-NB15/UNSW_NB15_training-set.csv"
        dataset = pd.read_csv(path)
        logging.info(dataset["attack_cat"].value_counts())
        logging.info("Dropping column [label, id] ...")
        dataset.drop(columns=["id", "label"], inplace=True)
        check_invalid_data(df=dataset)
        logging.info("Drop duplicate data ...")
        dataset.drop_duplicates(inplace=True)

        for column in dataset.columns:
            if column in ["proto", "service", "state"]:
                encode_text_dummy(dataset, column)
            elif column == "attack_cat":
                encode_text_index(dataset, column)
            else:
                if self.pre_processing == "z_score":
                    encode_numeric_zscore(dataset, name=column)
                elif self.pre_processing == "min_max":
                    encode_numeric_min_max_scale(dataset, name=column)
                else:
                    raise Exception("Not implemented Normalization")
        logging.info("After processing")
        check_invalid_data(df=dataset)
        logging.info(dataset["attack_cat"].value_counts())
        y = dataset["attack_cat"].to_numpy()
        dataset.drop(labels="attack_cat", axis=1, inplace=True)
        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            dataset.to_numpy(), y, test_size=0.2, random_state=42)

        self.train_data = self.train_data.astype(np.float32)
        self.test_data = self.test_data.astype(np.float32)
        del dataset
