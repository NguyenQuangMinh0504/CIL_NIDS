import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils.data import iData
from torchvision.transforms import ToTensor
from utils.helper import encode_text_index, encode_numeric_zscore


class CIC_IDS_2017(iData):
    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = [ToTensor()]

    def download_data(self):
        # path = "../../../Dataset/CIC-IDS-2017/Wednesday-workingHours.pcap_ISCX.csv"

        monday_working_path = "../../../Dataset/CIC-IDS-2017/Monday-WorkingHours.pcap_ISCX.csv"
        tuesday_working_hours_path = "../../../Dataset/CIC-IDS-2017/Tuesday-WorkingHours.pcap_ISCX.csv"
        wednesday_working_hours_path = "../../../Dataset/CIC-IDS-2017/Wednesday-workingHours.pcap_ISCX.csv"
        thursday_working_hours_morning_web_attacks_path = "../../../Dataset/CIC-IDS-2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
        thursday_working_hours_afternoon_infilteration_path = "../../../Dataset/CIC-IDS-2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
        friday_working_hours_morning_path = "../../../Dataset/CIC-IDS-2017/Friday-WorkingHours-Morning.pcap_ISCX.csv"
        friday_working_hours_afternoon_ddos_path = "../../../Dataset/CIC-IDS-2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        friday_working_hours_afternoon_port_scan_path = "../../../Dataset/CIC-IDS-2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

        monday_table = pd.read_csv(monday_working_path)
        tuesday_table = pd.read_csv(tuesday_working_hours_path)
        wednesday_table = pd.read_csv(wednesday_working_hours_path)
        thursday_morning_table = pd.read_csv(thursday_working_hours_morning_web_attacks_path)
        thurdays_afternoon_table = pd.read_csv(thursday_working_hours_afternoon_infilteration_path)
        friday_morning_table = pd.read_csv(friday_working_hours_morning_path)
        friday_afternoon_ddos_table = pd.read_csv(friday_working_hours_afternoon_ddos_path)
        friday_afternoon_port_scan_table = pd.read_csv(friday_working_hours_afternoon_port_scan_path)

        dataset = pd.concat(objs=[monday_table, tuesday_table, wednesday_table, thursday_morning_table, thurdays_afternoon_table,
                                  friday_morning_table, friday_afternoon_ddos_table, friday_afternoon_port_scan_table])

        # Drop 90% of benign traffic
        logging.info(dataset[" Label"].value_counts())
        logging.info(dataset[dataset[" Label"] == "BENIGN"].index)
        logging.info(len(dataset[dataset[" Label"] == "BENIGN"].index))
        dataset.drop(labels=dataset[dataset[" Label"] == "BENIGN"].index, inplace=True)

        dataset.drop(columns=[" Fwd Header Length.1"], inplace=True)  # duplicate of Fwd Header Length
        # drop unnecessary data
        # dataset.drop(columns=['Flow ID', ' Source IP', ' Source Port', ' Destination IP', ' Timestamp'], inplace=True)
        dataset.drop(labels=dataset[dataset[" Label"].isin(["Web Attack � Sql Injection", "Heartbleed", "Infiltration"])].index, inplace=True)
        logging.info(dataset[" Label"].value_counts())

        for column in dataset.columns:
            if column != " Label":
                encode_numeric_zscore(dataset, name=column)
            else:
                self.label_dict = encode_text_index(dataset, name=column)

        dataset.dropna(axis=1, inplace=True)

        y = dataset[" Label"].to_numpy()
        dataset.drop(labels=" Label", axis=1)

        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            dataset.to_numpy(), y, test_size=0.2, random_state=42)

        self.train_data = self.train_data.astype(np.float32)
        self.test_data = self.test_data.astype(np.float32)

        del dataset
