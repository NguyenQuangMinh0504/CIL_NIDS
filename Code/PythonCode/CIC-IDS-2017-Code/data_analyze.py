import pandas as pd

path = "/Users/nguyenquangminh/Desktop/Thesis/Dataset/CICDataset/CIC-IDS-2017/TrafficLabelling /Wednesday-workingHours.pcap_ISCX.csv"

dataset = pd.read_csv(filepath_or_buffer=path)


def get_statistic(column_name: str):
    print(column_name, "............")
    print(dataset[column_name])
    print(len(dataset[column_name].unique()))
    print(dataset[column_name].unique())
    print(dataset[column_name].describe())


# EDA
print(dataset)
print(dataset.columns)
print(f"Total features: {len(dataset.columns)}")
# Flow ID
print("Flow ID ...........")
print(dataset["Flow ID"])
print(dataset["Flow ID"].describe())
# Source Ip  -> spoof
# Too many IP ?
get_statistic(" Source IP")
# Source Port -> multiple port ?
print("Source port...........")
print(dataset[" Source Port"])
print(dataset[" Source Port"].unique())
print(dataset[" Source Port"].describe())
# Destination IP
# Too many IP ?
get_statistic(" Destination IP")
# Label
print("Label ..................")
print(dataset[" Label"])
print(dataset[" Label"].unique())
print(dataset[" Label"].describe())
# Destination Port
get_statistic(" Destination Port")
# Protocol
get_statistic(" Protocol")
print(dataset[" Protocol"].value_counts(normalize=False, sort=True))
# Checking more about Protocol of attack.
print((dataset[dataset[" Label"] == "BENIGN"])[" Protocol"].value_counts(normalize=False, sort=True))
print((dataset[dataset[" Label"] == "DoS slowloris"])[" Protocol"].value_counts(normalize=False, sort=True))
print((dataset[dataset[" Label"] == "DoS Slowhttptest"])[" Protocol"].value_counts(normalize=False, sort=True))
print((dataset[dataset[" Label"] == "DoS Hulk"])[" Protocol"].value_counts(normalize=False, sort=True))
print((dataset[dataset[" Label"] == "DoS GoldenEye"])[" Protocol"].value_counts(normalize=False, sort=True))

# Timestamp
get_statistic(" Timestamp")

# Flow duration
get_statistic(" Flow Duration")
