# How to run
## 1. Setup environment 
Setup requirement environments from https://github.com/NguyenQuangMinh0504/CIL_NIDS/blob/main/Code/requirements.txt
## 2. Install dataset
Install dataset, put them in the Dataset folder. For example: https://github.com/NguyenQuangMinh0504/CIL_NIDS/blob/main/Code/PythonCode/CIL/utils/cic_ids_2017_data.py
## 3. Run program
From this folder, run the scripts in script folder

```
python3 -m venv ./venv
source venv/bin/activate
./scripts/benchmark/nids_dataset/kdd99/kdd99_b0_2_der.sh
```