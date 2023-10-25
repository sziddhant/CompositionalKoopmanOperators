import h5py
import numpy as np

# Replace with the path to your stat.h5 file
stat_file_path = "./data/data_Rope/train_last/0.rollout.h5"

# Open the HDF5 file for reading
with h5py.File(stat_file_path, 'r') as hf:
    print("Keys: %s" % hf.keys())
    for name in hf.keys():
        data = hf[name][:]
        print(f"Dataset name: {name}")
        print(data)