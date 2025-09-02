import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np



class EnergyDataset(Dataset):
    def __init__(self,csv_path,input_window=168,output_window=24,split="train"):
        df =pd.read_csv(csv_path,parse_dates=["Datetime"], sep=",", on_bad_lines='skip')
        df = df.sort_values("Datetime")


        self.data = df["DUQ_MW"].values.astype(np.float32)
        self.input_window = input_window
        self.output_window = output_window


        # Normalization
        self.min_val = self.data.min()
        self.max_val = self.data.max()
        self.data = (self.data - self.min_val) / (self.max_val - self.min_val)

        split_point = int(len(self.data)*0.8)

        if split == "train":
            self.data = self.data[:split_point]
        else:
            self.data = self.data[split_point - input_window - output_window:]

    def __len__(self):
        return len(self.data) - self.input_window - self.output_window

    def __getitem__(self,idx):
        x = self.data[idx:idx + self.input_window]
        y = self.data[idx + self.input_window:idx + self.input_window + self.output_window]

        return torch.tensor(x).unsqueeze(1),torch.tensor(y)