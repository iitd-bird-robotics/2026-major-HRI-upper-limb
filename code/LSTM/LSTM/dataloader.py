import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class PhaseDataset(Dataset):
    def __init__(self, data_path, seq_length, transform=None, label_map=None):
        self.seq_length = seq_length
        # self.transform = transform
        # self.label_map = label_map or {
        #     "walking": 0, 
        #     "running": 1, 
        #     "going_down": 2, 
        #     "going_up": 3, 
        #     "standing": 4
        # }

        # # Columns to read
        # columns_to_use = [
        #     'accelerometer_right_thigh_x', 'accelerometer_right_thigh_y', 'accelerometer_right_thigh_z',
        #     'gyroscope_right_thigh_x', 'gyroscope_right_thigh_y', 'gyroscope_right_thigh_z',
        #     'activity'
        # ]

        self.data = data_path

        # Normalize features (min-max normalization)
        for col in self.data.columns[:-1]:
            min_col = self.data[col].min()
            max_col = self.data[col].max()
            self.data[col] = (self.data[col] - min_col) / (max_col - min_col)

        self.features = self.data.iloc[:, :-1].values.astype(np.float32)
        self.labels = self.data.iloc[:, -1].values.astype(np.int64)

        # Adjust dataset length so sequences fit
        self.num_samples = len(self.features) - self.seq_length + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Sequence of shape [seq_length, input_size=6]
        x = self.features[idx:idx + self.seq_length]
        # Label: take label of the last step in the sequence
        y = self.labels[idx + self.seq_length - 1]
        try:
            if self.transform:
                x = self.transform(x)
        except:
            None

        return torch.tensor(x), torch.tensor(y)
