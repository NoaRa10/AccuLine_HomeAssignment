import torch
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, dataframe):

        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)  # Total samples

    def __getitem__(self, idx):
        # Get the ECG signal and label from the dataframe
        signal = self.dataframe.iloc[idx]['sample']
        label = self.dataframe.iloc[idx]['label']
        sample_id = self.dataframe.iloc[idx]['sample_id']

        # Ensure the signal is a numpy array
        signal = np.array(signal)

        # Reshape to (1, signal_length) for 1D CNN (add channel dimension)
        signal = np.expand_dims(signal, axis=0)  # shape becomes (1, length)

        # Convert to tensor
        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        sample_id = torch.tensor(sample_id, dtype=torch.long)

        return signal, label, sample_id