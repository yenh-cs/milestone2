"""
LSTM
"""
import torch
import torch.nn as nn


class LSTM(nn.Module):
    INPUT_SIZE = 1

    def __init__(self, hidden_size, predict_len, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(self.INPUT_SIZE, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, predict_len)

    def forward(self, x):
        # input shape: (num of seqs, time steps per seq, num of features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out[:, :, None]
