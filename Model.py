import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pykalman import KalmanFilter


class AudioLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AudioLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # shape = (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        return out


class FeaturesEngineering:
    def __init__(self, methods: list):
        self.methods = methods
        self.methods_dict = {'kalman_filter': self.apply_fourier_transform,
                              'fourier_transform': self.apply_kalman_filter}

    @staticmethod
    def apply_kalman_filter(signal):
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        # You may need to adjust the parameters based on your specific needs
        filtered_signal, _ = kf.filter(signal)
        return filtered_signal.flatten()

    @staticmethod
    def apply_fourier_transform(signal, sample_rate):
        freqs = np.fft.rfftfreq(len(signal), d=1 / sample_rate)
        fft_spectrum = np.fft.rfft(signal)
        amplitude_spectrum = np.abs(fft_spectrum)
        return freqs, amplitude_spectrum

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for method in self.methods:
            if method in self.methods_dict:
                X[method] = method(X)

