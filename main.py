import os

import pandas as pd
from torch.utils.data import DataLoader

from AudioDataset import AudioDataset

train_labels = pd.read_csv('compare22-KSF/lab/train.csv')
test_labels = pd.read_csv('compare22-KSF/lab/test.csv')
devel_labels = pd.read_csv('compare22-KSF/lab/devel.csv')

directory = 'compare22-KSF/wav'
all_files = os.listdir(directory)
train_wav_files = [file for file in all_files if file.endswith('.wav') and file.startswith('train')]
test_wav_files = [file for file in all_files if file.endswith('.wav') and file.startswith('test')]
devel_wav_files = [file for file in all_files if file.endswith('.wav') and file.startswith('devel')]

# Create an instance of the AudioDataset class
train_dataset = AudioDataset(train_wav_files, train_labels['label'].values.tolist())
test_dataset = AudioDataset(test_wav_files, test_labels['label'].values.tolist())
devel_dataset = AudioDataset(devel_wav_files, devel_labels['label'].values.tolist())

# Create a DataLoader instance to load batches of data
train_data_loader = DataLoader(train_dataset, batch_size=32)
test_data_loader = DataLoader(test_dataset, batch_size=32)
devel_data_loader = DataLoader(devel_dataset, batch_size=32)

