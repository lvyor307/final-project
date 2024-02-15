import os

import pandas as pd
from torch.utils.data import DataLoader
from AudioDataset import AudioDataset
from HyperParameterTuning import HyperParameterTuning
import torch.nn as nn
import torch.optim as optim

train_labels = pd.read_csv('compare22-KSF/lab/train.csv').sort_values(by='filename')
test_labels = pd.read_csv('compare22-KSF/lab/test.csv').sort_values(by='filename')
devel_labels = pd.read_csv('compare22-KSF/lab/devel.csv').sort_values(by='filename')
# convert the test labels to nan
test_labels['label'] = float('nan')

directory = 'compare22-KSF/wav'
all_files = os.listdir(directory)
train_wav_files = sorted([os.path.join(directory, file) for file in all_files if file.startswith('train')])
devel_wav_files = sorted([os.path.join(directory, file) for file in all_files if file.startswith('devel')])
test_wav_files = sorted([os.path.join(directory, file) for file in all_files if file.startswith('test')])

# Create a label map dictionary for mapping the string labels to integers
label_map = {
    'Prolongation': 0,
    'no_disfluencies': 1,
    'Fillers': 2,
    'Block': 3,
    'Modified': 4,
    'SoundRepetition': 5,
    'WordRepetition': 6,
    'Garbage': 7
    }

train_labels['label'] = train_labels['label'].map(label_map)
devel_labels['label'] = devel_labels['label'].map(label_map)

# Create an instance of the AudioDataset class
train_dataset = AudioDataset(train_wav_files, train_labels['label'].values.tolist())
test_dataset = AudioDataset(test_wav_files, test_labels['label'].values.tolist())
devel_dataset = AudioDataset(devel_wav_files, devel_labels['label'].values.tolist())

# Create a DataLoader instance to load batches of data
train_data_loader = DataLoader(train_dataset, batch_size=32)
test_data_loader = DataLoader(test_dataset, batch_size=32)
devel_data_loader = DataLoader(devel_dataset, batch_size=32)

param_grid = {
    'input_size': [48000],
    'hidden_size':  [32, 64, 128],
    'num_layers': [2, 4, 6],
    'num_classes': [train_labels['label'].nunique()],
    'criterion': [nn.CrossEntropyLoss(), nn.BCELoss()],
    'learning_rate': [0.001, 0.01],
    'optimizer': [optim.Adam, optim.SGD],
    'num_epochs': [10, 15, 20, 25, 30]
    }

hpt = HyperParameterTuning(param_grid=param_grid)
hpt.fit(train_data_loader, train_data_loader)
aa = 1
