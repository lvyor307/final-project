import os

import pandas as pd
from torch.utils.data import DataLoader

import Utils
from AudioDataset import AudioDataset
from DescriptiveStatistics import DescriptiveStatistics
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

# Create an instance of the DescriptiveStatistics class and run the statistical tests and plots
# ds = DescriptiveStatistics(train_wav_files, test_wav_files, devel_wav_files)
# ds.run()
# Collect the features for the train and devel datasets
methods = ['tempo_and_beats', 'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast',
           'spectral_flatness', 'spectral_rolloff', 'zero_crossing_rate', 'rms_energy']
train_df = Utils.apply_methods(target_file=train_labels,
                               audio_files_list=train_wav_files,
                               methods=methods)
devel_df = Utils.apply_methods(target_file=devel_labels,
                               audio_files_list=devel_wav_files,
                               methods=methods)


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
    'num_layers': [4, 6],
    'num_classes': [train_labels['label'].nunique()],
    'criterion': [nn.CrossEntropyLoss()],
    'learning_rate': [0.001, 0.01],
    'optimizer': [optim.Adam],
    'num_epochs': [20, 30]
    }

hpt = HyperParameterTuning(param_grid=param_grid)
hpt.fit(train_data_loader, devel_data_loader)
hpt.print_best_model()
