import os

import pandas as pd
from torch.utils.data import DataLoader

from AudioDataset import AudioDataset
from Model import AudioLSTM
import torch
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
label_map = {'Prolongation': 0,
             'no_disfluencies': 1,
             'Fillers': 2,
             'Block': 3,
             'Modified': 4,
             'SoundRepetition': 5,
             'WordRepetition': 6,
             'Garbage': 7}

train_wav_files['label'] = train_wav_files['label'].map(label_map)
devel_wav_files['label'] = devel_wav_files['label'].map(label_map)

# Create an instance of the AudioDataset class
train_dataset = AudioDataset(train_wav_files, train_labels['label'].values.tolist())
test_dataset = AudioDataset(test_wav_files, test_labels['label'].values.tolist())
devel_dataset = AudioDataset(devel_wav_files, devel_labels['label'].values.tolist())

# Create a DataLoader instance to load batches of data
train_data_loader = DataLoader(train_dataset, batch_size=32)
test_data_loader = DataLoader(test_dataset, batch_size=32)
devel_data_loader = DataLoader(devel_dataset, batch_size=32)

# input_size=sample rate×duration=16000×3=48000
input_size = 48000
hidden_size = 64
num_layers = 2
num_classes = train_labels['label'].nunique()

# Define the model architecture (AudioLSTM in this case)
model = AudioLSTM(input_size=input_size,
                  hidden_size=hidden_size,
                  num_layers=num_layers,
                  num_classes=num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the number of epochs
num_epochs = 10

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    # Set model to training mode
    model.train()

    # Iterate over the training dataset
    for inputs, labels in train_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

    # Validate the model on the development set
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for inputs, labels in devel_data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f"Epoch [{epoch + 1}/{num_epochs}], Development Accuracy: {accuracy:.4f}")

# Evaluate the trained model on the test set
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for inputs, labels in test_data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")
