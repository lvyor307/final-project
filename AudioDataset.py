
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform, label


