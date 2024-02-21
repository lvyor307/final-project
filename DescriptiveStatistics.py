import numpy as np
import pandas as pd
import os
import torchaudio
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

class DescriptiveStatistics:
    def __init__(self):
        self.train_stats = None
        self.test_stats = None
        self.devel_stats = None
    def collect(self, audio_files_list: list, target_file_name: str, attr_name: str):
        ds_df = pd.DataFrame(columns=['label', 'sample_rate', 'duration',
                                      'mean_amplitude', 'std_amplitude', 'tempo', 'beat_times'])
        target_file = pd.read_csv(target_file_name)
        for file_path in audio_files_list:
            filename = file_path.split('/')[-1]
            target = target_file[target_file['filename'] == filename]['label']
            waveform, sample_rate = self.read_file(file_path)
            tempo, beat_times = self.calculate_tempo_and_beats(file_path, sample_rate)
            tmp_df = pd.DataFrame({'label': target, 'sample_rate': sample_rate, 'duration': waveform.shape[1] / sample_rate,
                                   'mean_amplitude': waveform.mean().item(), 'std_amplitude': waveform.std().item(),
                                   'tempo': tempo, 'beat_times': np.mean(beat_times)})
            ds_df = pd.concat([ds_df, tmp_df])

        setattr(self, attr_name, ds_df)

    @staticmethod
    def read_file(file_path: str):
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate

    @staticmethod
    def calculate_tempo_and_beats(file_path, sr):
        y, sr = librosa.load(file_path, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        return tempo, beat_times

    def plot_feature_distribution(self, feature_name):
        """
        Plots histograms and box plots for a specified feature across the datasets.
        """
        plt.figure(figsize=(14, 6))

        # Histograms
        plt.subplot(1, 2, 1)
        sns.histplot(self.train_stats[feature_name], label='Train', kde=True, stat="density", linewidth=0)
        sns.histplot(self.test_stats[feature_name], label='Test', kde=True, stat="density", linewidth=0, color='red')
        sns.histplot(self.devel_stats[feature_name], label='Devel', kde=True, stat="density", linewidth=0,
                     color='green')
        plt.title(f'Histogram of {feature_name}')
        plt.xlabel(feature_name)
        plt.ylabel('Density')
        plt.legend()

        # Box Plots
        plt.subplot(1, 2, 2)
        sns.boxplot(
            data=[self.train_stats[feature_name], self.test_stats[feature_name], self.devel_stats[feature_name]],
            notched=True,
            meanline=True,
            showmeans=True,
            meanprops={"color": "red", "ls": "-", "lw": 2},
            medianprops={"visible": False},
            whiskerprops={"color": "black", "ls": "--"},
            showcaps=True,
            showbox=True,
            flierprops={"marker": "o", "markersize": 4},
            labels=['Train', 'Test', 'Devel'])
        plt.title(f'Box Plot of {feature_name}')
        plt.xlabel('Dataset')
        plt.ylabel(feature_name)

        plt.tight_layout()
        plt.show()

    def compare_feature_distributions(self, feature_name):
        """
        Performs a Mann-Whitney U test to compare feature distributions between train and devel datasets.
        """
        stat, p = mannwhitneyu(self.train_stats[feature_name], self.devel_stats[feature_name])
        print(
            f'Mann-Whitney U test for {feature_name} between Train and Devel datasets:\nStatistics={stat:.3f}, p={p:.3f}')
        if p > 0.05:
            print('Same distribution (fail to reject H0)')
        else:
            print('Different distribution (reject H0)')

directory = 'compare22-KSF/wav'
all_files = os.listdir(directory)
train_wav_files = sorted([os.path.join(directory, file) for file in all_files if file.startswith('train')])
devel_wav_files = sorted([os.path.join(directory, file) for file in all_files if file.startswith('devel')])
test_wav_files = sorted([os.path.join(directory, file) for file in all_files if file.startswith('test')])

ds = DescriptiveStatistics()
ds.collect(train_wav_files, target_file_name='compare22-KSF/lab/train.csv', attr_name='train_stats')
ds.collect(devel_wav_files, target_file_name='compare22-KSF/lab/train.csv', attr_name='test_stats')
ds.collect(train_wav_files, target_file_name='compare22-KSF/lab/train.csv', attr_name='devel_stats')
a = 1
