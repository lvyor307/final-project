import numpy as np
import pandas as pd
import os
import torchaudio
import librosa
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
from scipy.stats import f_oneway, kruskal

import Utils


class DescriptiveStatistics:
    def __init__(self, directory: str):
        self.train_stats = None
        self.test_stats = None
        self.devel_stats = None
        self.directory = directory

    def collect(self, audio_files_list: list, target_file_name: str, attr_name: str):
        ds_df = pd.DataFrame(columns=['label', 'sample_rate', 'duration',
                                      'mean_amplitude', 'std_amplitude', 'tempo', 'beat_times'])
        target_file = pd.read_csv(target_file_name)
        for file_path in audio_files_list:
            filename = file_path.split('/')[-1]
            target = target_file[target_file['filename'] == filename]['label']
            waveform, sample_rate = Utils.read_file(file_path)
            tempo, beat_times = Utils.calculate_tempo_and_beats(file_path, sample_rate)
            tmp_df = pd.DataFrame(
                {'label': target, 'sample_rate': sample_rate, 'duration': waveform.shape[1] / sample_rate,
                 'mean_amplitude': waveform.mean().item(), 'std_amplitude': waveform.std().item(),
                 'tempo': tempo, 'beat_times': np.mean(beat_times)})
            ds_df = pd.concat([ds_df, tmp_df])

        setattr(self, attr_name, ds_df)

    def plot_feature_distribution(self, feature_name):
        """
        Plots histograms and box plots for a specified feature across the train, test, and devel datasets.
        """
        # Create subplots: one row, two columns
        fig = sp.make_subplots(rows=1, cols=2,
                               subplot_titles=(f"Histogram of {feature_name}", f"Box Plot of {feature_name}"))

        # Data preparation
        datasets = {'Train': self.train_stats, 'Test': self.test_stats, 'Devel': self.devel_stats}
        colors = {'Train': 'blue', 'Test': 'red', 'Devel': 'green'}

        # Histograms
        for name, df in datasets.items():
            fig.add_trace(go.Histogram(x=df[feature_name], name=name, opacity=0.6, marker_color=colors[name]), row=1,
                          col=1)

        # Box plots
        for name, df in datasets.items():
            fig.add_trace(go.Box(y=df[feature_name], name=name, boxmean='sd', marker_color=colors[name]), row=1, col=2)

        # Update layout for aesthetics
        fig.update_layout(
            title_text=f"Distribution of {feature_name} Across Datasets",
            bargap=0.2,  # Gap between bars of adjacent location coordinates
        )

        # Update x-axis properties for the histogram
        fig.update_xaxes(title_text=feature_name, row=1, col=1)

        # Update y-axis properties for the histogram
        fig.update_yaxes(title_text="Count", row=1, col=1)

        # Update y-axis properties for the box plot
        fig.update_yaxes(title_text=feature_name, row=1, col=2)

        fig.show()

    def plot_grouped_feature_distribution(self, feature_name, groupby_column):
        """
        Plots box plots for a specified feature grouped by another column across the datasets.

        :param feature_name: The name of the feature to plot.
        :param groupby_column: The name of the column to group by.
        """
        # Plot using Plotly Express
        fig = px.box(self.train_stats, x=groupby_column, y=feature_name,
                     title=f"Distribution of {feature_name} grouped by {groupby_column}",
                     labels={groupby_column: groupby_column, feature_name: feature_name})
        fig.update_layout(showlegend=True)
        fig.show()

    def anova_test(self, feature_name: str) -> float:
        """
        Performs a one-way ANOVA test to determine if there are significant differences
        in the means of a feature across the datasets.

        :param feature_name: The name of the feature to test.
        :return: The p-value of the test.
        """
        sample1 = self.train_stats[feature_name]
        sample2 = self.test_stats[feature_name]
        sample3 = self.devel_stats[feature_name]
        stat, p = f_oneway(sample1, sample2, sample3)
        print(f'ANOVA test for {feature_name}: F={stat}, p={p}')
        return p

    def kruskal_test_by_label(self, feature_name: str) -> float:
        """
        Performs the Kruskal-Wallis H-test to determine if the distribution of a continuous feature
        is the same across different labels within a given dataset.

        :param feature_name: The name of the feature to test across labels.
        """
        # Retrieve the dataset
        dataset = self.train_stats

        # Group data by label and collect the feature values for each group
        groups = dataset.groupby('label')[feature_name].apply(list).values

        # Perform Kruskal-Wallis test
        stat, p = kruskal(*groups)
        print(f'Kruskal-Wallis test for {feature_name}: H={stat}, p={p}')
        return p

    def run(self):
        """
        Run several methods.
        :return:
        """
        all_files = os.listdir(self.directory)
        train_wav_files = sorted([os.path.join(self.directory, file) for file in all_files if file.startswith('train')])
        devel_wav_files = sorted([os.path.join(self.directory, file) for file in all_files if file.startswith('devel')])
        test_wav_files = sorted([os.path.join(self.directory, file) for file in all_files if file.startswith('test')])
        self.collect(train_wav_files, target_file_name='compare22-KSF/lab/train.csv', attr_name='train_stats')
        self.collect(test_wav_files, target_file_name='compare22-KSF/lab/test.csv', attr_name='test_stats')
        self.collect(devel_wav_files, target_file_name='compare22-KSF/lab/devel.csv', attr_name='devel_stats')
        self.plot_feature_distribution('tempo')
        self.anova_test('tempo')
        self.kruskal_test_by_label('tempo')
        self.plot_grouped_feature_distribution(feature_name='tempo', groupby_column='label')


if __name__ == '__main__':
    ds = DescriptiveStatistics(directory='compare22-KSF/wav')
    ds.run()
