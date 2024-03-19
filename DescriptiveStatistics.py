from typing import Union

import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
from scipy.stats import f_oneway, kruskal

import Utils


class DescriptiveStatistics:
    def __init__(self, train_wav_files: list = None, test_wav_files: list = None,
                 devel_wav_files: list = None):
        self.train_stats = None
        self.test_stats = None
        self.devel_stats = None
        self.train_wav_files = train_wav_files
        self.test_wav_files = test_wav_files
        self.devel_wav_files = devel_wav_files

    @staticmethod
    def collect(audio_files_list: list, target_file_name: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Collects statistics from the audio files and stores them in a DataFrame.
        :param audio_files_list: a list of audio file paths
        :param target_file_name: a csv file containing the labels for the audio files or string name of the file
        :return:
        """
        if isinstance(target_file_name, str):
            target_file = pd.read_csv(target_file_name)
        else:
            target_file = target_file_name

        ds_df = Utils.apply_methods(target_file, audio_files_list, methods=['tempo_and_beats'])
        return ds_df

    @staticmethod
    def plot_feature_distribution(feature_name: str, train_df: pd.DataFrame,
                                  test_df: pd.DataFrame, devel_df: pd.DataFrame):
        """
        Plots histograms and box plots for a specified feature across the train, test, and devel datasets.
        :param feature_name: The name of the feature to plot.
        :param train_df: The DataFrame containing the statistics for the train dataset.
        :param test_df: The DataFrame containing the statistics for the test dataset.
        :param devel_df: The DataFrame containing the statistics for the devel dataset.
        """
        # Create subplots: one row, two columns
        fig = sp.make_subplots(rows=1, cols=2,
                               subplot_titles=(f"Histogram of {feature_name}", f"Box Plot of {feature_name}"))

        # Data preparation
        datasets = {'Train': train_df, 'Test': test_df, 'Devel': devel_df}
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

    @staticmethod
    def plot_grouped_feature_distribution(feature_name, groupby_column, train_df: pd.DataFrame):
        """
        Plots box plots for a specified feature grouped by another column across the datasets.
        :param train_df: The DataFrame containing the statistics for the train dataset.
        :param feature_name: The name of the feature to plot.
        :param groupby_column: The name of the column to group by.
        """
        # Plot using Plotly Express
        fig = px.box(train_df, x=groupby_column, y=feature_name,
                     title=f"Distribution of {feature_name} grouped by {groupby_column}",
                     labels={groupby_column: groupby_column, feature_name: feature_name})
        fig.update_layout(showlegend=True)
        fig.show()

    @staticmethod
    def anova_test(feature_name: str, train_df: pd.DataFrame,
                   test_df: pd.DataFrame, devel_df: pd.DataFrame) -> float:
        """
        Performs a one-way ANOVA test to determine if there are significant differences
        in the means of a feature across the datasets.
        :param train_df: The DataFrame containing the statistics for the train dataset.
        :param test_df: The DataFrame containing the statistics for the test dataset.
        :param devel_df: The DataFrame containing the statistics for the devel dataset.
        :param feature_name: The name of the feature to test.
        :return: The p-value of the test.
        """
        sample1 = train_df[feature_name]
        sample2 = test_df[feature_name]
        sample3 = devel_df[feature_name]
        stat, p = f_oneway(sample1, sample2, sample3)
        print(f'ANOVA test for {feature_name}: F={stat}, p={p}')
        return p

    def kruskal_test_by_label(self, feature_name: str, train_df: pd.DataFrame) -> float:
        """
        Performs the Kruskal-Wallis H-test to determine if the distribution of a continuous feature
        is the same across different labels within a given dataset.
        :param train_df: The DataFrame containing the statistics for the train dataset.
        :param feature_name: The name of the feature to test across labels.
        """
        # Group data by label and collect the feature values for each group
        groups = train_df.groupby('label')[feature_name].apply(list).values

        # Perform Kruskal-Wallis test
        stat, p = kruskal(*groups)
        print(f'Kruskal-Wallis test for {feature_name}: H={stat}, p={p}')
        return p

    def run(self):
        """
        Run several methods.
        :return:
        """
        train_stats = self.collect(self.train_wav_files, target_file_name='compare22-KSF/lab/train.csv')
        test_stats = self.collect(self.test_wav_files, target_file_name='compare22-KSF/lab/test.csv')
        devel_stats = self.collect(self.devel_wav_files, target_file_name='compare22-KSF/lab/devel.csv')
        self.plot_feature_distribution('tempo', train_df=train_stats,
                                       test_df=test_stats, devel_df=devel_stats)
        self.anova_test('tempo', train_df=train_stats, test_df=test_stats, devel_df=devel_stats)
        self.kruskal_test_by_label('tempo', train_df=train_stats)
        self.plot_grouped_feature_distribution(feature_name='tempo', groupby_column='label', train_df=train_stats)
