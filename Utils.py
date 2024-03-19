import librosa
import numpy as np
import pandas as pd
import torchaudio


def read_file(file_path: str):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate


def calculate_tempo_and_beats(file_path, sr) -> dict:
    """
    Calculate the tempo and beats of an audio file
    :param file_path: a string representing the path to the audio file
    :param sr: sample rate
    :return: dictionary containing the tempo and mean beat time
    """
    y, sr = librosa.load(file_path, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return {'tempo': tempo, 'mean_beat_time': np.mean(beat_times)}


def calculate_spectral_centroid(file_path, sr) -> dict:
    """
    Calculate the spectral centroid of an audio file
    :param file_path: a string representing the path to the audio file
    :param sr: sample rate
    :return: dictionary containing the spectral centroid
    """
    y, sr = librosa.load(file_path, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return {'spectral_centroid': np.mean(spectral_centroid)}


def apply_methods(target_file: pd.DataFrame, audio_files_list: list, methods: list) -> pd.DataFrame:
    """
    Apply a list of methods to a list of audio files and store the results in DataFrame
    :param target_file: a DataFrame containing the labels for the audio files
    :param audio_files_list: a list of audio file paths
    :param methods: a list of methods to apply to the audio files
    :return:
    """
    methods_dict = {'tempo_and_beats': calculate_tempo_and_beats,
                    'spectral_centroid': calculate_spectral_centroid}

    all_data = []  # Initialize an empty list to store data for each file
    for file_path in audio_files_list:
        filename = file_path.split('/')[-1]
        target = target_file[target_file['filename'] == filename]['label'].values[0]  # Assuming one label per filename
        waveform, sample_rate = read_file(file_path)
        # Initialize a dictionary for this file's features
        features = {
            'filename': filename,
            'label': target,
            'mean_amplitude': waveform.mean().item(),
            'std_amplitude': waveform.std().item()
        }
        # Apply each method and update the features dictionary
        for method_name in methods:
            method = methods_dict[method_name]
            features.update(method(file_path, sample_rate))

        all_data.append(features)
        # Convert the list of dictionaries to a DataFrame
    ds_df = pd.DataFrame(all_data)
    return ds_df
