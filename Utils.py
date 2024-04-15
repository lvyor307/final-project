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


def calculate_spectral_bandwidth(file_path, sr) -> dict:
    """
    Calculate the spectral bandwidth of an audio file
    :param file_path: Path to the audio file
    :param sr: Sample rate
    :return: Dictionary containing the mean spectral bandwidth
    """
    y, sr = librosa.load(file_path, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return {'spectral_bandwidth': np.mean(spectral_bandwidth)}


def calculate_spectral_contrast(file_path, sr) -> dict:
    """
    Calculate the spectral contrast of an audio file
    :param file_path: Path to the audio file
    :param sr: Sample rate
    :return: Dictionary containing the mean spectral contrast
    """
    y, sr = librosa.load(file_path, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    n_features = spectral_contrast.shape[0]
    features_names = [f'spectral_contrast_{i}' for i in range(n_features)]
    res = dict(zip(features_names, np.mean(spectral_contrast, axis=1)))
    return res


def calculate_spectral_flatness(file_path, sr) -> dict:
    """
    Calculate the spectral flatness of an audio file
    :param file_path: Path to the audio file
    :param sr: Sample rate
    :return: Dictionary containing the spectral flatness
    """
    y, sr = librosa.load(file_path, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    return {'spectral_flatness': np.mean(spectral_flatness)}


def calculate_spectral_rolloff(file_path, sr) -> dict:
    """
    Calculate the spectral rolloff of an audio file
    :param file_path: Path to the audio file
    :param sr: Sample rate
    :return: Dictionary containing the mean spectral rolloff
    """
    y, sr = librosa.load(file_path, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return {'spectral_rolloff': np.mean(spectral_rolloff)}


def calculate_zero_crossing_rate(file_path, sr) -> dict:
    """
    Calculate the zero crossing rate of an audio file
    :param file_path: Path to the audio file
    :param sr: Sample rate
    :return: Dictionary containing the mean zero crossing rate
    """
    y, sr = librosa.load(file_path, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    return {'zero_crossing_rate': np.mean(zero_crossing_rate)}


def calculate_rms_energy(file_path, sr) -> dict:
    """
    Calculate the RMS energy of an audio file
    :param file_path: Path to the audio file
    :param sr: Sample rate
    :return: Dictionary containing the RMS energy
    """
    y, sr = librosa.load(file_path, sr=sr)
    rms_energy = librosa.feature.rms(y=y)
    return {'rms_energy': np.mean(rms_energy)}


def apply_methods(target_file: pd.DataFrame, audio_files_list: list, methods: list) -> pd.DataFrame:
    """
    Apply a list of methods to a list of audio files and store the results in DataFrame
    :param target_file: a DataFrame containing the labels for the audio files
    :param audio_files_list: a list of audio file paths
    :param methods: a list of methods to apply to the audio files
    :return:
    """
    methods_dict = {
        'tempo_and_beats': calculate_tempo_and_beats,
        'spectral_centroid': calculate_spectral_centroid,
        'spectral_bandwidth': calculate_spectral_bandwidth,
        'spectral_contrast': calculate_spectral_contrast,
        'spectral_flatness': calculate_spectral_flatness,
        'spectral_rolloff': calculate_spectral_rolloff,
        'zero_crossing_rate': calculate_zero_crossing_rate,
        'rms_energy': calculate_rms_energy
    }

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
    res = pd.DataFrame(all_data)
    return res
