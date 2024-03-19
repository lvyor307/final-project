import librosa
import torchaudio


def read_file(file_path: str):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate


def calculate_tempo_and_beats(file_path, sr):
    y, sr = librosa.load(file_path, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return tempo, beat_times
