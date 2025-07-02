import numpy as np
import librosa

def stft_features(audio_path,sr=22050,n_fft=2024,hop_length=512):
    y, sr = librosa.load(audio_path, sr=sr)
    stft_matrix = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    stft_magnitude = np.abs(stft_matrix)
    stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    return stft_db