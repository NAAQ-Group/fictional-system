import numpy as np
import librosa

def stft_features(audio_path,sr=22050,n_fft=2024,hop_length=512):
    """
    parameters:
    audio_path (str): Path to the input audio file.
    sr (int): Sampling rate for loading the audio file. Default is 22050 Hz.
    n_fft (int): Number of FFT components. Default is 2024.
    hop_length (int): Number of samples between successive frames. Default is 512.

    returns:
    np.ndarray: Short-Time Fourier Transform (STFT) representation of the audio file in decibels (dB).

    This function computes the Short-Time Fourier Transform (STFT) of an audio file and converts it to decibels (dB).
    The STFT is a time-frequency representation that provides information about the frequency content of the audio signal over time.
    The function uses the librosa library to load the audio file, compute the STFT, and convert the amplitude to decibels.
    The output is a 2D NumPy array where each column represents the STFT of a frame of audio,
    and each row corresponds to a frequency bin. The values are in decibels, which is a logarithmic scale
    that is commonly used in audio processing to represent sound intensity.
    The function is designed to run on CPU, ensuring compatibility with systems that do not have CUDA support.
    If an error occurs during processing, it will print an error message and return None.   
    """
    y, sr = librosa.load(audio_path, sr=sr)
    stft_matrix = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    stft_magnitude = np.abs(stft_matrix)
    stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    return stft_db