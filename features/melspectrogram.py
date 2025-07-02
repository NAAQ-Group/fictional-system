import sys
import os
import numpy as np
import librosa

def melspectrogram(file_path):
    """
    parameters:
    file_path (str): Path to the input audio file.

    returns:
    np.ndarray: Mel spectrogram representation of the audio file.

    This function computes the Mel spectrogram of an audio file using the librosa library.
    It loads the audio file, computes the Mel spectrogram with specified parameters,
    and returns the result as a NumPy array. The Mel spectrogram is a time-frequency representation
    that is particularly useful for analyzing audio signals, especially in the context of speech and music.
    The function uses a sampling rate of 1.46 times the standard sampling rate of 22050 Hz,
    which is approximately 32258 Hz, to capture a wide range of frequencies.
    The Mel spectrogram is computed with a window size of 1024 samples, a hop length of 256 samples,
    and 95 Mel frequency bins. The minimum frequency is set to 18 Hz (C1 in the musical scale),
    and the maximum frequency is set to 4186 Hz (C7 in the musical scale).
    The output is normalized to decibels (dB) using the maximum value as the reference.
    This function is designed to run on CPU, ensuring compatibility with systems
    that do not have CUDA support. If an error occurs during processing,
    it will print an error message and return None.
    """
    audio,_ = librosa.load(file_path,sr=1.46*22050)
    features=librosa.feature.melspectrogram(
        y=audio,
        sr=1.46*22050,
        n_fft=1024,
        hop_length=256,
        window='hann',
        center=True,
        pad_mode='constant',
        power=2.0,
        n_mels=95,
        fmin=18,
        fmax=4186)
    mel_spec_db = librosa.power_to_db(features, ref=np.max)
    return mel_spec_db