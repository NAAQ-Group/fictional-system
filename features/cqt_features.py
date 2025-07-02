import sys
import os
import numpy as np
import librosa

def cqt_features(file_path):
    """
    parameters:
    file_path (str): Path to the input audio file.

    returns:
    cqt_db (np.ndarray): The Constant-Q Transform (CQT) of the audio file, converted to decibels (dB).

    This function computes the Constant-Q Transform (CQT) of an audio file and converts it to decibels.
    The CQT is a time-frequency representation that is particularly useful for analyzing musical signals.
    The function uses a sampling rate of 1.46 times the standard sampling rate of 22050 Hz,
    which is approximately 32258 Hz, to capture a wide range of frequencies.
    The CQT is computed with a hop length of 256 samples, a minimum frequency of 18 Hz (C1 in the musical scale),
    and a total of 95 frequency bins, with 12 bins per octave.
    The Hann window is used for the CQT computation"""
    y, sr = librosa.load(file_path, sr=1.46*22050)
    cqt = librosa.cqt(
    y,
    sr=sr,
    hop_length=256,      # Number of samples between successive frames
    fmin=18,           # Minimum frequency (C1 in musical scale)           # Maximum frequency
    n_bins=95,           # Total number of frequency bins
    bins_per_octave=12,  # Number of bins per octave
    window="hann"        # Type of window
)

    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return cqt_db