import sys
import os
import numpy as np
import librosa

def cqt_features(file_path):
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