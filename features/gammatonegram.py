import sys
import os
import numpy as np
import librosa
from nnAudio2 import Spectrogram

gammatonegram = Spectrogram.Gammatonegram(
    sr=1.46*22050,
    n_fft=1024,         # FFT size
    n_bins=95,          # Number of Gammatone filters (bins)
    hop_length=256,     # Hop length
    window='hann',      # Window type
    center=True,        # Center frames
    pad_mode='reflect', # Padding mode
    htk=False,          # Use HTK normalization (set to False here)
    fmin=18,          # Minimum frequency for Gammatone filter
    fmax=4186,          # Maximum frequency (None means Nyquist)
    norm=1,             # Normalization
    trainable_bins=False, # Non-trainable filter bins
    trainable_STFT=False, # Non-trainable STFT
    device='cpu'        # Force CPU usage instead of CUDA
)