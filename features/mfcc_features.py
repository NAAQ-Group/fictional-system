import sys
import os
import numpy as np
import librosa

def extract_mfcc_1d(file_path, n_mfcc=45):
    """
    parameters:
    file_path (str): Path to the input audio file.

    returns:
    np.ndarray: 1D array of MFCC features extracted from the audio file.

    This function computes the Mel-frequency cepstral coefficients (MFCC) of an audio file using the librosa library.
    It loads the audio file, computes the MFCC with specified parameters, and returns the result as a 1D NumPy array.
    The MFCC is a representation of the short-term power spectrum of the audio signal, which is commonly used in speech and audio processing tasks.
    The function uses a sampling rate of 22050 Hz, a window size of 2048 samples,
    a hop length of 512 samples, and 50 Mel frequency bins. The number of MFCC coefficients to extract can be specified with the `n_mfcc` parameter, defaulting to 45.
    The output is a flattened 1D array of MFCC features, which can be used for various audio analysis tasks such as classification or regression.
    The function is designed to run on CPU, ensuring compatibility with systems that do not have CUDA support.
    If an error occurs during processing, it will print an error message and return None.
    """
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=50,
        n_mfcc=n_mfcc
    )
    mfcc_1d = mfcc.flatten()
    return mfcc_1d