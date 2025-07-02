import numpy as np
import librosa
from kymatio.torch import Scattering1D

def wst_features(audio_path, sr=22050, J=4, Q=6,T=32768):
    """
    parameters:
    audio_path (str): Path to the input audio file.
    sr (int): Sampling rate for loading the audio file. Default is 22050 Hz.
    J (int): Number of scales for the scattering transform. Default is 4.
    Q (int): Number of wavelets per octave. Default is 6.
    T (int): Length of the signal to be processed. Default is 32768 samples.

    returns:
    np.ndarray: Scattering transform representation of the audio file.

    This function computes the scattering transform of an audio file using the kymatio library.
    It loads the audio file, pads it to a specified length, and applies the scattering transform.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    y = np.pad(y, (0, T - len(y)), mode='constant')
    scattering = Scattering1D(J=J, shape=(T,), Q=Q)
    S = scattering(y)
    return S