import numpy as np
import librosa
from kymatio.torch import Scattering1D

def wst_features(audio_path, sr=22050, J=4, Q=6,T=32768):
    y, sr = librosa.load(audio_path, sr=sr)
    y = np.pad(y, (0, T - len(y)), mode='constant')
    scattering = Scattering1D(J=J, shape=(T,), Q=Q)
    S = scattering(y)
    return S