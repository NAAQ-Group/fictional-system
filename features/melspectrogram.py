import sys
import os
import numpy as np
import librosa

def melspectrogram(file_path):
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