import sys
import os
import numpy as np
import librosa

def extract_mfcc_1d(file_path, n_mfcc=45):
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