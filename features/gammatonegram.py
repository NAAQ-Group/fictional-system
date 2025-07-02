import sys
import os
import numpy as np
import librosa
from nnAudio2 import Spectrogram

def gammatonegram(input_file_path):
    """
    Generate a Gammatonegram from an audio file.

    Parameters:
    input_file_path (str): Path to the input audio file.

    Returns:
    np.ndarray: Gammatonegram representation of the audio.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(input_file_path, sr=None)

        # Create a Gammatonegram instance
        gammatonegram = Spectrogram.Gammatonegram(
            sr=sr,
            n_fft=1024,
            n_bins=95,
            hop_length=256,
            window='hann',
            center=True,
            pad_mode='reflect',
            htk=False,
            fmin=18,
            fmax=None,  # None means Nyquist frequency
            norm=1,
            trainable_bins=False,
            trainable_STFT=False,
            device='cpu'  # Force CPU usage instead of CUDA
        )

        # Compute the Gammatonegram
        spec = gammatonegram(y)

        return spec.numpy()  # Convert to numpy array for consistency
    except Exception as e:
        print(f"Error processing {input_file_path}: {e}")
        return None