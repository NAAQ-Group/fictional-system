import sys
import os
import numpy as np
import librosa
from nnAudio2 import Spectrogram

def gammatonegram(input_file_path):
    """
    parameters:
    input_file_path (str): Path to the input audio file.

    returns:
    np.ndarray: Gammatonegram representation of the audio.

    This function computes the Gammatonegram of an audio file using the nnAudio2 library.
    It loads the audio file, computes the Gammatonegram with specified parameters,
    and returns the result as a NumPy array. The Gammatonegram is a time
    frequency representation that is particularly useful for analyzing audio signals.
    The function uses a sampling rate of the audio file, a window size of 1024
    samples, 95 frequency bins, and a hop length of 256 samples.
    The Gammatonegram is computed with a minimum frequency of 18 Hz and a maximum
    frequency set to None, which means it will use the Nyquist frequency.
    The output is normalized and converted to a NumPy array for consistency.
    The function is designed to run on CPU, ensuring compatibility with systems
    that do not have CUDA support.
    If an error occurs during processing, it will print an error message and return None.
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