import os
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
from features import melspectrogram, gammatonegram, extract_mfcc_1d, stft_features, wst_features, cqt_features

# Create output directory if it doesn't exist

def process_file(input_file_path, input_folder, output_folder, mode="melspectrogram"):
    """
    parameters:
    input_file_path (str): Path to the input audio file.
    input_folder (str): Root folder containing the input audio files.
    output_folder (str): Root folder where the output files will be saved.
    mode (str): The type of feature extraction to perform. Options are:
        - "melspectrogram"
        - "gammatonegram"
        - "mfcc"
        - "stft"
        - "wst"
        - "cqt"

    returns:
    None

    This function processes a single audio file and saves the extracted features to the output folder.
    It creates the necessary directory structure in the output folder to mirror the input folder structure.
    If the input file is not a valid audio file or if an error occurs during processing,
    it will print an error message but continue processing other files.
    """
    try:
        # Create the equivalent output path
        relative_path = os.path.relpath(os.path.dirname(input_file_path), input_folder)
        output_dir = os.path.join(output_folder, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        # Generate Mel spectrogram and save
        if mode == "melspectrogram":
            spec = melspectrogram(input_file_path)
        elif mode == "gammatonegram":
            spec = gammatonegram(input_file_path)
        elif mode == "mfcc":
            spec = extract_mfcc_1d(input_file_path)
        elif mode == "stft":
            spec = stft_features(input_file_path)
        elif mode == "wst":
            spec = wst_features(input_file_path)
        elif mode == "cqt":
            spec = cqt_features(input_file_path)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if spec is not None:
            output_file_path = os.path.join(output_dir, os.path.basename(input_file_path).replace('.wav', '.npy'))
            np.save(output_file_path, spec)
            print(f"Saved Mel spectrogram: {output_file_path}")
    except Exception as e:
        print(f"Failed to process {input_file_path}: {e}")

def process_dataset_parallel(input_folder, output_folder, max_workers=8):
    """
    parameters:
    input_folder (str): Root folder containing the input audio files.
    output_folder (str): Root folder where the output files will be saved.
    max_workers (int): Maximum number of threads to use for parallel processing.

    returns:
    None

    This function processes all audio files in the input folder in parallel using multiple threads.
    It searches for all `.wav` files in the input folder and its subdirectories,
    processes each file to extract features, and saves the results in the output folder.
    The output folder structure mirrors the input folder structure.
    If an error occurs while processing a file, it will print an error message but continue processing
    other files.
    """
    wav_files = []
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith('.wav'):
                wav_files.append(os.path.join(root, file_name))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, f, input_folder, output_folder) for f in wav_files]
        for _ in as_completed(futures):
            pass  # Results are printed inside process_file

def process_dataset_sequential(input_folder, output_folder):
    """
    parameters:
    input_folder (str): Root folder containing the input audio files.
    output_folder (str): Root folder where the output files will be saved.

    returns:
    None

    This function processes all audio files in the input folder sequentially.
    It creates the necessary directory structure in the output folder to mirror the input folder structure.
    """
    wav_files = []
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith('.wav'):
                wav_files.append(os.path.join(root, file_name))

    for wav_file in wav_files:
        process_file(wav_file, input_folder, output_folder)