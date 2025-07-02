import os
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
from features.melspectrogram import melspectrogram

# Create output directory if it doesn't exist

def process_file(input_file_path, input_folder, output_folder):
    try:
        # Create the equivalent output path
        relative_path = os.path.relpath(os.path.dirname(input_file_path), input_folder)
        output_dir = os.path.join(output_folder, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        # Generate Mel spectrogram and save
        mel_spec = melspectrogram(input_file_path)
        if mel_spec is not None:
            output_file_path = os.path.join(output_dir, os.path.basename(input_file_path).replace('.wav', '.npy'))
            np.save(output_file_path, mel_spec)
            print(f"Saved Mel spectrogram: {output_file_path}")
    except Exception as e:
        print(f"Failed to process {input_file_path}: {e}")

def process_dataset_mel(input_folder, output_folder, max_workers=8):
    wav_files = []
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith('.wav'):
                wav_files.append(os.path.join(root, file_name))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, f, input_folder, output_folder) for f in wav_files]
        for _ in as_completed(futures):
            pass  # Results are printed inside process_file