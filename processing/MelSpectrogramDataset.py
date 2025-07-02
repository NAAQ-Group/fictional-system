from torch.utils.data import Dataset
import os
import numpy as np
import torch

class MelSpectrogramDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.classes = sorted(os.listdir(data_dir))  # Get class names
        self.file_paths = []
        self.labels = []

        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)

            # Skip system folders like `.ipynb_checkpoints`
            if class_name == ".ipynb_checkpoints":
                continue

            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)

                # Check if it's a valid `.npy` file
                if file.endswith(".npy"):
                    self.file_paths.append(file_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        mel_spec = np.load(file_path)

        mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
        mel_spec = mel_spec.unsqueeze(0)  # Shape: (1, 193)# Convert to Tensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return mel_spec, label