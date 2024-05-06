import os
import shutil
import random

def split_dataset(input_folder, train_folder, valid_folder, test_folder, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    # Create directories if they don't exist
    for folder in [train_folder, valid_folder, test_folder]:
        os.makedirs(folder, exist_ok=True)

    # Iterate through subfolders in the input folder
    for root, dirs, _ in os.walk(input_folder):
        for dir in dirs:
            dir_path = os.path.join(root, dir)

            # List all files in the subfolder
            files = [file for file in os.listdir(dir_path) if file.endswith('.wav')]
            random.shuffle(files)

            # Calculate split points
            train_split = int(len(files) * train_ratio)
            valid_split = int(len(files) * (train_ratio + valid_ratio))

            # Copy files to respective folders
            for i, file in enumerate(files):
                src = os.path.join(dir_path, file)
                if i < train_split:
                    dst = os.path.join(train_folder, dir, file)
                elif i < valid_split:
                    dst = os.path.join(valid_folder, dir, file)
                else:
                    dst = os.path.join(test_folder, dir, file)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)

# Example usage
input_folder = "/Users/aarontaub/Library/CloudStorage/Box-Box/Aaron-Personal/School/masters/Thesis/Datasets/VCTK/VCTK-Corpus/wav8-all"
train_folder = "/Users/aarontaub/Library/CloudStorage/Box-Box/Aaron-Personal/School/masters/Thesis/Datasets/VCTK/VCTK-Corpus/wav8-train"
valid_folder = "/Users/aarontaub/Library/CloudStorage/Box-Box/Aaron-Personal/School/masters/Thesis/Datasets/VCTK/VCTK-Corpus/wav8-valid"
test_folder = "/Users/aarontaub/Library/CloudStorage/Box-Box/Aaron-Personal/School/masters/Thesis/Datasets/VCTK/VCTK-Corpus/wav8-test"

split_dataset(input_folder, train_folder, valid_folder, test_folder)
