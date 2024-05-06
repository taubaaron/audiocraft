import os
import torchaudio
from concurrent.futures import ThreadPoolExecutor

def resample_wav(input_path, output_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(input_path)
    # Resample the audio file to 8kHz
    resampled_waveform = torchaudio.transforms.Resample(sample_rate, 8000)(waveform)
    # Save the resampled audio file
    torchaudio.save(output_path, resampled_waveform, 8000)
    print(f"saved file {output_path}")


def resample_wav_files(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate list of input-output paths
    input_output_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)
                output_subfolder = os.path.relpath(root, input_folder)
                output_subfolder_path = os.path.join(output_folder, output_subfolder)
                os.makedirs(output_subfolder_path, exist_ok=True)
                output_path = os.path.join(output_subfolder_path, file)
                input_output_paths.append((input_path, output_path))

    # Process files using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        executor.map(lambda x: resample_wav(*x), input_output_paths)


# Example usage
input_folder = "/Users/aarontaub/Library/CloudStorage/Box-Box/Aaron-Personal/School/masters/Thesis/Datasets/noises_1/audio"
output_folder = "/Users/aarontaub/Library/CloudStorage/Box-Box/Aaron-Personal/School/masters/Thesis/Datasets/noises_1/audio-8khz"
# input_folder = "/Users/aarontaub/Library/CloudStorage/Box-Box/Aaron-Personal/School/masters/Thesis/Datasets/VCTK/VCTK-Corpus/wav48"
# output_folder = "/Users/aarontaub/Library/CloudStorage/Box-Box/Aaron-Personal/School/masters/Thesis/Datasets/VCTK/VCTK-Corpus/wav8"

resample_wav_files(input_folder, output_folder)
