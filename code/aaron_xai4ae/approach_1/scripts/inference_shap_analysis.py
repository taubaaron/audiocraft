import os
import torch
import torchaudio
import numpy as np
import shap
import matplotlib.pyplot as plt
from audiocraft.solvers import CompressionSolver

# --- Helper Functions ---
def convert_flac_to_wav(input_dir, output_dir):
    """
    Converts all .flac files in the input directory to .wav files 
    and saves them in the output directory with the suffix '_raw.wav'.
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                input_path = os.path.join(root, file)
                output_file = os.path.splitext(file)[0] + "_raw.wav"
                output_path = os.path.join(output_dir, output_file)
                
                # Load .flac file
                wav, sr = torchaudio.load(input_path)
                
                # Save as .wav file
                if wav.ndim == 1:  # Mono audio
                    wav = wav.unsqueeze(0)  # Add channel dimension
                torchaudio.save(output_path, wav, sr)
                print(f"Converted: {input_path} -> {output_path}")

def compute_mel_spectrogram(waveform, sample_rate=32000, n_fft=1024, n_mels=128, hop_length=320):
    """
    Computes a Mel-spectrogram from the waveform.
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spectrogram = mel_transform(waveform).squeeze().numpy()  # Shape (n_mels, time_frames)
    return mel_spectrogram.T  # Transpose to (time_frames, n_mels)

def save_mel_spectrogram_image(mel_spectrogram, output_path, file_name):
    """
    Saves the Mel-spectrogram as an image for visualization.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label="Log-Mel Amplitude")
    plt.title("Mel-Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency Bins")
    plt.savefig(f"{output_path}/{file_name}_mel_spectrogram.png")
    plt.close()

class EnCodecRVQWrapper:
    """
    A wrapper around the EnCodec model to simulate forward passes for SHAP.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs).float()
        if inputs.dim() == 2:  # (batch_size, input_length)
            inputs = inputs.unsqueeze(1)  # Add channel dimension: (batch_size, 1, input_length)
        elif inputs.dim() == 3 and inputs.size(1) == 1:  # Ensure single-channel (batch_size, 1, input_length)
            pass
        else:
            raise ValueError(f"Unexpected input shape for model: {inputs.shape}")

        with torch.no_grad():
            tokens, _ = self.model.encode(inputs)  # Ensure model's output matches SHAP expectation
        return tokens.cpu().numpy().reshape(inputs.size(0), -1)  # Reshape to (batch_size, feature_size)

def prepare_background_data(wav, frame_size=1024, hop_length=512):
    """
    Prepares background data from a raw waveform by slicing it into overlapping frames.
    """
    wav = wav.squeeze(0)  # Remove batch dimension if present
    frames = []
    for start in range(0, wav.size(0) - frame_size + 1, hop_length):
        frame = wav[start : start + frame_size]
        frames.append(frame.numpy())
    return np.array(frames)  # Shape: (num_frames, frame_size)



# --- Inference Workflow ---
def run_inference(input_dir, output_dir, model):
    """
    Processes all FLAC files in a directory, extracts tokens, and reconstructs audio.
    """
    convert_flac_to_wav(input_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                print(f"Running inference for {file} ..")
                input_path = os.path.join(root, file)
                wav, sr = torchaudio.load(input_path)
                wav = torchaudio.transforms.Resample(sr, model.sample_rate)(wav)
                wav = wav.mean(dim=0, keepdim=True)  # Convert to mono

                # Compute Mel-spectrogram
                mel_spectrogram = compute_mel_spectrogram(wav, sample_rate=model.sample_rate)

                # Save Mel-spectrogram
                base_name = os.path.basename(file).replace(".flac", "")
                mel_spec_path = os.path.join(output_dir, f"{base_name}_mel_spec.pt")
                torch.save(mel_spectrogram, mel_spec_path)
                save_mel_spectrogram_image(mel_spectrogram, output_dir, base_name)

                # Encode and decode
                tokens, _ = model.encode(wav.unsqueeze(0))
                reconstructed_audio = model.decode(tokens, None)


                # Save tokens and reconstructed audio
                torch.save(tokens, os.path.join(output_dir, f"{base_name}_tokens.pt"))
                torchaudio.save(
                    os.path.join(output_dir, f"{base_name}_reconstructed.wav"),
                    reconstructed_audio.squeeze(0),
                    model.sample_rate
                )

def run_shap_analysis(model, wav, tokens, output_dir, file_name_prefix, frame_size=1024, hop_length=512):
    """
    Runs SHAP analysis for the given raw waveform and saves the results.
    """
    print(f"Running SHAP for: {file_name_prefix}...")

    # Prepare background data from raw waveform
    background_data = prepare_background_data(wav, frame_size, hop_length)
    print(f"Original background data shape: {background_data.shape}")

    # Reduce background data using K-means clustering
    from shap import kmeans
    background_data_reduced = kmeans(background_data, 5)  # Reduce to 5 clusters
    print(f"Reduced background data shape: {background_data_reduced.data.shape}")

    # Initialize SHAP explainer
    wrapper = EnCodecRVQWrapper(model)
    explainer = shap.KernelExplainer(wrapper, background_data_reduced.data)

    # Compute SHAP values
    test_data = background_data[:5]  # Use a subset of the original data for explanation
    shap_values = explainer.shap_values(test_data)  # Compute SHAP values
    print(f"Computed SHAP values.")

    # Save SHAP values
    shap_output_path = os.path.join(output_dir, f"{file_name_prefix}_shap_values.pt")
    torch.save(shap_values, shap_output_path)

    # Visualize SHAP overlayed on waveform
    plt.figure(figsize=(10, 4))
    plt.plot(wav.squeeze(0)[:1000].numpy(), label="Waveform")
    plt.plot(np.mean(shap_values, axis=0), label="SHAP Influence")
    plt.legend()
    plt.title("SHAP Influence on Raw Waveform")
    plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_shap_waveform.png"))
    plt.close()
    print(f"Saved SHAP analysis for {file_name_prefix}.")



# --- Main Function ---
def main(mode="inference"):
    print(f"Starting {mode} mode...")
    model = CompressionSolver.model_from_checkpoint("//pretrained/facebook/encodec_32khz")
    input_dir = "aaron_xai4ae/dataset/vctk_sub_dataset"
    output_dir = "aaron_xai4ae/results"

    if mode == "inference":
        run_inference(input_dir, os.path.join(output_dir, "inference_outputs"), model)
    elif mode == "analysis":
        for file in os.listdir(os.path.join(output_dir, "inference_outputs")):
            if file.endswith("_mel_spec.pt"):
                mel_spectrogram = torch.load(os.path.join(output_dir, "inference_outputs", file))
                wav_path = os.path.join(output_dir, "inference_outputs", file.replace("_mel_spec.pt", "_raw.wav"))
                wav, _ = torchaudio.load(wav_path)
                file_name_prefix = file.replace("_mel_spec.pt", "")
                tokens = torch.load(os.path.join(output_dir, "inference_outputs", file.replace("_mel_spec.pt", "_tokens.pt")))
                run_shap_analysis(model, wav, tokens, os.path.join(output_dir, "shap_outputs"), file_name_prefix)

    print(f"{mode.capitalize()} complete.")

if __name__ == "__main__":
    # main(mode="inference")  # Change to "analysis" for SHAP analysis
    main(mode="analysis")  # Change to "analysis" for SHAP analysis
