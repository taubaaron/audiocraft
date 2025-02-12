import os
import torch
import torchaudio
import numpy as np
import shap
import matplotlib.pyplot as plt
from audiocraft.solvers import CompressionSolver

# --- Helper Functions ---
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

def normalize_mel_spectrogram(mel_spectrogram):
    """
    Normalizes a Mel-spectrogram to ensure values are within a reasonable range.
    """
    mel_spectrogram = np.maximum(mel_spectrogram, 1e-9)  # Avoid zeros
    mel_spectrogram = mel_spectrogram / mel_spectrogram.max()  # Scale to [0, 1]
    return mel_spectrogram

def mel_to_audio(mel_spectrogram, sample_rate=32000, n_fft=1024, hop_length=320, n_mels=128):
    """
    Converts a Mel-spectrogram back to a waveform.
    """
    mel_spectrogram = normalize_mel_spectrogram(mel_spectrogram)

    # Interpolate or pad to match `n_mels` if needed
    if mel_spectrogram.shape[1] != n_mels:
        mel_spectrogram = torch.nn.functional.interpolate(
            torch.tensor(mel_spectrogram).unsqueeze(0).unsqueeze(0),
            size=(mel_spectrogram.shape[0], n_mels),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0).numpy()

    # Inverse Mel-spectrogram transform
    mel_transform = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate
    )
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length)
    linear_spectrogram = mel_transform(torch.tensor(mel_spectrogram.T))
    waveform = griffin_lim(linear_spectrogram)
    return waveform

# --- Wrapper Class ---
class EnCodecRVQWrapper:
    """
    A wrapper around the EnCodec model to simulate forward passes for SHAP.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs).float()
        if inputs.dim() == 2:  # (batch_size, n_mels)
            inputs = mel_to_audio(inputs).unsqueeze(0).unsqueeze(0)  # Convert back to waveform
        with torch.no_grad():
            tokens, _ = self.model.encode(inputs)
        return tokens.cpu().numpy().flatten()

# --- Inference Workflow ---
def run_inference(input_dir, output_dir, model):
    """
    Processes all FLAC files in a directory, extracts tokens, and reconstructs audio.
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
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
                tokens, _ = model.encode(wav)
                reconstructed_audio = model.decode(tokens, None)

                # Save tokens and reconstructed audio
                torch.save(tokens, os.path.join(output_dir, f"{base_name}_tokens.pt"))
                torchaudio.save(
                    os.path.join(output_dir, f"{base_name}_reconstructed.wav"),
                    reconstructed_audio.squeeze(0),
                    model.sample_rate
                )

# --- SHAP Analysis ---
def prepare_background_data(mel_spectrogram, num_samples=10):
    """
    Prepares background data by sampling frames from the Mel-spectrogram.
    """
    return mel_spectrogram[:num_samples, :]  # Select first `num_samples` frames

def run_shap_analysis(model, mel_spectrogram, output_dir, file_name_prefix):
    """
    Runs SHAP analysis for the given Mel-spectrogram and saves the results.
    """
    print(f"Running SHAP for: {file_name_prefix}...")

    # Prepare background data
    background_data = prepare_background_data(mel_spectrogram)

    # Initialize SHAP explainer
    wrapper = EnCodecRVQWrapper(model)
    explainer = shap.KernelExplainer(wrapper, background_data)

    # Compute SHAP values
    shap_values = explainer.shap_values(background_data)

    # Save SHAP values
    shap_output_path = os.path.join(output_dir, f"{file_name_prefix}_shap_values.pt")
    torch.save(shap_values, shap_output_path)

    # Visualize SHAP overlayed on Mel-spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label="Log-Mel Amplitude")
    plt.title("SHAP Influence Overlay on Mel-Spectrogram")
    for i, shap_val in enumerate(shap_values):
        plt.plot(np.mean(shap_val, axis=1), label=f"SHAP Influence {i}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{file_name_prefix}_shap_overlay.png"))
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
                file_name_prefix = file.replace("_mel_spec.pt", "")
                run_shap_analysis(model, mel_spectrogram, os.path.join(output_dir, "shap_outputs"), file_name_prefix)

    print(f"{mode.capitalize()} complete.")

if __name__ == "__main__":
    main(mode="analysis")  # Change to "analysis" for SHAP analysis
