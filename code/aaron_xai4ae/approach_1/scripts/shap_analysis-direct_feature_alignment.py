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
    Saves the Mel-spectrogram as an image with normalization and logarithmic scaling for better visualization.
    """
    # Normalize the spectrogram to a range of [0, 1]
    mel_spectrogram = mel_spectrogram / np.max(mel_spectrogram)

    # Apply a logarithmic scale to enhance contrast
    log_mel_spectrogram = np.log1p(mel_spectrogram)  # log(1 + x) to avoid log(0)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(log_mel_spectrogram.T, aspect='auto', origin='lower', cmap='magma')  # Use 'magma' for better contrast
    plt.colorbar(label="Log-Mel Amplitude")
    plt.title("Mel-Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency Bins")
    plt.savefig(f"{output_path}/{file_name}_mel_spectrogram.png")
    plt.close()


class EnCodecRVQWrapper:
    """
    A wrapper around the EnCodec model for direct feature alignment with SHAP.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs).float()

        # Reshape flat inputs back to (batch_size, 1, n_mels)
        if inputs.dim() == 2:  # Shape: (batch_size, n_mels)
            batch_size, num_features = inputs.shape
            n_mels = 128  # Assuming fixed n_mels
            inputs = inputs.view(batch_size, 1, n_mels)

        if inputs.dim() != 3:  # Model expects (batch_size, channels, input_length)
            raise ValueError(f"Invalid input shape: {inputs.shape}")

        with torch.no_grad():
            tokens, _ = self.model.encode(inputs)
        return tokens.cpu().numpy()


# --- Inference Workflow ---
def run_inference(input_dir, output_dir, model):
    """
    Processes all FLAC files in a directory, computes Mel-spectrograms, and encodes tokens.
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                print(f"Running inference for {file} ...")
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

# --- SHAP Analysis ---
def prepare_background_data(mel_spectrogram, num_samples=10):
    """
    Prepares background data directly from the Mel-spectrogram.
    Flattens the data to 2D for SHAP compatibility.
    """
    background_data = mel_spectrogram[:num_samples, :]  # Select the first `num_samples` frames
    return background_data.reshape(num_samples, -1)  # Flatten to (num_samples, n_mels)

def run_shap_analysis(model, mel_spectrogram, output_dir, file_name_prefix):
    """
    Runs SHAP analysis directly on the Mel-spectrogram and saves the results.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Running SHAP for: {file_name_prefix}...")

    # Prepare background data
    background_data = prepare_background_data(mel_spectrogram)
    print(f"Background data shape: {background_data.shape}")

    # Initialize SHAP explainer
    wrapper = EnCodecRVQWrapper(model)
    explainer = shap.KernelExplainer(wrapper, background_data)

    # Compute SHAP values
    shap_values = explainer.shap_values(background_data)

    # Save SHAP values
    shap_output_path = os.path.join(output_dir, f"{file_name_prefix}_shap_values.pt")
    torch.save(shap_values, shap_output_path)

    # Reshape SHAP values to align with spectrogram time frames
    shap_values_reshaped = np.mean(np.array(shap_values), axis=0)  # Averaging SHAP values
    shap_values_rescaled = shap_values_reshaped[:mel_spectrogram.shape[0]]  # Truncate to match time frames

    # Normalize Mel-spectrogram for better visualization
    mel_spectrogram_normalized = mel_spectrogram / (np.max(mel_spectrogram) + 1e-9)

    # Plotting SHAP overlay
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    # Plot normalized spectrogram
    img = ax1.imshow(
        mel_spectrogram_normalized.T,
        aspect="auto",
        origin="lower",
        cmap="plasma",  # Change colormap for better contrast
        interpolation="nearest",
    )
    ax1.set_title("SHAP Influence Overlay on Mel-Spectrogram")
    ax1.set_xlabel("Time Frames")
    ax1.set_ylabel("Mel Frequency Bins")
    cbar = fig.colorbar(img, ax=ax1, label="Normalized Amplitude")
    cbar.ax.set_ylabel("Linear Amplitude")

    # Plot SHAP influence
    shap_normalized = shap_values_rescaled / (np.max(np.abs(shap_values_rescaled)) + 1e-9)  # Normalize SHAP
    ax2.plot(shap_normalized, color="cyan", label="Normalized SHAP Influence", linewidth=2)
    ax2.set_ylabel("Normalized SHAP Influence", color="cyan")
    ax2.tick_params(axis="y", labelcolor="cyan")

    # Add legend and save
    ax2.legend(loc="upper right")
    output_image_path = os.path.join(output_dir, f"{file_name_prefix}_shap_overlay.png")
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"Saved SHAP overlay for {file_name_prefix} at {output_image_path}")

# --- Main Function ---
def main(mode="inference"):
    print(f"Starting {mode} mode...")
    model = CompressionSolver.model_from_checkpoint("//pretrained/facebook/encodec_32khz")
    input_dir = "aaron_xai4ae/dataset/vctk_sub_dataset"
    output_dir = "aaron_xai4ae/results-direct_feature_alignment"

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
    # main(mode="inference")  # Change to "analysis" for SHAP analysis
    main(mode="analysis")  # Change to "analysis" for SHAP analysis
