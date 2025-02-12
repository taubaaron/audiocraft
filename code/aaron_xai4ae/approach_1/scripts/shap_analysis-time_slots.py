import os
import torch
import torchaudio
import numpy as np
import shap
import matplotlib.pyplot as plt
from audiocraft.solvers import CompressionSolver

# --- Helper Functions ---
def compute_spectrogram(waveform, sample_rate=32000, n_fft=1024, hop_length=320):
    """ Computes a spectrogram (STFT) from the waveform """
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
    spectrogram = spectrogram_transform(waveform).squeeze().numpy()  # Shape (frequency_bins, time_frames)
    return spectrogram.T  # Transpose to (time_frames, frequency_bins)

def save_wav_plot(waveform, sample_rate, output_path, file_name):
    """ Saves a plot of the waveform """
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(waveform) / sample_rate, len(waveform)), waveform)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig(os.path.join(output_path, f"{file_name}_waveform.png"))
    plt.close()
    print(f"Saved spectrogram analysis as: {file_name}_waveform.png")

def save_combined_plot(wav, sample_rate, spectrogram, shap_contributions, output_dir, file_name):
    """ Saves a combined plot of the audio waveform, spectrogram, and SHAP contributions """
    plt.figure(figsize=(10, 12))

    # Plot audio waveform
    plt.subplot(3, 1, 1)
    time = np.linspace(0, len(wav) / sample_rate, num=len(wav))
    plt.plot(time, wav)
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot spectrogram
    plt.subplot(3, 1, 2)
    spectrogram_db = torchaudio.transforms.AmplitudeToDB()(torch.tensor(spectrogram))
    plt.imshow(spectrogram_db.T, aspect='auto', origin='lower', cmap='inferno', vmin=-80, vmax=0)
    plt.colorbar(label="Spectrogram Amplitude", orientation='horizontal', pad=0.2)
    plt.title("Spectrogram")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")

    # Plot SHAP contributions
    if shap_contributions.ndim == 1:
        shap_contributions = shap_contributions.reshape(spectrogram.shape)  # Ensure correct shape
    plt.subplot(3, 1, 3)
    shap_overlay = np.mean(shap_contributions, axis=1)  # Averaging over frequency bins
    plt.plot(shap_overlay, color="cyan", label="SHAP Influence")
    plt.title("SHAP Contributions")
    plt.xlabel("Time Frames")
    plt.ylabel("Mean SHAP Influence")
    plt.legend()

    # Save the combined plot
    combined_output_path = os.path.join(output_dir, f"{file_name}_combined_plot.png")
    plt.tight_layout()
    plt.savefig(combined_output_path)
    plt.close()
    print(f"Saved combined analysis as: {file_name}_combined_plot.png")


# --- Wrapper Class ---
class EnCodecRVQWrapper:
    """ A wrapper around the EnCodec model to focus on the first codebook's output for one frame """
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs).float()
        if inputs.dim() == 2:  # (batch_size, time_length)
            inputs = inputs.unsqueeze(1)  # Add channel dimension (batch_size, 1, time_length)
        with torch.no_grad():
            tokens, _ = self.model.encode(inputs)
            first_codebook_frame = tokens[:, 0, 0]  # Focus on the first codebook for one frame
        return first_codebook_frame.cpu().numpy()  # Shape: (batch_size,)


# --- Inference Workflow ---
def run_inference(input_dir, output_dir, model):
    """
    Processes all FLAC files in a directory, extracts tokens, and reconstructs audio.
    """
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                print(f"Processing {file=}")
                input_path = os.path.join(root, file)
                wav, sr = torchaudio.load(input_path)
                wav = torchaudio.transforms.Resample(sr, model.sample_rate)(wav)
                wav = wav.mean(dim=0, keepdim=True)  # Convert to mono
                wav = wav.squeeze().numpy()

                base_name = os.path.basename(file).replace(".flac", "")
                spectrogram = compute_spectrogram(torch.tensor(wav), sample_rate=model.sample_rate)
                torch.save(spectrogram, os.path.join(output_dir, f"{base_name}_spectrogram.pt"))
                save_wav_plot(wav, model.sample_rate, output_dir, base_name)
                
                # Reconstruct and save output audio
                wav_tensor = torch.tensor(wav).unsqueeze(0).unsqueeze(0)
                tokens, _ = model.encode(wav_tensor)
                reconstructed_audio = model.decode(tokens, None)
                torchaudio.save(
                    os.path.join(output_dir, f"{base_name}_reconstructed.wav"),
                    reconstructed_audio.squeeze(0),
                    model.sample_rate
                )

# --- SHAP Analysis ---
                
from sklearn.decomposition import PCA
def reduce_dimensionality(data, num_components=50):
    """
    Reduces the dimensionality of the input data using PCA.
    Automatically adjusts num_components to fit the sample size.
    """
    max_components = min(data.shape[0], data.shape[1])
    if num_components > max_components:
        print(f"Reducing number of components from {num_components} to {max_components} due to sample size.")
        num_components = max_components

    pca = PCA(n_components=num_components)
    reduced_data = pca.fit_transform(data)
    print(f"Reduced data shape: {reduced_data.shape}")
    return reduced_data


def prepare_background_data(wav, frame_size, num_samples=10):
    """
    Prepares background data by splitting the waveform into overlapping time slots.
    """
    slots = [wav[i:i + frame_size] for i in range(0, len(wav) - frame_size, frame_size // 2)]
    slots = [slot for slot in slots if len(slot) == frame_size]  # Ensure full slots only
    background_data = np.array(slots[:num_samples])

    from shap import kmeans
    # After preparing the background data
    background_data = kmeans(background_data, 5)  # Reduce to 5 clusters

    return background_data.data

def run_shap_analysis(model, wav, spectrogram, output_dir, file_name_prefix, frame_size):
    """
    Runs SHAP analysis per time slot and visualizes contributions.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prepare background data and reduce its dimensionality
    background_data = prepare_background_data(wav, frame_size)
    background_data = reduce_dimensionality(background_data, num_components=50)  # Reduce to 50 features
    print(f"Background data shape after PCA: {background_data.shape}")

    # Initialize SHAP KernelExplainer with reduced background data
    wrapper = EnCodecRVQWrapper(model)
    explainer = shap.KernelExplainer(wrapper, background_data)

    # Perform SHAP analysis with fewer slots
    shap_contributions = np.zeros(len(wav))
    for i in range(0, len(wav) - frame_size, frame_size // 2):
        time_slot = wav[i:i + frame_size].reshape(1, -1)
        shap_values = explainer.shap_values(time_slot)
        contribution = np.mean(np.abs(shap_values))
        shap_contributions[i:i + frame_size] += contribution

    shap_contributions /= np.max(np.abs(shap_contributions))  # Normalize

    print(f"Spectrogram shape: {spectrogram.shape}")
    print(f"SHAP contributions shape (before reshape): {shap_contributions.shape}")
    # Resize contributions to match the spectrogram
    shap_contributions = np.resize(shap_contributions, spectrogram.shape)

    # Save and plot results
    torch.save(shap_contributions, os.path.join(output_dir, f"{file_name_prefix}_shap_contributions.pt"))
    save_combined_plot(wav, model.sample_rate, spectrogram, shap_contributions, output_dir, file_name_prefix)

    
# --- Main Function ---x
def main(mode="inference"):
    print(f"Starting {mode} mode...")
    model = CompressionSolver.model_from_checkpoint("//pretrained/facebook/encodec_32khz")
    input_dir = "aaron_xai4ae/dataset/vctk_sub_dataset"
    output_dir = "aaron_xai4ae/approach_1/results-time_slots"

    if mode == "inference":
        run_inference(input_dir, os.path.join(output_dir, "inference_outputs"), model)
    elif mode == "analysis":
        for file in os.listdir(os.path.join(output_dir, "inference_outputs")):
            if file.endswith("_spectrogram.pt"):
                spectrogram = torch.load(os.path.join(output_dir, "inference_outputs", file))
                wav_path = os.path.join(output_dir, "inference_outputs", file.replace("_spectrogram.pt", "_reconstructed.wav"))
                wav, _ = torchaudio.load(wav_path)
                wav = wav.squeeze().numpy()
                file_name_prefix = file.replace("_spectrogram.pt", "")
                run_shap_analysis(model, wav, spectrogram, os.path.join(output_dir, "shap_outputs"), file_name_prefix, frame_size=1600)

    print(f"{mode.capitalize()} complete.")

if __name__ == "__main__":
    main(mode="analysis")  # Change to "analysis" for SHAP analysis
