import os
import torch
import torchaudio
import numpy as np
import shap
import matplotlib.pyplot as plt
from audiocraft.solvers import CompressionSolver
from sklearn.decomposition import PCA

# --- Helper Functions ---
def compute_spectrogram(waveform, sample_rate=32000, n_fft=1024, hop_length=320):
    """Computes a spectrogram (STFT) from the waveform."""
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
    spectrogram = spectrogram_transform(waveform).squeeze().numpy()
    return spectrogram.T  # Shape: (time_frames, frequency_bins)

def save_combined_plot(wav, sample_rate, spectrogram, shap_contributions, output_dir, file_name, speaker, codebook_num, chosen_code):
    """
    Saves a combined plot of the audio waveform, spectrogram, and SHAP contributions.
    """
    plt.figure(figsize=(10, 12))

    # Plot 1: Audio waveform
    plt.subplot(3, 1, 1)
    time = np.linspace(0, len(wav) / sample_rate, num=len(wav))
    plt.plot(time, wav)
    plt.title(f"Audio Waveform - {speaker.capitalize()}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot 2: Spectrogram
    plt.subplot(3, 1, 2)
    spectrogram_db = torchaudio.transforms.AmplitudeToDB()(torch.tensor(spectrogram))
    img = plt.imshow(spectrogram_db.T, aspect='auto', origin='lower', cmap='inferno', vmin=-80, vmax=0)
    plt.colorbar(img, orientation='horizontal', pad=0.2)
    plt.title(f"Spectrogram (Speaker: {speaker.capitalize()}, Codebook: {codebook_num}, Code: {chosen_code})")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")

    # Plot 3: SHAP contributions
    plt.subplot(3, 1, 3)
    
    # Align SHAP contributions to spectrogram time frames
    num_frames = spectrogram.shape[0]  # Number of time frames in the spectrogram
    samples_per_frame = len(wav) // num_frames  # Samples per spectrogram time frame
    aligned_shap_contributions = np.array([
        np.mean(shap_contributions[i * samples_per_frame: (i + 1) * samples_per_frame])
        for i in range(num_frames)
    ])

    plt.plot(aligned_shap_contributions, color="cyan", label="SHAP Influence")
    plt.title("SHAP Contributions")
    plt.xlabel("Time Frames")
    plt.ylabel("Mean SHAP Influence")
    plt.legend()

    combined_output_path = os.path.join(output_dir, f"{file_name}_combined_plot_{speaker}.png")
    plt.tight_layout()
    plt.savefig(combined_output_path)
    plt.close()
    print(f"Saved combined analysis as: {combined_output_path}")


from scipy.interpolate import interp1d
def aggregate_shap_values(shap_values_per_file, target_length=1000):
    """
    Aggregates SHAP values across multiple files by interpolating them to a common length.
    """
    aligned_shap_values = []

    for shap_values in shap_values_per_file:
        current_length = len(shap_values)
        if current_length == target_length:
            aligned_shap_values.append(shap_values)
        else:
            # Interpolate to target length
            x_original = np.linspace(0, 1, current_length)
            x_target = np.linspace(0, 1, target_length)
            interpolator = interp1d(x_original, shap_values, kind='linear')
            aligned_shap_values.append(interpolator(x_target))

    # Convert to numpy array for averaging
    aligned_shap_values = np.array(aligned_shap_values)
    return np.mean(aligned_shap_values, axis=0)

def prepare_background_data(wav, frame_size, num_samples=10):
    """Prepares background data by splitting the waveform into overlapping time slots."""
    slots = [wav[i:i + frame_size] for i in range(0, len(wav) - frame_size, frame_size // 2)]
    slots = [slot for slot in slots if len(slot) == frame_size]
    background_data = np.array(slots[:num_samples])
    from shap import kmeans
    background_data = kmeans(background_data, 5)  # Reduce to 5 clusters
    return background_data.data

def reduce_dimensionality(data, num_components=50):
    """Reduces the dimensionality of the input data using PCA."""
    max_components = min(data.shape[0], data.shape[1])
    num_components = min(num_components, max_components)
    pca = PCA(n_components=num_components)
    reduced_data = pca.fit_transform(data)
    print(f"Reduced data shape: {reduced_data.shape}")
    return reduced_data

# --- Wrapper Class ---
class EnCodecRVQWrapper:
    """A wrapper around the EnCodec model to focus on a specific codebook's output."""
    def __init__(self, model, codebook_num):
        self.model = model
        self.codebook_num = codebook_num

    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs).float()
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        with torch.no_grad():
            tokens, _ = self.model.encode(inputs)
            codebook_frame = tokens[:, self.codebook_num, 0]
        return codebook_frame.cpu().numpy()


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

def run_shap_analysis(model, wav, spectrogram, output_dir, file_name_prefix, speaker, codebook_num, frame_size):
    os.makedirs(output_dir, exist_ok=True)

    # Prepare background data
    background_data = prepare_background_data(wav, frame_size)
    background_data = reduce_dimensionality(background_data, num_components=50)  # Reduce to 50 features
    print(f"Background data shape after PCA: {background_data.shape}")

    wrapper = EnCodecRVQWrapper(model, codebook_num)
    explainer = shap.KernelExplainer(wrapper, background_data)

    # Run SHAP analysis
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

    # Save combined plots
    chosen_code = "N/A"  # Can be updated to show the chosen code in future updates
    save_combined_plot(wav, model.sample_rate, spectrogram, shap_contributions, output_dir, file_name_prefix, speaker, codebook_num, chosen_code)

def run_comparison(input_dir, output_dir, model):
    """
    Compares SHAP analysis for each audio file across male and female speakers.
    """
    speakers = {"p225": "female", "p226": "male"}
    frame_size = 1600

    for file in os.listdir(os.path.join(input_dir, "p225")):
        if not file.endswith(".flac"):
            continue

        # Extract the common audio file name without speaker prefix (e.g., "195_mic1")
        common_audio_name = "_".join(file.split("_")[1:])
        audio_name = file.replace(".flac", "")  # Full file name (e.g., "p225_195_mic1")
        comparison_dir = os.path.join(output_dir, "comparison", audio_name)
        os.makedirs(comparison_dir, exist_ok=True)

        for codebook_num in range(4):  # Codebook 1 to 4
            codebook_dir = os.path.join(comparison_dir, f"codebook_{codebook_num + 1}")
            os.makedirs(codebook_dir, exist_ok=True)

            for speaker_id, speaker_label in speakers.items():
                # Construct the file name with the correct speaker prefix
                audio_file = f"{speaker_id}_{common_audio_name}"
                audio_path = os.path.join(input_dir, speaker_id, audio_file)

                if not os.path.exists(audio_path):
                    print(f"Warning: File {audio_path} not found.")
                    continue

                # Load audio
                wav, sr = torchaudio.load(audio_path)
                wav = torchaudio.transforms.Resample(sr, model.sample_rate)(wav)
                wav = wav.mean(dim=0).numpy()  # Convert to mono
                spectrogram = compute_spectrogram(torch.tensor(wav), sample_rate=model.sample_rate)

                # Encode to get tokens and chosen codes
                wav_tensor = torch.tensor(wav).unsqueeze(0).unsqueeze(0)  # (1, 1, length)
                tokens, _ = model.encode(wav_tensor)
                chosen_code = tokens[0, codebook_num, 0].item()  # First time frame of chosen code

                print(f"Chosen code for {audio_file} (codebook {codebook_num + 1}): {chosen_code}")

                # Run SHAP analysis
                run_shap_analysis(
                    model, wav, spectrogram,
                    codebook_dir, f"{audio_name}_{speaker_label}_code_{chosen_code}",
                    speaker_label, codebook_num, frame_size
                )

# --- Aggregate Shap Values ---
def pad_or_truncate_shap_values(shap_values_list):
    """Pads or truncates SHAP contributions to have the same length."""
    max_length = max(len(shap) for shap in shap_values_list)
    uniform_shap_values = []

    for shap in shap_values_list:
        if len(shap) < max_length:
            # Pad with zeros
            padded_shap = np.pad(shap, (0, max_length - len(shap)), mode='constant')
        else:
            # Truncate to match max length
            padded_shap = shap[:max_length]
        uniform_shap_values.append(padded_shap)

    return np.array(uniform_shap_values)

def run_aggregate_shap_comparison(input_dir, output_dir, model, num_files=None):
    """Runs SHAP analysis and aggregates contributions for each codebook across selected audio files."""
    speakers = {"p225": "female", "p226": "male"}
    frame_size = 1600
    audio_files = os.listdir(os.path.join(input_dir, "p225"))

    if num_files is not None:
        audio_files = audio_files[:num_files]

    aggregated_shap_values_per_speaker = {speaker_id: {codebook_num: [] for codebook_num in range(4)} for speaker_id in speakers}

    for audio_file in audio_files:
        if not audio_file.endswith(".flac"):
            continue

        # Extract the common audio file name without speaker prefix (e.g., "195_mic1")
        common_audio_name = "_".join(audio_file.split("_")[1:])
        audio_name = audio_file.replace(".flac", "")

        for speaker_id, speaker_label in speakers.items():
            audio_path = os.path.join(input_dir, speaker_id, f"{speaker_id}_{common_audio_name}")
            if not os.path.exists(audio_path):
                print(f"Warning: File {audio_path} not found.")
                continue

            wav, sr = torchaudio.load(audio_path)
            wav = torchaudio.transforms.Resample(sr, model.sample_rate)(wav)
            wav = wav.mean(dim=0).numpy()  # Convert to mono
            spectrogram = compute_spectrogram(torch.tensor(wav), sample_rate=model.sample_rate)

            for codebook_num in range(4):
                # Encode to get tokens and chosen codes
                wav_tensor = torch.tensor(wav).unsqueeze(0).unsqueeze(0)
                tokens, _ = model.encode(wav_tensor)
                chosen_code = tokens[0, codebook_num, 0].item()

                background_data = prepare_background_data(wav, frame_size)
                background_data = reduce_dimensionality(background_data, num_components=50)

                wrapper = EnCodecRVQWrapper(model, codebook_num)
                explainer = shap.KernelExplainer(wrapper, background_data)

                shap_contributions = np.zeros(len(wav))
                for i in range(0, len(wav) - frame_size, frame_size // 2):
                    time_slot = wav[i:i + frame_size].reshape(1, -1)
                    shap_values = explainer.shap_values(time_slot)
                    contribution = np.mean(np.abs(shap_values))
                    shap_contributions[i:i + frame_size] += contribution

                aggregated_shap_values_per_speaker[speaker_id][codebook_num].append(shap_contributions)

    # Ensure uniform shapes and calculate average SHAP values
    for speaker_id, speaker_label in speakers.items():
        for codebook_num, shap_values_list in aggregated_shap_values_per_speaker[speaker_id].items():
            if len(shap_values_list) == 0:
                print(f"No SHAP values found for {speaker_label} (codebook {codebook_num + 1}). Skipping.")
                continue

            # Pad or truncate SHAP values to ensure uniform shape
            uniform_shap_values = pad_or_truncate_shap_values(shap_values_list)
            avg_shap_contributions = np.mean(uniform_shap_values, axis=0)

            aggregate_plot_path = os.path.join(output_dir, f"aggregate_plots/{speaker_id}/")
            os.makedirs(aggregate_plot_path, exist_ok=True)
            plot_aggregate_shap(avg_shap_contributions, aggregate_plot_path, codebook_num, speaker_label)



def plot_aggregate_shap(aggregate_shap_contributions, output_dir, codebook_num, speaker_label):
    """Plots the average SHAP contributions for each time slot for a specific speaker."""
    plt.figure(figsize=(12, 6))
    plt.plot(aggregate_shap_contributions, label=f"Average SHAP Contributions (Speaker: {speaker_label.capitalize()}, Codebook {codebook_num + 1})", color="cyan")
    plt.title(f"Average SHAP Contributions for {speaker_label.capitalize()} - Codebook {codebook_num + 1}")
    plt.xlabel("Time Frames")
    plt.ylabel("Average SHAP Influence")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"average_shap_contributions_{speaker_label}_codebook_{codebook_num + 1}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved average SHAP contribution plot: {plot_path}")

# --- Main Function ---
def main(mode="aggregate", num_files=5):
    print(f"Starting {mode} mode...")
    model = CompressionSolver.model_from_checkpoint("//pretrained/facebook/encodec_32khz")
    input_dir = "aaron_xai4ae/dataset/vctk_sub_dataset"
    output_dir = "aaron_xai4ae/results-time_slots"

    if mode == "inference":
        run_inference(input_dir, os.path.join(output_dir, "inference_outputs"), model)
    
    elif mode == "analysis":
        run_comparison(input_dir, os.path.join(output_dir, "shap_outputs"), model)  # No extra parameters
    
    elif mode == "aggregate":
        run_aggregate_shap_comparison(input_dir, output_dir, model, num_files)

    print(f"{mode.capitalize()} complete.")

if __name__ == "__main__":
    main(mode="aggregate", num_files=5)  # inference , analysis , aggregate
