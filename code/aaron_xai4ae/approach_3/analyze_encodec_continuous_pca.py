import os
import json
import torch
import torchaudio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from encodec import EncodecModel
import plotly.express as px
import pandas as pd
import random

# --- Configuration ---
AUDIO_DIR = "aaron_xai4ae/approach_3/results-attempt_2/vctk_sub_dataset-attempt_2"
# AUDIO_DIR = "aaron_xai4ae/common/dataset/vctk_sub_dataset"
OUTPUT_DIR = "aaron_xai4ae/approach_3/results-attempt_2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_AUDIOS_PER_SPEAKER = 10
NUM_PCA_ITERATIONS = 10  # Outer loop for PCA calculations

SPEAKER_GENDER_MAP = {
    "p236": "female", 
    "p238": "female", 
    "p248": "female", 
    "p364": "male", 
    "p374": "male", 
    "p376": "male",
}
# SPEAKER_GENDER_MAP = {
#     "p225": "female",
#     "p226": "male",
#     "p227": "male",
#     "p228": "female",
#     "p236": "female",
#     "p237": "male",
#     # Add more speakers as needed
# }


def create_color_map():
    """Creates a consistent color mapping for speakers, grouped by gender."""
    female_speakers = [spk for spk, gender in SPEAKER_GENDER_MAP.items() if gender == "female"]
    male_speakers = [spk for spk, gender in SPEAKER_GENDER_MAP.items() if gender == "male"]

    female_colors = ["#FF0000", "#B22222", "#FF7F50", "#FFA500", "#FFD700", "#FF6347", "#FF8C00", "#E9967A", "#DC143C", "#FF4500"]
    male_colors = ["#0000FF", "#1E90FF", "#00CED1", "#20B2AA", "#3CB371", "#32CD32", "#00FF7F", "#7FFFD4", "#87CEFA", "#4682B4" ]

    # Assign colors to speakers
    color_map = {}
    for i, spk in enumerate(female_speakers):
        color_map[spk] = female_colors[i]
    for i, spk in enumerate(male_speakers):
        color_map[spk] = male_colors[i]
    return color_map

COLOR_MAP = create_color_map()


def resample_audio(file_path, target_sample_rate=24000):
    """Preprocesses audio by resampling to the target sample rate."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
    return waveform


def extract_latent_representations(metadata, audio_dir, model, output_dir):
    """Extracts and caches latent representations from audio files."""
    os.makedirs(output_dir, exist_ok=True)
    cached_data = []

    for entry in metadata:
        file_path = os.path.join(audio_dir, entry["audio_path"])
        print(f"extracting latent representation for {file_path}")
        gender     = entry["gender"]
        speaker_id = entry["speaker_id"]

        # Preprocess audio
        waveform = resample_audio(file_path).to(DEVICE)

        # Extract latent representation
        with torch.no_grad():
            latent_rep = model.encoder(waveform.unsqueeze(0)).cpu().numpy()

        # Save to cache
        cache_entry = {
            "audio_path"            : entry["audio_path"],
            "latent_representation" : latent_rep.tolist(),
            "gender"                : gender,
            "speaker_id"            : speaker_id,
        }
        cached_data.append(cache_entry)

    # Save cache to file
    cache_file = os.path.join(output_dir, "latent_cache.json")
    with open(cache_file, "w") as f:
        json.dump(cached_data, f)

    return cached_data


def normalize_latents(cached_data, target_length=None, mode="pad"):
    """Normalizes latent representations to a consistent length.
    Args:
        cached_data (list): List of cached entries with latent representations.
        target_length (int): Target number of frames. If None, infer from data.
        mode (str): "pad" to pad shorter sequences, "trim" to truncate longer ones, or "mean" for aggregation.
    Returns:
        np.ndarray: Normalized latent representations."""
    latents = []
    for entry in cached_data:
        latent = np.array(entry["latent_representation"])  # (1, 128, frames)
        latent = latent.squeeze(0)  # Remove the first dimension -> (128, frames)

        if mode == "mean":
            # Aggregate over frames
            normalized_latent = latent.mean(axis=-1)  # (128,)
        else:
            # Determine target length
            if target_length is None:
                target_length = max(latent.shape[1] for entry in cached_data)

            if latent.shape[1] > target_length:  # Trim
                normalized_latent = latent[:, :target_length]
            elif latent.shape[1] < target_length:  # Pad
                pad_width = target_length - latent.shape[1]
                normalized_latent = np.pad(latent, ((0, 0), (0, pad_width)), mode="constant")
            else:
                normalized_latent = latent  # Already matches the target length
        latents.append(normalized_latent.flatten())  # Flatten for PCA
    return np.array(latents)


def load_metadata(audio_dir, num_clips_per_speaker):
    """Generates metadata for VCTK Corpus from directory structure."""
    metadata = []
    for speaker_id in os.listdir(audio_dir):
        speaker_path = os.path.join(audio_dir, speaker_id)
        if os.path.isdir(speaker_path):
            gender = SPEAKER_GENDER_MAP.get(speaker_id, "unknown")
            audio_files = [
                f for f in os.listdir(speaker_path) if f.endswith(".flac")
            ]
            random.shuffle(audio_files)  # Shuffle to randomize selection
            selected_files = audio_files[:num_clips_per_speaker]
            for audio_file in selected_files:
                metadata.append({
                    "audio_path": os.path.join(speaker_id, audio_file),
                    "speaker_id": speaker_id,
                    "gender": gender,
                })
    return metadata


def perform_pca(cached_data, n_components=2):
    """Applies PCA on latent representations and prepares speaker-specific labels."""
    latents = normalize_latents(cached_data, mode="mean")
    speaker_labels = [f"{entry['speaker_id']}-{entry['gender']}" for entry in cached_data]

    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)

    pca = PCA(n_components=n_components)
    latents_pca = pca.fit_transform(latents_scaled)

    return latents_pca, speaker_labels, pca


def plot_pca_2d(latents_pca, speaker_labels, pca, output_dir, iteration):
    """Visualizes PCA results with a scatter plot, color-coded by speaker."""
    plt.figure(figsize=(12, 10))
    unique_speakers = sorted(set(speaker_labels))  # Ensure consistent ordering
    speaker_to_color = {label.split("-")[0]: COLOR_MAP[label.split("-")[0]] for label in unique_speakers}
    for i, (x, y) in enumerate(latents_pca):
        speaker_id = speaker_labels[i].split("-")[0]
        plt.scatter(x, y, color=speaker_to_color[speaker_id], s=50)

    # Create a legend with one entry per speaker
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color=speaker_to_color[speaker_id], markersize=8, linestyle='',
                   label=f"{speaker_id} ({SPEAKER_GENDER_MAP[speaker_id]})")
        for speaker_id in speaker_to_color.keys()
    ]
    plt.legend(handles=legend_handles, title="Speakers", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f"PCA 2D (Iteration {iteration})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pca_2d_iter_{iteration}.png"))
    print(f"Explained Variance Ratio 2D: {pca.explained_variance_ratio_}")


def plot_pca_3d(latents_pca, speaker_labels, pca, output_dir, iteration):
    """Visualizes PCA results interactively in 3D using Plotly."""
    df = pd.DataFrame({
        'PCA Component 1': latents_pca[:, 0],
        'PCA Component 2': latents_pca[:, 1],
        'PCA Component 3': latents_pca[:, 2],
        'Speaker': speaker_labels
    })

    fig = px.scatter_3d(
        df,
        x='PCA Component 1', y='PCA Component 2', z='PCA Component 3',
        color='Speaker',
        color_discrete_map={f"{spk}-{SPEAKER_GENDER_MAP[spk]}": COLOR_MAP[spk] for spk in SPEAKER_GENDER_MAP.keys()},
        title=f"3D PCA (Iteration {iteration})"
    )
    fig.write_html(os.path.join(output_dir, f"pca_3d_iter_{iteration}.html"), include_plotlyjs="cdn")
    print(f"Explained Variance Ratio 3D: {pca.explained_variance_ratio_}")


def main(audio_dir, output_dir, device):
    for iteration in range(1, NUM_PCA_ITERATIONS + 1):
        print(f"Starting PCA Iteration {iteration}...")
        metadata = load_metadata(audio_dir, MAX_AUDIOS_PER_SPEAKER)

        print("Loading EnCodec model...")
        model = EncodecModel.encodec_model_24khz()
        model.eval().to(device)

        print("Extracting latent representations...")
        cached_data = extract_latent_representations(metadata, audio_dir, model, output_dir)

        print("Performing PCA 2D...")
        latents_pca, speaker_labels, pca = perform_pca(cached_data, n_components=2)
        print("Plotting PCA 2D results...")
        plot_pca_2d(latents_pca, speaker_labels, pca, output_dir, iteration)

        print("Performing PCA 3D...")
        latents_pca, speaker_labels, pca = perform_pca(cached_data, n_components=3)
        print("Plotting PCA 3D results...")
        plot_pca_3d(latents_pca, speaker_labels, pca, output_dir, iteration)

    print("All PCA iterations completed.")


if __name__ == "__main__":
    main(AUDIO_DIR, OUTPUT_DIR, DEVICE)
