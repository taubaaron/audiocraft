# import os
# import torch
# import torchaudio
# import numpy as np
# import matplotlib.pyplot as plt
# import plotly.express as px
# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from encodec import EncodecModel
# import scipy.ndimage
# from scipy.interpolate import UnivariateSpline
# import json
# import random

# # --- Config ---
# NOTE_JSON_PATH = "aaron_xai4ae/approach_3/attempt_4/sine_wave/note_frequencies.json"
# OUTPUT_DIR = "aaron_xai4ae/approach_3/works/general_spline_functions/results"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_COMPONENTS = 20
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Load frequency values
# with open(NOTE_JSON_PATH, "r") as f:
#     note_freqs = json.load(f)
# sorted_notes = sorted(note_freqs.items(), key=lambda x: x[1])
# frequencies = [f for _, f in sorted_notes]

# # Save config
# config = {
#     "num_components": NUM_COMPONENTS,
#     "frequencies": frequencies,
#     "note_json_path": NOTE_JSON_PATH
# }
# with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
#     json.dump(config, f, indent=4)

# # --- Functions ---
# def generate_sine_wave(freq, duration=2.0, sr=24000):
#     t = np.linspace(0, duration, int(sr * duration), endpoint=False)
#     wave = 0.5 * np.sin(2 * np.pi * freq * t)
#     return torch.tensor(wave, dtype=torch.float32).unsqueeze(0)

# def extract_latents(model, waveform):
#     with torch.no_grad():
#         z = model.encoder(waveform.unsqueeze(0).to(DEVICE)).cpu().numpy()
#     return z.squeeze(0).T

# def decode_latents(model, latents):
#     latents = torch.tensor(latents.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         return model.decoder(latents).cpu()

# def smooth_latents(latents, sigma=1.0):
#     return scipy.ndimage.gaussian_filter1d(latents, sigma=sigma, axis=0)

# def plot_pca_components_2d_3d(latents_3d, freqs, new_point, output_dir, file_name):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(latents_3d[:, 0], latents_3d[:, 1], c=freqs, cmap='viridis', edgecolors='k')
#     plt.scatter(new_point[0], new_point[1], color='black', marker='X', s=100, label='Predicted Point')
#     plt.colorbar(label="Frequency (Hz)")
#     plt.xlabel("PC1")
#     plt.ylabel("PC2")
#     plt.title("2D PCA - Sine Wave Trajectory")
#     plt.grid()
#     plt.legend()
#     plt.savefig(os.path.join(output_dir, f"{file_name}_pca_2d.png"))
#     plt.close()

#     df = pd.DataFrame({
#         'PC1': latents_3d[:, 0],
#         'PC2': latents_3d[:, 1],
#         'PC3': latents_3d[:, 2],
#         'Frequency': freqs
#     })
#     fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Frequency', title="3D PCA - Sine Wave Trajectory")
#     fig.add_scatter3d(x=[new_point[0]], y=[new_point[1]], z=[new_point[2]],
#                       mode='markers', marker=dict(size=6, color='black', symbol='x'),
#                       name='Predicted Point')
#     fig.write_html(os.path.join(output_dir, f"{file_name}_pca_3d.html"))

# def plot_spline_fits(x_vals, component_values, target_x, output_dir):
#     fig, axs = plt.subplots(1, 3, figsize=(18, 5))
#     for i in range(3):
#         spline = UnivariateSpline(x_vals, component_values[:, i], k=3, s=0)
#         dense_x = np.linspace(min(x_vals), max(x_vals), 500)
#         axs[i].plot(x_vals, component_values[:, i], 'o', label='Data')
#         axs[i].plot(dense_x, spline(dense_x), '--', label='Spline')
#         axs[i].axvline(target_x, color='red', linestyle='--', label='Target')
#         axs[i].set_title(f"Spline Fit for PC{i+1}")
#         axs[i].legend()
#         axs[i].grid()
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"spline_fits_freq_{int(target_x)}.png"))
#     plt.close()

# # --- Main Execution ---
# def main():
#     model = EncodecModel.encodec_model_24khz().to(DEVICE).eval()

#     all_latents = []
#     for freq in frequencies:
#         print(f"Generating latent for sine wave {freq} Hz")
#         waveform = generate_sine_wave(freq)
#         latents = extract_latents(model, waveform)
#         all_latents.append(latents)

#     all_latents = np.stack(all_latents)  # (n_freqs, frames, 128)
#     n_freqs, n_frames, latent_dim = all_latents.shape

#     avg_latents = np.mean(all_latents, axis=1)  # (n_freqs, 128)
#     scaler = StandardScaler()
#     avg_scaled = scaler.fit_transform(avg_latents)
#     pca_for_plot = PCA(n_components=3)
#     pca_traj = pca_for_plot.fit_transform(avg_scaled)

#     # Random 10 shifts (one of them being identity)
#     target_idxs = random.sample(range(len(frequencies)), 9)
#     base_freq = 220

#     random.shuffle(target_idxs)

#     for target_idx in target_idxs:
#         target_freq = frequencies[target_idx]
#         shift_dir = os.path.join(OUTPUT_DIR, f"shift_{int(base_freq)}_to_{int(target_freq)}")
#         os.makedirs(shift_dir, exist_ok=True)

#         predicted_latents = []
#         first_spline_data = []

#         for f in range(n_frames):
#             frame_vectors = all_latents[:, f, :]
#             local_scaler = StandardScaler()
#             scaled = local_scaler.fit_transform(frame_vectors)
#             pca = PCA(n_components=NUM_COMPONENTS)
#             pca_latents = pca.fit_transform(scaled)

#             if f == 0:
#                 first_spline_data = pca_latents[:, :3]

#             predicted = []
#             for c in range(NUM_COMPONENTS):
#                 spline = UnivariateSpline(frequencies, pca_latents[:, c], k=3, s=0)
#                 pred = spline(target_freq)
#                 predicted.append(pred)

#             predicted = np.array(predicted).reshape(1, -1)
#             restored = local_scaler.inverse_transform(pca.inverse_transform(predicted))
#             predicted_latents.append(restored.squeeze(0))

#         predicted_latents = np.stack(predicted_latents)
#         smoothed = smooth_latents(predicted_latents, sigma=1.0)
#         reconstructed = decode_latents(model, smoothed)

#         audio_path = os.path.join(shift_dir, f"reconstructed_{int(base_freq)}_to_{int(target_freq)}.wav")
#         torchaudio.save(audio_path, reconstructed.squeeze(0), 24000)
#         print(f"✅ Saved reconstructed audio: {audio_path}")

#         predicted_avg = np.mean(predicted_latents, axis=0)
#         pred_point = pca_for_plot.transform(scaler.transform(predicted_avg.reshape(1, -1))).squeeze(0)
#         file_stub = f"sine_{int(base_freq)}_to_{int(target_freq)}"
#         plot_pca_components_2d_3d(pca_traj, frequencies, pred_point, shift_dir, file_stub)
#         plot_spline_fits(frequencies, first_spline_data, target_freq, shift_dir)

# if __name__ == "__main__":
#     main()




import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from encodec import EncodecModel
import scipy.ndimage
from scipy.interpolate import UnivariateSpline
import json

# --- Config ---
AUDIO_FILE = "440_sine_wav_shifted_0.wav"
AUDIO_DIR = "aaron_xai4ae/approach_3/attempt_3/pitch_shifts/sine_wave/"
OUTPUT_DIR = "aaron_xai4ae/approach_3/works/general_spline_functions/results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_COMPONENTS = 30
PITCH_RANGE = list(range(-20, 21))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save config
config = {
    "audio_file": AUDIO_FILE,
    "audio_dir": AUDIO_DIR,
    "num_components": NUM_COMPONENTS,
    "pitch_range": PITCH_RANGE
}
with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

# --- Functions ---
def resample_audio(path):
    waveform, sr = torchaudio.load(path)
    if sr != 24000:
        waveform = torchaudio.transforms.Resample(sr, 24000)(waveform)
    return waveform

def shift_pitch(waveform, semitones):
    return torchaudio.transforms.PitchShift(24000, semitones)(waveform)

def extract_latents(model, waveform):
    with torch.no_grad():
        z = model.encoder(waveform.unsqueeze(0).to(DEVICE)).cpu().numpy()
    return z.squeeze(0).T  # (frames, 128)

def decode_latents(model, latents):
    latents = torch.tensor(latents.T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return model.decoder(latents).cpu()

def smooth_latents(latents, sigma=1.0):
    return scipy.ndimage.gaussian_filter1d(latents, sigma=sigma, axis=0)

def plot_pca_components_2d_3d(latents_3d, pitch_shifts, new_point, output_dir, file_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(latents_3d[:, 0], latents_3d[:, 1], c=pitch_shifts, cmap='coolwarm', edgecolors='k')
    plt.scatter(new_point[0], new_point[1], color='black', marker='X', s=100, label='Predicted Point')
    plt.colorbar(label="Pitch Shift (Semitones)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D PCA - Pitch Shift Trajectory")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{file_name}_pca_2d.png"))
    plt.close()

    df = pd.DataFrame({
        'PC1': latents_3d[:, 0],
        'PC2': latents_3d[:, 1],
        'PC3': latents_3d[:, 2],
        'Pitch Shift': pitch_shifts
    })
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Pitch Shift', title="3D PCA - Pitch Shift Trajectory")
    fig.add_scatter3d(x=[new_point[0]], y=[new_point[1]], z=[new_point[2]],
                      mode='markers', marker=dict(size=6, color='black', symbol='x'),
                      name='Predicted Point')
    fig.write_html(os.path.join(output_dir, f"{file_name}_pca_3d.html"))

def plot_spline_fits(pitch_range, component_values, target_shift, output_dir):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i in range(3):
        spline = UnivariateSpline(pitch_range, component_values[:, i], k=3, s=0)
        dense_x = np.linspace(min(pitch_range), max(pitch_range), 500)
        axs[i].plot(pitch_range, component_values[:, i], 'o', label='Data')
        axs[i].plot(dense_x, spline(dense_x), '--', label='Spline')
        axs[i].axvline(target_shift, color='red', linestyle='--', label='Target Shift')
        axs[i].set_title(f"Spline Fit for PC{i+1}")
        axs[i].legend()
        axs[i].grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"spline_fits_shift_{target_shift}.png"))
    plt.close()

# --- Main Execution ---
def main():
    model = EncodecModel.encodec_model_24khz().to(DEVICE).eval()
    audio_path = os.path.join(AUDIO_DIR, AUDIO_FILE)
    waveform = resample_audio(audio_path)

    # Manual pitch shifting.
    all_latents = []
    for shift in PITCH_RANGE:
        print(f"Pitch shift: {shift}")
        shifted_waveform = shift_pitch(waveform, shift)
        latents = extract_latents(model, shifted_waveform)
        all_latents.append(latents)

    all_latents = np.stack(all_latents)  # (shifts, frames, 128) shifts = 41.
    n_shifts, n_frames, latent_dim = all_latents.shape

    # PCA over averaged latents for trajectory plotting. Only preparing here, used at the end of the code
    average_latents = np.mean(all_latents, axis=1)  # (shifts, 128). For each pitch we get and averaged vector size [1*128]
    scaler = StandardScaler()
    avg_scaled = scaler.fit_transform(average_latents) # Normalize
    pca_for_plot = PCA(n_components=3) # This "num_components" is just for the skae of plotting the latent space
    pca_traj = pca_for_plot.fit_transform(avg_scaled)

    pr = list(range(-10, 11))
    for TARGET_SHIFT in pr:
        shift_dir = os.path.join(OUTPUT_DIR, f"shift_{TARGET_SHIFT}")
        os.makedirs(shift_dir, exist_ok=True)

        predicted_latents = []
        first_spline_data = []

        for f in range(n_frames):
            frame_vectors = all_latents[:, f, :]  # shifts x encodeing = [41 * 128]
            local_scaler = StandardScaler()
            scaled = local_scaler.fit_transform(frame_vectors)
            pca = PCA(n_components=NUM_COMPONENTS)
            pca_latents = pca.fit_transform(scaled)

            if f == 0:
                first_spline_data = pca_latents[:, :3]  # save for spline plotting

            predicted = []
            for c in range(NUM_COMPONENTS):
                spline = UnivariateSpline(PITCH_RANGE, pca_latents[:, c], k=3, s=0)
                pred = spline(TARGET_SHIFT)  # predict latents for my desired shift
                predicted.append(pred)

            predicted = np.array(predicted).reshape(1, -1) # size 30
            restored = local_scaler.inverse_transform(pca.inverse_transform(predicted)) # 30 -> 128
            predicted_latents.append(restored.squeeze(0)) 

        predicted_latents = np.stack(predicted_latents)
        smoothed = smooth_latents(predicted_latents, sigma=1.0)
        reconstructed = decode_latents(model, smoothed)

        audio_path = os.path.join(shift_dir, f"reconstructed_pitch_{TARGET_SHIFT}.wav")
        torchaudio.save(audio_path, reconstructed.squeeze(0), 24000)
        print(f"✅ Saved reconstructed audio: {audio_path}")

        predicted_avg = np.mean(predicted_latents, axis=0)
        pred_point = pca_for_plot.transform(scaler.transform(predicted_avg.reshape(1, -1))).squeeze(0)
        file_stub = f"{AUDIO_FILE[5:-4]}_shift_{TARGET_SHIFT}"
        plot_pca_components_2d_3d(pca_traj, PITCH_RANGE, pred_point, shift_dir, file_stub)

        plot_spline_fits(PITCH_RANGE, first_spline_data, TARGET_SHIFT, shift_dir)

if __name__ == "__main__":
    main()
