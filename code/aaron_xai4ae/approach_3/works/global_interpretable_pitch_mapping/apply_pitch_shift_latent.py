import os
import torch
import torchaudio
import numpy as np
import pickle
import matplotlib.pyplot as plt
from encodec import EncodecModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Config ---
AUDIO_FILE = "aaron_xai4ae/common/dataset/vctk_sub_dataset/p236/p236_004_mic2.flac"
SAVE_DIR = "aaron_xai4ae/approach_3/works/global_interpretable_pitch_mapping/results/output"
MODEL_DIR = "aaron_xai4ae/approach_3/works/global_interpretable_pitch_mapping/results/model"
TARGET_SHIFT = 2  # semitones
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Load Model ---
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler: StandardScaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "pca_model.pkl"), "rb") as f:
    pca: PCA = pickle.load(f)
splines = np.load(os.path.join(MODEL_DIR, "splines.npy"), allow_pickle=True)

# --- Helpers ---
def resample_audio(path):
    waveform, sr = torchaudio.load(path)
    if sr != 24000:
        waveform = torchaudio.transforms.Resample(sr, 24000)(waveform)
    return waveform

def extract_latents(model, waveform):
    with torch.no_grad():
        z = model.encoder(waveform.unsqueeze(0).to(DEVICE)).cpu().numpy()
    return z.squeeze(0).T.mean(axis=0)  # average over time

def decode_latents(model, latent):
    latent = torch.tensor(latent, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(DEVICE)
    latent = latent.repeat(1, 1, 100)  # fill temporal dim with constant vector
    with torch.no_grad():
        wav = model.decoder(latent).cpu()
    return wav.squeeze(0)

def visualize_pca_splines():
    os.makedirs(os.path.join(SAVE_DIR, "plots"), exist_ok=True)
    x = list(range(-20, 21))
    for c, splines_c in enumerate(splines[:5]):  # visualize first 5 PCA components
        plt.figure(figsize=(8, 4))
        for spline in splines_c[:10]:  # plot for first 10 samples
            y = [spline(xi) for xi in x]
            plt.plot(x, y, alpha=0.5)
        plt.title(f"PCA Component {c} Spline Curves")
        plt.xlabel("Pitch Shift (semitones)")
        plt.ylabel("Component Value")
        plt.grid(True)
        plt.savefig(os.path.join(SAVE_DIR, "plots", f"pca_component_{c}_splines.png"))
        plt.close()

# --- Main ---
def main():
    model = EncodecModel.encodec_model_24khz().to(DEVICE).eval()
    waveform = resample_audio(AUDIO_FILE)
    orig_latent = extract_latents(model, waveform)

    # Project to PCA space
    latent_scaled = scaler.transform(orig_latent.reshape(1, -1))
    latent_pca = pca.transform(latent_scaled).squeeze(0)

    # Predict new latent using average spline across samples
    new_pca = []
    for c, splines_c in enumerate(splines):
        all_vals = [spline(TARGET_SHIFT) for spline in splines_c]
        avg_val = np.mean(all_vals)
        new_pca.append(avg_val)

    new_pca = np.array(new_pca).reshape(1, -1)
    restored_scaled = pca.inverse_transform(new_pca)
    restored_latent = scaler.inverse_transform(restored_scaled).squeeze(0)

    output_audio = decode_latents(model, restored_latent)
    torchaudio.save(os.path.join(SAVE_DIR, f"shifted_{TARGET_SHIFT:+d}st.wav"), output_audio, 24000)
    print("✅ Audio saved.")

    visualize_pca_splines()
    print("📊 PCA spline plots saved.")

if __name__ == "__main__":
    main()