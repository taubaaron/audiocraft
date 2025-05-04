import os
import torch
import torchaudio
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import UnivariateSpline
import scipy.ndimage
from encodec import EncodecModel

# --- Config ---
AUDIO_FILE = "p236/p236_004_mic2.flac"
AUDIO_DIR = "aaron_xai4ae/approach_3/attempt_2/vctk_sub_dataset-attempt_2/"
OUTPUT_DIR = "aaron_xai4ae/approach_3/works/compare_shifting_stages/results_+0"
PITCH_RANGE = list(range(-20, 21))
TARGET_SHIFT = 0
NUM_COMPONENTS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def get_predicted_latents(all_latents, shift_idx, target_shift):
    n_frames = all_latents.shape[1]
    predicted_latents = []

    for f in range(n_frames):
        frame_vectors = all_latents[:, f, :]  # (shifts, 128)
        local_scaler = StandardScaler()
        scaled = local_scaler.fit_transform(frame_vectors)
        pca = PCA(n_components=NUM_COMPONENTS)
        pca_latents = pca.fit_transform(scaled)

        predicted = []
        for c in range(NUM_COMPONENTS):
            spline = UnivariateSpline(PITCH_RANGE, pca_latents[:, c], k=3, s=0)
            pred = spline(target_shift)
            predicted.append(pred)

        predicted = np.array(predicted).reshape(1, -1)
        restored = local_scaler.inverse_transform(pca.inverse_transform(predicted))
        predicted_latents.append(restored.squeeze(0))

    return np.stack(predicted_latents)

# --- Main ---
def main():
    model = EncodecModel.encodec_model_24khz().to(DEVICE).eval()
    audio_path = os.path.join(AUDIO_DIR, AUDIO_FILE)
    waveform_orig = resample_audio(audio_path)

    # Load all shifted latents
    all_latents = []
    for shift in PITCH_RANGE:
        shifted_waveform = shift_pitch(waveform_orig, shift)
        latents = extract_latents(model, shifted_waveform)
        all_latents.append(latents)
    all_latents = np.stack(all_latents)  # (shifts, frames, 128)

    # --- Option 1: shift first then do spline shift by 0 (stay at +2) ---
    opt1_dir = os.path.join(OUTPUT_DIR, "option_1_shift_first")
    os.makedirs(opt1_dir, exist_ok=True)
    waveform_shifted = shift_pitch(waveform_orig, TARGET_SHIFT)
    torchaudio.save(os.path.join(opt1_dir, f"pitch_shifted_input.wav"), waveform_shifted, 24000)

    latents_shifted = extract_latents(model, waveform_shifted)
    shift_idx = PITCH_RANGE.index(TARGET_SHIFT)
    pred_latents_1 = get_predicted_latents(all_latents, shift_idx, TARGET_SHIFT)
    smoothed_1 = smooth_latents(pred_latents_1)
    audio_1 = decode_latents(model, smoothed_1)
    torchaudio.save(os.path.join(opt1_dir, f"spline_0_at_plus2.wav"), audio_1.squeeze(0), 24000)

    # --- Option 2: no shift, then move to +2 via spline ---
    opt2_dir = os.path.join(OUTPUT_DIR, "option_2_shift_later")
    os.makedirs(opt2_dir, exist_ok=True)
    torchaudio.save(os.path.join(opt2_dir, f"original_input.wav"), waveform_orig, 24000)

    latents_orig = extract_latents(model, waveform_orig)
    shift_idx_0 = PITCH_RANGE.index(0)
    pred_latents_2 = get_predicted_latents(all_latents, shift_idx_0, TARGET_SHIFT)
    smoothed_2 = smooth_latents(pred_latents_2)
    audio_2 = decode_latents(model, smoothed_2)
    torchaudio.save(os.path.join(opt2_dir, f"spline_shift_to_plus2.wav"), audio_2.squeeze(0), 24000)

    # --- Option 3: manual shift + PCA cycle (no spline) ---
    opt3_dir = os.path.join(OUTPUT_DIR, "option_3_shift_then_pca_cycle")
    os.makedirs(opt3_dir, exist_ok=True)

    # Reuse shifted waveform from Option 1
    torchaudio.save(os.path.join(opt3_dir, f"pitch_shifted_input.wav"), waveform_shifted, 24000)

    latents = extract_latents(model, waveform_shifted)  # (frames, 128)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(latents)              # normalize
    pca = PCA(n_components=NUM_COMPONENTS)
    reduced = pca.fit_transform(scaled)                 # reduce
    restored = pca.inverse_transform(reduced)           # expand back
    restored = scaler.inverse_transform(restored)       # denormalize

    smoothed = smooth_latents(restored)
    reconstructed = decode_latents(model, smoothed)
    torchaudio.save(os.path.join(opt3_dir, f"manual_shift_plus_pca_cycle.wav"), reconstructed.squeeze(0), 24000)

    print("✅ All outputs saved!")

if __name__ == "__main__":
    main()
