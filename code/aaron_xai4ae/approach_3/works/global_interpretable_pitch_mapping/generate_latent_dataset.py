import os
import torch
import torchaudio
import numpy as np
from encodec import EncodecModel
from tqdm import tqdm

# --- Config ---
AUDIO_DIR = "aaron_xai4ae/common/dataset/vctk_sub_dataset"
OUTPUT_DIR = "aaron_xai4ae/approach_3/works/global_interpretable_pitch_mapping/results/latent_data"
PITCH_RANGE = list(range(-5, 6))
MAX_FILES_PER_SPEAKER = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helpers ---
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

# --- Main ---
def main():
    model = EncodecModel.encodec_model_24khz().to(DEVICE).eval()
    speakers = sorted(os.listdir(AUDIO_DIR))

    for speaker in speakers:
        speaker_path = os.path.join(AUDIO_DIR, speaker)
        if not os.path.isdir(speaker_path):
            continue

        print(f"🔊 Processing speaker: {speaker}")
        mic2_files = [f for f in sorted(os.listdir(speaker_path)) if f.endswith("mic2.flac")]

        for fname in tqdm(mic2_files[:MAX_FILES_PER_SPEAKER], desc=f"{speaker} files"):
            print(f"\tProcessing audio: {fname}")
            audio_id = fname.replace(".flac", "")
            path = os.path.join(speaker_path, fname)
            waveform = resample_audio(path)

            latents_all_shifts = []
            for shift in PITCH_RANGE:
                print(f"\tPitch shift: {shift:+d}")
                shifted = shift_pitch(waveform, shift)
                latents = extract_latents(model, shifted)
                avg_latent = latents.mean(axis=0)
                latents_all_shifts.append(avg_latent)
                torch.cuda.empty_cache()  # clear memory if using GPU

            latents_all_shifts = np.stack(latents_all_shifts)  # (41, 128)
            np.savez(os.path.join(OUTPUT_DIR, f"{audio_id}_latents.npz"), latents=latents_all_shifts)
            print(f"✅ Saved: {audio_id}_latents.npz")

    print("✅ Latent dataset saved.")

if __name__ == "__main__":
    main()
